
/* TbonCL.cc
   Temporal Branch-on-Need tree (T-BON) - curvilinear grid implementation
   Packages/Philip Sutton
   June 1999

  Copyright (C) 2000 SCI Group, University of Utah
*/

#include "TbonTreeCL.h"
#include "Clock.h"
#include "TriGroup.h"

#include <Core/Util/NotFinished.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Geom/Material.h>
#include <Core/Thread/CrowdMonitor.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>

namespace Phil {
using namespace SCIRun;
using namespace std;

typedef float type;

class TbonCL : public Module {

public:

  // Functions required by Module inheritance
  TbonCL(const clString& id);
  virtual ~TbonCL();
  virtual void execute();
  virtual void tcl_command(TCLArgs& args, void* userdata);

protected:

private:
  TbonTreeCL<type>* tbon;
  GeomTriGroup* surface;

  Material* matl;
  char** treefiles;
  char** datafiles;

  GuiString metafilename;
  GuiInt nodebricksize, databricksize;
  GuiDouble isovalue;
  GuiInt timevalue;
  GuiInt resolution;

  double miniso, maxiso;
  double res;
  int timesteps;
  double curriso;
  int currtime;
  int currres;

  GeometryOPort* geomout;
  CrowdMonitor* crowdmonitor;

  void processQuery( );
  void preprocess(ifstream& metafile);

}; // class TbonCL


// More required stuff...
extern "C" Module* make_TbonCL(const clString& id){
  return new TbonCL(id);
}


// Constructor
TbonCL::TbonCL(const clString& id) 
  : Module("TbonCL", id, Filter), metafilename("metafilename",id,this),
  nodebricksize("nodebricksize",id,this), 
  databricksize("databricksize",id,this), isovalue("isovalue",id,this),
  timevalue("timevalue",id,this), resolution("resolution",id,this)
{
  timevalue.set(0);
  isovalue.set(0);
  nodebricksize.set(1024);
  databricksize.set(2048);
  resolution.set(0);
  curriso = -1.0;
  currtime = -1;
  currres = -1;
  tbon = 0;

  matl =  new Material( Color(0,0,0),
			Color(0.25,0.75,0.75),
			Color(0.8,0.8,0.8),
			10 );
  crowdmonitor = new CrowdMonitor("TbonCL crowdmonitor");

  geomout = new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport( geomout );
}

// Destructor
TbonCL::~TbonCL(){
}

// Execute - set up tree, process first isovalue query
void 
TbonCL::execute()
{
  char filetype[20];
  char geomfile[80];
  char treesmeta[80];
  char trees[80];
  ifstream metafile( metafilename.get()(), ios::in );
  if( !metafile ) {
    cerr << "Error: cannot open file " << metafilename.get() << endl;
    return;
  }

  init_clock();
  metafile >> filetype;
  if( strcmp( filetype, "PREPROCESS" ) == 0 ) {

    // do preprocess
    cout << "Preprocessing . . ." << endl;
    preprocess( metafile );
    cout << "Done." << endl;

  } else if( strcmp( filetype, "EXECUTE" ) == 0 ) {

    // read parameters from metafile
    metafile >> treesmeta;
    metafile >> geomfile;
    metafile >> trees;
    metafile >> miniso >> maxiso >> res;
    metafile >> timesteps;

   // create list of files
    ifstream files( trees, ios::in );
    if( !files ) {
      cerr << "Error: cannot open file " << trees << endl;
      return;
    }

    treefiles = new char*[timesteps];
    datafiles = new char*[timesteps];
    for( int i = 0; i < timesteps; i++ ) {
      treefiles[i] = new char[80];
      datafiles[i] = new char[80];
      files >> treefiles[i] >> datafiles[i];
    }
    files.close();

    // recreate tree skeleton in memory
    tbon = new TbonTreeCL<type>( treesmeta, geomfile );

    // update UI to reflect current values
    TCL::execute( id + " updateFrames");
    /////////////////
    // uncomment for immediate full resolution
    resolution.set( tbon->getDepth() );
    // uncomment for immediate 1/2 resolution
    //    resolution.set( (int)(tbon->getDepth() / 2) );
    /////////////////

    processQuery();
  } else {
    cerr << "Error: metafile type not recognized" << endl;
  } 
} // execute


// tcl_command - routes changes from the UI
//    update:  time or isovalue slider was changed - recompute isosurface
//    getVars: metafile changed - get extreme values of iso, time
void
TbonCL::tcl_command(TCLArgs& args, void* userdata) {

  if( args[1] == "update" ) {
    if( tbon != 0 ) {
      // recompute isosurface
      processQuery();
    }
  } else if( args[1] == "getVars" ) {
    // construct list of variable values, as strings
    char var[80];
    sprintf(var,"%lf",miniso);
    clString svar = clString( var );
    clString result = svar;
    
    sprintf(var,"%lf",maxiso);
    svar = clString( var );
    result += clString( " " + svar );
    sprintf(var,"%lf",res);
    svar = clString( var );
    result += clString( " " + svar );
    sprintf(var,"%d",timesteps-1);
    svar = clString( var );
    result += clString( " " + svar );

    sprintf(var,"%d",tbon->getDepth());
    svar = clString( var );
    result += clString( " " + svar );

    // return result
    args.result( result );
  } else {
    // message not for us - propagate up
    Module::tcl_command( args, userdata );
  }
}


// preprocess - set up data structures
void
TbonCL::preprocess( ifstream& metafile ) {
  char filename[80];
  char geomfile[80];
  char newgeomfile[80];
  char treesmeta[80];
  char treebase[80];
  int numzones, zone;
  int* nx;
  int* ny;
  int* nz;
  int* c;

  metafile >> filename;
  metafile >> geomfile;
  metafile >> newgeomfile;
  metafile >> treesmeta;
  metafile >> treebase;

  ifstream datafiles( filename, ios::in );
  if( !datafiles ) {
    cerr << "Error: cannot open file " << filename << endl;
    return;
  }

  metafile >> numzones;
  nx = new int[numzones];
  ny = new int[numzones];
  nz = new int[numzones];
  c = new int[numzones];
  for( zone = 0; zone < numzones; zone++ ) {
    char circularity[80];

    metafile >> circularity;
    metafile >> nx[zone] >> ny[zone] >> nz[zone];

    if( !strcmp( circularity, "NONCIRCULAR" ) ) {
      c[zone] = TbonTreeCL<type>::NONCIRCULAR;
    } else if ( !strcmp( circularity, "CIRCULAR_X" ) ) {
      c[zone] = TbonTreeCL<type>::CIRCULAR_X;
    } else if ( !strcmp( circularity, "CIRCULAR_Y" ) ) {
      c[zone] = TbonTreeCL<type>::CIRCULAR_Y;
      cout << "Notice: CIRCULAR_Y not supported!" << endl;
    } else if ( !strcmp( circularity, "CIRCULAR_Z" ) ) {
      c[zone] = TbonTreeCL<type>::CIRCULAR_Z;
      cout << "Notice: CIRCULAR_Z not supported!" << endl;
    } else {
      cerr << "Error: undefined circularity" << endl;
      return;
    }
  }

  // create T-BON tree
  tbon = new TbonTreeCL<type>( nx, ny, nz, geomfile, numzones, c, 
			       nodebricksize.get(), databricksize.get() );
  
  // fill in time steps
  int i = 0;
  while( datafiles >> filename ) {
    char newfilename[80];
    // read data from disk
    tbon->readData( filename );
    // construct T-BON tree and save to disk
    tbon->fillTree( );
    datafiles >> newfilename;
    tbon->writeTree( treesmeta, treebase, i++, newfilename, newgeomfile );
  }

  // clean up
  delete tbon;
}

// processQuery - called whenever the time/iso value changes
//  processes a single (time,iso) query
void
TbonCL::processQuery() {
  static int id = -1;
  static int gotasurface = 0;
  iotimer_t t0, t1;

  //0 = read_time();
  timevalue.reset();
  isovalue.reset();
  resolution.reset();

  if( timevalue.get() == currtime && isovalue.get() == curriso &&
      resolution.get() == currres ) {
    // do nothing - nothing has changed
  } else {
#if 1
    curriso = isovalue.get();
    currres = resolution.get();
    int res = tbon->getDepth() - currres;

    int havemonitor = gotasurface;
    if( havemonitor )
      crowdmonitor->writeLock();

    // calculate new surface
    if( timevalue.get() != currtime ) {
      currtime = timevalue.get();
      surface = tbon->search( curriso, treefiles[currtime], 
			      datafiles[currtime], 
			      TbonTreeCL<type>::TIME_CHANGED, res );    
    } else {
      surface = tbon->search( curriso, treefiles[currtime],
			      datafiles[currtime], 
			      TbonTreeCL<type>::TIME_SAME, res );
    }
    
    if( surface != 0 )
      gotasurface = 1;

    // send surface to output port
    if( havemonitor )
      crowdmonitor->writeUnlock();
    if( gotasurface && id == -1 ) {
      id = geomout->addObj( new GeomMaterial( surface, matl ), "Geometry",
			    crowdmonitor );
    }
    if( gotasurface ) {
      geomout->flushViews();
    }
#endif
#if 0
  currres = tbon->getDepth();
  //    for( curriso = 1.10; curriso < 1.46; curriso += 0.10 ) {
  for( currtime = 0; currtime < 10; currtime += 1 ) {
    for( curriso = 0.995; curriso < 1.005; curriso += 0.001 ) {
      t0 = read_time();
      isovalue.set(curriso);
      timevalue.set(currtime);
      resolution.set(currres);
      reset_vars();
      int res = tbon->getDepth() - currres;
      
      //      cout << "time = " << currtime << endl;
      
      int havemonitor = gotasurface;
      if( havemonitor ) 
      	crowdmonitor->writeLock();
    
      if( curriso == 0.995 )
	surface = tbon->search( curriso, treefiles[currtime],
				datafiles[currtime], 
				TbonTreeCL<type>::TIME_CHANGED, res ); 
      else
 	surface = tbon->search( curriso, treefiles[currtime],
 				datafiles[currtime], 
 				TbonTreeCL<type>::TIME_SAME, res ); 
	
      if( surface != 0 )
	gotasurface = 1;

      if( havemonitor )
	crowdmonitor->writeUnlock();
      if( gotasurface && id == -1 ) {
      	id = geomout->addObj( new GeomMaterial( surface, matl ), "Geometry", 
      			      crowdmonitor );
      }
      if( gotasurface ) {
	geomout->flushViews();
      }
      t1 = read_time();
      PrintTime(t0,t1,"Isosurface time: ");
    }
  }
#endif    
  }
  //  t1 = read_time();
  //  PrintTime(t0,t1,"Isosurface time: ");
}
} // End namespace Phil


