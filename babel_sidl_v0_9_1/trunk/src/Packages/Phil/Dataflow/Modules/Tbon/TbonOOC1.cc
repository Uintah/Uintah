
/* TbonOOC1.cc
   Temporal Branch-on-Need tree (T-BON) implementation
     Out-of-Core algorithm #1
   Packages/Philip Sutton
   September 1999

  Copyright (C) 2000 SCI Group, University of Utah
*/

#include "TbonTreeOOC1.h"
#include "TriGroup.h"
#include "Clock.h"

#include <Core/Util/NotFinished.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Thread/CrowdMonitor.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>

namespace Phil {
// MINIMUM settings (very important!)
// NODEBRICKS = 1
// DATABRICKS = 8
static const int NODEBRICKS = 1;
static const int DATABRICKS = 8;

using namespace SCIRun;
using namespace std;

//typedef unsigned char type;
typedef float type;

class TbonOOC1 : public Module {

public:

  // Functions required by Module inheritance
  TbonOOC1(const clString& id);
  virtual ~TbonOOC1();
  virtual void execute();
  virtual void tcl_command(TCLArgs& args, void* userdata);

protected:

private:
  TbonTreeOOC1<type>* tbon;
  GeomTriGroup* surface;
  
  Material* matl;
  char** treefiles;
  char** datafiles;

  GuiString metafilename;
  GuiInt nodebricksize, databricksize;
  GuiDouble isovalue;
  GuiInt timevalue;

  double miniso, maxiso;
  double res;
  int timesteps;
  double curriso;
  int currtime;

  GeometryOPort* geomout;
  CrowdMonitor* crowdmonitor;

  void processQuery( );
  void preprocess(ifstream& metafile);

}; // class TbonOOC1


// More required stuff...
extern "C" Module* make_TbonOOC1(const clString& id){
  return new TbonOOC1(id);
}


// Constructor
TbonOOC1::TbonOOC1(const clString& id) 
  : Module("TbonOOC1", id, Filter), metafilename("metafilename",id,this), 
    nodebricksize("nodebricksize",id,this), 
    databricksize("databricksize",id,this), isovalue("isovalue",id,this),
    timevalue("timevalue",id,this)
{
  timevalue.set(0);
  isovalue.set(0);

  nodebricksize.set(1024);
  databricksize.set(2048);
  curriso = -1.0;
  currtime = -1;
  tbon = 0;
  matl =  new Material( Color(0,0,0),
			Color(0.25,0.75,0.75),
			Color(0.8,0.8,0.8),
			10 );
  crowdmonitor = new CrowdMonitor("TbonOOC1 crowdmonitor");

  geomout = new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport( geomout );
}

// Destructor
TbonOOC1::~TbonOOC1(){
}


// Execute - set up tree, process first isovalue query
void 
TbonOOC1::execute()
{
  char filetype[20];
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
    tbon = new TbonTreeOOC1<type>( treesmeta, NODEBRICKS, DATABRICKS );
    
    // update UI to reflect current values
    TCL::execute( id + " updateFrames");

    processQuery();
  } else {
    cerr << "Error: metafile type not recognized" << endl;
  } 
}

// tcl_command - routes changes from the UI
//    update:  time or isovalue slider was changed - recompute isosurface
//    getVars: metafile changed - get extreme values of iso, time
void
TbonOOC1::tcl_command(TCLArgs& args, void* userdata) {

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
TbonOOC1::preprocess( ifstream& metafile ) {
  char filename[80];
  char treesmeta[80];
  char treebase[80];
  int nx, ny, nz;

  metafile >> filename;
  metafile >> nx >> ny >> nz;
  metafile >> treesmeta;
  metafile >> treebase;

  ifstream datafiles( filename, ios::in );
  if( !datafiles ) {
    cerr << "Error: cannot open file " << filename << endl;
    return;
  }

  // create T-BON tree
  tbon = new TbonTreeOOC1<type>( nx, ny, nz, nodebricksize.get(), 
			      databricksize.get() );
   
  // fill in time steps
  int i = 0;
  while( datafiles >> filename ) {
    char newfile[80];
    // read data from disk
    tbon->readData( filename );
    // construct T-BON tree and save to disk
    datafiles >> newfile;
    tbon->fillTree( );
    tbon->writeTree( treesmeta, treebase, newfile, i++ ); 
  }
  
  // clean up
  delete tbon;
}

// processQuery - called whenever the time/iso value changes
//  processes a single (time,iso) query
void
TbonOOC1::processQuery() {
  static int id = -1;
  static int gotasurface = 0;
  iotimer_t t0, t1;

  t0 = read_time();
  timevalue.reset();
  isovalue.reset();
  
  if( timevalue.get() == currtime && isovalue.get() == curriso ) {
    // do nothing - nothing has changed
  } else {
    
    curriso = isovalue.get();

    int havemonitor = gotasurface;
    if( havemonitor )
      crowdmonitor->writeLock();

    // calculate new surface
    if( timevalue.get() != currtime ) {
      currtime = timevalue.get();
      surface = tbon->search( curriso, treefiles[currtime], 
			      datafiles[currtime], 
			      TbonTreeOOC1<type>::TIME_CHANGED );    
    } else {
      surface = tbon->search( curriso, treefiles[currtime],
			      datafiles[currtime], 
			      TbonTreeOOC1<type>::TIME_SAME );
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

#if 0
    // output the geometry to disk
    static int num = 0;
    char filename[80];
    sprintf(filename, "output%d.obj", num++);

    if( surface != 0 )
      surface->write( filename );
#endif
#if 0
    curriso = 1.25;
    for( curriso = 1.10; curriso < 1.46; curriso += 0.10 ) {
    //    for( curriso = 20.5; curriso < 232; curriso += 60.0 ) {
    for( currtime = 0; currtime < 5; currtime += 1 ) {
      isovalue.set(curriso);
      timevalue.set(currtime);
      reset_vars();

      //      cout << "time = " << currtime << endl;

      havemonitor = gotasurface;
      if( havemonitor ) 
	crowdmonitor->writeLock();

      surface = tbon->search( curriso, treefiles[currtime],
			      datafiles[currtime], 
			      TbonTreeOOC1<type>::TIME_CHANGED ); 
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
    }
    } 
#endif    
  }

  t1 = read_time();
  PrintTime(t0,t1,"Isosurface time: ");
}
} // End namespace Phil


