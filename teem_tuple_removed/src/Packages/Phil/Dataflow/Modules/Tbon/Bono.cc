
/* Bono.cc
   Branch-on-Need Octree (BONO) implementation
   Packages/Philip Sutton
   July 1999

   Copyright (C) 2000 SCI Group, University of Utah
*/

#include "BonoTree.h"
#include "Clock.h"
#include "TriGroup.h"

#include <Core/Util/NotFinished.h>
#include <Core/GuiInterface/GuiVar.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Geom/Material.h>
#include <Core/Thread/CrowdMonitor.h>

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>

namespace Phil {
using namespace SCIRun;
using namespace std;

//typedef float type;
typedef unsigned char type;

class Bono : public Module {

public:

  // Functions required by Module inheritance
  Bono(const clString& id);
  virtual ~Bono();
  virtual void execute();
  virtual void tcl_command(TCLArgs& args, void* userdata);

protected:

private:
  BonoTree<type>* bono;
  GeomTriGroup* surface;
  
  char** treefiles;
  char** datafiles;

  // interface variables
  GuiString metafilename;
  GuiInt nodeblocksize, datablocksize;
  GuiDouble isovalue;
  GuiInt timevalue;

  double miniso, maxiso;
  double res;
  int timesteps;
  double curriso;
  int currtime;

  Material* matl;
  int portid;
  CrowdMonitor* crowdmonitor;
  GeomMaterial* theIsosurface;

  GeometryOPort* geomout;

  void processQuery( );
  void preprocess(ifstream& metafile);

}; // class Bono


// More required stuff...
extern "C" Module* make_Bono(const clString& id){
  return new Bono(id);
}


// Constructor
Bono::Bono(const clString& id) 
  : Module("Bono", id, Filter), metafilename("metafilename",id,this), 
    nodeblocksize("nodeblocksize",id,this), 
    datablocksize("datablocksize",id,this), isovalue("isovalue",id,this),
    timevalue("timevalue",id,this)
{
  portid = -1;
  
  timevalue.set(0);
  isovalue.set(0);
  nodeblocksize.set(1024);
  datablocksize.set(4096);
  curriso = -1.0;
  currtime = -1;
  bono = 0;
  crowdmonitor = new CrowdMonitor("Bono crowdmonitor");

  // default material (cyan)
  matl =  new Material( Color(0,0,0),
			Color(0.25,0.75,0.75),
			Color(0.8,0.8,0.8),
			10 );

  geomout = new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport( geomout );
}

// Destructor
Bono::~Bono(){
}

// Execute - set up tree, process first isovalue query
void 
Bono::execute()
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

    // update UI to reflect current values
    TCL::execute( id + " updateFrames");

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
    bono = new BonoTree<type>( treesmeta, datablocksize.get() );

    processQuery();
  } else {
    cerr << "Error: metafile type not recognized" << endl;
  }
}

// tcl_command - routes changes from the UI
//    update:  time or isovalue slider was changed - recompute isosurface
//    getVars: metafile changed - get extreme values of iso, time
void
Bono::tcl_command(TCLArgs& args, void* userdata) {

  if( args[1] == "update" ) {
    if( bono != 0 ) {
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

    // return result
    args.result( result );
  } else {
    // message not for us - propagate up
    Module::tcl_command( args, userdata );
  }
}


// preprocess - set up data structures
void
Bono::preprocess( ifstream& metafile ) {
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
  bono = new BonoTree<type>( nx, ny, nz );

  // fill in time steps
  int i = 0;
  while( datafiles >> filename ) {
    // read data from disk
    bono->readData( filename );
    // construct T-BON tree and save to disk
    bono->fillTree( );
    bono->writeTree( treesmeta, treebase, i++ );
  }

  // clean up
  delete bono;
}

// processQuery - called whenever the time/iso value changes
//  processes a single (time,iso) query
void
Bono::processQuery() {
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
      surface = bono->search( curriso, treefiles[currtime], 
			      datafiles[currtime], 
			      BonoTree<type>::TIME_CHANGED );    
    } else {
      surface = bono->search( curriso, treefiles[currtime],
			      datafiles[currtime], 
			      BonoTree<type>::TIME_SAME );
    }

    if( surface != 0 )
      gotasurface = 1;

    if( havemonitor )
      crowdmonitor->writeUnlock();
    // send surface to output port
    if( gotasurface && portid == -1 ) {
      theIsosurface = new GeomMaterial( surface, matl );
      portid = geomout->addObj( theIsosurface, "Geometry", crowdmonitor);
    }
    if( gotasurface ) 
      geomout->flushViews();
  } 
#if 0
  for( curriso = -0.50; curriso <= 0.0; curriso += 0.10 ) {
    for( currtime = 0; currtime < 10; currtime++ ) {
      isovalue.set(curriso);
      timevalue.set(currtime);
      reset_vars();
      surface = bono->search( curriso, treefiles[currtime],
			      datafiles[currtime], 
			      BonoTree<type>::TIME_CHANGED ); 
      if( id > 0 )
	geomout->delObj( id );
      id = geomout->addObj( surface, "Geometry" );
      geomout->flush();
    }
  } 
#endif  
  t1 = read_time();
  PrintTime(t0,t1,"Isosurface time: ");
}
} // End namespace Phil


