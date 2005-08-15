
/*
  BonoP.cc
  Parallel Branch-on-Need Octree (BONO) implementation

  Packages/Philip Sutton
  July 1999

   Copyright (C) 2000 SCI Group, University of Utah
*/

#include "BonoTreeP.h"
#include "Clock.h"
#include "TriGroup.h"

#include <Core/Util/NotFinished.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/CrowdMonitor.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>

namespace Phil {
using namespace SCIRun;

typedef unsigned char type;
//typedef float type;

class BonoP : public Module {

public:
  // functions required by Module inheritance
  BonoP( const clString& id);
  virtual ~BonoP();
  virtual void execute();
  virtual void tcl_command(TCLArgs& args, void* userdata);

  void parallel(int);
protected:

private:
  BonoTreeP<type>* bono;
  GeomGroup* group;
  Mutex* grouplock;

  Material* matl;
  char** treefiles;
  char** datafiles;

  GuiString metafilename;
  GuiInt nodebricksize, databricksize;
  GuiDouble isovalue;
  GuiInt timevalue;
  GuiInt resolution;
  GuiInt go;

  double miniso, maxiso;
  double stepsize;
  int timesteps;
  double curriso;
  int currtime;
  int currres;
  int currlevel;

  int np;
  int isocells, rez;
  int sametime, sameiso, sameres;
  int gotasurface;

  GeometryOPort* geomout;
  CrowdMonitor* crowdmonitor;

  Semaphore* waitStart;
  Semaphore* waitDone;

  void processQuery( );
  void preprocess(ifstream& metafile);
}; // class BonoP

// More required stuff...
extern "C" Module* make_BonoP(const clString& id){
  return new BonoP(id);
}

// Constructor
BonoP::BonoP( const clString& id )
  : Module("BonoP", id, Filter), metafilename("metafilename",id,this), 
    nodebricksize("nodebricksize",id,this), go("go",id,this),
    databricksize("databricksize",id,this), isovalue("isovalue",id,this),
    timevalue("timevalue",id,this), resolution("resolution",id,this)
{
  timevalue.set(0);
  isovalue.set(0);
  nodebricksize.set(1024);
  databricksize.set(2048);
  resolution.set(0);
  go.set(0);

  curriso = -1.0;
  currtime = -1;
  currres = -1;
  currlevel = -1;
  bono = 0;
  matl = new Material( Color(0,0,0),
		       Color(0.25,0.7,0.7),
		       Color(0.8,0.8,0.8),
		       10 );
  grouplock = new Mutex("BonoP grouplock");
  group = new GeomGroup();
  gotasurface = 0;
  crowdmonitor = new CrowdMonitor("BonoP crowdmonitor");
  waitStart = new Semaphore("BonoP waitStart",0);
  waitDone = new Semaphore("BonoP waitDone",0);

  geomout = new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport( geomout );
}

// Destructor
BonoP::~BonoP() {
}

// Execute - set up tree, process first isovalue query
void
BonoP::execute() {
  char filetype[20];
  char treesmeta[80];
  char trees[80];

  // statically set number of processors
  // this avoids having to restart the WorkQueue when np changes
  np = Thread::numProcessors();
  if( np > 16)
    np = 16;
  
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
    metafile >> miniso >> maxiso >> stepsize;
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
    bono = new BonoTreeP<type>( treesmeta, databricksize.get(), np );
    
    // update UI to reflect current values
    TCL::execute( id + " updateFrames");

    // This deals with "resolution" i.e. the complexity of the surface
    // lower res means cruder, larger triangles.
    // See thesis for (a few) more details.

    /////////////////
    // uncomment for immediate full resolution (more accurate)
    resolution.set( bono->getDepth() );
    // uncomment for immediate 1/2 resolution (faster)
    //    resolution.set( (int)(bono->getDepth() / 2) );
    /////////////////

    processQuery();
  } else {
    cerr << "Error: metafile type not recognized" << endl;
  } 
}


// tcl_command - routes changes from the UI
//    update:  time or isovalue slider was changed - recompute isosurface
//    getVars: metafile changed - get extreme values of iso, time
void
BonoP::tcl_command(TCLArgs& args, void* userdata) {

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
    sprintf(var,"%lf",stepsize);
    svar = clString( var );
    result += clString( " " + svar );
    sprintf(var,"%d",timesteps-1);
    svar = clString( var );
    result += clString( " " + svar );

    sprintf(var,"%d",bono->getDepth());
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
BonoP::preprocess( ifstream& metafile ) {
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

  // create BONO tree
  bono = new BonoTreeP<type>( nx, ny, nz );
  
  // fill in time steps
  int i = 0;
  while( datafiles >> filename ) {
    // read data from disk
    bono->readData( filename );
    // construct BONO tree and save to disk
    bono->fillTree( );
    bono->writeTree( treesmeta, treebase, i++ ); 
  }
  
  // clean up
  delete bono;
}

// processQuery - called whenever the time/iso value changes
//  processes a single (time,iso) query
void
BonoP::processQuery() {
  static int id = -1;
  static int firsttime = 1;
  iotimer_t t0, t1;
  iotimer_t t2, t3;

  t0 = read_time();
  
  timevalue.reset();
  isovalue.reset();
  resolution.reset();

  sametime = ( timevalue.get() == currtime ) ? 1 : 0;
  sameiso = ( isovalue.get() == curriso ) ? 1 : 0;
  sameres = ( resolution.get() == currres ) ? 1 : 0;
  currlevel = bono->getDepth() - 1;
  curriso = isovalue.get();
  currres = resolution.get();
  currtime = timevalue.get();

  //  rez = bono->getDepth() - currres;

  if( sametime && sameiso && sameres ) {
    // do nothing - nothing has changed
  } else {
    // calculate new surface
    if( !sametime ) {
      t2 = read_time();
      isocells = bono->search1( curriso, 
				treefiles[currtime], datafiles[currtime], 
				BonoTreeP<type>::TIME_CHANGED, currlevel, 
				np );
      t3 = read_time();
      PrintTime( t2, t3, "1st pass" );
    } else {
      t2 = read_time();
      isocells = bono->search1( curriso, 
				treefiles[currtime], datafiles[currtime], 
				BonoTreeP<type>::TIME_SAME, currlevel, 
				np );
      t3 = read_time();
      PrintTime( t2, t3, "1st pass" );
    }

    t2 = read_time();
    int havemonitor = gotasurface;
    if( havemonitor ) 
      crowdmonitor->writeLock();

    bono->resize( isocells, np );
    if( firsttime ) {
      // create worker threads
      Thread::parallel(Parallel<BonoP>(this, &BonoP::parallel), np, false);
      firsttime = 0;
    }
    // wait for worker threads to execute
    waitStart->up(np);
    waitDone->down(np);
    bono->cleanup();

    if( havemonitor )
      crowdmonitor->writeUnlock();

    t3 = read_time();
    PrintTime( t2, t3, "2nd pass" );

    // send surface to output port
    if( gotasurface && id == -1 ) {
      id = geomout->addObj( new GeomMaterial( group, matl ), "Geometry", 
			    crowdmonitor );
    }
    if( gotasurface ) {
      geomout->flushViews();
    }
  
#if 0
    sleep(15);
    for( int counter = 5; counter > 0; counter-- ) {
      cout << counter << "..." << endl;
      fflush(0);
      sleep(1);
    }
    cout << "0!" << endl;
    fflush(0);
    //    for( curriso = 1.10; curriso < 1.46; curriso += 0.05 ) {
    curriso = 155.5;
    //    for( curriso = 20.5; curriso < 232; curriso += 30.0 ) {
    for( currtime = 60; currtime < 90; currtime += 10 ) {
    for( curriso = 20.5; curriso < 232; curriso += 10.0 ) {
      isovalue.set(curriso);
      timevalue.set(currtime);
      reset_vars();
      iotimer_t t0, t1;

      t0 = read_time();
      if( curriso == 20.5 ) {
	isocells = bono->search1( curriso, 
				  treefiles[currtime], datafiles[currtime], 
				  BonoTreeP<type>::TIME_CHANGED, currlevel, 
				  np );
      } else {
	isocells = bono->search1( curriso, 
				  treefiles[currtime], datafiles[currtime], 
				  BonoTreeP<type>::TIME_SAME, currlevel, 
				  np );
      }
      t1 = read_time();
      PrintTime( t0, t1, "1st pass" );


      t0 = read_time();

      int havemonitor = gotasurface;
      if( havemonitor ) 
	crowdmonitor->writeLock();

      bono->resize( isocells, np );
      waitStart->up(np);
      waitDone->down(np);      
      bono->cleanup();

      if( havemonitor )
	crowdmonitor->writeUnlock();

      t1 = read_time();
      PrintTime( t0, t1, "2nd pass" );

      if( gotasurface && id == -1 ) {
	id = geomout->addObj( new GeomMaterial( group, matl ), "Geometry", 
			      crowdmonitor );
      }
      if( gotasurface ) {
	geomout->flushViews();
      }
    }
    }
    //  }
#endif
  }

  t1 = read_time();
  PrintTime(t0,t1,"Isosurface time: ");
}

// parallel - each thread loops indefinitely, waiting for the semaphore
//   to tell it to start extracting a surface
void
BonoP::parallel( int rank ) {
  GeomTriGroup* surface;
  int havesurface = 0;
  
  for(;;) {
    waitStart->down();
    // create surface
    surface = bono->search3( curriso, currlevel, rank, isocells );
    if( !havesurface && surface != 0 ) {
      // if THIS thread has not contributed yet, it adds its portion
      // of the surface to the group
      grouplock->lock();
      group->add( surface );
      havesurface = 1;
      gotasurface = 1;
      grouplock->unlock();
    }

    waitDone->up();
  }
}


} // End namespace Phil


