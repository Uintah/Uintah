
/*
  TbonP.cc
  Parallel Temporal Branch-on-Need Octree (T-BON)

  Packages/Philip Sutton
  July 1999

  Copyright (C) 2000 SCI Group, University of Utah
*/

#include "TbonTreeP.h"
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
#include <unistd.h>

namespace Phil {
using namespace SCIRun;
using namespace std;

typedef unsigned char type;
//typedef float type;

static const float DX = 1.0;
static const float DY = 1.0;
static const float DZ = 1.0;

class TbonP : public Module {

public:
  // functions required by Module inheritance
  TbonP( const clString& id);
  virtual ~TbonP();
  virtual void execute();
  virtual void tcl_command(TCLArgs& args, void* userdata);

  void parallel(int);
protected:

private:
  TbonTreeP<type>* tbon;
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
  GuiString red, green, blue, alpha;

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
  GeomMaterial* theIsosurface;

  Semaphore* waitStart;
  Semaphore* waitDone;

  void processQuery( );
  void preprocess(ifstream& metafile);
}; // class TbonP

// More required stuff...
extern "C" Module* make_TbonP(const clString& id){
  return new TbonP(id);
}

// Constructor
TbonP::TbonP( const clString& id )
  : Module("TbonP", id, Filter), metafilename("metafilename",id,this), 
    nodebricksize("nodebricksize",id,this), 
    databricksize("databricksize",id,this), isovalue("isovalue",id,this),
    timevalue("timevalue",id,this), resolution("resolution",id,this),
    red("red",id,this), green("green",id,this), blue("blue",id,this),
    alpha("alpha",id,this)
{
  timevalue.set(0);
  isovalue.set(0);
  nodebricksize.set(1024);
  databricksize.set(2048);
  resolution.set(0);
  red.set(".25");
  green.set(".7");
  blue.set(".7");
  alpha.set("1.0");

  curriso = -1.0;
  currtime = -1;
  currres = -1;
  currlevel = -1;
  tbon = 0;
  matl = new Material( Color(0,0,0),
		       Color(0.25,0.7,0.7),
		       Color(0.8,0.8,0.8),
		       10 );
  grouplock = new Mutex("TbonP grouplock");
  group = new GeomGroup();
  gotasurface = 0;
  crowdmonitor = new CrowdMonitor("TbonP crowdmonitor");
  waitStart = new Semaphore("TbonP waitStart",0);
  waitDone = new Semaphore("TbonP waitDone",0);

  geomout = new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport( geomout );
}

// Destructor
TbonP::~TbonP() {
}

// Execute
void
TbonP::execute() {
  char filetype[20];
  char treesmeta[80];
  char trees[80];

  np = Thread::numProcessors();
  if( np > 8 )
    np = 8;
  
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
    tbon = new TbonTreeP<type>( treesmeta, np, DX, DY, DZ );
    
    // update UI to reflect current values
    TCL::execute( id + " updateFrames");
    /////////////////
    // uncomment for immediate full resolution
    resolution.set( tbon->getDepth() );
    // uncomment for immediate 1/2 resolution
    //    resolution.set( (int)(tbon->getDepth() / 2) );

    processQuery();
  } else {
    cerr << "Error: metafile type not recognized" << endl;
  } 
}


// tcl_command - routes changes from the UI
//    update:  time or isovalue slider was changed - recompute isosurface
//    getVars: metafile changed - get extreme values of iso, time
void
TbonP::tcl_command(TCLArgs& args, void* userdata) {

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
    sprintf(var,"%lf",stepsize);
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
  } else if( args[1] == "changeColor" ) {
    reset_vars();
    matl->diffuse = Color( atof(red.get()()), atof(green.get()()), 
			   atof(blue.get()()) );
    matl->transparency = 1.0 - atof(alpha.get()());
    theIsosurface->setMaterial(matl);
    geomout->flushViews();
  } else {
    // message not for us - propagate up
    Module::tcl_command( args, userdata );
  }
}


// preprocess - set up data structures
void
TbonP::preprocess( ifstream& metafile ) {
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
  tbon = new TbonTreeP<type>( nx, ny, nz, nodebricksize.get(), 
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
TbonP::processQuery() {
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
  currlevel = tbon->getDepth() - 1;
  curriso = isovalue.get();
  currres = resolution.get();
  currtime = timevalue.get();

  rez = tbon->getDepth() - currres;

  if( sametime && sameiso && sameres ) {
    // do nothing - nothing has changed
  } else {
    // calculate new surface
    if( !sametime ) {
      t2 = read_time();
      isocells = tbon->search1( curriso, 
				treefiles[currtime], datafiles[currtime], 
				TbonTreeP<type>::TIME_CHANGED, currlevel, 
				np, rez );
      t3 = read_time();
      PrintTime( t2, t3, "1st pass" );
      t2 = read_time();
      tbon->search2( TbonTreeP<type>::TIME_CHANGED );
      t3 = read_time();
      PrintTime( t2, t3, "read data" );
    } else {
      t2 = read_time();
      isocells = tbon->search1( curriso, 
				treefiles[currtime], datafiles[currtime], 
				TbonTreeP<type>::TIME_SAME, currlevel, 
				np, rez );
      t3 = read_time();
      PrintTime( t2, t3, "1st pass" );
      t2 = read_time();
      tbon->search2( TbonTreeP<type>::TIME_SAME );
      t3 = read_time();
      PrintTime( t2, t3, "read data" );
    }

    t2 = read_time();
    int havemonitor = gotasurface;
    if( havemonitor ) 
      crowdmonitor->writeLock();

    //    tbon->resize( rez, isocells, np );
    if( firsttime ) {
      // create worker threads
      Thread::parallel(Parallel<TbonP>(this, &TbonP::parallel), np, false);
      firsttime = 0;
    }
    // wait for worker threads to execute
    waitStart->up(np);
    waitDone->down(np);
    tbon->cleanup();

    if( havemonitor )
      crowdmonitor->writeUnlock();

    t3 = read_time();
    PrintTime( t2, t3, "2nd pass" );

    // send surface to output port
    if( gotasurface && id == -1 ) {
      theIsosurface = new GeomMaterial( group, matl );
      id = geomout->addObj( theIsosurface, "Geometry", crowdmonitor );
    }
    if( gotasurface ) {
      geomout->flushViews();
    }

#if 0
//     sleep(15);
//     for( int counter = 5; counter > 0; counter-- ) {
//       cout << counter << "..." << endl;
//       fflush(0);
//       sleep(1);
//     }
//     cout << "0!" << endl;
//     fflush(0);
    //    for( curriso = 1.10; curriso < 1.46; curriso += 0.05 ) {
    //    curriso = 155.5;
    //    for( curriso = 20.5; curriso < 232; curriso += 30.0 ) {
    //    for( currtime = 60; currtime < 90; currtime += 10 ) {
    //    for( curriso = 20.5; curriso < 232; curriso += 10.0 ) {
    for( curriso = 1.10; curriso < 1.45; curriso += 0.05 ) {
      for( currtime = 0; currtime < 5; currtime++ ) {
      isovalue.set(curriso);
      timevalue.set(currtime);
      reset_vars();
      iotimer_t t0, t1;
      
      //      cout << "time = " << currtime << endl;
      
      t0 = read_time();
      //      if( curriso == 20.5 ) {
	isocells = tbon->search1( curriso, 
				  treefiles[currtime], datafiles[currtime], 
				  TbonTreeP<type>::TIME_CHANGED, currlevel, 
				  np, rez );
//       } else {
// 	isocells = tbon->search1( curriso, 
// 				  treefiles[currtime], datafiles[currtime], 
// 				  TbonTreeP<type>::TIME_SAME, currlevel, 
// 				  np, rez );
//       }
      t1 = read_time();
      PrintTime( t0, t1, "1st pass" );

      t0 = read_time();
      tbon->search2( TbonTreeP<type>::TIME_CHANGED );
      t1 = read_time();
      PrintTime( t0, t1, "read data" );

      t0 = read_time();

      havemonitor = gotasurface;
      if( havemonitor ) 
	crowdmonitor->writeLock();

      //      tbon->resize( rez, isocells, np );
      waitStart->up(np);
      waitDone->down(np);      
      tbon->cleanup();

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
#endif
  }

  t1 = read_time();
  PrintTime(t0,t1,"Isosurface time: ");
}

// parallel - each thread loops indefinitely, waiting for the semaphore
//   to tell it to start extracting a surface
void
TbonP::parallel( int rank ) {
  GeomTriGroup* surface;
  int havesurface = 0;
  
  for(;;) {
    waitStart->down();
    tbon->resize( rez, isocells, np, rank );
    // create surface
    surface = tbon->search3( curriso, currlevel, rez, rank, isocells );
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


