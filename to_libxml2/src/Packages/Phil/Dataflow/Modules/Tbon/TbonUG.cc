
/* TbonUG.cc
   Temporal Branch-on-Need tree (T-BON) - Unstructured Grid implementation
   Packages/Philip Sutton
   May 1999

  Copyright (C) 2000 SCI Group, University of Utah
*/

#include "TbonTreeUG.h"
#include "Clock.h"
#include "TriGroup.h"

#include <Core/Geom/Material.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomCylinder.h>

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

class TbonUG : public Module {

public:

  // Functions required by Module inheritance
  TbonUG(const clString& id);
  virtual ~TbonUG();
  virtual void execute();
  virtual void tcl_command(TCLArgs& args, void* userdata);

protected:

private:
  TbonTreeUG<type>* tbon;
  GeomTriGroup* surface;
  
  char** treefiles;
  char** datafiles;

  GuiString metafilename;
  GuiInt nodebricksize, databricksize;
  GuiDouble isovalue;
  GuiInt timevalue;
  GuiInt threshold;

  double miniso, maxiso;
  double res;
  int timesteps;
  double curriso;
  int currtime;

  GeometryOPort* geomout;

  void processQuery( );
  void preprocess(ifstream& metafile);

}; // class Tbon


// More required stuff...
extern "C" Module* make_TbonUG(const clString& id){
  return new TbonUG(id);
}


// Constructor
TbonUG::TbonUG(const clString& id) 
  : Module("TbonUG", id, Filter), metafilename("metafilename",id,this), 
    nodebricksize("nodebricksize",id,this), 
    databricksize("databricksize",id,this), isovalue("isovalue",id,this),
    timevalue("timevalue",id,this), threshold("threshold",id,this)
{
  timevalue.set(0);
  isovalue.set(0);
  nodebricksize.set(1024);
  databricksize.set(4096);
  threshold.set(0);

  curriso = -1.0;
  currtime = -1;
  tbon = 0;

  geomout = new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport( geomout );
}

// Destructor
TbonUG::~TbonUG(){
}

// Execute - set up tree, process first isovalue query
void 
TbonUG::execute()
{
  char filetype[20];
  char treesmeta[80];
  char geomfile[80];
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
      tbon = new TbonTreeUG<type>( treesmeta, geomfile, databricksize.get() );

      processQuery();
  } else {
    cerr << "Error: metafile type not recognized" << endl;
  }

}

// tcl_command - routes changes from the UI
//    update:  time or isovalue slider was changed - recompute isosurface
//    getVars: metafile changed - get extreme values of iso, time
void
TbonUG::tcl_command(TCLArgs& args, void* userdata) {

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

    // return result
    args.result( result );
  } else {
    // message not for us - propagate up
    Module::tcl_command( args, userdata );
  }
}


// preprocess - set up data structures
void
TbonUG::preprocess( ifstream& metafile ) {
  char filename[80];
  char geomfile[80];
  char treesmeta[80];
  char treebase[80];
  char newgeom[80];
  int nx, ny, nz;

  metafile >> filename;
  metafile >> geomfile;
  metafile >> nx >> ny >> nz;
  metafile >> treesmeta;
  metafile >> treebase;
  metafile >> newgeom;

  ifstream datafiles( filename, ios::in );
  if( !datafiles ) {
    cerr << "Error: cannot open file " << filename << endl;
    return;
  }

  // create T-BON tree
  tbon = new TbonTreeUG<type>( nx, ny, nz, geomfile, nodebricksize.get() );

  // fill in time steps
  int i = 0;
  while( datafiles >> filename ) {
    // read data from disk
    tbon->readData( filename );
    // construct T-BON tree and save to disk
    tbon->fillTree( threshold.get() );
    tbon->writeTree( treesmeta, treebase, newgeom, i++ );
  }

  // clean up
  delete tbon;
}

// processQuery - called whenever the time/iso value changes
//  processes a single (time,iso) query
void
TbonUG::processQuery() {
  static int id = -1;
  iotimer_t t0, t1;

  t0 = read_time();
  timevalue.reset();
  isovalue.reset();
  
  if( timevalue.get() == currtime && isovalue.get() == curriso ) {
    // do nothing - nothing has changed
  } else {
    curriso = isovalue.get();

    // calculate new surface
    if( timevalue.get() != currtime ) {
      currtime = timevalue.get();
      surface = tbon->search( curriso, treefiles[currtime], 
			      datafiles[currtime], 
			      TbonTreeUG<type>::TIME_CHANGED );    
    } else {
      surface = tbon->search( curriso, treefiles[currtime], 
			      datafiles[currtime], 
			      TbonTreeUG<type>::TIME_SAME );
    }

    // send surface to output port
    if( id > 0 )
      geomout->delObj( id );
    id = geomout->addObj( new GeomMaterial( surface, 
					    new Material( Color(0,0,0),
							  Color(0.25,0.7,0.7),
							  Color(0.8,0.8,0.8),
							  20 )
					    ),
			  "Geometry" );
    geomout->flush();
    /*
    for( curriso = -15.0; curriso < 20.0; curriso += 1.0 ) {
      //    for( currtime = 0; currtime < 374; currtime++ ) {
      isovalue.set( curriso );
      reset_vars();
      surface = tbon->search( curriso, treefiles[currtime],
			      datafiles[currtime], 
			      TbonTreeUG<type>::TIME_CHANGED ); 
      if( id > 0 )
	geomout->delObj( id );
      id = geomout->addObj( surface, "Geometry" );
      geomout->flush();
      //  } 
    }
    */
  }
  t1 = read_time();
  PrintTime(t0,t1,"Isosurface time: ");
}
} // End namespace Phil


