/****************************************
CLASS
    VectorFieldExtractor

    Visualization control for simulation data that contains
    information on both a regular grid in particle sets.

OVERVIEW TEXT
    This module receives a ParticleGridReader object.  The user
    interface is dynamically created based information provided by the
    ParticleGridReader.  The user can then select which variables he/she
    wishes to view in a visualization.



KEYWORDS
    ParticleGridReader, Material/Particle Method

AUTHOR
    Kurt Zimmerman
    Department of Computer Science
    University of Utah
    January 1999

    Copyright (C) 1999 SCI Group

LOG
    Created January 5, 1999
****************************************/
#include "VectorFieldExtractor.h"

#include <Core/Util/NotFinished.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/BBox.h>
#include <Core/Util/Timer.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Datatypes/LevelMesh.h>
#include <Packages/Uintah/Core/Datatypes/LevelField.h>
#include <Packages/Uintah/Core/Datatypes/PatchDataThread.h>
#include <Packages/Uintah/CCA/Ports/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Containers/ConsecutiveRangeSet.h>

#include <iostream> 
#include <sstream>
#include <string>

namespace Uintah {

using std::cerr;
using std::endl;
using std::vector;
using std::string;
using std::ostringstream;

using namespace SCIRun;

extern "C" Module* make_VectorFieldExtractor( const string& id ) {
  return scinew VectorFieldExtractor( id ); 
}

//--------------------------------------------------------------- 
VectorFieldExtractor::VectorFieldExtractor(const string& id) 
  : Module("VectorFieldExtractor", id, Filter, "Selectors", "Uintah"),
    tcl_status("tcl_status", id, this), sVar("sVar", id, this),
    sMatNum("sMatNum", id, this), type(0)
{ 
} 

//------------------------------------------------------------ 
VectorFieldExtractor::~VectorFieldExtractor(){} 

//------------------------------------------------------------- 

void VectorFieldExtractor::setVars()
{
  string command;

  DataArchive& archive = *((*(archiveH.get_rep()))());

  vector< string > names;
  vector< const TypeDescription *> types;
  archive.queryVariables(names, types);


  vector< double > times;
  vector< int > indices;
  archive.queryTimesteps( indices, times );

  string sNames("");
  int index = -1;
  bool matches = false;


  // get all of the VectorField Variables
  for( int i = 0; i < (int)names.size(); i++ ){
    const TypeDescription *td = types[i];
    const TypeDescription *subtype = td->getSubType();
    //cerr << "\tVariable: " << names[i] << ", type " << td->getName() << "\n";
    if( td->getType() ==  TypeDescription::NCVariable ||
	td->getType() ==  TypeDescription::CCVariable ){

      if( subtype->getType() == TypeDescription::Vector){
	if( sNames.size() != 0 )
	  sNames += " ";
	sNames += names[i];
	if( sVar.get() == "" ){ sVar.set( names[i].c_str() ); }
	if( sVar.get() == names[i].c_str()){
	  type = td;
	  matches = true;
	} else {
	  if( index == -1) {index = i;}
	}
      }	
    }
  }

  if( !matches && index != -1 ) {
    sVar.get() = names[index].c_str();
    type = types[index];
  }

  // get the number of materials for the VectorField Variables
  GridP grid = archive.queryGrid(times[0]);
  LevelP level = grid->getLevel( 0 );
  Patch* r = *(level->patchesBegin());
  ConsecutiveRangeSet matls = 
    archive.queryMaterials(sVar.get(), r, times[0]);

  string visible;
  TCL::eval(id + " isVisible", visible);
  if( visible == "1"){
    TCL::execute(id + " destroyFrames");
    TCL::execute(id + " build");
    
    TCL::execute(id + " buildMaterials " + matls.expandedString().c_str());

    TCL::execute(id + " setVectors " + sNames.c_str());
    TCL::execute(id + " buildVarList");

    TCL::execute("update idletasks");
    reset_vars();
  }
}



void VectorFieldExtractor::execute() 
{ 
  tcl_status.set("Calling VectorFieldExtractor!"); 
  in = (ArchiveIPort *) get_iport("Data Archive");
  vfout = (FieldOPort *) get_oport("Vector Field");
  
  ArchiveHandle handle;
  if(!in->get(handle)){
    std::cerr<<"VectorFieldExtractor::execute:Didn't get a handle\n";
    return;
  }
  
  if (archiveH.get_rep()  == 0 ){
    string visible;
    TCL::eval(id + " isVisible", visible);
    if( visible == "0" ){
      TCL::execute(id + " buildTopLevel");
    }
  }
  
  archiveH = handle;
  DataArchive& archive = *((*(archiveH.get_rep()))());
  cerr << "Calling setVars\n";
  setVars();
  cerr << "done with setVars\n";
  
  
  
  
  // what time is it?
  vector< double > times;
  vector< int > indices;
  archive.queryTimesteps( indices, times );
  
  // set the index for the correct timestep.
  int idx = handle->timestep();
  double dt = -1;
  if (idx < (int)times.size() - 1)
    dt = times[idx+1] - times[idx];
  else if (times.size() > 1)
    dt = times[idx] - times[idx-1];
  
  // set the generation of the grid
  int generation = (*(archiveH.get_rep())).generation;
  
  int max_workers = Max(Thread::numProcessors()/2, 8);
  Semaphore* thread_sema = scinew Semaphore( "vector extractor semahpore",
					     max_workers); 

  WallClockTimer my_timer;
  GridP grid = archive.queryGrid(times[idx]);
  LevelP level = grid->getLevel( 0 );
  const TypeDescription* subtype = type->getSubType();
  string var(sVar.get());
  int mat = sMatNum.get();
  double time = times[idx];
  switch( type->getType() ) {
  case TypeDescription::NCVariable:
    switch ( subtype->getType() ) {
    case TypeDescription::Vector:
      {	
	my_timer.start();
	LevelMeshHandle mesh = scinew LevelMesh( grid, 0 );
	LevelField<Vector> *vfd =
	  scinew LevelField<Vector>( mesh, Field::NODE );
	vector<ShareAssignArray3<Vector> >& data = vfd->fdata();
	data.resize(level->numPatches());
	double size = data.size();
	int count = 0;
	vector<ShareAssignArray3<Vector> >::iterator it = data.begin();
	for(Level::const_patchIterator r = level->patchesBegin();
	    r != level->patchesEnd(); r++, ++it ){
	  update_progress(count++/size, my_timer);
	    thread_sema->down();
	    Thread *thrd =scinew Thread(scinew PatchDataThread<NCVariable<Vector>,
			                 vector<ShareAssignArray3<Vector> >::iterator>
			  (archive, it, var, mat, *r, time, thread_sema),
					"patch_data_worker");
	    thrd->detach();
	}
	thread_sema->down(max_workers);
	if( thread_sema ) delete thread_sema;
	// set the generation and timestep in the field
	vfd->store("varname",string(var));
	vfd->store("generation",generation);
	vfd->store("timestep",idx);
	vfd->store("delta_t",dt);
	// do timer stuff
	timer.add( my_timer.time() );
	my_timer.stop();
	// send the field out to the port
	vfout->send(vfd);
	return;
      }
      break;
    default:
      cerr<<"NCVariable<?>  Unknown vector type\n";
      return;
    }
    break;
  case TypeDescription::CCVariable:
    switch ( subtype->getType() ) {
    case TypeDescription::Vector:
      {	
	my_timer.start();
	LevelMeshHandle mesh = scinew LevelMesh( grid, 0 );
	LevelField<Vector> *vfd =
	  scinew LevelField<Vector>( mesh, Field::CELL );
	vector<ShareAssignArray3<Vector> >& data = vfd->fdata();
	data.resize(level->numPatches());
	double size = data.size();
	int count = 0;
	vector<ShareAssignArray3<Vector> >::iterator it = data.begin();
	for(Level::const_patchIterator r = level->patchesBegin();
	    r != level->patchesEnd(); r++, ++it ){
	  update_progress(count++/size, my_timer);
	    thread_sema->down();
	    Thread *thrd =
	      scinew Thread(scinew PatchDataThread<CCVariable<Vector>,
			    vector<ShareAssignArray3<Vector> >::iterator>
			    (archive, it, var, mat, *r, time, thread_sema),
			    "patch_data_worker");
	    thrd->detach();
	}
	thread_sema->down(max_workers);
	if( thread_sema ) delete thread_sema;
	// set the generation and timestep in the field
	vfd->store("varname",string(var));
	vfd->store("generation",generation);
	vfd->store("timestep",idx);
	vfd->store("delta_t",dt);
	// do timer stuff
	timer.add( my_timer.time() );
	my_timer.stop();
	// send the field out to the port
	vfout->send(vfd);
	return;
      }
      break;
    default:
      cerr<<"CCVariable<?> Unknown vector type\n";
      return;
    }
    break;
  default:
    cerr<<"Not a VectorField\n";
    return;
  }
  return;
}
} // End namespace Uintah
