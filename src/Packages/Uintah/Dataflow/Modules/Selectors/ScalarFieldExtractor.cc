/****************************************
CLASS
    ScalarFieldExtractor

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
#include "ScalarFieldExtractor.h"

#include <Core/Util/NotFinished.h>
#include <Packages/Uintah/CCA/Ports/DataArchive.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/BBox.h>
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
#include <Packages/Uintah/Core/Grid/ShareAssignArray3.h>
//#include <Packages/Uintah/Core/Grid/NodeIterator.h>
 
#include <iostream> 
#include <sstream>
#include <string>

using std::cerr;
using std::endl;
using std::vector;
using std::string;
using std::ostringstream;

namespace Uintah {

using namespace SCIRun;

  //using DumbScalarField;

extern "C" Module* make_ScalarFieldExtractor( const string& id ) {
  return scinew ScalarFieldExtractor( id ); 
}

//--------------------------------------------------------------- 
ScalarFieldExtractor::ScalarFieldExtractor(const string& id) 
  : Module("ScalarFieldExtractor", id, Filter, "Selectors", "Uintah"),
    tcl_status("tcl_status", id, this), sVar("sVar", id, this),
    sMatNum("sMatNum", id, this), type(0)
{ 

} 

//------------------------------------------------------------ 
ScalarFieldExtractor::~ScalarFieldExtractor(){} 

//------------------------------------------------------------- 

void ScalarFieldExtractor::setVars()
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


  // get all of the ScalarField Variables
  for( int i = 0; i < (int)names.size(); i++ ){
    const TypeDescription *td = types[i];
    const TypeDescription *subtype = td->getSubType();
    cerr << "\tVariable: " << names[i] << ", type " << td->getName() << "\n";
    if( td->getType() ==  TypeDescription::NCVariable ||
	td->getType() ==  TypeDescription::CCVariable ){

      if( subtype->getType() == TypeDescription::double_type ||
	  subtype->getType() == TypeDescription::int_type ||
	  subtype->getType() == TypeDescription::long_type){

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
    sVar.set(names[index].c_str());
    type = types[index];
  }

  // get the number of materials for the ScalarField Variables
  
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

    TCL::execute(id + " setScalars " + sNames.c_str());
    TCL::execute(id + " buildVarList");

    TCL::execute("update idletasks");
    reset_vars();
  }
}



void ScalarFieldExtractor::execute() 
{ 
  tcl_status.set("Calling ScalarFieldExtractor!"); 

  in = (ArchiveIPort *) get_iport("Data Archive");
  sfout = (FieldOPort *) get_oport("Scalar Field");
  
  ArchiveHandle handle;
  if(!in->get(handle)){
    std::cerr<<"Didn't get a handle\n";
    return;
  }
   
  if (archiveH.get_rep() == 0 ){
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


  int max_workers = Max(Thread::numProcessors()/2, 8);
  Semaphore* thread_sema = scinew Semaphore( "scalar extractor semahpore",
					     max_workers); 

  GridP grid = archive.queryGrid(times[idx]);
  LevelP level = grid->getLevel( 0 );
  const TypeDescription* subtype = type->getSubType();
  string var(sVar.get());
  int mat = sMatNum.get();
  double time = times[idx];
  if(var != ""){
    switch( type->getType() ) {
    case TypeDescription::NCVariable:
      switch ( subtype->getType() ) {
      case TypeDescription::double_type:
	{	
	  LevelMeshHandle mesh = scinew LevelMesh( grid, 0 );
	  LevelField<double> *sfd =
	    scinew LevelField<double>( mesh, Field::NODE );
	  vector<ShareAssignArray3<double> > &data = sfd->fdata();
	  data.resize(level->numPatches());
	  vector<ShareAssignArray3<double> >::iterator it = data.begin();
	  for(Level::const_patchIterator r = level->patchesBegin();
	      r != level->patchesEnd(); r++, ++it){
	    thread_sema->down();
	    Thread *thrd =
	      scinew Thread(new PatchDataThread<NCVariable<double>,
			    vector<ShareAssignArray3<double> >::iterator>
			    (archive, it, var, mat, *r, time, thread_sema),
			    "patch_data_worker");
	    thrd->detach();
	  }
	  thread_sema->down(max_workers);
	  if( thread_sema ) delete thread_sema;
	  sfout->send(sfd);
	  return;
	}
      case TypeDescription::int_type:
	{
	  LevelMeshHandle mesh = scinew LevelMesh( grid, 0 );
	  LevelField<int> *sfd =
	    scinew LevelField<int>( mesh, Field::NODE );
	  vector<ShareAssignArray3<int> > &data = sfd->fdata();
	  data.resize(level->numPatches());
	  vector<ShareAssignArray3<int> >::iterator it = data.begin();
	  for(Level::const_patchIterator r = level->patchesBegin();
	      r != level->patchesEnd(); r++, ++it){
	    thread_sema->down();
	    Thread *thrd =
	      scinew Thread(new PatchDataThread<NCVariable<int>,
			    vector<ShareAssignArray3<int> >::iterator>
			    (archive, it, var, mat, *r, time, thread_sema),
			    "patch_data_worker");
	    thrd->detach();
	  }
	  thread_sema->down(max_workers);
	  if( thread_sema ) delete thread_sema;
	  sfout->send(sfd);
	  return;
	}
     case TypeDescription::long_type:
	{
	  LevelMeshHandle mesh = scinew LevelMesh( grid, 0 );
	  LevelField<long> *sfd =
	    scinew LevelField<long>( mesh, Field::NODE );
	  vector<ShareAssignArray3<long> > &data = sfd->fdata();
	  data.resize(level->numPatches());
	  vector<ShareAssignArray3<long> >::iterator it = data.begin();
	  for(Level::const_patchIterator r = level->patchesBegin();
	      r != level->patchesEnd(); r++, ++it){
	    thread_sema->down();
	    Thread *thrd =scinew Thread(new PatchDataThread<NCVariable<long>,
			                 vector<ShareAssignArray3<long> >::iterator>
			  (archive, it, var, mat, *r, time, thread_sema),
					"patch_data_worker");
	    thrd->detach();
	  }
	  thread_sema->down(max_workers);
	  if( thread_sema ) delete thread_sema;
	  sfout->send(sfd);
	  return;
	}
      default:
	cerr<<"NCScalarField<?>  Unknown scalar type\n";
	return;
      }
      break;
    case TypeDescription::CCVariable:
      switch ( subtype->getType() ) {
      case TypeDescription::double_type:
	{
	  LevelMeshHandle mesh = scinew LevelMesh( grid, 0 );
	  LevelField<double> *sfd =
	    scinew LevelField<double>( mesh, Field::CELL );
	  vector<ShareAssignArray3<double> >& data = sfd->fdata();
	  data.resize(level->numPatches());
	  vector<ShareAssignArray3<double> >::iterator it = data.begin();
	  for(Level::const_patchIterator r = level->patchesBegin();
	      r != level->patchesEnd(); r++, ++it ){
	    thread_sema->down();
	    Thread *thrd =scinew Thread(new PatchDataThread<CCVariable<double>,
			                 vector<ShareAssignArray3<double> >::iterator>
			  (archive, it, var, mat, *r, time, thread_sema),
			  "patch_data_worker");
	    thrd->detach();
	  }

	  thread_sema->down(max_workers);
	  if( thread_sema ) delete thread_sema;
	  sfout->send(sfd);
	  return;
	}
      case TypeDescription::int_type:
	{
	  LevelMeshHandle mesh = scinew LevelMesh( grid, 0 );
	  LevelField<int> *sfd =
	    scinew LevelField<int>( mesh, Field::CELL );
	  vector<ShareAssignArray3<int> >& data = sfd->fdata();
	  data.resize(level->numPatches());
	  vector<ShareAssignArray3<int> >::iterator it = data.begin();
	  for(Level::const_patchIterator r = level->patchesBegin();
	      r != level->patchesEnd(); r++, ++it ){
	    thread_sema->down();
	    Thread *thrd =scinew Thread(new PatchDataThread<CCVariable<int>,
			                 vector<ShareAssignArray3<int> >::iterator>
			  (archive, it, var, mat, *r, time, thread_sema),
			  "patch_data_worker");
	    thrd->detach();
	  }

	  thread_sema->down(max_workers);
	  if( thread_sema ) delete thread_sema;
	  sfout->send(sfd);
	  return;
	}
      case TypeDescription::long_type:
	{
	  LevelMeshHandle mesh = scinew LevelMesh( grid, 0 );
	  LevelField<long> *sfd =
	    scinew LevelField<long>( mesh, Field::CELL );
	  vector<ShareAssignArray3<long> >& data = sfd->fdata();
	  data.resize(level->numPatches());
	  vector<ShareAssignArray3<long> >::iterator it = data.begin();
	  for(Level::const_patchIterator r = level->patchesBegin();
	      r != level->patchesEnd(); r++, ++it ){
	    thread_sema->down();
	    Thread *thrd =scinew Thread(new PatchDataThread<CCVariable<long>,
			                 vector<ShareAssignArray3<long> >::iterator>
			  (archive, it, var, mat, *r, time, thread_sema),
			  "patch_data_worker");
	    thrd->detach();
	  }

	  thread_sema->down(max_workers);
	  if( thread_sema ) delete thread_sema;
	  sfout->send(sfd);
	  return;
	}
      default:
	cerr<<"CCScalarField<?> Unknown scalar type\n";
	return;
      }
      break;
    default:
      cerr<<"Not a ScalarField\n";
      return;
    }
  }
}
} // End namespace Uintah
