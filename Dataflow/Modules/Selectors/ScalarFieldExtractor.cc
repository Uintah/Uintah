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
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/BBox.h>
#include <Core/Util/Timer.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Packages/Uintah/Core/Grid/ShareAssignArray3.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
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

  DECLARE_MAKER(ScalarFieldExtractor)

//--------------------------------------------------------------- 
ScalarFieldExtractor::ScalarFieldExtractor(GuiContext* ctx) 
  : FieldExtractor("ScalarFieldExtractor", ctx, "Selectors", "Uintah"),
    tcl_status(ctx->subVar("tcl_status")), sVar(ctx->subVar("sVar")),
    sMatNum(ctx->subVar("sMatNum")), type(0)
{ 
} 

//------------------------------------------------------------ 
ScalarFieldExtractor::~ScalarFieldExtractor(){} 

//------------------------------------------------------------- 
void ScalarFieldExtractor::get_vars(vector< string >& names,
				   vector< const TypeDescription *>& types)
{
  string command;
  // Set up data to build or rebuild GUI interface
  string sNames("");
  int index = -1;
  bool matches = false;
  // get all of the ScalarField Variables
  for( int i = 0; i < (int)names.size(); i++ ){
    const TypeDescription *td = types[i];
    const TypeDescription *subtype = td->getSubType();
    //  only handle NC and CC Vars
    if( td->getType() ==  TypeDescription::NCVariable ||
	td->getType() ==  TypeDescription::CCVariable ||
 	td->getType() ==  TypeDescription::SFCXVariable ||
	td->getType() ==  TypeDescription::SFCYVariable ||
	td->getType() ==  TypeDescription::SFCZVariable )
    {
      // supported scalars double, int, long64, long long, short, bool
      if( subtype->getType() == TypeDescription::double_type ||
	  subtype->getType() == TypeDescription::int_type ||
  	  subtype->getType() == TypeDescription::long64_type ||
  	  subtype->getType() == TypeDescription::bool_type) {
//  	  subtype->getType() == TypeDescription::short_int_type ||
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

  // inherited from FieldExtractor
  update_GUI(sVar.get(), sNames);
}
//------------------------------------------------------------- 

void ScalarFieldExtractor::execute() 
{ 
  tcl_status.set("Calling ScalarFieldExtractor!"); 
  in = (ArchiveIPort *) get_iport("Data Archive");
  sfout = (FieldOPort *) get_oport("Scalar Field");
  
  ArchiveHandle handle;
  if(!in->get(handle)){
    warning("ScalarFieldExtractor::execute() Didn't get a handle.");
    return;
  }
   
  if (archiveH.get_rep() == 0 ){
    // first time through a frame must be built
    build_GUI_frame();
  }

  
  archiveH = handle;
  DataArchive& archive = *((*(archiveH.get_rep()))());

  // get time, set timestep, set generation, update grid and update gui
  double time = field_update(); // yeah it does all that

  if(type == 0){
    warning( "No variables found.");
    return;
  }
  
  LevelP level = grid->getLevel( level_.get() );
  IntVector hi, low, range;
  level->findIndexRange(low, hi);
  range = hi - low;
  BBox box;
  level->getSpatialRange(box);
  //  cerr<<"level  = "<<level_.get()<<" box: min("<<box.min()<<"), max("<<box.max()<<"), index range is imin = "<<low<<", imax = "<<hi<<", range = "<<range<<"\n";

  IntVector cellHi, cellLo;
  level->findCellIndexRange(cellLo, cellHi);

  const TypeDescription* subtype = type->getSubType();
  string var(sVar.get());
  int mat = sMatNum.get();
  if(var != ""){
    switch( type->getType() ) {
    case TypeDescription::NCVariable:
      if( mesh_handle_.get_rep() == 0 ){
	mesh_handle_ = scinew LatVolMesh(range.x(), range.y(),
					 range.z(), box.min(),
					 box.max());
      }else if(mesh_handle_->get_ni() != range.x() ||
	       mesh_handle_->get_nj() != range.y() ||
	       mesh_handle_->get_nk() != range.z() ){
	mesh_handle_ = scinew LatVolMesh(range.x(), range.y(),
					 range.z(), box.min(),
					 box.max());
      }
      switch ( subtype->getType() ) {
      case TypeDescription::double_type:
	{
	  NCVariable<double> gridVar;
	  LatVolField<double> *sfd =
	    scinew LatVolField<double>( mesh_handle_, Field::NODE );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "vartype",int(TypeDescription::NCVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  sfout->send(sfd);
	  return;
	}
      case TypeDescription::int_type:
	{
	  NCVariable<int> gridVar;
	  LatVolField<int> *sfd =
	    scinew LatVolField<int>( mesh_handle_, Field::NODE );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "vartype",int(TypeDescription::NCVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  sfout->send(sfd);
	  return;
	}
     case TypeDescription::long64_type:
	{
	  NCVariable<long64> gridVar;
	  LatVolField<long64> *sfd =
	    scinew LatVolField<long64>( mesh_handle_, Field::NODE );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "vartype",int(TypeDescription::NCVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  sfout->send(sfd);
	  return;
	}
      default:
	error("NCScalarField<?>  Unknown scalar type.");
	return;
      }
    case TypeDescription::CCVariable:
      if( mesh_handle_.get_rep() == 0 ){
	if(cellHi == hi){
	  mesh_handle_ = scinew LatVolMesh(range.x()+1, range.y()+1,
					   range.z()+1, box.min(),
					   box.max());
	} else {
	  mesh_handle_ = scinew LatVolMesh(range.x(), range.y(),
					   range.z(), box.min(),
					   box.max());
	}
      } else if(mesh_handle_->get_ni() != range.x() ||
		mesh_handle_->get_nj() != range.y() ||
		mesh_handle_->get_nk() != range.z() ){
	if(cellHi == hi){
	  mesh_handle_ = scinew LatVolMesh(range.x()+1, range.y()+1,
					   range.z()+1, box.min(),
					   box.max());
	} else {
	  mesh_handle_ = scinew LatVolMesh(range.x(), range.y(),
					   range.z(), box.min(),
					   box.max());
	}
      }	
      switch ( subtype->getType() ) {
      case TypeDescription::double_type:
	{
	  CCVariable<double> gridVar;
	  LatVolField<double> *sfd =
	    scinew LatVolField<double>( mesh_handle_, Field::CELL );
	  
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "vartype",int(TypeDescription::CCVariable),true);
 	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
 	  sfout->send(sfd);
	  return;
	}
      case TypeDescription::int_type:
	{
	  CCVariable<int> gridVar;
	  LatVolField<int> *sfd =
	    scinew LatVolField<int>( mesh_handle_, Field::CELL );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "vartype",int(TypeDescription::CCVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  sfout->send(sfd);
	  return;
	}
      case TypeDescription::bool_type:
	{
	  CCVariable<unsigned char> gridVar;
	  LatVolField<unsigned char> *sfd =
	    scinew LatVolField<unsigned char>( mesh_handle_, Field::CELL );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "vartype",int(TypeDescription::CCVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  sfout->send(sfd);
	  return;
	}
      case TypeDescription::long64_type:
      case TypeDescription::long_type:
	{
	  CCVariable<long64> gridVar;
	  LatVolField<long64> *sfd =
	    scinew LatVolField<long64>( mesh_handle_, Field::CELL );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "vartype",int(TypeDescription::CCVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  sfout->send(sfd);
	  return;
	}
      default:
	error("CCScalarField<?> Unknown scalar type.");
	return;
      }
    case TypeDescription::SFCXVariable:
      if( mesh_handle_.get_rep() == 0 ){
	mesh_handle_ = scinew LatVolMesh(range.x(), range.y() - 1,
					 range.z() - 1, box.min(),
					 box.max());
	  } else if(mesh_handle_->get_ni() != range.x() ||
		    mesh_handle_->get_nj() != range.y() -1 ||
		    mesh_handle_->get_nk() != range.z() -1 ){
	    mesh_handle_ = scinew LatVolMesh(range.x(), range.y() - 1,
					     range.z()-1, box.min(),
					     box.max());
	  }
      switch ( subtype->getType() ) {
      case TypeDescription::double_type:
	{
	  SFCXVariable<double> gridVar;
	  LatVolField<double> *sfd =
	    scinew LatVolField<double>( mesh_handle_, Field::NODE );
	  
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "vartype",
			     int(TypeDescription::SFCXVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  sfout->send(sfd);
	  return;
	}
      case TypeDescription::int_type:
	{
	  SFCXVariable<int> gridVar;
	  LatVolField<int> *sfd =
	    scinew LatVolField<int>( mesh_handle_, Field::NODE );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "vartype",
			     int(TypeDescription::SFCXVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  sfout->send(sfd);
	  return;
	}
      case TypeDescription::long64_type:
      case TypeDescription::long_type:
	{
	  SFCXVariable<long64> gridVar;
	  LatVolField<long64> *sfd =
	    scinew LatVolField<long64>( mesh_handle_, Field::NODE );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "vartype",
			     int(TypeDescription::SFCXVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  sfout->send(sfd);
	  return;
	}
      default:
	error("SFCXScalarField<?> Unknown scalar type.");
	return;
      }
    case TypeDescription::SFCYVariable:
	  if( mesh_handle_.get_rep() == 0 ){
	    mesh_handle_ = scinew LatVolMesh(range.x()-1, range.y(),
					     range.z()-1, box.min(),
					     box.max());
	  } else if(mesh_handle_->get_ni() != range.x() -1 ||
		    mesh_handle_->get_nj() != range.y() ||
		    mesh_handle_->get_nk() != range.z() -1 ){
	    mesh_handle_ = scinew LatVolMesh(range.x()-1, range.y(),
					     range.z()-1, box.min(),
					     box.max());
	  }
      switch ( subtype->getType() ) {
      case TypeDescription::double_type:
	{
	  SFCYVariable<double> gridVar;
	  LatVolField<double> *sfd =
	    scinew LatVolField<double>( mesh_handle_, Field::NODE );
	  
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "vartype",
			     int(TypeDescription::SFCYVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  sfout->send(sfd);
	  return;
	}
      case TypeDescription::int_type:
	{
	  SFCYVariable<int> gridVar;
	  LatVolField<int> *sfd =
	    scinew LatVolField<int>( mesh_handle_, Field::NODE );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "vartype",
			     int(TypeDescription::SFCYVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  sfout->send(sfd);
	  return;
	}
      case TypeDescription::long64_type:
      case TypeDescription::long_type:
	{
	  SFCYVariable<long64> gridVar;
	  LatVolField<long64> *sfd =
	    scinew LatVolField<long64>( mesh_handle_, Field::NODE );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "vartype",
			     int(TypeDescription::SFCYVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  sfout->send(sfd);
	  return;
	}
      default:
	error("SFCYScalarField<?> Unknown scalar type.");
	return;
      }
    case TypeDescription::SFCZVariable:
	  if( mesh_handle_.get_rep() == 0 ){
	    mesh_handle_ = scinew LatVolMesh(range.x()-1, range.y()-1,
					     range.z(), box.min(),
					     box.max());
 	  } else if(mesh_handle_->get_ni() != range.x() -1 ||
		    mesh_handle_->get_nj() != range.y() -1 ||
		    mesh_handle_->get_nk() != range.z() ){
	    mesh_handle_ = scinew LatVolMesh(range.x()-1, range.y()-1,
					     range.z(), box.min(),
					     box.max());
	  }     switch ( subtype->getType() ) {
      case TypeDescription::double_type:
	{
	  SFCZVariable<double> gridVar;
	  LatVolField<double> *sfd =
	    scinew LatVolField<double>( mesh_handle_, Field::NODE );
	  
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "vartype",
			     int(TypeDescription::SFCZVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  sfout->send(sfd);
	  return;
	}
      case TypeDescription::int_type:
	{
	  SFCZVariable<int> gridVar;
	  LatVolField<int> *sfd =
	    scinew LatVolField<int>( mesh_handle_, Field::NODE );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "vartype",
			     int(TypeDescription::SFCZVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  sfout->send(sfd);
	  return;
	}
      case TypeDescription::long64_type:
      case TypeDescription::long_type:
	{
	  SFCZVariable<long64> gridVar;
	  LatVolField<long64> *sfd =
	    scinew LatVolField<long64>( mesh_handle_, Field::NODE );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "vartype",
			     int(TypeDescription::SFCZVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  sfout->send(sfd);
	  return;
	}
      default:
	error("SFCZScalarField<?> Unknown scalar type.");
	return;
      }
    default:
      error("Not a ScalarField.");
      return;
    }
  }
}

} // End namespace Uintah
