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
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>

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

  DECLARE_MAKER(VectorFieldExtractor)

//--------------------------------------------------------------- 
VectorFieldExtractor::VectorFieldExtractor(GuiContext* ctx) 
  : FieldExtractor("VectorFieldExtractor", ctx, "Selectors", "Uintah"),
    tcl_status(ctx->subVar("tcl_status")), sVar(ctx->subVar("sVar")),
    sMatNum(ctx->subVar("sMatNum")), type(0)
{ 
} 

//------------------------------------------------------------ 
VectorFieldExtractor::~VectorFieldExtractor(){} 

//------------------------------------------------------------- 

void VectorFieldExtractor::get_vars(vector< string >& names,
				   vector< const TypeDescription *>& types)
{
  string command;
  // Set up data to build or rebuild GUI interface
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
    sVar.set(names[index].c_str());
    type = types[index];
  }

  if( names.size() == 0 ) type = 0;
  // inherited from FieldExtractor
  update_GUI(sVar.get(), sNames);
}
//------------------------------------------------------------- 

void VectorFieldExtractor::execute() 
{ 
  tcl_status.set("Calling VectorFieldExtractor!"); 
  in = (ArchiveIPort *) get_iport("Data Archive");
  vfout = (FieldOPort *) get_oport("Vector Field");
  
  ArchiveHandle handle;
  if(!in->get(handle)){
    warning("VectorFieldExtractor::execute() Didn't get a handle");
    grid = 0;
    return;
  }
  
  if (archiveH.get_rep()  == 0 ){
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
  
  // set the index for the correct timestep.
  double dt = -1;
  if (timestep < (int)times.size() - 1)
    dt = times[timestep+1] - times[timestep];
  else if (times.size() > 1)
    dt = times[timestep] - times[timestep-1];
  
  LevelP level = grid->getLevel( level_.get() );
  IntVector hi, low, range;
  level->findIndexRange(low, hi);
  range = hi - low;
  BBox box;
  level->getSpatialRange(box);

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
      case TypeDescription::Vector:
	{	
	  NCVariable<Vector> gridVar;
	  LatVolField<Vector> *vfd =
	    scinew LatVolField<Vector>( mesh_handle_, Field::NODE );
	  // set the generation and timestep in the field
	  vfd->set_property("varname",string(var), true);
	  vfd->set_property("generation",generation, true);
	  vfd->set_property("timestep",timestep, true);
	  vfd->set_property("delta_t",dt, true);
	  vfd->set_property( "vartype",
			     int(TypeDescription::NCVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, vfd );
	  // send the field out to the port
	  vfout->send(vfd);
	  return;
	}
      default:
	warning("NCVariable<?>  Unknown vector type.");
	return;
      }
    case TypeDescription::CCVariable:
      if( mesh_handle_.get_rep() == 0 ){
	if(is_periodic_bcs(cellHi, hi)){
	  IntVector newrange(0,0,0);
	  get_periodic_bcs_range( cellHi, hi, range, newrange);
	  mesh_handle_ = scinew LatVolMesh(newrange.x(), newrange.y(),
					   newrange.z(), box.min(),
					   box.max());
	} else {
	  mesh_handle_ = scinew LatVolMesh(range.x(), range.y(),
					   range.z(), box.min(),
					   box.max());
	}
      } else if(mesh_handle_->get_ni() != range.x() ||
		mesh_handle_->get_nj() != range.y() ||
		mesh_handle_->get_nk() != range.z() ){
	if(is_periodic_bcs(cellHi, hi)){
	  IntVector newrange(0,0,0);
	  get_periodic_bcs_range( cellHi, hi, range, newrange);
	  mesh_handle_ = scinew LatVolMesh(newrange.x(), newrange.y(),
					   newrange.z(), box.min(),
					   box.max());
	} else {
	  mesh_handle_ = scinew LatVolMesh(range.x(), range.y(),
					   range.z(), box.min(),
					   box.max());
	}
      }
      switch ( subtype->getType() ) {
      case TypeDescription::Vector:
	{	
	  CCVariable<Vector> gridVar;
	  LatVolField<Vector> *vfd =
	    scinew LatVolField<Vector>( mesh_handle_, Field::CELL );
	  // set the generation and timestep in the field
	  vfd->set_property("varname",string(var), true);
	  vfd->set_property("generation",generation, true);
	  vfd->set_property("timestep",timestep, true);
	  vfd->set_property("delta_t",dt, true);
	  vfd->set_property( "vartype",
			     int(TypeDescription::CCVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, vfd );
	  // send the field out to the port
	  vfout->send(vfd);
	  return;
	}
      default:
	warning("CCVariable<?> Unknown vector type.");
	return;
      }
    case TypeDescription::SFCXVariable:
      if( mesh_handle_.get_rep() == 0 ){
	mesh_handle_ = scinew LatVolMesh(range.x(), range.y()-1,
					 range.z()-1, box.min(),
					 box.max());
      } else if(mesh_handle_->get_ni() != range.x() ||
	       mesh_handle_->get_nj() != range.y()-1 ||
	       mesh_handle_->get_nk() != range.z()-1 ){
	mesh_handle_ = scinew LatVolMesh(range.x(), range.y()-1,
					 range.z()-1, box.min(),
					 box.max());
      }
      switch ( subtype->getType() ) {
      case TypeDescription::Vector:
	{	
	  SFCXVariable<Vector> gridVar;
	  LatVolField<Vector> *vfd =
	    scinew LatVolField<Vector>( mesh_handle_, Field::NODE );
	  // set the generation and timestep in the field
	  vfd->set_property("varname",string(var), true);
	  vfd->set_property("generation",generation, true);
	  vfd->set_property("timestep",timestep, true);
	  vfd->set_property("delta_t",dt, true);
	  vfd->set_property( "vartype",
			     int(TypeDescription::SFCXVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, vfd );
	  // send the field out to the port
	  vfout->send(vfd);
	  return;
	}
      default:
	warning("SFCXVariable<?> Unknown vector type.");
	return;
      }
    case TypeDescription::SFCYVariable:
      if( mesh_handle_.get_rep() == 0 ){
	mesh_handle_ = scinew LatVolMesh(range.x()-1, range.y(),
					 range.z()-1, box.min(),
					 box.max());
      } else if(mesh_handle_->get_ni() != range.x()-1 ||
	       mesh_handle_->get_nj() != range.y() ||
	       mesh_handle_->get_nk() != range.z()-1 ){
	mesh_handle_ = scinew LatVolMesh(range.x()-1, range.y(),
					 range.z()-1, box.min(),
					 box.max());
      }
      switch ( subtype->getType() ) {
      case TypeDescription::Vector:
	{	
	  SFCYVariable<Vector> gridVar;
	  LatVolField<Vector> *vfd =
	    scinew LatVolField<Vector>( mesh_handle_, Field::NODE );
	  // set the generation and timestep in the field
	  vfd->set_property("varname",string(var), true);
	  vfd->set_property("generation",generation, true);
	  vfd->set_property("timestep",timestep, true);
	  vfd->set_property("delta_t",dt, true);
	  vfd->set_property( "vartype",
			     int(TypeDescription::SFCYVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, vfd );
	  // send the field out to the port
	  vfout->send(vfd);
	  return;
	}
      default:
	warning("SFCYVariable<?> Unknown vector type.");
	return;
      }
    case TypeDescription::SFCZVariable:
      if( mesh_handle_.get_rep() == 0 ){
	mesh_handle_ = scinew LatVolMesh(range.x()-1, range.y()-1,
					 range.z(), box.min(),
					 box.max());
      } else if(mesh_handle_->get_ni() != range.x()-1 ||
	       mesh_handle_->get_nj() != range.y()-1 ||
	       mesh_handle_->get_nk() != range.z() ){
	mesh_handle_ = scinew LatVolMesh(range.x()-1, range.y()-1,
					 range.z(), box.min(),
					 box.max());
      }
      switch ( subtype->getType() ) {
      case TypeDescription::Vector:
	{	
	  SFCZVariable<Vector> gridVar;
	  LatVolField<Vector> *vfd =
	    scinew LatVolField<Vector>( mesh_handle_, Field::NODE );
	  // set the generation and timestep in the field
	  vfd->set_property("varname",string(var), true);
	  vfd->set_property("generation",generation, true);
	  vfd->set_property("timestep",timestep, true);
	  vfd->set_property("delta_t",dt, true);
	  vfd->set_property( "vartype",
			     int(TypeDescription::SFCZVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, vfd );
	  // send the field out to the port
	  vfout->send(vfd);
	  return;
	}
      default:
	warning("SFCZVariable<?> Unknown vector type.");
	return;
      }

    default:
      warning("Not a VectorField.");
      return;
    }
  }
}
} // End namespace Uintah
