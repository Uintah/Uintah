/****************************************
CLASS
    FieldExtractor

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
#include <Core/Geometry/Transform.h>
#include <Packages/Uintah/Core/Grid/ShareAssignArray3.h>
//#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include "FieldExtractor.h"
 
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

  //using DumbField;



//--------------------------------------------------------------- 
  FieldExtractor::FieldExtractor(const string& name,
				 GuiContext* ctx,
				 const string& cat,
				 const string& pack)
  : Module(name, ctx, Filter, cat, pack),
    generation(-1),  timestep(-1), material(-1), levelnum(0),
    level_(ctx->subVar("level")), grid(0), 
    archiveH(0), mesh_handle_(0),
    tcl_status(ctx->subVar("tcl_status")), 
    sVar(ctx->subVar("sVar")),
    sMatNum(ctx->subVar("sMatNum")),
    type(0)

{ 

} 

//------------------------------------------------------------ 
FieldExtractor::~FieldExtractor(){} 

//------------------------------------------------------------- 
void FieldExtractor::build_GUI_frame()
{
  // create the variable extractor interface.
  string visible;
  gui->eval(id + " isVisible", visible);
  if( visible == "0" ){
    gui->execute(id + " buildTopLevel");
  }
}

//------------------------------------------------------------- 
// get time, set timestep, set generation, update grid and update gui
double FieldExtractor::field_update()
{
   DataArchive& archive = *((*(archiveH.get_rep()))());
   // set the index for the correct timestep.
   int new_timestep = archiveH->timestep();
   vector< const TypeDescription *> types;
   vector< string > names;
   vector< int > indices;
   double time;
   // check to see if we have a new Archive
   archive.queryVariables(names, types);
   int new_generation = archiveH->generation;
   bool archive_dirty =  new_generation != generation;
   if (archive_dirty) {
     generation = new_generation;
     timestep = -1; // make sure old timestep is different from current
     times.clear();
     mesh_handle_ = 0;
     archive.queryTimesteps( indices, times );
   }

   if (timestep != new_timestep) {
     time = times[new_timestep];
     grid = archive.queryGrid(time);
//      BBox gbox; grid->getSpatialRange(gbox);
     //     cerr<<"box: min("<<gbox.min()<<"), max("<<gbox.max()<<")\n";
     timestep = new_timestep;
   } else {
     time = times[timestep];
   }

   // Deal with changed level information
   int n = grid->numLevels();
   if (level_.get() >= (n-1)){
     level_.set(n-1);
   }
   if (levelnum != level_.get() ){
     mesh_handle_ = 0;
     levelnum = level_.get();
   }

   get_vars( names, types );
   return time;
}

//------------------------------------------------------------- 
void FieldExtractor::update_GUI(const string& var,
			       const string& varnames)
  // update the variable list for the GUI
{
  DataArchive& archive = *((*(archiveH.get_rep()))());
  int levels = grid->numLevels();
  LevelP level = grid->getLevel( level_.get() );

  Patch* r = *(level->patchesBegin());
  ConsecutiveRangeSet matls = 
    archive.queryMaterials(var, r, times[timestep]);

  ostringstream os;
  os << levels;

  string visible;
  gui->eval(id + " isVisible", visible);
  if( visible == "1"){
    gui->execute(id + " destroyFrames");
    gui->execute(id + " build");
    gui->execute(id + " buildLevels "+ os.str());
    gui->execute(id + " buildMaterials " + matls.expandedString().c_str());
      
    gui->execute(id + " setVars " + varnames.c_str());
    gui->execute(id + " buildVarList");
      
    gui->execute("update idletasks");
    reset_vars();
  }
}

bool 
FieldExtractor::is_periodic_bcs(IntVector cellir, IntVector ir)
{
  if( cellir.x() == ir.x() ||
      cellir.y() == ir.y() ||
      cellir.z() == ir.z() )
    return true;
  else
    return false;
}

void 
FieldExtractor::get_periodic_bcs_range(IntVector cellmax, IntVector datamax,
				       IntVector range, IntVector& newrange)
{
  if( cellmax.x() == datamax.x())
    newrange.x( range.x() + 1 );
  else
    newrange.x( range.x() );
  if( cellmax.y() == datamax.y())
    newrange.y( range.y() + 1 );
  else
    newrange.y( range.y() );
  if( cellmax.z() == datamax.z())
    newrange.z( range.z() + 1 );
  else
    newrange.z( range.z() );
}

void
FieldExtractor::execute()
{ 
  tcl_status.set("Calling FieldExtractor!"); 
  ArchiveIPort *in = (ArchiveIPort *) get_iport("Data Archive");
  FieldOPort *fout = (FieldOPort *) get_oport("Field");

  ArchiveHandle handle;
  if(!in->get(handle)){
    warning("VariableExtractor::execute() Didn't get a handle.");
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
      }else if(mesh_handle_->get_ni() != (unsigned int) range.x() ||
	       mesh_handle_->get_nj() != (unsigned int) range.y() ||
	       mesh_handle_->get_nk() != (unsigned int) range.z() ){
	mesh_handle_ = scinew LatVolMesh(range.x(), range.y(),
					 range.z(), box.min(),
					 box.max());
      }
      switch ( subtype->getType() ) {
      case TypeDescription::double_type:
	{
	  NCVariable<double> gridVar;
	  LatVolField<double> *sfd =
	    scinew LatVolField<double>( mesh_handle_, 1 );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  // 	  sfd->set_property( "offset", IntVector(low), true);
	  sfd->set_property( "vartype",int(TypeDescription::NCVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  fout->send(sfd);
	  return;
	}
      case TypeDescription::float_type:
	{
	  NCVariable<float> gridVar;
	  LatVolField<float> *sfd =
	    scinew LatVolField<float>( mesh_handle_, 1 );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "offset", IntVector(low), true);
	  sfd->set_property( "vartype",int(TypeDescription::NCVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  fout->send(sfd);
	  return;
	}
      case TypeDescription::int_type:
	{
	  NCVariable<int> gridVar;
	  LatVolField<int> *sfd =
	    scinew LatVolField<int>( mesh_handle_, 1 );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "offset", IntVector(low), true);
	  sfd->set_property( "vartype",int(TypeDescription::NCVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  fout->send(sfd);
	  return;
	}
      case TypeDescription::long64_type:
	{
	  NCVariable<long64> gridVar;
	  LatVolField<long64> *sfd =
	    scinew LatVolField<long64>( mesh_handle_, 1 );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "offset", IntVector(low), true);
	  sfd->set_property( "vartype",int(TypeDescription::NCVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  fout->send(sfd);
	  return;
	}
      case TypeDescription::Vector:
	{	
	  NCVariable<Vector> gridVar;
	  LatVolField<Vector> *vfd =
	    scinew LatVolField<Vector>( mesh_handle_, 1 );
	  // set the generation and timestep in the field
	  vfd->set_property("varname",string(var), true);
	  vfd->set_property("generation",generation, true);
	  vfd->set_property("timestep",timestep, true);
	  vfd->set_property( "offset", IntVector(low), true);
	  vfd->set_property("delta_t",dt, true);
	  vfd->set_property( "vartype",
			     int(TypeDescription::NCVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, vfd );
	  // send the field out to the port
	  fout->send(vfd);
	  return;
	}
      case TypeDescription::Matrix3:
	{	
	  NCVariable<Matrix3> gridVar;
	  LatVolField<Matrix3> *tfd =
	    scinew LatVolField<Matrix3>( mesh_handle_, 1 );
	  // set the generation and timestep in the field
	  tfd->set_property( "vartype",
			     int(TypeDescription::NCVariable),true);
	  tfd->set_property( "offset", IntVector(low), true);
	  build_field( archive, level, low, var, mat, time, gridVar, tfd );
	  // send the field out to the port
	  fout->send(tfd);
	  // 	DumpAllocator(default_allocator, "TensorDump.allocator");
	  return;
	}
      default:
	error("NCVariable<?>  Unknown scalar type.");
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
      } else if(mesh_handle_->get_ni() != (unsigned int) range.x() ||
		mesh_handle_->get_nj() != (unsigned int) range.y() ||
		mesh_handle_->get_nk() != (unsigned int) range.z() ){
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
      case TypeDescription::double_type:
	{
	  CCVariable<double> gridVar;
	  LatVolField<double> *sfd =
	    scinew LatVolField<double>( mesh_handle_, 0 );
	  
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "offset", IntVector(low), true);
	  sfd->set_property( "vartype",int(TypeDescription::CCVariable),true);
 	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
 	  fout->send(sfd);
	  return;
	}
      case TypeDescription::float_type:
	{
	  CCVariable<float> gridVar;
	  LatVolField<float> *sfd =
	    scinew LatVolField<float>( mesh_handle_, 0 );
	  
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "offset", IntVector(low), true);
	  sfd->set_property( "vartype",int(TypeDescription::CCVariable),true);
 	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
 	  fout->send(sfd);
	  return;
	}
      case TypeDescription::int_type:
	{
	  CCVariable<int> gridVar;
	  LatVolField<int> *sfd =
	    scinew LatVolField<int>( mesh_handle_, 0 );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "offset", IntVector(low), true);
	  sfd->set_property( "vartype",int(TypeDescription::CCVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  fout->send(sfd);
	  return;
	}
      case TypeDescription::bool_type:
	{
	  CCVariable<unsigned char> gridVar;
	  LatVolField<unsigned char> *sfd =
	    scinew LatVolField<unsigned char>( mesh_handle_, 0 );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "offset", IntVector(low), true);
	  sfd->set_property( "vartype",int(TypeDescription::CCVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  fout->send(sfd);
	  return;
	}
      case TypeDescription::long64_type:
      case TypeDescription::long_type:
	{
	  CCVariable<long64> gridVar;
	  LatVolField<long64> *sfd =
	    scinew LatVolField<long64>( mesh_handle_, 0 );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "offset", IntVector(low), true);
	  sfd->set_property( "vartype",int(TypeDescription::CCVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  fout->send(sfd);
	  return;
	}
      case TypeDescription::Vector:
	{	
	  CCVariable<Vector> gridVar;
	  LatVolField<Vector> *vfd =
	    scinew LatVolField<Vector>( mesh_handle_, 0 );
	  // set the generation and timestep in the field
	  vfd->set_property("varname",string(var), true);
	  vfd->set_property("generation",generation, true);
	  vfd->set_property("timestep",timestep, true);
	  vfd->set_property( "offset", IntVector(low), true);
	  vfd->set_property("delta_t",dt, true);
	  vfd->set_property( "vartype",
			     int(TypeDescription::CCVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, vfd );
	  // send the field out to the port
	  fout->send(vfd);
	  return;
	}
      case TypeDescription::Matrix3:
	{
	  CCVariable<Matrix3> gridVar;
	  LatVolField<Matrix3> *tfd =
	    scinew LatVolField<Matrix3>( mesh_handle_, 0 );
	  // set the generation and timestep in the field
	  tfd->set_property( "vartype",
			     int(TypeDescription::CCVariable),true);
	  tfd->set_property( "offset", IntVector(low), true);
	  build_field( archive, level, low, var, mat, time, gridVar, tfd );
	  // send the field out to the port
	  fout->send(tfd);
	  // 	DumpAllocator(default_allocator, "TensorDump.allocator");
	  return;
	}
      default:
	error("CCVariable<?> Unknown scalar type.");
	return;
      }
    case TypeDescription::SFCXVariable:
      if( mesh_handle_.get_rep() == 0 ){
	mesh_handle_ = scinew LatVolMesh(range.x(), range.y() - 1,
					 range.z() - 1, box.min(),
					 box.max());
      } else if(mesh_handle_->get_ni() != (unsigned int) range.x() ||
		mesh_handle_->get_nj() != (unsigned int) range.y() -1 ||
		mesh_handle_->get_nk() != (unsigned int) range.z() -1 ){
	mesh_handle_ = scinew LatVolMesh(range.x(), range.y() - 1,
					 range.z()-1, box.min(),
					 box.max());
      }
      switch ( subtype->getType() ) {
      case TypeDescription::double_type:
	{
	  SFCXVariable<double> gridVar;
	  LatVolField<double> *sfd =
	    scinew LatVolField<double>( mesh_handle_, 1 );
	  
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "offset", IntVector(low), true);
	  sfd->set_property( "vartype",
			     int(TypeDescription::SFCXVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  fout->send(sfd);
	  return;
	}
      case TypeDescription::float_type:
	{
	  SFCXVariable<float> gridVar;
	  LatVolField<float> *sfd =
	    scinew LatVolField<float>( mesh_handle_, 1 );
	  
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "offset", IntVector(low), true);
	  sfd->set_property( "vartype",
			     int(TypeDescription::SFCXVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  fout->send(sfd);
	  return;
	}
      case TypeDescription::int_type:
	{
	  SFCXVariable<int> gridVar;
	  LatVolField<int> *sfd =
	    scinew LatVolField<int>( mesh_handle_, 1 );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "offset", IntVector(low), true);
	  sfd->set_property( "vartype",
			     int(TypeDescription::SFCXVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  fout->send(sfd);
	  return;
	}
      case TypeDescription::long64_type:
      case TypeDescription::long_type:
	{
	  SFCXVariable<long64> gridVar;
	  LatVolField<long64> *sfd =
	    scinew LatVolField<long64>( mesh_handle_, 1 );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "offset", IntVector(low), true);
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "vartype",
			     int(TypeDescription::SFCXVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  fout->send(sfd);
	  return;
	}
      case TypeDescription::Vector:
	{	
	  SFCXVariable<Vector> gridVar;
	  LatVolField<Vector> *vfd =
	    scinew LatVolField<Vector>( mesh_handle_, 1 );
	  // set the generation and timestep in the field
	  vfd->set_property("varname",string(var), true);
	  vfd->set_property("generation",generation, true);
	  vfd->set_property("timestep",timestep, true);
	  vfd->set_property( "offset", IntVector(low), true);
	  vfd->set_property("delta_t",dt, true);
	  vfd->set_property( "vartype",
			     int(TypeDescription::SFCXVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, vfd );
	  // send the field out to the port
	  fout->send(vfd);
	  return;
	}
      case TypeDescription::Matrix3:
	{
	  SFCXVariable<Matrix3> gridVar;
	  LatVolField<Matrix3> *tfd =
	    scinew LatVolField<Matrix3>( mesh_handle_, 0 );
	  // set the generation and timestep in the field
	  tfd->set_property( "vartype",
			     int(TypeDescription::SFCXVariable),true);
	  tfd->set_property( "offset", IntVector(low), true);
	  build_field( archive, level, low, var, mat, time, gridVar, tfd );
	  // send the field out to the port
	  fout->send(tfd);
	  // 	DumpAllocator(default_allocator, "TensorDump.allocator");
	  return;
	}
      default:
	error("SFCXVariable<?> Unknown scalar type.");
	return;
      }
    case TypeDescription::SFCYVariable:
      if( mesh_handle_.get_rep() == 0 ){
	mesh_handle_ = scinew LatVolMesh(range.x()-1, range.y(),
					 range.z()-1, box.min(),
					 box.max());
      } else if(mesh_handle_->get_ni() != (unsigned int) range.x() -1 ||
		mesh_handle_->get_nj() != (unsigned int) range.y() ||
		mesh_handle_->get_nk() != (unsigned int) range.z() -1 ){
	mesh_handle_ = scinew LatVolMesh(range.x()-1, range.y(),
					 range.z()-1, box.min(),
					 box.max());
      }
      switch ( subtype->getType() ) {
      case TypeDescription::double_type:
	{
	  SFCYVariable<double> gridVar;
	  LatVolField<double> *sfd =
	    scinew LatVolField<double>( mesh_handle_, 1 );
	  
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "offset", IntVector(low), true);
	  sfd->set_property( "vartype",
			     int(TypeDescription::SFCYVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  fout->send(sfd);
	  return;
	}
      case TypeDescription::float_type:
	{
	  SFCYVariable<float> gridVar;
	  LatVolField<float> *sfd =
	    scinew LatVolField<float>( mesh_handle_, 1 );
	  
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "offset", IntVector(low), true);
	  sfd->set_property( "vartype",
			     int(TypeDescription::SFCYVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  fout->send(sfd);
	  return;
	}
      case TypeDescription::int_type:
	{
	  SFCYVariable<int> gridVar;
	  LatVolField<int> *sfd =
	    scinew LatVolField<int>( mesh_handle_, 1 );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "offset", IntVector(low), true);
	  sfd->set_property( "vartype",
			     int(TypeDescription::SFCYVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  fout->send(sfd);
	  return;
	}
      case TypeDescription::long64_type:
      case TypeDescription::long_type:
	{
	  SFCYVariable<long64> gridVar;
	  LatVolField<long64> *sfd =
	    scinew LatVolField<long64>( mesh_handle_, 1 );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "offset", IntVector(low), true);
	  sfd->set_property( "vartype",
			     int(TypeDescription::SFCYVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  fout->send(sfd);
	  return;
	}
      case TypeDescription::Vector:
	{	
	  SFCYVariable<Vector> gridVar;
	  LatVolField<Vector> *vfd =
	    scinew LatVolField<Vector>( mesh_handle_, 1 );
	  // set the generation and timestep in the field
	  vfd->set_property("varname",string(var), true);
	  vfd->set_property("generation",generation, true);
	  vfd->set_property("timestep",timestep, true);
	  vfd->set_property( "offset", IntVector(low), true);
	  vfd->set_property("delta_t",dt, true);
	  vfd->set_property( "vartype",
			     int(TypeDescription::SFCYVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, vfd );
	  // send the field out to the port
	  fout->send(vfd);
	  return;
	}
      case TypeDescription::Matrix3:
	{
	  SFCYVariable<Matrix3> gridVar;
	  LatVolField<Matrix3> *tfd =
	    scinew LatVolField<Matrix3>( mesh_handle_, 1 );
	  // set the generation and timestep in the field
	  tfd->set_property( "vartype",
			     int(TypeDescription::SFCYVariable),true);
	  tfd->set_property( "offset", IntVector(low), true);
	  build_field( archive, level, low, var, mat, time, gridVar, tfd );
	  // send the field out to the port
	  fout->send(tfd);
	  // 	DumpAllocator(default_allocator, "TensorDump.allocator");
	  return;
	}
      default:
	error("SFCYVariable<?> Unknown scalar type.");
	return;
      }
    case TypeDescription::SFCZVariable:
      if( mesh_handle_.get_rep() == 0 ){
	mesh_handle_ = scinew LatVolMesh(range.x()-1, range.y()-1,
					 range.z(), box.min(),
					 box.max());
      } else if(mesh_handle_->get_ni() != (unsigned int) range.x() -1 ||
		mesh_handle_->get_nj() != (unsigned int) range.y() -1 ||
		mesh_handle_->get_nk() != (unsigned int) range.z() ){
	mesh_handle_ = scinew LatVolMesh(range.x()-1, range.y()-1,
					 range.z(), box.min(),
					 box.max());
      }     
      switch ( subtype->getType() ) {
      case TypeDescription::double_type:
	{
	  SFCZVariable<double> gridVar;
	  LatVolField<double> *sfd =
	    scinew LatVolField<double>( mesh_handle_, 1 );
	  
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "offset", IntVector(low), true);
	  sfd->set_property( "vartype",
			     int(TypeDescription::SFCZVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  fout->send(sfd);
	  return;
	}
      case TypeDescription::float_type:
	{
	  SFCZVariable<float> gridVar;
	  LatVolField<float> *sfd =
	    scinew LatVolField<float>( mesh_handle_, 1 );
	  
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "offset", IntVector(low), true);
	  sfd->set_property( "vartype",
			     int(TypeDescription::SFCZVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  fout->send(sfd);
	  return;
	}
      case TypeDescription::int_type:
	{
	  SFCZVariable<int> gridVar;
	  LatVolField<int> *sfd =
	    scinew LatVolField<int>( mesh_handle_, 1 );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "offset", IntVector(low), true);
	  sfd->set_property( "vartype",
			     int(TypeDescription::SFCZVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  fout->send(sfd);
	  return;
	}
      case TypeDescription::long64_type:
      case TypeDescription::long_type:
	{
	  SFCZVariable<long64> gridVar;
	  LatVolField<long64> *sfd =
	    scinew LatVolField<long64>( mesh_handle_, 1 );
	  sfd->set_property( "variable", string(var), true );
	  sfd->set_property( "time", double( time ), true);
	  sfd->set_property( "offset", IntVector(low), true);
	  sfd->set_property( "vartype",
			     int(TypeDescription::SFCZVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, sfd );
	  fout->send(sfd);
	  return;
	}
      case TypeDescription::Vector:
	{	
	  SFCZVariable<Vector> gridVar;
	  LatVolField<Vector> *vfd =
	    scinew LatVolField<Vector>( mesh_handle_, 1 );
	  // set the generation and timestep in the field
	  vfd->set_property("varname",string(var), true);
	  vfd->set_property("generation",generation, true);
	  vfd->set_property("timestep",timestep, true);
	  vfd->set_property( "offset", IntVector(low), true);
	  vfd->set_property("delta_t",dt, true);
	  vfd->set_property( "vartype",
			     int(TypeDescription::SFCZVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, vfd );
	  // send the field out to the port
	  fout->send(vfd);
	  return;
	}
      case TypeDescription::Matrix3:
	{
	  SFCZVariable<Matrix3> gridVar;
	  LatVolField<Matrix3> *tfd =
	    scinew LatVolField<Matrix3>( mesh_handle_, 1 );
	  // set the generation and timestep in the field
	  tfd->set_property( "vartype",
			     int(TypeDescription::SFCZVariable),true);
	  build_field( archive, level, low, var, mat, time, gridVar, tfd );
	  // send the field out to the port
	  fout->send(tfd);
	  // 	DumpAllocator(default_allocator, "TensorDump.allocator");
	  return;
	}
      default:
	error("SFCZVariable<?> Unknown type.");
	return;
      }
    default:
      error("Not a Uintah type.");
      return;
    }
  }
}



} // End namespace Uintah
