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
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/LocallyComputedPatchVarMap.h>
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
   if (level_.get() > (n-1)){
     level_.set(n);
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
  int guilevel = level_.get();
  LevelP level = grid->getLevel((guilevel == levels ? 0 : guilevel) );

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

  cerr<<"Gui say levels = "<<level_.get()<<"\n";
  // get time, set timestep, set generation, update grid and update gui
  double time = field_update(); // yeah it does all that
  cerr<<"Field updated; do we get here\n";

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

  bool get_all_levels = false;
  int lev = level_.get();
  if (lev == grid->numLevels()){
    get_all_levels = true;
  }

  cerr<<"grid->numLevels() = "<<grid->numLevels()<<" and get_all_levels = "<<
    get_all_levels<<"\n";
  string var(sVar.get());
  int mat = sMatNum.get();
  if(var != ""){
    
    const TypeDescription* subtype = type->getSubType();
    LevelP level;
    
    if( get_all_levels ){ 
      level = grid->getLevel( 0 );
    } else {
      level = grid->getLevel( level_.get() );
    }
    IntVector hi, low, range;
    level->findIndexRange(low, hi);
    range = hi - low;
    BBox box;
    level->getSpatialRange(box);
//      IntVector cellHi, cellLo;
//      level->findCellIndexRange(cellLo, cellHi);

    cerr<<"before anything data range is:  "<<range.x()<<"x"<<range.y()<<"x"<<
      range.z()<<"  size:  "<<box.min()<<", "<<box.max()<<"\n";

    switch( type->getType() ) {

    case TypeDescription::NCVariable:
      switch ( subtype->getType() ) {
      case TypeDescription::double_type:
	{
	  NCVariable<double> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<double>* mrfield = 0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::NCVariable,
				     TypeDescription::double_type, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::NCVariable, mesh_handle_);
	    LatVolField<double> *sfd =
	      scinew LatVolField<double>( mesh_handle_, 1 );
	    set_scalar_properties( sfd, var, time, low, 
				   TypeDescription::NCVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, sfd );
	    fout->send(sfd);
	    return;
	  }
	}
      case TypeDescription::float_type:
	{
	  NCVariable<float> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<float>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::NCVariable,
				     TypeDescription::float_type, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::NCVariable, mesh_handle_);
	    LatVolField<float> *sfd =
	      scinew LatVolField<float>( mesh_handle_, 1 );
	    set_scalar_properties( sfd, var, time, low, 
				   TypeDescription::NCVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, sfd );
	    fout->send(sfd);
	    return;
	  }
	}
      case TypeDescription::int_type:
	{
	  NCVariable<int> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<int>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::NCVariable,
				     TypeDescription::int_type, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::NCVariable, mesh_handle_);
	    LatVolField<int> *sfd =
	      scinew LatVolField<int>( mesh_handle_, 1 );
	    set_scalar_properties( sfd, var, time, low, 
				   TypeDescription::NCVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, sfd );
	    fout->send(sfd);
	    return;
	  }
	}
      case TypeDescription::long64_type:
	{
	  NCVariable<long64> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<long64>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::NCVariable,
				     TypeDescription::long64_type, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::NCVariable, mesh_handle_);
	    LatVolField<long64> *sfd =
	      scinew LatVolField<long64>( mesh_handle_, 1 );
	    set_scalar_properties( sfd, var, time, low, 
				   TypeDescription::NCVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, sfd );
	    fout->send(sfd);
	    return;
	  }
	}
      case TypeDescription::Vector:
	{	
	  NCVariable<Vector> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<Vector>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::NCVariable,
				     TypeDescription::Vector, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::NCVariable, mesh_handle_);
	    LatVolField<Vector> *vfd =
	      scinew LatVolField<Vector>( mesh_handle_, 1 );
	    // set the generation and timestep in the field
	    set_vector_properties( vfd, var, generation, timestep, low, dt,
				   TypeDescription::NCVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, vfd );
	    // send the field out to the port
	    fout->send(vfd);
	    return;
	  }
	}
      case TypeDescription::Matrix3:
	{	
	  NCVariable<Matrix3> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<Matrix3>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::NCVariable,
				     TypeDescription::Matrix3, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::NCVariable, mesh_handle_);
	    LatVolField<Matrix3> *tfd =
	      scinew LatVolField<Matrix3>( mesh_handle_, 1 );
	    // set the generation and timestep in the field
	    set_tensor_properties( tfd, low, TypeDescription::NCVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, tfd );
	    // send the field out to the port
	    fout->send(tfd);
	    // 	DumpAllocator(default_allocator, "TensorDump.allocator");
	    return;
	  }
	}
      default:
	error("NCVariable<?>  Unknown scalar type.");
	return;
      }
    case TypeDescription::CCVariable:
      switch ( subtype->getType() ) {
      case TypeDescription::double_type:
	{
	  CCVariable<double> gridVar;
	  if(get_all_levels){
	    cerr<<"getting all levels\n";
	    cerr.flush();
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<double>* mrfield =0;
	    cerr<<"building multi-level field\n";
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 0,
				     TypeDescription::CCVariable,
				     TypeDescription::double_type, mrfield);
	    cerr<<"multi-level field built\n";
	    fout->send(mrfield);
	    return;
	  } else {
	    cerr<<"Before update_mesh_handled: type = "<<
	      TypeDescription::CCVariable<<"\n";
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::CCVariable, mesh_handle_);
	    LatVolField<double> *sfd =
	      scinew LatVolField<double>( mesh_handle_, 0 );
	    cerr<<"setting scalar properties\n";
	    set_scalar_properties( sfd, var, time, low, 
				   TypeDescription::CCVariable);
	    cerr<<"properties set...building field\n";
	    build_field( archive, level, low, var, mat, time, gridVar, sfd );
	    cerr<<"field built\n";
	    fout->send(sfd);
	    return;
	  }
	}
      case TypeDescription::float_type:
	{
	  CCVariable<float> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<float>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 0,
				     TypeDescription::CCVariable,
				     TypeDescription::float_type, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::CCVariable, mesh_handle_);
	    LatVolField<float> *sfd =
	      scinew LatVolField<float>( mesh_handle_, 0 );
	    set_scalar_properties( sfd, var, time, low, 
				   TypeDescription::CCVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, sfd );
	    fout->send(sfd);
	    return;
	  }
	}
      case TypeDescription::int_type:
	{
	  CCVariable<int> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<int>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 0,
				     TypeDescription::CCVariable,
				     TypeDescription::int_type, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::CCVariable, mesh_handle_);
	    LatVolField<int> *sfd =
	      scinew LatVolField<int>( mesh_handle_, 0 );
	    set_scalar_properties( sfd, var, time, low, 
				   TypeDescription::CCVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, sfd );
	    fout->send(sfd);
	    return;
	  }
	}
      case TypeDescription::bool_type:
	{
	  CCVariable<unsigned char> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<unsigned char>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 0,
				     TypeDescription::CCVariable,
				     TypeDescription::bool_type, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::CCVariable, mesh_handle_);
	    LatVolField<unsigned char> *sfd =
	      scinew LatVolField<unsigned char>( mesh_handle_, 0 );
	    set_scalar_properties( sfd, var, time, low, 
				   TypeDescription::CCVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, sfd );
	    fout->send(sfd);
	    return;
	  }
	}
      case TypeDescription::long64_type:
      case TypeDescription::long_type:
	{
	  CCVariable<long64> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<long64>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 0,
				     TypeDescription::CCVariable,
				     TypeDescription::long64_type, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::CCVariable, mesh_handle_);
	    LatVolField<long64> *sfd =
	      scinew LatVolField<long64>( mesh_handle_, 0 );
	    set_scalar_properties( sfd, var, time, low, 
				   TypeDescription::CCVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, sfd );
	    fout->send(sfd);
	    return;
	  }
	}
      case TypeDescription::Vector:
	{	
	  CCVariable<Vector> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<Vector>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 0,
				     TypeDescription::CCVariable,
				     TypeDescription::Vector, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::CCVariable, mesh_handle_);
	    LatVolField<Vector> *vfd =
	      scinew LatVolField<Vector>( mesh_handle_, 0 );
	    // set the generation and timestep in the field
	    set_vector_properties( vfd, var, generation, timestep, low, dt,
				   TypeDescription::CCVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, vfd );
	    // send the field out to the port
	    fout->send(vfd);
	    return;
	  }
	}
      case TypeDescription::Matrix3:
	{
	  CCVariable<Matrix3> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<Matrix3>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 0,
				     TypeDescription::CCVariable,
				     TypeDescription::Matrix3, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::CCVariable, mesh_handle_);
	    LatVolField<Matrix3> *tfd =
	      scinew LatVolField<Matrix3>( mesh_handle_, 0 );
	    // set the generation and timestep in the field
	    set_tensor_properties( tfd, low, TypeDescription::CCVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, tfd );
	    // send the field out to the port
	    fout->send(tfd);
	    // 	DumpAllocator(default_allocator, "TensorDump.allocator");
	    return;
	  }
	}
      default:
	error("CCVariable<?> Unknown scalar type.");
	return;
      }
    case TypeDescription::SFCXVariable:
      switch ( subtype->getType() ) {
      case TypeDescription::double_type:
	{
	  SFCXVariable<double> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<double>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::SFCXVariable,
				     TypeDescription::double_type, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::SFCXVariable, mesh_handle_);
	    LatVolField<double> *sfd =
	      scinew LatVolField<double>( mesh_handle_, 1 );
	    set_scalar_properties( sfd, var, time, low, 
				   TypeDescription::SFCXVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, sfd );
	    fout->send(sfd);
	    return;
	  }
	}
      case TypeDescription::float_type:
	{
	  SFCXVariable<float> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<float>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::SFCXVariable,
				     TypeDescription::float_type, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::SFCXVariable, mesh_handle_);
	    LatVolField<float> *sfd =
	      scinew LatVolField<float>( mesh_handle_, 1 );
	    set_scalar_properties( sfd, var, time, low, 
				   TypeDescription::SFCXVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, sfd );
	    fout->send(sfd);
	    return;
	  }
	}
      case TypeDescription::int_type:
	{
	  SFCXVariable<int> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<int>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::SFCXVariable,
				     TypeDescription::int_type, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::SFCXVariable, mesh_handle_);
	    LatVolField<int> *sfd =
	      scinew LatVolField<int>( mesh_handle_, 1 );
	    set_scalar_properties( sfd, var, time, low, 
				   TypeDescription::SFCXVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, sfd );
	    fout->send(sfd);
	    return;
	  }
	}
      case TypeDescription::long64_type:
      case TypeDescription::long_type:
	{
	  SFCXVariable<long64> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<long64>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::SFCXVariable,
				     TypeDescription::long64_type, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::SFCXVariable, mesh_handle_);
	    LatVolField<long64> *sfd =
	      scinew LatVolField<long64>( mesh_handle_, 1 );
	    set_scalar_properties( sfd, var, time, low, 
				   TypeDescription::SFCXVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, sfd );
	    fout->send(sfd);
	    return;
	  }
	}
      case TypeDescription::Vector:
	{	
	  SFCXVariable<Vector> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<Vector>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::SFCXVariable,
				     TypeDescription::Vector, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::SFCXVariable, mesh_handle_);
	    LatVolField<Vector> *vfd =
	      scinew LatVolField<Vector>( mesh_handle_, 1 );
	    // set the generation and timestep in the field
	    set_vector_properties( vfd, var, generation, timestep, low, dt,
				   TypeDescription::SFCXVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, vfd );
	    // send the field out to the port
	    fout->send(vfd);
	    return;
	  }
	}
      case TypeDescription::Matrix3:
	{
	  SFCXVariable<Matrix3> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<Matrix3>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::SFCXVariable,
				     TypeDescription::Matrix3, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::SFCXVariable, mesh_handle_);
	    LatVolField<Matrix3> *tfd =
	      scinew LatVolField<Matrix3>( mesh_handle_, 1 );
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
	}
      default:
	error("SFCXVariable<?> Unknown scalar type.");
	return;
      }
      
    case TypeDescription::SFCYVariable:
      switch ( subtype->getType() ) {
      case TypeDescription::double_type:
	{
	  SFCYVariable<double> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<double>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::SFCYVariable,
				     TypeDescription::double_type, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::SFCYVariable, mesh_handle_);
	    LatVolField<double> *sfd =
	      scinew LatVolField<double>( mesh_handle_, 1 );
	    set_scalar_properties( sfd, var, time, low, 
				   TypeDescription::SFCYVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, sfd );
	    fout->send(sfd);
	    return;
	  }
	}
      case TypeDescription::float_type:
	{
	  SFCYVariable<float> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<float>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::SFCYVariable,
				     TypeDescription::float_type, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::SFCYVariable, mesh_handle_);
	    LatVolField<float> *sfd =
	      scinew LatVolField<float>( mesh_handle_, 1 );
	    set_scalar_properties( sfd, var, time, low, 
				   TypeDescription::SFCYVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, sfd );
	    fout->send(sfd);
	    return;
	  }
	}
      case TypeDescription::int_type:
	{
	  SFCYVariable<int> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<int>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::SFCYVariable,
				     TypeDescription::int_type, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::SFCYVariable, mesh_handle_);
	    LatVolField<int> *sfd =
	      scinew LatVolField<int>( mesh_handle_, 1 );
	    set_scalar_properties( sfd, var, time, low, 
				   TypeDescription::SFCYVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, sfd );
	    fout->send(sfd);
	    return;
	  }
	}
      case TypeDescription::long64_type:
      case TypeDescription::long_type:
	{
	  SFCYVariable<long64> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<long64>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::SFCYVariable,
				     TypeDescription::long64_type, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::SFCYVariable, mesh_handle_);
	    LatVolField<long64> *sfd =
	      scinew LatVolField<long64>( mesh_handle_, 1 );
	    set_scalar_properties( sfd, var, time, low, 
				   TypeDescription::SFCYVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, sfd );
	    fout->send(sfd);
	    return;
	  }
	}
      case TypeDescription::Vector:
	{	
	  SFCYVariable<Vector> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<Vector>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::SFCYVariable,
				     TypeDescription::Vector, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::SFCYVariable, mesh_handle_);
	    LatVolField<Vector> *vfd =
	      scinew LatVolField<Vector>( mesh_handle_, 1 );
	    // set the generation and timestep in the field
	    set_vector_properties( vfd, var, generation, timestep, low, dt,
				   TypeDescription::SFCYVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, vfd );
	    // send the field out to the port
	    fout->send(vfd);
	    return;
	  }
	}
      case TypeDescription::Matrix3:
	{
	  SFCYVariable<Matrix3> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<Matrix3>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::SFCYVariable,
				     TypeDescription::Matrix3, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::SFCYVariable, mesh_handle_);
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
	}
      default:
	error("SFCYVariable<?> Unknown scalar type.");
	return;
      }
    case TypeDescription::SFCZVariable:
      switch ( subtype->getType() ) {
      case TypeDescription::double_type:
	{
	  SFCZVariable<double> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<double>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::SFCZVariable,
				     TypeDescription::double_type, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::SFCZVariable, mesh_handle_);
	    LatVolField<double> *sfd =
	      scinew LatVolField<double>( mesh_handle_, 1 );
	    set_scalar_properties( sfd, var, time, low, 
				   TypeDescription::SFCZVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, sfd );
	    fout->send(sfd);
	    return;
	  }
	}
      case TypeDescription::float_type:
	{
	  SFCZVariable<float> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<float>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::SFCZVariable,
				     TypeDescription::float_type, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::SFCZVariable, mesh_handle_);
	    LatVolField<float> *sfd =
	      scinew LatVolField<float>( mesh_handle_, 1 );
	    set_scalar_properties( sfd, var, time, low, 
				   TypeDescription::SFCZVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, sfd );
	    fout->send(sfd);
	    return;
	  }
	}
      case TypeDescription::int_type:
	{
	  SFCZVariable<int> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<int>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::SFCZVariable,
				     TypeDescription::int_type, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::SFCZVariable, mesh_handle_);
	    LatVolField<int> *sfd =
	      scinew LatVolField<int>( mesh_handle_, 1 );
	    set_scalar_properties( sfd, var, time, low, 
				   TypeDescription::SFCZVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, sfd );
	    fout->send(sfd);
	    return;
	  }
	}
      case TypeDescription::long64_type:
      case TypeDescription::long_type:
	{
	  SFCZVariable<long64> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<long64>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::SFCZVariable,
				     TypeDescription::long64_type, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::SFCZVariable, mesh_handle_);
	    LatVolField<long64> *sfd =
	      scinew LatVolField<long64>( mesh_handle_, 1 );
	    set_scalar_properties( sfd, var, time, low, 
				   TypeDescription::SFCZVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, sfd );
	    fout->send(sfd);
	    return;
	  }
	}
      case TypeDescription::Vector:
	{	
	  SFCZVariable<Vector> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<Vector>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::SFCZVariable,
				     TypeDescription::Vector, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::SFCZVariable, mesh_handle_);
	    LatVolField<Vector> *vfd =
	      scinew LatVolField<Vector>( mesh_handle_, 1 );
	    // set the generation and timestep in the field
	    set_vector_properties( vfd, var, generation, timestep, low, dt,
				   TypeDescription::SFCZVariable);
	    build_field( archive, level, low, var, mat, time, gridVar, vfd );
	    // send the field out to the port
	    fout->send(vfd);
	    return;
	  }
	}
      case TypeDescription::Matrix3:
	{
	  SFCZVariable<Matrix3> gridVar;
	  if(get_all_levels){
	    GridP newGrid = build_minimal_patch_grid( grid );
	    MRLatVolField<Matrix3>* mrfield =0;
	    build_multi_level_field( archive, newGrid, var, gridVar,
				     mat, generation, time, timestep,
				     dt, 1,
				     TypeDescription::SFCZVariable,
				     TypeDescription::Matrix3, mrfield);
	    fout->send(mrfield);
	    return;
	  } else {
	    update_mesh_handle( level, hi, range, box,
				TypeDescription::SFCZVariable, mesh_handle_);
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

GridP 
FieldExtractor::build_minimal_patch_grid( GridP oldGrid )
{
  int nlevels = oldGrid->numLevels();
  GridP newGrid = scinew Grid();
  const SuperPatchContainer* superPatches;

  for( int i = 0; i < nlevels; i++ ){
    LevelP level = oldGrid->getLevel(i);
    LocallyComputedPatchVarMap patchGrouper;
    const PatchSubset* patches = level->allPatches()->getUnion();
    patchGrouper.addComputedPatchSet(0, patches);
    patchGrouper.makeGroups();
    superPatches = patchGrouper.getSuperPatches(0, level.get_rep());
    ASSERT(superPatches != 0);

    LevelP newLevel =
      newGrid->addLevel(level->getAnchor(), level->dCell());

    cerr<<"Level "<<i<<":\n";
    int count = 0;
    SuperPatchContainer::const_iterator superIter;
    for (superIter = superPatches->begin();
	 superIter != superPatches->end(); superIter++) {
      IntVector low = (*superIter)->getLow();
      IntVector high = (*superIter)->getHigh();
      IntVector inLow = high; // taking min values starting at high
      IntVector inHigh = low; // taking max values starting at low

      cerr<<"\tcombined patch "<<count++<<" is "<<low<<", "<<high<<"\n";

      for (unsigned int p = 0; p < (*superIter)->getBoxes().size(); p++) {
	const Patch* patch = (*superIter)->getBoxes()[p];
	inLow = Min(inLow, patch->getInteriorCellLowIndex());
	inHigh = Max(inHigh, patch->getInteriorCellHighIndex());
      }
      
      Patch* newPatch =
	newLevel->addPatch(low, high, inLow, inHigh);
      for (unsigned int p = 0; p < (*superIter)->getBoxes().size(); p++) {
	const Patch* patch = (*superIter)->getBoxes()[p];
	new2OldPatchMap_[newPatch].push_back(patch);
      }
    }
    newLevel->finalizeLevel();
  }
  return newGrid;
}

void FieldExtractor::update_mesh_handle( LevelP& level,
					 IntVector& hi,
					 IntVector& range,
					 BBox& box,
					 TypeDescription::Type type,
					 LatVolMeshHandle& mesh_handle)
{
  cerr<<"In update_mesh_handled: type = "<<type<<"\n";
  
  switch ( type ){
  case TypeDescription::CCVariable:
    {
      IntVector cellHi, cellLo, levelHi, levelLo;
      level->findCellIndexRange(cellLo, cellHi);
      level->findIndexRange( levelLo, levelHi);
      if( mesh_handle.get_rep() == 0 ){
	if(is_periodic_bcs(cellHi, hi) && is_periodic_bcs(cellHi, levelHi)){
	  IntVector newrange(0,0,0);
	  get_periodic_bcs_range( cellHi, hi, range, newrange);
	  mesh_handle = scinew LatVolMesh(newrange.x(),newrange.y(),
					   newrange.z(), box.min(),
					   box.max());
	} else {
	  mesh_handle = scinew LatVolMesh(range.x(), range.y(),
					   range.z(), box.min(),
					   box.max());
	  cerr<<"mesh built:  "<<range.x()<<"x"<<range.y()<<"x"<<
	    range.z()<<"  size:  "<<box.min()<<", "<<box.max()<<"\n";
	}
      } else if(mesh_handle->get_ni() != (unsigned int) range.x() ||
		mesh_handle->get_nj() != (unsigned int) range.y() ||
		mesh_handle->get_nk() != (unsigned int) range.z() ){
	if(is_periodic_bcs(cellHi, hi) && is_periodic_bcs(cellHi, levelHi)){
	  IntVector newrange(0,0,0);
	  get_periodic_bcs_range( cellHi, hi, range, newrange);
	  mesh_handle = scinew LatVolMesh(newrange.x(),newrange.y(),
					   newrange.z(), box.min(),
					   box.max());
	} else {
	  mesh_handle = scinew LatVolMesh(range.x(), range.y(),
					   range.z(), box.min(),
					   box.max());
	  cerr<<"mesh built:  "<<range.x()<<"x"<<range.y()<<"x"<<
	    range.z()<<"  size:  "<<box.min()<<", "<<box.max()<<"\n";
	}
      }	
      return;
    }
  case TypeDescription::NCVariable:
    {
      if( mesh_handle.get_rep() == 0 ){
	mesh_handle = scinew LatVolMesh(range.x(), range.y(),
					 range.z(), box.min(),
					 box.max());
	cerr<<"mesh built:  "<<range.x()<<"x"<<range.y()<<"x"<<
	  range.z()<<"  size:  "<<box.min()<<", "<<box.max()<<"\n";
      }else if(mesh_handle->get_ni() != (unsigned int) range.x() ||
	       mesh_handle->get_nj() != (unsigned int) range.y() ||
	       mesh_handle->get_nk() != (unsigned int) range.z() ){
	mesh_handle = scinew LatVolMesh(range.x(), range.y(),
					 range.z(), box.min(),
					 box.max());
	cerr<<"mesh built:  "<<range.x()<<"x"<<range.y()<<"x"<<
	  range.z()<<"  size:  "<<box.min()<<", "<<box.max()<<"\n";
      }
      return;
    }
  case TypeDescription::SFCXVariable:
    {
      if( mesh_handle.get_rep() == 0 ){
	mesh_handle = scinew LatVolMesh(range.x(), range.y() - 1,
					 range.z() - 1, box.min(),
					 box.max());
      } else if(mesh_handle->get_ni() != (unsigned int) range.x() ||
		mesh_handle->get_nj() != (unsigned int) range.y() -1 ||
		mesh_handle->get_nk() != (unsigned int) range.z() -1 ){
	mesh_handle = scinew LatVolMesh(range.x(), range.y() - 1,
					 range.z()-1, box.min(),
					 box.max());
      }
      return;
    }
  case TypeDescription::SFCYVariable:
    {
      if( mesh_handle.get_rep() == 0 ){
	mesh_handle = scinew LatVolMesh(range.x()-1, range.y(),
					 range.z()-1, box.min(),
					 box.max());
      } else if(mesh_handle->get_ni() != (unsigned int) range.x() -1 ||
		mesh_handle->get_nj() != (unsigned int) range.y() ||
		mesh_handle->get_nk() != (unsigned int) range.z() -1 ){
	mesh_handle = scinew LatVolMesh(range.x()-1, range.y(),
					 range.z()-1, box.min(),
					 box.max());
      }
      return;
    }
  case TypeDescription::SFCZVariable:
    {
      if( mesh_handle.get_rep() == 0 ){
	mesh_handle = scinew LatVolMesh(range.x()-1, range.y()-1,
					 range.z(), box.min(),
					 box.max());
      } else if(mesh_handle->get_ni() != (unsigned int) range.x() -1 ||
		mesh_handle->get_nj() != (unsigned int) range.y() -1 ||
		mesh_handle->get_nk() != (unsigned int) range.z() ){
	mesh_handle = scinew LatVolMesh(range.x()-1, range.y()-1,
					 range.z(), box.min(),
					 box.max());
      }     
      return;
    }
  default:
    error("in update_mesh_handle:: Not a Uintah type.");
  }
}

  


} // End namespace Uintah
