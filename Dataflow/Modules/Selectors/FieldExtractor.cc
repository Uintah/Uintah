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
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Variables/LocallyComputedPatchVarMap.h>
#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Geometry/Transform.h>
#include <Packages/Uintah/Core/Grid/Variables/ShareAssignArray3.h>
//#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include "FieldExtractor.h"
 
#include <iostream> 
#include <sstream>
#include <string>

using std::cerr;
using std::endl;
using std::vector;
using std::string;
using std::ostringstream;

using namespace Uintah;
using namespace SCIRun;

//--------------------------------------------------------------- 
FieldExtractor::FieldExtractor(const string& name,
                               GuiContext* ctx,
                               const string& cat,
                               const string& pack) :
  Module(name, ctx, Filter, cat, pack),
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
FieldExtractor::~FieldExtractor()
{} 

//------------------------------------------------------------- 

void
FieldExtractor::build_GUI_frame()
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

double
FieldExtractor::field_update()
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
    grid = 0;
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
// update the variable list for the GUI

void
FieldExtractor::update_GUI(const string& var,
                           const string& varnames)
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

#if 0
template<class T>
void getVariable(QueryInfo &qinfo, IntVector &low,
                 IntVector &range, BBox &box, string &filename) {
}

#endif

void
FieldExtractor::execute()
{
  //  const char* old_tag1 = AllocatorSetDefaultTag("FieldExtractor::execute");
  tcl_status.set("Calling FieldExtractor!"); 
  ArchiveIPort *in = (ArchiveIPort *) get_iport("Data Archive");
  FieldOPort *fout = (FieldOPort *) get_oport("Field");

  ArchiveHandle handle;
  if (!(in->get(handle) && handle.get_rep())) {
    warning("VariableExtractor::execute() - No data from input port.");
    //    AllocatorSetDefaultTag(old_tag1);
    return;
  }
   
  if (archiveH.get_rep() == 0 ){
    // first time through a frame must be built
    build_GUI_frame();
  }
  
  archiveH = handle;
  DataArchive& archive = *((*(archiveH.get_rep()))());

  //   cerr<<"Gui say levels = "<<level_.get()<<"\n";
  // get time, set timestep, set generation, update grid and update gui
  double time = field_update(); // yeah it does all that

  if(type == 0){
    warning( "No variables found.");
    //    AllocatorSetDefaultTag(old_tag1);
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

  //   cerr<<"grid->numLevels() = "<<grid->numLevels()<<" and get_all_levels = "<<
  //     get_all_levels<<"\n";
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
    
    //     cerr<<"before anything data range is:  "<<range.x()<<"x"<<range.y()<<"x"<<
    //       range.z()<<"  size:  "<<box.min()<<", "<<box.max()<<"\n";
    
    FieldHandle fHandle_;
#ifdef TEMPLATE_FUN
    update_mesh_handle( level, hi, range, box, type->getType(), mesh_handle_);
    QueryInfo qinfo(&archive, generation, grid, level, var, mat, type,
                    get_all_levels, time, timestep, dt);
    // get_all_levels, generation, timestep, dt
    switch( subtype->getType() ) {
    case TypeDescription::double_type:
      fHandle_ = getVariable<double>(qinfo, low, mesh_handle_);
      break;
    case TypeDescription::float_type:
      fHandle_ = getVariable<float>(qinfo, low, mesh_handle_);
      break;
    case TypeDescription::int_type:
      fHandle_ = getVariable<int>(qinfo, low, mesh_handle_);
      break;
    case TypeDescription::bool_type:
      fHandle_ = getVariable<unsigned char>(qinfo, low, mesh_handle_);
      break;
    case Uintah::TypeDescription::long_type:
    case Uintah::TypeDescription::long64_type:
      fHandle_ = getVariable<long64>(qinfo, low, mesh_handle_);
      break;
    case TypeDescription::Vector:
      fHandle_ = getVariable<Vector>(qinfo, low, mesh_handle_);
      break;
    case TypeDescription::Matrix3:
      fHandle_ = getVariable<Matrix3>(qinfo, low, mesh_handle_);
      break;
    case Uintah::TypeDescription::short_int_type:
    default:
      error("Subtype " + subtype->getName() + " is not implemented\n");
      return;
    }
#else
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::NCVariable, mesh_handle_);
            LatVolField<double> *sfd =
              scinew LatVolField<double>( mesh_handle_, 1 );
            set_scalar_properties( sfd, var, time, low, 
                                   TypeDescription::NCVariable);
            build_field( archive, level, low, var, mat, time, gridVar, sfd );
            fHandle_ = sfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::NCVariable, mesh_handle_);
            LatVolField<float> *sfd =
              scinew LatVolField<float>( mesh_handle_, 1 );
            set_scalar_properties( sfd, var, time, low, 
                                   TypeDescription::NCVariable);
            build_field( archive, level, low, var, mat, time, gridVar, sfd );
            fHandle_ = sfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::NCVariable, mesh_handle_);
            LatVolField<int> *sfd =
              scinew LatVolField<int>( mesh_handle_, 1 );
            set_scalar_properties( sfd, var, time, low, 
                                   TypeDescription::NCVariable);
            build_field( archive, level, low, var, mat, time, gridVar, sfd );
            fHandle_ = sfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::NCVariable, mesh_handle_);
            LatVolField<long64> *sfd =
              scinew LatVolField<long64>( mesh_handle_, 1 );
            set_scalar_properties( sfd, var, time, low, 
                                   TypeDescription::NCVariable);
            build_field( archive, level, low, var, mat, time, gridVar, sfd );
            fHandle_ = sfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
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
            fHandle_ = vfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::NCVariable, mesh_handle_);
            LatVolField<Matrix3> *tfd =
              scinew LatVolField<Matrix3>( mesh_handle_, 1 );
            // set the generation and timestep in the field
            set_tensor_properties( tfd, low, TypeDescription::NCVariable);
            build_field( archive, level, low, var, mat, time, gridVar, tfd );
            // send the field out to the port
            fHandle_ = tfd;
            //  DumpAllocator(default_allocator, "TensorDump.allocator");
          }
        }
        break;
      default:
        error("NCVariable<?>  Unknown scalar type.");
        //        AllocatorSetDefaultTag(old_tag1);
        return;
      }
      break;
    case TypeDescription::CCVariable:
      switch ( subtype->getType() ) {
      case TypeDescription::double_type:
        {
          CCVariable<double> gridVar;
          if(get_all_levels){
            //      cerr<<"getting all levels\n";
//             cerr.flush();
            GridP newGrid = build_minimal_patch_grid( grid );
            MRLatVolField<double>* mrfield =0;
            //      cerr<<"building multi-level field\n";
            build_multi_level_field( archive, newGrid, var, gridVar,
                                     mat, generation, time, timestep,
                                     dt, 0,
                                     TypeDescription::CCVariable,
                                     TypeDescription::double_type, mrfield);
            //      cerr<<"multi-level field built\n";
            fHandle_ = mrfield;
          } else {
            //      cerr<<"Before update_mesh_handled: type = "<<
            //        TypeDescription::CCVariable<<"\n";
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::CCVariable, mesh_handle_);
            LatVolField<double> *sfd =
              scinew LatVolField<double>( mesh_handle_, 0 );
            //      cerr<<"setting scalar properties\n";
            set_scalar_properties( sfd, var, time, low, 
                                   TypeDescription::CCVariable);
            //      cerr<<"properties set...building field\n";
            build_field( archive, level, low, var, mat, time, gridVar, sfd );
            //      cerr<<"field built\n";
            fHandle_ = sfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::CCVariable, mesh_handle_);
            LatVolField<float> *sfd =
              scinew LatVolField<float>( mesh_handle_, 0 );
            set_scalar_properties( sfd, var, time, low, 
                                   TypeDescription::CCVariable);
            build_field( archive, level, low, var, mat, time, gridVar, sfd );
            fHandle_ = sfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::CCVariable, mesh_handle_);
            LatVolField<int> *sfd =
              scinew LatVolField<int>( mesh_handle_, 0 );
            set_scalar_properties( sfd, var, time, low, 
                                   TypeDescription::CCVariable);
            build_field( archive, level, low, var, mat, time, gridVar, sfd );
            fHandle_ = sfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::CCVariable, mesh_handle_);
            LatVolField<unsigned char> *sfd =
              scinew LatVolField<unsigned char>( mesh_handle_, 0 );
            set_scalar_properties( sfd, var, time, low, 
                                   TypeDescription::CCVariable);
            build_field( archive, level, low, var, mat, time, gridVar, sfd );
            fHandle_ = sfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::CCVariable, mesh_handle_);
            LatVolField<long64> *sfd =
              scinew LatVolField<long64>( mesh_handle_, 0 );
            set_scalar_properties( sfd, var, time, low, 
                                   TypeDescription::CCVariable);
            build_field( archive, level, low, var, mat, time, gridVar, sfd );
            fHandle_ = sfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
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
            fHandle_ = vfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::CCVariable, mesh_handle_);
            LatVolField<Matrix3> *tfd =
              scinew LatVolField<Matrix3>( mesh_handle_, 0 );
            // set the generation and timestep in the field
            set_tensor_properties( tfd, low, TypeDescription::CCVariable);
            build_field( archive, level, low, var, mat, time, gridVar, tfd );
            // send the field out to the port
            fHandle_ = tfd;
            //  DumpAllocator(default_allocator, "TensorDump.allocator");
          }
        }
        break;
      default:
        error("CCVariable<?> Unknown scalar type.");
        //        AllocatorSetDefaultTag(old_tag1);
        return;
      }
      break;
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::SFCXVariable, mesh_handle_);
            LatVolField<double> *sfd =
              scinew LatVolField<double>( mesh_handle_, 1 );
            set_scalar_properties( sfd, var, time, low, 
                                   TypeDescription::SFCXVariable);
            build_field( archive, level, low, var, mat, time, gridVar, sfd );
            fHandle_ = sfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::SFCXVariable, mesh_handle_);
            LatVolField<float> *sfd =
              scinew LatVolField<float>( mesh_handle_, 1 );
            set_scalar_properties( sfd, var, time, low, 
                                   TypeDescription::SFCXVariable);
            build_field( archive, level, low, var, mat, time, gridVar, sfd );
            fHandle_ = sfd;
          }
        }
        break; 
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::SFCXVariable, mesh_handle_);
            LatVolField<int> *sfd =
              scinew LatVolField<int>( mesh_handle_, 1 );
            set_scalar_properties( sfd, var, time, low, 
                                   TypeDescription::SFCXVariable);
            build_field( archive, level, low, var, mat, time, gridVar, sfd );
            fHandle_ = sfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::SFCXVariable, mesh_handle_);
            LatVolField<long64> *sfd =
              scinew LatVolField<long64>( mesh_handle_, 1 );
            set_scalar_properties( sfd, var, time, low, 
                                   TypeDescription::SFCXVariable);
            build_field( archive, level, low, var, mat, time, gridVar, sfd );
            fHandle_ = sfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
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
            fHandle_ = vfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
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
            fHandle_ = tfd;
            //  DumpAllocator(default_allocator, "TensorDump.allocator");
          }
        }
        break;
      default:
        error("SFCXVariable<?> Unknown scalar type.");
        //        AllocatorSetDefaultTag(old_tag1);
        return;
      }
      break;
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::SFCYVariable, mesh_handle_);
            LatVolField<double> *sfd =
              scinew LatVolField<double>( mesh_handle_, 1 );
            set_scalar_properties( sfd, var, time, low, 
                                   TypeDescription::SFCYVariable);
            build_field( archive, level, low, var, mat, time, gridVar, sfd );
            fHandle_ = sfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::SFCYVariable, mesh_handle_);
            LatVolField<float> *sfd =
              scinew LatVolField<float>( mesh_handle_, 1 );
            set_scalar_properties( sfd, var, time, low, 
                                   TypeDescription::SFCYVariable);
            build_field( archive, level, low, var, mat, time, gridVar, sfd );
            fHandle_ = sfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::SFCYVariable, mesh_handle_);
            LatVolField<int> *sfd =
              scinew LatVolField<int>( mesh_handle_, 1 );
            set_scalar_properties( sfd, var, time, low, 
                                   TypeDescription::SFCYVariable);
            build_field( archive, level, low, var, mat, time, gridVar, sfd );
            fHandle_ = sfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::SFCYVariable, mesh_handle_);
            LatVolField<long64> *sfd =
              scinew LatVolField<long64>( mesh_handle_, 1 );
            set_scalar_properties( sfd, var, time, low, 
                                   TypeDescription::SFCYVariable);
            build_field( archive, level, low, var, mat, time, gridVar, sfd );
            fHandle_ = sfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
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
            fHandle_ = vfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
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
            fHandle_ = tfd;
            //  DumpAllocator(default_allocator, "TensorDump.allocator");
          }
        }
        break;
      default:
        error("SFCYVariable<?> Unknown scalar type.");
        //        AllocatorSetDefaultTag(old_tag1);
        return;
      }
      break;
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::SFCZVariable, mesh_handle_);
            LatVolField<double> *sfd =
              scinew LatVolField<double>( mesh_handle_, 1 );
            set_scalar_properties( sfd, var, time, low, 
                                   TypeDescription::SFCZVariable);
            build_field( archive, level, low, var, mat, time, gridVar, sfd );
            fHandle_ = sfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::SFCZVariable, mesh_handle_);
            LatVolField<float> *sfd =
              scinew LatVolField<float>( mesh_handle_, 1 );
            set_scalar_properties( sfd, var, time, low, 
                                   TypeDescription::SFCZVariable);
            build_field( archive, level, low, var, mat, time, gridVar, sfd );
            fHandle_ = sfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::SFCZVariable, mesh_handle_);
            LatVolField<int> *sfd =
              scinew LatVolField<int>( mesh_handle_, 1 );
            set_scalar_properties( sfd, var, time, low, 
                                   TypeDescription::SFCZVariable);
            build_field( archive, level, low, var, mat, time, gridVar, sfd );
            fHandle_ = sfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
          } else {
            update_mesh_handle( level, hi, range, box,
                                TypeDescription::SFCZVariable, mesh_handle_);
            LatVolField<long64> *sfd =
              scinew LatVolField<long64>( mesh_handle_, 1 );
            set_scalar_properties( sfd, var, time, low, 
                                   TypeDescription::SFCZVariable);
            build_field( archive, level, low, var, mat, time, gridVar, sfd );
            fHandle_ = sfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
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
            fHandle_ = vfd;
          }
        }
        break;
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
            fHandle_ = mrfield;
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
            fHandle_ = tfd;
            //  DumpAllocator(default_allocator, "TensorDump.allocator");
          }
        }
        break;
      default:
        error("SFCZVariable<?> Unknown type.");
        //        AllocatorSetDefaultTag(old_tag1);
        return;
      }
      break;
    default:
      error("Not a Uintah type.");
      //      AllocatorSetDefaultTag(old_tag1);
      return;
    }
#endif // ifdef TEMPLATE_FUN
    new2OldPatchMap_.clear();
    fout->send(fHandle_);
  }
  //  AllocatorSetDefaultTag(old_tag1);
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

//     cerr<<"Level "<<i<<":\n";
//    int count = 0;
    SuperPatchContainer::const_iterator superIter;
    for (superIter = superPatches->begin();
         superIter != superPatches->end(); superIter++) {
      IntVector low = (*superIter)->getLow();
      IntVector high = (*superIter)->getHigh();
      IntVector inLow = high; // taking min values starting at high
      IntVector inHigh = low; // taking max values starting at low

//       cerr<<"\tcombined patch "<<count++<<" is "<<low<<", "<<high<<"\n";

      for (unsigned int p = 0; p < (*superIter)->getBoxes().size(); p++) {
        const Patch* patch = (*superIter)->getBoxes()[p];
        inLow = Min(inLow, patch->getInteriorCellLowIndex());
        inHigh = Max(inHigh, patch->getInteriorCellHighIndex());
      }
      
      Patch* newPatch =
        newLevel->addPatch(low, high, inLow, inHigh);
      list<const Patch*> oldPatches; 
      for (unsigned int p = 0; p < (*superIter)->getBoxes().size(); p++) {
        const Patch* patch = (*superIter)->getBoxes()[p];
        oldPatches.push_back(patch);
      }
      new2OldPatchMap_[newPatch] = oldPatches;
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
  //   cerr<<"In update_mesh_handled: type = "<<type<<"\n";
  
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
          //      cerr<<"mesh built:  "<<range.x()<<"x"<<range.y()<<"x"<<
          //        range.z()<<"  size:  "<<box.min()<<", "<<box.max()<<"\n";
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
          //      cerr<<"mesh built:  "<<range.x()<<"x"<<range.y()<<"x"<<
          //        range.z()<<"  size:  "<<box.min()<<", "<<box.max()<<"\n";
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
        //      cerr<<"mesh built:  "<<range.x()<<"x"<<range.y()<<"x"<<
        //        range.z()<<"  size:  "<<box.min()<<", "<<box.max()<<"\n";
      }else if(mesh_handle->get_ni() != (unsigned int) range.x() ||
               mesh_handle->get_nj() != (unsigned int) range.y() ||
               mesh_handle->get_nk() != (unsigned int) range.z() ){
        mesh_handle = scinew LatVolMesh(range.x(), range.y(),
                                        range.z(), box.min(),
                                        box.max());
        //      cerr<<"mesh built:  "<<range.x()<<"x"<<range.y()<<"x"<<
        //        range.z()<<"  size:  "<<box.min()<<", "<<box.max()<<"\n";
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

template <class T, class Var>
void
FieldExtractor::build_multi_level_field( DataArchive& archive, GridP grid,
                                         string& var, Var& v, int mat,
                                         int generation,  double time, 
                                         int timestep, double dt,
                                         int loc,
                                         TypeDescription::Type type,
                                         TypeDescription::Type subtype,
                                         MRLatVolField<T>*& mrfield)

{
  vector<MultiResLevel<T>*> levelfields;
  for(int i = 0; i < grid->numLevels(); i++){
    LevelP level = grid->getLevel( i );
    vector<LockingHandle<LatVolField<T> > > patchfields;
    int count = 0;
    
    // At this point we should have a mimimal patch set in our grid
    // And we want to make a LatVolField for each patch
    for(Level::const_patchIterator patch_it = level->patchesBegin();
        patch_it != level->patchesEnd(); ++patch_it){
      
      IntVector hi, low, range;
      low = (*patch_it)->getLowIndex();
      hi = (*patch_it)->getHighIndex(); 

      // ***** This seems like a hack *****
      range = hi - low + IntVector(1,1,1); 
      // **********************************

      BBox pbox;
      pbox.extend((*patch_it)->getBox().lower());
      pbox.extend((*patch_it)->getBox().upper());
      
      //       cerr<<"before mesh update: range is "<<range.x()<<"x"<<
      //      range.y()<<"x"<< range.z()<<",  low index is "<<low<<
      //      "high index is "<<hi<<" , size is  "<<
      //      pbox.min()<<", "<<pbox.max()<<"\n";
      
      LatVolMeshHandle mh = 0;
      update_mesh_handle(level, hi, range, pbox, type, mh);
      LatVolField<T> *fd = 
        scinew LatVolField<T>( mh, loc );
      if( subtype == TypeDescription::Vector ) {
        set_vector_properties( fd, var, generation, timestep, low, dt, type);
      } else if( subtype == TypeDescription::Matrix3 ){
        set_tensor_properties( fd, low, type);
      } else {
        set_scalar_properties( fd, var, time, low, type);
      }
      //       cerr<<"Field "<<count<<", level "<<i<<" ";
      build_patch_field(archive, (*patch_it), low, var, mat, time, v, fd);
      patchfields.push_back( fd );
      count++;
    }
    //     cerr<<"Added "<<count<<" fields to level "<<i<<"\n";
    MultiResLevel<T> *mrlevel = 
      new MultiResLevel<T>( patchfields, i );
    levelfields.push_back(mrlevel);
  }
  //        MRLatVolField<double>* mrfield =
  mrfield =  scinew MRLatVolField<T>( levelfields );
}
  
template <class T, class Var>
void
FieldExtractor::build_patch_field(DataArchive& archive,
                                  const Patch* patch,
                                  IntVector& new_low,
                                  const string& varname,
                                  int mat,
                                  double time,
                                  Var& /*var*/,
                                  LatVolField<T>*& sfd)
{
  // Initialize the data
  sfd->fdata().initialize(T(0));

  int max_workers = Max(Thread::numProcessors()/2, 2);
  Semaphore* thread_sema = scinew Semaphore( "extractor semaphore",
                                             max_workers);
  int count = 0;
  map<const Patch*, list<const Patch*> >::iterator oldPatch_it =
    new2OldPatchMap_.find(patch);
  if( oldPatch_it != new2OldPatchMap_.end() ){
    list<const Patch*> oldPatches = (*oldPatch_it).second;
    for(list<const Patch*>::iterator patch_it = oldPatches.begin();
        patch_it != oldPatches.end(); ++patch_it){
      IntVector old_low, old_hi;
      Var v;
      int vartype;
      archive.query( v, varname, mat, *patch_it, time);
      if( sfd->basis_order() == 0){
        old_low = (*patch_it)->getCellLowIndex();
        old_hi = (*patch_it)->getCellHighIndex();
      } else if(sfd->get_property("vartype", vartype)){
        old_low = (*patch_it)->getNodeLowIndex();
        switch (vartype) {
        case TypeDescription::SFCXVariable:
          old_hi = (*patch_it)->getSFCXHighIndex();
          break;
        case TypeDescription::SFCYVariable:
          old_hi = (*patch_it)->getSFCYHighIndex();
          break;
        case TypeDescription::SFCZVariable:
          old_hi = (*patch_it)->getSFCZHighIndex();
          break;
        default:
          old_hi = (*patch_it)->getNodeHighIndex();     
        } 
      } 

      IntVector range = old_hi - old_low;

      int z_min = old_low.z();
      int z_max = old_low.z() + old_hi.z() - old_low.z();
      int z_step, z, N = 0;
      if ((z_max - z_min) >= max_workers){
        // in case we have large patches we'll divide up the work 
        // for each patch, if the patches are small we'll divide the
        // work up by patch.
        unsigned long cs = 25000000;  
        unsigned long S = range.x() * range.y() * range.z() * sizeof(T);
        N = Min(Max(S/cs, 1), (max_workers-1));
      }
      N = Max(N,2);
      z_step = (z_max - z_min)/(N - 1);
      for(z = z_min ; z < z_max; z += z_step) {
      
        IntVector min_i(old_low.x(), old_low.y(), z);
        IntVector max_i(old_hi.x(), old_hi.y(), Min(z+z_step, z_max));
        thread_sema->down();
        PatchToFieldThread<Var, T>* ptft = 
          scinew PatchToFieldThread<Var, T>(sfd, v, new_low, min_i, max_i,
                                            // old_low, old_hi,
                                            thread_sema);
#if 1
        // Non threaded version
        ptft->run();
        delete ptft;
#else
        // Threaded version
        Thread *thrd = scinew Thread( ptft, "patch_to_field_worker");
        thrd->detach();
#endif
      }
      count++;
    }
  } else {
    error("No mapping from old patches to new patches.");
  }
  //   cerr<<"used "<<count<<" patches to fill field\n";
  thread_sema->down(max_workers);
  if( thread_sema ) delete thread_sema;
}

template <class T, class Var>
void FieldExtractor::build_field(DataArchive& archive,
                                 const LevelP& level,
                                 IntVector& lo,
                                 const string& varname,
                                 int mat,
                                 double time,
                                 Var& /*var_in*/,
                                 LatVolField<T>*& sfd)
{
  // Initialize the data
  sfd->fdata().initialize(T(0));

  int max_workers = Max(Thread::numProcessors()/2, 2);
  Semaphore* thread_sema = scinew Semaphore( "extractor semaphore",
                                             max_workers);
  //  WallClockTimer my_timer;
  //  my_timer.start();
  
  //   double size = level->numPatches();
  //   int count = 0;
  
  for( Level::const_patchIterator patch_it = level->patchesBegin();
       patch_it != level->patchesEnd(); ++patch_it){
    IntVector low, hi;
    Var var;
    int vartype;
    const TypeDescription* td = var.virtualGetTypeDescription();
    cerr << "build_field::varname = "<<varname<<", type = "<<td->getName()<<"\n";
    archive.query( var, varname, mat, *patch_it, time);
    if( sfd->basis_order() == 0){
      low = (*patch_it)->getCellLowIndex();
      hi = (*patch_it)->getCellHighIndex();
      //       low = (*patch_it)->getNodeLowIndex();
      //       hi = (*patch_it)->getNodeHighIndex() - IntVector(1,1,1);

      //       cerr<<"var.getLowIndex() = "<<var.getLowIndex()<<"\n";
      //       cerr<<"var.getHighIndex() = "<<var.getHighIndex()<<"\n";
      //       cerr<<"getCellLowIndex() = "<< (*patch_it)->getCellLowIndex()
      //        <<"\n";
      //       cerr<<"getCellHighIndex() = "<< (*patch_it)->getCellHighIndex()
      //        <<"\n";
      //       cerr<<"getInteriorCellLowIndex() = "<< (*patch_it)->getInteriorCellLowIndex()
      //        <<"\n";
      //       cerr<<"getInteriorCellHighIndex() = "<< (*patch_it)->getInteriorCellHighIndex()
      //        <<"\n";
      //       cerr<<"getNodeLowIndex() = "<< (*patch_it)->getNodeLowIndex()
      //        <<"\n";
      //       cerr<<"getNodeHighIndex() = "<< (*patch_it)->getNodeHighIndex()
      //        <<"\n";
      //       cerr<<"getInteriorNodeLowIndex() = "<< (*patch_it)->getInteriorNodeLowIndex()
      //        <<"\n";
      //       cerr<<"getInteriorNodeHighIndex() = "<< (*patch_it)->getInteriorNodeHighIndex()
      //        <<"\n\n";
    } else if(sfd->get_property("vartype", vartype)){
      low = (*patch_it)->getNodeLowIndex();
      switch (vartype) {
      case TypeDescription::SFCXVariable:
        hi = (*patch_it)->getSFCXHighIndex();
        break;
      case TypeDescription::SFCYVariable:
        hi = (*patch_it)->getSFCYHighIndex();
        break;
      case TypeDescription::SFCZVariable:
        hi = (*patch_it)->getSFCZHighIndex();
        break;
      default:
        hi = (*patch_it)->getNodeHighIndex();   
      } 
    } 

    IntVector range = hi - low;

    int z_min = low.z();
    int z_max = low.z() + hi.z() - low.z();
    int z_step, z, N = 0;
    if ((z_max - z_min) >= max_workers){
      // in case we have large patches we'll divide up the work 
      // for each patch, if the patches are small we'll divide the
      // work up by patch.
      unsigned long cs = 25000000;  
      unsigned long S = range.x() * range.y() * range.z() * sizeof(T);
      N = Min(Max(S/cs, 1), (max_workers-1));
    }
    N = Max(N,2);
    z_step = (z_max - z_min)/(N - 1);
    for(z = z_min ; z < z_max; z += z_step) {
      
      IntVector min_i(low.x(), low.y(), z);
      IntVector max_i(hi.x(), hi.y(), Min(z+z_step, z_max));
      //      update_progress((count++/double(N))/size, my_timer);
      
      thread_sema->down();
      PatchToFieldThread<Var, T> *ptft = 
        scinew PatchToFieldThread<Var, T>(sfd, var, lo, min_i, max_i,
                                          thread_sema); 
      //        cerr<<"low = "<<low<<", hi = "<<hi<<", min_i = "<<min_i 
      //        <<", max_i = "<<max_i<<endl; 

#if 1
      // Non threaded version
      ptft->run();
      delete ptft;
#else
      // Threaded version
      Thread *thrd = scinew Thread( ptft, "patch_to_field_worker");
      thrd->detach();
#endif
    }
  }
  thread_sema->down(max_workers);
  if( thread_sema ) delete thread_sema;
  //  timer.add( my_timer.time());
  //  my_timer.stop();
}

template <class T>
void 
FieldExtractor::set_scalar_properties(LatVolField<T>*& sfd,
                                      string& varname,
                                      double time, IntVector& low,
                                      TypeDescription::Type type)
{  
  sfd->set_property( "variable", string(varname), true );
  sfd->set_property( "time",     double( time ), true);
  sfd->set_property( "offset",   IntVector(low), true);
  sfd->set_property( "vartype",  int(type),true);
}

template <class T>
void
FieldExtractor::set_vector_properties(LatVolField<T>*& vfd, string& var,
                                      int generation, int timestep,
                                      IntVector& low, double dt,
                                      TypeDescription::Type type)
{
  vfd->set_property( "varname",    string(var), true);
  vfd->set_property( "generation", generation, true);
  vfd->set_property( "timestep",   timestep, true);
  vfd->set_property( "offset",     IntVector(low), true);
  vfd->set_property( "delta_t",    dt, true);
  vfd->set_property( "vartype",    int(type),true);
}

template <class T>
void 
FieldExtractor::set_tensor_properties(LatVolField<T>*& tfd,  IntVector& low,
                                      TypeDescription::Type /*type*/)
{
  tfd->set_property( "vartype", int(TypeDescription::CCVariable),true);
  tfd->set_property( "offset",  IntVector(low), true);
}

#ifdef TEMPLATE_FUN

void
FieldExtractor::set_field_properties(Field* field, QueryInfo& qinfo,
                                     IntVector& offset) {
  field->set_property( "varname",    string(qinfo.varname), true);
  field->set_property( "generation", qinfo.generation, true);
  field->set_property( "timestep",   qinfo.timestep, true);
  field->set_property( "offset",     IntVector(offset), true);
  field->set_property( "delta_t",    qinfo.dt, true);
  field->set_property( "vartype",    int(qinfo.type->getType()),true);
}

template <class Var, class T>
FieldHandle
FieldExtractor::build_multi_level_field( QueryInfo& qinfo, int loc)
{
  // Build the minimal patch set.  build_minimal_patch_grid should
  // eventually return the map rather than have it as a member
  // variable to map with all the other parameters that aren't being
  // used by the class.
  GridP grid_minimal = build_minimal_patch_grid( qinfo.grid );
  
  vector<MultiResLevel<T>*> levelfields;
  for(int i = 0; i < grid_minimal->numLevels(); i++){
    LevelP level = grid_minimal->getLevel( i );
    vector<LockingHandle<LatVolField<T> > > patchfields;
    
    // At this point we should have a mimimal patch set in our
    // grid_minimal, and we want to make a LatVolField for each patch.
    for(Level::const_patchIterator patch_it = level->patchesBegin();
        patch_it != level->patchesEnd(); ++patch_it){
      
      IntVector patch_low, patch_high, range;
      patch_low = (*patch_it)->getLowIndex();
      patch_high = (*patch_it)->getHighIndex(); 

      // ***** This seems like a hack *****
      range = patch_high - patch_low + IntVector(1,1,1); 
      // **********************************

      BBox pbox;
      pbox.extend((*patch_it)->getBox().lower());
      pbox.extend((*patch_it)->getBox().upper());
      
      //       cerr<<"before mesh update: range is "<<range.x()<<"x"<<
      //      range.y()<<"x"<< range.z()<<",  low index is "<<low<<
      //      "high index is "<<hi<<" , size is  "<<
      //      pbox.min()<<", "<<pbox.max()<<"\n";
      
      LatVolMeshHandle mh = 0;
      update_mesh_handle(qinfo.level, patch_high, range, pbox,
                         qinfo.type->getType(), mh);
      LatVolField<T> *field = scinew LatVolField<T>( mh, loc );
      set_field_properties(field, qinfo, patch_low);

      build_patch_field<Var, T>(qinfo, (*patch_it), patch_low, field);
      patchfields.push_back( field );
    }
    MultiResLevel<T> *mrlevel = scinew MultiResLevel<T>( patchfields, i );
    levelfields.push_back(mrlevel);
  }
  return scinew MRLatVolField<T>( levelfields );
}

template <class Var, class T>
void
FieldExtractor::getPatchData(QueryInfo& qinfo, IntVector& offset,
                             LatVolField<T>* sfield, const Patch* patch)
{
  cerr << "getPatchData:: offset = "<<offset<<", sfield = "<<sfield<<", patch = "<<patch<<"\n";
  IntVector patch_low, patch_high;
  Var patch_data;
  int vartype;
  //  const TypeDescription* td = var.virtualGetTypeDescription();
  //    cerr << "build_field::varname = "<<varname<<", type = "<<td->getName()<<"\n";
  if( sfield->basis_order() == 0) {
    patch_low = patch->getCellLowIndex();
    patch_high = patch->getCellHighIndex();
    //       patch_low = patch->getNodeLowIndex();
    //       patch_high = patch->getNodeHighIndex() - IntVector(1,1,1);
    
    //       cerr<<"patch_data.getLowIndex() = "<<patch_data.getLowIndex()<<"\n";
    //       cerr<<"patch_data.getHighIndex() = "<<patch_data.getHighIndex()<<"\n";
    //       cerr<<"getCellLowIndex() = "<< patch->getCellLowIndex()
    //        <<"\n";
    //       cerr<<"getCellHighIndex() = "<< patch->getCellHighIndex()
    //        <<"\n";
    //       cerr<<"getInteriorCellLowIndex() = "<< patch->getInteriorCellLowIndex()
    //        <<"\n";
    //       cerr<<"getInteriorCellHighIndex() = "<< patch->getInteriorCellHighIndex()
    //        <<"\n";
    //       cerr<<"getNodeLowIndex() = "<< patch->getNodeLowIndex()
    //        <<"\n";
    //       cerr<<"getNodeHighIndex() = "<< patch->getNodeHighIndex()
    //        <<"\n";
    //       cerr<<"getInteriorNodeLowIndex() = "<< patch->getInteriorNodeLowIndex()
    //        <<"\n";
    //       cerr<<"getInteriorNodeHighIndex() = "<< patch->getInteriorNodeHighIndex()
    //        <<"\n\n";
  } else if(sfield->get_property("vartype", vartype)){
    patch_low = patch->getNodeLowIndex();
    switch (vartype) {
    case TypeDescription::SFCXVariable:
      patch_high = patch->getSFCXHighIndex();
      break;
    case TypeDescription::SFCYVariable:
      patch_high = patch->getSFCYHighIndex();
      break;
    case TypeDescription::SFCZVariable:
      patch_high = patch->getSFCZHighIndex();
      break;
    default:
      patch_high = patch->getNodeHighIndex();   
    } 
  } 

  try {
    qinfo.archive->query(patch_data, qinfo.varname, qinfo.mat, patch,
                         qinfo.time);
  } catch (Exception& e) {
    error("query caused an exception");
    cerr << "getPatchData::error in query function\n";
    cerr << e.message()<<"\n";
    return;
  }
  
  PatchToFieldThread<Var, T> *ptft = 
    scinew PatchToFieldThread<Var, T>(sfield, patch_data, offset,
                                      patch_low, patch_high);
  //        cerr<<"patch_low = "<<patch_low<<", patch_high = "<<patch_high<<", min_i = "<<min_i 
  //        <<", max_i = "<<max_i<<endl; 
  
  ptft->run();
  delete ptft;
}

// For help with build_multi_level_field
template <class Var, class T>
void
FieldExtractor::build_patch_field(QueryInfo& qinfo,
                                  const Patch* patch,
                                  IntVector& offset,
                                  LatVolField<T>* field)
{
  // Initialize the data
  field->fdata().initialize(T(0));

  map<const Patch*, list<const Patch*> >::iterator oldPatch_it =
    new2OldPatchMap_.find(patch);
  if( oldPatch_it == new2OldPatchMap_.end() ) {
    error("No mapping from old patches to new patches.");
    return;
  }
    
  list<const Patch*> oldPatches = (*oldPatch_it).second;
  for(list<const Patch*>::iterator patch_it = oldPatches.begin();
      patch_it != oldPatches.end(); ++patch_it){
    getPatchData<Var, T>(qinfo, offset, field, *patch_it);
  }
}

template <class Var, class T>
void
FieldExtractor::build_field(QueryInfo& qinfo, IntVector& offset,
                            LatVolField<T>* field)
{
  // Initialize the data
  field->fdata().initialize(T(0));

  //  WallClockTimer my_timer;
  //  my_timer.start();
  
  for( Level::const_patchIterator patch_it = qinfo.level->patchesBegin();
       patch_it != qinfo.level->patchesEnd(); ++patch_it){
  //      update_progress(somepercentage, my_timer);
    getPatchData<Var, T>(qinfo, offset, field, *patch_it);
  }

  //  timer.add( my_timer.time());
  //  my_timer.stop();
}

template<class Var, class T>
FieldHandle FieldExtractor::getData(QueryInfo& qinfo, IntVector& offset,
                                    LatVolMeshHandle mesh_handle,
                                    int basis_order)
{
  if (qinfo.get_all_levels) {
    return build_multi_level_field<Var, T>(qinfo, basis_order);
  } else {
    LatVolField<T>* sf = scinew LatVolField<T>(mesh_handle, basis_order);
    set_field_properties(sf, qinfo, offset);
//     int vartype;
//     if (sf->get_property("vartype", vartype)) {
//       cerr << "getData::get_property:: vartype =  "<<vartype<<"\n";
//     } else {
//       cerr << "getData::get_property:: couldn't get vartype\n";
//     }
//     sleep(4);
    build_field<Var, T>(qinfo, offset, sf);
    return sf;
  }
}

template<class T>
FieldHandle FieldExtractor::getVariable(QueryInfo& qinfo, IntVector& offset,
                                        LatVolMeshHandle mesh_handle)
{
  switch( qinfo.type->getType() ) {
  case TypeDescription::CCVariable:
    return getData<CCVariable<T>, T>(qinfo, offset, mesh_handle, 0);
  case TypeDescription::NCVariable:
    return getData<NCVariable<T>, T>(qinfo, offset, mesh_handle, 1);
  case TypeDescription::SFCXVariable:
    return getData<SFCXVariable<T>, T>(qinfo, offset, mesh_handle, 1);
  case TypeDescription::SFCYVariable:
    return getData<SFCYVariable<T>, T>(qinfo, offset, mesh_handle, 1);
  case TypeDescription::SFCZVariable:
    return getData<SFCZVariable<T>, T>(qinfo, offset, mesh_handle, 1);
  default:
    cerr << "Type is unknown.\n";
    return 0;
  }
}
#endif



