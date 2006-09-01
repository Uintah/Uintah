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

#include <Packages/Uintah/Dataflow/Modules/Selectors/FieldExtractor.h>

#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Transform.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Timer.h>

#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Variables/LocallyComputedPatchVarMap.h>
#include <Packages/Uintah/Core/Grid/Variables/ShareAssignArray3.h>
#include <Packages/Uintah/Dataflow/Ports/ArchivePort.h>
 
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
  level_(get_ctx()->subVar("level")), grid(0), 
  archiveH(0), mesh_handle_(0),
  tcl_status(get_ctx()->subVar("tcl_status")), 
  sVar(get_ctx()->subVar("sVar")),
  sMatNum(get_ctx()->subVar("sMatNum")),
  remove_boundary_cells(get_ctx()->subVar("remove_boundary_cells")),
  type(0)
{ 
} 

//------------------------------------------------------------ 
FieldExtractor::~FieldExtractor()
{} 

// //------------------------------------------------------------- 

void
FieldExtractor::build_GUI_frame()
{
  // create the variable extractor interface.
  string visible;
  get_gui()->eval(get_id() + " isVisible", visible);
  if( visible == "0" ){
    get_gui()->execute(get_id() + " buildTopLevel");
  }
}

//------------------------------------------------------------- 
// get time, set timestep, set generation, update grid and update gui

double
FieldExtractor::field_update()
{
  DataArchiveHandle archive = archiveH->getDataArchive();
  // set the index for the correct timestep.
  int new_timestep = archiveH->timestep();
  vector< const TypeDescription *> types;
  vector< string > names;
  vector< int > indices;
  double time;
  // check to see if we have a new Archive
  archive->queryVariables(names, types);
  int new_generation = archiveH->generation;
  bool archive_dirty =  new_generation != generation;
  if (archive_dirty) {
    generation = new_generation;
    timestep = -1; // make sure old timestep is different from current
    times.clear();
    mesh_handle_ = 0;
    archive->queryTimesteps( indices, times );
  }

  if (timestep != new_timestep) {
    time = times[new_timestep];
    grid = 0;
    grid = archive->queryGrid(time);
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
  DataArchiveHandle archive = archiveH->getDataArchive();
  int levels = grid->numLevels();
  int guilevel = level_.get();
  LevelP level = grid->getLevel((guilevel == levels ? 0 : guilevel) );

  Patch* r = *(level->patchesBegin());
  ConsecutiveRangeSet matls = 
    archive->queryMaterials(var, r, times[timestep]);

  ostringstream os;
  os << levels;

  string visible;
  get_gui()->eval(get_id() + " isVisible", visible);
  if( visible == "1"){
    get_gui()->execute(get_id() + " destroyFrames");
    get_gui()->execute(get_id() + " build");
    get_gui()->execute(get_id() + " buildLevels "+ os.str());
    get_gui()->execute(get_id() + " buildMaterials " + matls.expandedString().c_str());
      
    get_gui()->execute(get_id() + " setVars " + varnames.c_str());
    get_gui()->execute(get_id() + " buildVarList");
      
    get_gui()->execute("update idletasks");
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
  if (!(in->get(handle) && handle.get_rep())) {
    warning("VariableExtractor::execute() - No data from input port.");
    return;
  }
   
  if (archiveH.get_rep() == 0 ){
    // first time through a frame must be built
    build_GUI_frame();
  }
  
  archiveH = handle;

  //   cerr<<"Gui say levels = "<<level_.get()<<"\n";
  // get time, set timestep, set generation, update grid and update gui
  double time = field_update(); // yeah it does all that

  if(type == 0){
    warning( "No variables found.");
    return;
  }
  
  string var(sVar.get());
  int mat = sMatNum.get();
  if(var == "") {
    warning("empty variable specified.  Doing nothing.");
    return;
  }

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

//   cerr<<"grid->numLevels() = "<<grid->numLevels()<<" and get_all_levels = "<<get_all_levels<<"\n";
    
  const TypeDescription* subtype = type->getSubType();
  LevelP level;
  
  if( get_all_levels ){ 
    level = grid->getLevel( 0 );
  } else {
    level = grid->getLevel( level_.get() );
  }
  IntVector hi, low, range;
  BBox box;

  if( remove_boundary_cells.get() == 1 ){
    level->findInteriorIndexRange(low, hi);
    level->getInteriorSpatialRange(box);
    // ***** This seems like a hack *****
    range = hi - low;// + IntVector(1,1,1); 
    // **********************************
  } else {
    level->findIndexRange(low, hi);
    level->getSpatialRange(box);
    range = hi - low;
  }
  
  IntVector cellHi, cellLo;
  level->findCellIndexRange(cellLo, cellHi);
  
//   cerr<<"before anything data range is:  "<<range.x()<<"x"<<range.y()<<"x"<<
//      range.z()<<"  size:  "<<box.min()<<", "<<box.max()<<"\n";
//   cerr<<"and hi index is "<<hi<<", and low index is "<<low<<"\n";
  
  if( !update_mesh_handle( level, hi, range, box, 
                           type->getType(), mesh_handle_, 
                           remove_boundary_cells.get()) ) {
        error("in update_mesh_handle:: Not a Uintah type.");
        return;
  }


  QueryInfo qinfo(archiveH->getDataArchive(),
                  generation, grid, level, var, mat, type,
                  get_all_levels, time, timestep, dt);

  CompileInfoHandle ci = FieldExtractorAlgo::get_compile_info(type, subtype);
  SCIRun::Handle<FieldExtractorAlgo> algo;
  if( !module_dynamic_compile(ci, algo) ){
    error("dynamic compile failed.");
    return;
  }

  FieldHandle fHandle_ = algo->execute(qinfo, low, mesh_handle_,
                                       remove_boundary_cells.get());

  fout->send(fHandle_);
}

CompileInfoHandle
FieldExtractorAlgo::get_compile_info( const Uintah::TypeDescription *vt,
                                      const Uintah::TypeDescription *t )
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(SCIRun::TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("FieldExtractorAlgoT");
  static const string base_class_name("FieldExtractorAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       vt->getFileName() + ".",
                       base_class_name, 
                       template_class_name, 
                       vt->getName() + ", " +
                       t->getName() );

  
  // Add in the include path to compile this obj
  rval->add_include(include_path);
  // Add namespace
  rval->add_namespace("Uintah");
//   vt->fill_compile_info(rval);
//   t->fill_compile_info(rval);
  return rval;
}


GridP 
FieldExtractorAlgo::build_minimal_patch_grid( GridP oldGrid )
{
  int nlevels = oldGrid->numLevels();
  GridP newGrid = scinew Grid();
  const SuperPatchContainer* superPatches;

  for( int i = 0; i < nlevels; i++ ){
    LevelP level = oldGrid->getLevel(i);
    LocallyComputedPatchVarMap patchGrouper;
    const PatchSubset* patches = level->allPatches()->getUnion();
    patchGrouper.addComputedPatchSet(patches);
    patchGrouper.makeGroups();
    superPatches = patchGrouper.getSuperPatches(level.get_rep());
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
      newLevel->setExtraCells( level->getExtraCells() );
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

bool FieldExtractor::update_mesh_handle( LevelP& level,
                                         IntVector& hi,
                                         IntVector& range,
                                         BBox& box,
                                         Uintah::TypeDescription::Type type,
                                         LVMeshHandle& mesh_handle,
                                         int remove_boundary)
{
  //   cerr<<"In update_mesh_handled: type = "<<type<<"\n";
  
  switch ( type ){
  case TypeDescription::CCVariable:
    {
      IntVector cellHi, cellLo, levelHi, levelLo;
      if( remove_boundary == 1){
        level->findInteriorCellIndexRange(cellLo, cellHi);
        level->findInteriorIndexRange( levelLo, levelHi);
      } else {
        level->findCellIndexRange(cellLo, cellHi);
        level->findIndexRange( levelLo, levelHi);
      }
      if( mesh_handle.get_rep() == 0 ){
        if(is_periodic_bcs(cellHi, hi) && is_periodic_bcs(cellHi, levelHi)){
          IntVector newrange(0,0,0);
          get_periodic_bcs_range( cellHi, hi, range, newrange);
          mesh_handle = scinew LVMesh(newrange.x(),newrange.y(),
                                          newrange.z(), box.min(),
                                          box.max());
        } else {
          mesh_handle = scinew LVMesh(range.x(), range.y(),
                                          range.z(), box.min(),
                                          box.max());
//                cerr<<"mesh built:  "<<range.x()<<"x"<<range.y()<<"x"<<
//                  range.z()<<"  size:  "<<box.min()<<", "<<box.max()<<"\n";
        }
      } else if(mesh_handle->get_ni() != (unsigned int) range.x() ||
                mesh_handle->get_nj() != (unsigned int) range.y() ||
                mesh_handle->get_nk() != (unsigned int) range.z() ){
        if(is_periodic_bcs(cellHi, hi) && is_periodic_bcs(cellHi, levelHi)){
          IntVector newrange(0,0,0);
          get_periodic_bcs_range( cellHi, hi, range, newrange);
          mesh_handle = scinew LVMesh(newrange.x(),newrange.y(),
                                          newrange.z(), box.min(),
                                          box.max());
        } else {
          mesh_handle = scinew LVMesh(range.x(), range.y(),
                                          range.z(), box.min(),
                                          box.max());
//                cerr<<"mesh built:  "<<range.x()<<"x"<<range.y()<<"x"<<
//                  range.z()<<"  size:  "<<box.min()<<", "<<box.max()<<"\n";
        }
      } 
      return true;
    }
  case TypeDescription::NCVariable:
    {
      if( mesh_handle.get_rep() == 0 ){
        mesh_handle = scinew LVMesh(range.x(), range.y(),
                                        range.z(), box.min(),
                                        box.max());
//              cerr<<"mesh built:  "<<range.x()<<"x"<<range.y()<<"x"<<
//                range.z()<<"  size:  "<<box.min()<<", "<<box.max()<<"\n";
      }else if(mesh_handle->get_ni() != (unsigned int) range.x() ||
               mesh_handle->get_nj() != (unsigned int) range.y() ||
               mesh_handle->get_nk() != (unsigned int) range.z() ){
        mesh_handle = scinew LVMesh(range.x(), range.y(),
                                        range.z(), box.min(),
                                        box.max());
//              cerr<<"mesh built:  "<<range.x()<<"x"<<range.y()<<"x"<<
//                range.z()<<"  size:  "<<box.min()<<", "<<box.max()<<"\n";
      }
      return true;
    }
  case TypeDescription::SFCXVariable:
    {
      IntVector cellHi, cellLo, levelHi, levelLo;
      if( remove_boundary == 1){
        level->findInteriorCellIndexRange(cellLo, cellHi);
        level->findInteriorIndexRange( levelLo, levelHi);
      } else {
        level->findCellIndexRange(cellLo, cellHi);
        level->findIndexRange( levelLo, levelHi);
      }
      if( mesh_handle.get_rep() == 0 ){
        if(is_periodic_bcs(cellHi, hi) && is_periodic_bcs(cellHi, levelHi)){
          IntVector newrange(0,0,0);
          get_periodic_bcs_range( cellHi, hi, range, newrange);
          mesh_handle = scinew LVMesh(newrange.x(),newrange.y() - 1,
                                          newrange.z() - 1, box.min(),
                                          box.max());
        } else {
          mesh_handle = scinew LVMesh(range.x(), range.y() - 1,
                                      range.z() - 1, box.min(),
                                      box.max());
        }
      } else if(mesh_handle->get_ni() != (unsigned int) range.x() ||
                mesh_handle->get_nj() != (unsigned int) range.y() -1 ||
                mesh_handle->get_nk() != (unsigned int) range.z() -1 ){
        if(is_periodic_bcs(cellHi, hi) && is_periodic_bcs(cellHi, levelHi)){
          IntVector newrange(0,0,0);
          get_periodic_bcs_range( cellHi, hi, range, newrange);
          mesh_handle = scinew LVMesh(newrange.x(),newrange.y() - 1,
                                          newrange.z() - 1, box.min(),
                                          box.max());
        } else {
          mesh_handle = scinew LVMesh(range.x(), range.y() - 1,
                                      range.z()-1, box.min(),
                                      box.max());
        }
      }
      return true;
    }
  case TypeDescription::SFCYVariable:
    {
      IntVector cellHi, cellLo, levelHi, levelLo;
      if( remove_boundary == 1){
        level->findInteriorCellIndexRange(cellLo, cellHi);
        level->findInteriorIndexRange( levelLo, levelHi);
      } else {
        level->findCellIndexRange(cellLo, cellHi);
        level->findIndexRange( levelLo, levelHi);
      }
      if( mesh_handle.get_rep() == 0 ){
        if(is_periodic_bcs(cellHi, hi) && is_periodic_bcs(cellHi, levelHi)){
          IntVector newrange(0,0,0);
          get_periodic_bcs_range( cellHi, hi, range, newrange);
          mesh_handle = scinew LVMesh(newrange.x() - 1,newrange.y(),
                                          newrange.z() - 1, box.min(),
                                          box.max());
        } else {
          mesh_handle = scinew LVMesh(range.x()-1, range.y(),
                                      range.z()-1, box.min(),
                                      box.max());
        }
      } else if(mesh_handle->get_ni() != (unsigned int) range.x() -1 ||
                mesh_handle->get_nj() != (unsigned int) range.y() ||
                mesh_handle->get_nk() != (unsigned int) range.z() -1 ){
        if(is_periodic_bcs(cellHi, hi) && is_periodic_bcs(cellHi, levelHi)){
          IntVector newrange(0,0,0);
          get_periodic_bcs_range( cellHi, hi, range, newrange);
          mesh_handle = scinew LVMesh(newrange.x() - 1,newrange.y(),
                                          newrange.z() - 1, box.min(),
                                          box.max());
        } else {
          mesh_handle = scinew LVMesh(range.x()-1, range.y(),
                                      range.z()-1, box.min(),
                                      box.max());
        }
      }
      return true;
    }
  case TypeDescription::SFCZVariable:
     {
      IntVector cellHi, cellLo, levelHi, levelLo;
      if( remove_boundary == 1){
        level->findInteriorCellIndexRange(cellLo, cellHi);
        level->findInteriorIndexRange( levelLo, levelHi);
      } else {
        level->findCellIndexRange(cellLo, cellHi);
        level->findIndexRange( levelLo, levelHi);
      }
      if( mesh_handle.get_rep() == 0 ){
        if(is_periodic_bcs(cellHi, hi) && is_periodic_bcs(cellHi, levelHi)){
          IntVector newrange(0,0,0);
          get_periodic_bcs_range( cellHi, hi, range, newrange);
          mesh_handle = scinew LVMesh(newrange.x() - 1,newrange.y() - 1,
                                          newrange.z(), box.min(),
                                          box.max());
        } else {
          mesh_handle = scinew LVMesh(range.x()-1, range.y()-1,
                                      range.z(), box.min(),
                                      box.max());
        }
      } else if(mesh_handle->get_ni() != (unsigned int) range.x() -1 ||
                mesh_handle->get_nj() != (unsigned int) range.y() -1 ||
                mesh_handle->get_nk() != (unsigned int) range.z() ){
        if(is_periodic_bcs(cellHi, hi) && is_periodic_bcs(cellHi, levelHi)){
          IntVector newrange(0,0,0);
          get_periodic_bcs_range( cellHi, hi, range, newrange);
          mesh_handle = scinew LVMesh(newrange.x() - 1,newrange.y() - 1,
                                          newrange.z(), box.min(),
                                          box.max());
        } else {
          mesh_handle = scinew LVMesh(range.x()-1, range.y()-1,
                                      range.z(), box.min(),
                                      box.max());
        }
      }     
      return true;
    }
  default:
    return false;
  }
}

// Sets all sorts of properties using the PropertyManager facility
// of the Field.  This is called for all types of Fields.
void
FieldExtractorAlgo::set_field_properties( Field* field, 
                                          QueryInfo& qinfo,
                                          IntVector& offset )
{
  BBox b;
  qinfo.grid->getInteriorSpatialRange( b );

  field->set_property( "spacial_min", b.min(), true);
  field->set_property( "spacial_max", b.max(), true);
  field->set_property( "name",        string(qinfo.varname), true);
  field->set_property( "generation",  qinfo.generation, true);
  field->set_property( "timestep",    qinfo.timestep, true);
  field->set_property( "offset",      IntVector(offset), true);
  field->set_property( "delta_t",     qinfo.dt, true);
  field->set_property( "time",        qinfo.time, true);
  field->set_property( "vartype",     int(qinfo.type->getType()),true);
}


