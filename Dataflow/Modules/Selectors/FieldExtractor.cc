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
  DataArchive& archive = *((*(archiveH.get_rep()))());

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
  level->findIndexRange(low, hi);
  range = hi - low;
  BBox box;
  level->getSpatialRange(box);
  
//   IntVector cellHi, cellLo;
//   level->findCellIndexRange(cellLo, cellHi);
  
//   cerr<<"before anything data range is:  "<<range.x()<<"x"<<range.y()<<"x"<<
//     range.z()<<"  size:  "<<box.min()<<", "<<box.max()<<"\n";
  
  update_mesh_handle( level, hi, range, box, type->getType(), mesh_handle_);

  QueryInfo qinfo(&archive, generation, grid, level, var, mat, type,
                  get_all_levels, time, timestep, dt);

  FieldHandle fHandle_;

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
  new2OldPatchMap_.clear();
  fout->send(fHandle_);
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

// Sets all sorts of properties using the PropertyManager facility
// of the Field.  This is called for all types of Fields.
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

// Creates a MRLatVolField.
template <class Var, class T>
FieldHandle
FieldExtractor::build_multi_level_field( QueryInfo& qinfo, int basis_order)
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
      LatVolField<T> *field = scinew LatVolField<T>( mh, basis_order );
      set_field_properties(field, qinfo, patch_low);

      build_patch_field<Var, T>(qinfo, (*patch_it), patch_low, field);
      patchfields.push_back( field );
    }
    MultiResLevel<T> *mrlevel = scinew MultiResLevel<T>( patchfields, i );
    levelfields.push_back(mrlevel);
  }
  return scinew MRLatVolField<T>( levelfields );
}

// This does the actuall work of getting the data from the
// DataArchive for a single patch and filling the field.  This is
// called by both build_field and build_patch_field.
template <class Var, class T>
void
FieldExtractor::getPatchData(QueryInfo& qinfo, IntVector& offset,
                             LatVolField<T>* sfield, const Patch* patch)
{
  IntVector patch_low, patch_high;
  Var patch_data;

  try {
    qinfo.archive->query(patch_data, qinfo.varname, qinfo.mat, patch,
                         qinfo.time);
  } catch (Exception& e) {
    error("query caused an exception: " + string(e.message()));
    cerr << "getPatchData::error in query function\n";
    cerr << e.message()<<"\n";
    return;
  }
  
  int vartype;
  if( sfield->basis_order() == 0) {
    patch_low = patch->getCellLowIndex();
    patch_high = patch->getCellHighIndex();
#if 0
    cerr<<"patch_data.getLowIndex() = "<<patch_data.getLowIndex()<<"\n";
    cerr<<"patch_data.getHighIndex() = "<<patch_data.getHighIndex()<<"\n";
    cerr<<"getCellLowIndex() = "<< patch->getCellLowIndex()<<"\n";
    cerr<<"getCellHighIndex() = "<< patch->getCellHighIndex()<<"\n";
    cerr<<"getInteriorCellLowIndex() = "<< patch->getInteriorCellLowIndex()<<"\n";
    cerr<<"getInteriorCellHighIndex() = "<< patch->getInteriorCellHighIndex()<<"\n";
    cerr<<"getNodeLowIndex() = "<< patch->getNodeLowIndex()<<"\n";
    cerr<<"getNodeHighIndex() = "<< patch->getNodeHighIndex()<<"\n";
    cerr<<"getInteriorNodeLowIndex() = "<< patch->getInteriorNodeLowIndex()<<"\n";
    cerr<<"getInteriorNodeHighIndex() = "<< patch->getInteriorNodeHighIndex()<<"\n\n";
#endif
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
  } else {
    error("getPatchData::Problem with getting vartype from field");
    return;
  }

  PatchToFieldThread<Var, T> *ptft = 
    scinew PatchToFieldThread<Var, T>(sfield, patch_data, offset,
                                      patch_low, patch_high);
  ptft->run();
  delete ptft;
}

// Similar to build_field, but is called from build_multi_level_field.
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

// Calls query for a single-level data set.
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
    getPatchData<Var, T>(qinfo, offset, field, *patch_it);
  //      update_progress(somepercentage, my_timer);
  }

  //  timer.add( my_timer.time());
  //  my_timer.stop();
}

// This function makes a switch between building multi-level data or
// single-level data.  Makes a call to either build_field or or
// build_multi_level_field.  The basis_order pertains to whether the
// data is node or cell centerd.  Type Var should look something
// like CCVariable<T> or NCVariable<T>.
template<class Var, class T>
FieldHandle
FieldExtractor::getData(QueryInfo& qinfo, IntVector& offset,
                        LatVolMeshHandle mesh_handle,
                        int basis_order)
{
  if (qinfo.get_all_levels) {
    return build_multi_level_field<Var, T>(qinfo, basis_order);
  } else {
    LatVolField<T>* sf = scinew LatVolField<T>(mesh_handle, basis_order);
    set_field_properties(sf, qinfo, offset);
    build_field<Var, T>(qinfo, offset, sf);
    return sf;
  }
}

// This is the first function on your way to getting a field.  This
// makes a template switch on the type of variable (CCVariable,
// NCVariable, etc.).  It then calls getData.  The type of T is
// double, int, Vector, etc.
template<class T>
FieldHandle
FieldExtractor::getVariable(QueryInfo& qinfo, IntVector& offset,
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



