/****************************************
Class
    FieldExtractor

    

OVERVIEW TEXT
    This module receives a DataArchive object.  The user
    interface is dynamically created based information provided by the
    DataArchive.  The user can then select which variables he/she
    wishes to view in a visualization.



KEYWORDS
    ParticleGridReader, Material/Particle Method

AUTHOR
    Kurt Zimmerman, James Bigler
    Department of Computer Science
    University of Utah
    June, 2000

    Copyright (C) 2000 SCI Group

LOG
    Created June 27, 2000
****************************************/
#ifndef FIELDEXTRACTOR_H
#define FIELDEXTRACTOR_H 1

#include <Core/Math/Matrix3.h>
#include <SCIRun/Core/Basis/Constant.h>
#include <SCIRun/Core/Basis/HexTrilinearLgn.h>
#include <SCIRun/Core/Datatypes/LatVolMesh.h>
#include <SCIRun/Core/Datatypes/MultiLevelField.h>
#include <SCIRun/Core/Geometry/IntVector.h>
#include <SCIRun/Core/Geometry/Point.h>
#include <SCIRun/Dataflow/GuiInterface/GuiVar.h> 
#include <SCIRun/Core/Math/MinMax.h>
#include <SCIRun/Core/Util/Timer.h>
#include <SCIRun/Core/Util/TypeDescription.h>
#include <SCIRun/Core/Util/DynamicLoader.h>

#include <SCIRun/Dataflow/Network/Module.h> 
#include <SCIRun/Dataflow/Network/Ports/FieldPort.h>

#include <Core/Datatypes/Archive.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Dataflow/Modules/Selectors/PatchToField.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>


#include <Dataflow/Modules/Selectors/share.h>
namespace Uintah {
using namespace SCIRun;

typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh;
typedef LVMesh::handle_type LVMeshHandle;

// This struct is used to store parameters passed up the chain to the
// build field functions.  This makes it easier to pass a large
// number of parameters.
struct QueryInfo {
public:
  QueryInfo() {}
  QueryInfo(const DataArchiveHandle& archive, int generation,
            GridP& grid, LevelP& level,
            string varname,
            int mat,
            const Uintah::TypeDescription *type,
            bool get_all_levels,
            double time, int timestep, double dt):
    archive(archive), generation(generation),
    grid(grid), level(level),
    varname(varname), mat(mat), type(type),
    get_all_levels(get_all_levels),
    time(time), timestep(timestep), dt(dt)
  {}
  
  DataArchiveHandle archive;
  int generation;
  GridP grid;
  LevelP level;
  string varname;
  int mat;
  const Uintah::TypeDescription *type;
  bool get_all_levels;
  double time;
  int timestep;
  double dt;
};

class SCISHARE FieldExtractor : public Module { 
  
public: 
  
  
  // GROUP: Constructors
  //////////
  FieldExtractor(const string& name,
                 GuiContext* ctx,
                 const string& cat="unknown",
                 const string& pack="unknown");
  // GROUP: Destructors
  //////////
  virtual ~FieldExtractor(); 
  
  // GROUP: cloning and execution 
  ////////// 
  virtual void execute(); 

  // helper functions
  static bool is_periodic_bcs(IntVector cellir, IntVector ir);
  static  void get_periodic_bcs_range(IntVector cellir, IntVector ir,
                              IntVector range, IntVector& newir);
  
  
  static bool update_mesh_handle( LevelP& level,
                                   IntVector& hi,
                                   IntVector& range,
                                   BBox& box,
                                   TypeDescription::Type type,
                                   LVMeshHandle& mesh_handle,
                                  int remove_boundary);
    
protected:
  virtual void get_vars(vector< string >&,
                        vector< const TypeDescription *>&) = 0;
  void build_GUI_frame();
  void update_GUI(const string& var,
                  const string& varnames);
  double field_update();

  
  vector< double > times;
  int generation;
  int timestep;
  int material;
  int levelnum;
  GuiInt level_;
  GridP grid;
  ArchiveHandle  archiveH;
  LVMeshHandle mesh_handle_;
  
  GuiString tcl_status;
  
  GuiString sVar;
  GuiInt sMatNum;
  GuiInt remove_boundary_cells;
  const TypeDescription *type;
  
  
}; //class 


class SCISHARE FieldExtractorAlgo: public DynamicAlgoBase
{
public:
  virtual FieldHandle  execute(QueryInfo& quinfo, IntVector& offset,
                               LVMeshHandle mh, int remove_boundary) = 0;
  static CompileInfoHandle get_compile_info( const Uintah::TypeDescription *vt,
                                             const Uintah::TypeDescription *t);
//   static void update_mesh_handle( LevelP& level,
//                                   IntVector& hi,
//                                   IntVector& range,
//                                   BBox& box,
//                                   TypeDescription::Type type,
//                                   LVMeshHandle& mesh_handle,
//                                   int remove_boundary);

protected:
  // Sets all sorts of properties using the PropertyManager facility
  // of the Field.  This is called for all types of Fields.
  void set_field_properties(Field* field, QueryInfo& qinfo,
                            IntVector& offset);  

  GridP build_minimal_patch_grid( GridP oldGrid );
  map<const Patch*, list<const Patch*> > new2OldPatchMap_;

};

template<class T>
class FieldExtractorAlgoT: public FieldExtractorAlgo
{
public:
   virtual FieldHandle
   execute(QueryInfo& quinfo, IntVector& offset,
           LVMeshHandle mh, int remove_boundary);
protected:
  // This function makes a switch between building multi-level data or
  // single-level data.  Makes a call to either build_field or or
  // build_multi_level_field.  The basis_order pertains to whether the
  // data is node or cell centerd.
  //  template<class T>
  FieldHandle getData(QueryInfo& qinfo, IntVector& offset,
                      LVMeshHandle mesh_handle,
                      int remove_boundary, int basis_order);
  // Calls query for a single-level data set.
  template <class FIELD>
  void build_field(QueryInfo& qinfo, IntVector& offset,
                   FIELD* field, int remove_boundary);
  // This does the actuall work of getting the data from the
  // DataArchive for a single patch and filling the field.  This is
  // called by both build_field and build_patch_field.
  template <class FIELD>
  void getPatchData(QueryInfo& qinfo, IntVector& offset,
                    FIELD* sfield, const Patch* patch, int remove_boundary);

//   // Similar to build_field, but is called from build_multi_level_field.
  template <class FIELD>
  void build_patch_field(QueryInfo& qinfo,
                         const Patch* patch,
                         IntVector& offset,
                         FIELD* field,
                         int remove_boundary);
//   // Creates a MultiLevelField.
   FieldHandle build_multi_level_field( QueryInfo& qinfo, int basis_order,
                                        int remove_boundary);

  
};

template<class T>
FieldHandle
FieldExtractorAlgoT<T>::execute(QueryInfo& qinfo,
                                IntVector& offset,
                                LVMeshHandle mh,
                                int remove_boundary)
{
  FieldHandle f = 0;

  if( qinfo.type->getType() == Uintah::TypeDescription::CCVariable ){
    new2OldPatchMap_.clear();
    return getData(qinfo, offset, mh, remove_boundary, 0);
  } else {
    new2OldPatchMap_.clear();
    return getData(qinfo, offset, mh, remove_boundary, 1);
  }
  return f;
}


// This does the actuall work of getting the data from the
// DataArchive for a single patch and filling the field.  This is
//  called by both build_field and build_patch_field.
template <class T>
 template<class FIELD>
void
FieldExtractorAlgoT<T>::getPatchData(QueryInfo& qinfo, IntVector& offset,
                                     FIELD* sfield, const Patch* patch,
                                     int remove_boundary)
{
  IntVector patch_low, patch_high;
  GridVariable<T>* patch_data = dynamic_cast<GridVariable<T>*>(qinfo.type->createInstance());

  try {
    qinfo.archive->query(*patch_data, qinfo.varname, qinfo.mat, patch,
                         qinfo.timestep);
  } catch (Exception& e) {
//     error("query caused an exception: " + string(e.message()));
    cerr << "getPatchData::error in query function\n";
    cerr << e.message()<<"\n";
    return;
  }


  int vartype;
  if( remove_boundary == 1 ){
#if 0
      cerr<<"patch_data->getLowIndex() = "<<patch_data->getLowIndex()<<"\n";
      cerr<<"patch_data->getHighIndex() = "<<patch_data->getHighIndex()<<"\n";
      cerr<<"getCellLowIndex() = "<< patch->getCellLowIndex()<<"\n";
      cerr<<"getCellHighIndex() = "<< patch->getCellHighIndex()<<"\n";
      cerr<<"getInteriorCellLowIndex() = "<< patch->getInteriorCellLowIndex()<<"\n";
      cerr<<"getInteriorCellHighIndex() = "<< patch->getInteriorCellHighIndex()<<"\n";
      cerr<<"getNodeLowIndex() = "<< patch->getNodeLowIndex()<<"\n";
      cerr<<"getNodeHighIndex() = "<< patch->getNodeHighIndex()<<"\n";
      cerr<<"getInteriorNodeLowIndex() = "<< patch->getInteriorNodeLowIndex()<<"\n";
      cerr<<"getInteriorNodeHighIndex() = "<< patch->getInteriorNodeHighIndex()<<"\n\n";

      cerr<<"getHighIndex(Patch::XFaceBased, IntVector(0,0,0)) = "<<patch->getHighIndex(Patch::XFaceBased, IntVector(0,0,0))<<"\n";
      cerr<<"getHighIndex(Patch::YFaceBased, IntVector(0,0,0)) = "<<patch->getHighIndex(Patch::YFaceBased, IntVector(0,0,0))<<"\n";
      cerr<<"getHighIndex(Patch::ZFaceBased, IntVector(0,0,0)) = "<<patch->getHighIndex(Patch::ZFaceBased, IntVector(0,0,0))<<"\n\n";
      cerr<<"getInteriorHighIndex(Patch::XFaceBased) = "<<patch->getInteriorHighIndex(Patch::XFaceBased)<<"\n";
      cerr<<"getInteriorHighIndex(Patch::YFaceBased) = "<<patch->getInteriorHighIndex(Patch::YFaceBased)<<"\n";
      cerr<<"getInteriorHighIndex(Patch::ZFaceBased) = "<<patch->getInteriorHighIndex(Patch::ZFaceBased)<<"\n\n";
#endif
    if(sfield->basis_order() == 0){
      patch_low = patch->getInteriorCellLowIndex();
      patch_high = patch->getInteriorCellHighIndex();
    } else if(sfield->get_property("vartype", vartype)){
      patch_low = patch->getInteriorNodeLowIndex();
      switch (vartype) {
      case TypeDescription::SFCXVariable:
        patch_high = patch->getInteriorHighIndex(Patch::XFaceBased);
        break;
      case TypeDescription::SFCYVariable:
        patch_high = patch->getInteriorHighIndex(Patch::YFaceBased);
        break;
      case TypeDescription::SFCZVariable:
        patch_high = patch->getInteriorHighIndex(Patch::ZFaceBased);
        break;
      default:
        patch_high = patch->getInteriorNodeHighIndex();   
      } 
    } else {
//       error("getPatchData::Problem with getting vartype from field");
      return;
    }
    if( !patch_data->rewindow( patch_low, patch_high ) ) {
//       warning("patch data thinks it needs reallocation, this will fail.");
    }

  } else {
#if 0
      cerr<<"patch_data->getLowIndex() = "<<patch_data->getLowIndex()<<"\n";
      cerr<<"patch_data->getHighIndex() = "<<patch_data->getHighIndex()<<"\n";
      cerr<<"getCellLowIndex() = "<< patch->getCellLowIndex()<<"\n";
      cerr<<"getCellHighIndex() = "<< patch->getCellHighIndex()<<"\n";
      cerr<<"getInteriorCellLowIndex() = "<< patch->getInteriorCellLowIndex()<<"\n";
      cerr<<"getInteriorCellHighIndex() = "<< patch->getInteriorCellHighIndex()<<"\n";
      cerr<<"getNodeLowIndex() = "<< patch->getNodeLowIndex()<<"\n";
      cerr<<"getNodeHighIndex() = "<< patch->getNodeHighIndex()<<"\n";
      cerr<<"getInteriorNodeLowIndex() = "<< patch->getInteriorNodeLowIndex()<<"\n";
      cerr<<"getInteriorNodeHighIndex() = "<< patch->getInteriorNodeHighIndex()<<"\n";
      cerr<<"Patch low position is "<< patch->cellPosition(patch->getCellLowIndex())<<"\n";
      cerr<<"Patch high position is "<< patch->cellPosition(patch->getCellHighIndex())<<"\n";
      cerr<<"getHighIndex(Patch::XFaceBased, IntVector(0,0,0)) = "<<patch->getHighIndex(Patch::XFaceBased, IntVector(0,0,0))<<"\n";
      cerr<<"getHighIndex(Patch::YFaceBased, IntVector(0,0,0)) = "<<patch->getHighIndex(Patch::YFaceBased, IntVector(0,0,0))<<"\n";
      cerr<<"getHighIndex(Patch::ZFaceBased, IntVector(0,0,0)) = "<<patch->getHighIndex(Patch::ZFaceBased, IntVector(0,0,0))<<"\n\n";
      cerr<<"getInteriorHighIndex(Patch::XFaceBased) = "<<patch->getInteriorHighIndex(Patch::XFaceBased)<<"\n";
      cerr<<"getInteriorHighIndex(Patch::YFaceBased) = "<<patch->getInteriorHighIndex(Patch::YFaceBased)<<"\n";
      cerr<<"getInteriorHighIndex(Patch::ZFaceBased) = "<<patch->getInteriorHighIndex(Patch::ZFaceBased)<<"\n\n";

      BBox b;
      qinfo.grid->getInteriorSpatialRange(b);
      cerr<<"grid low position is "<<b.min() <<"\n";
      cerr<<"grid high position is "<< b.max()<<"\n\n";
#endif
    if( sfield->basis_order() == 0) {
      patch_low = patch->getCellLowIndex();
      patch_high = patch->getCellHighIndex();
    } else if(sfield->get_property("vartype", vartype)){
      patch_low = patch->getNodeLowIndex();
      switch (vartype) {
      case TypeDescription::SFCXVariable:
        patch_high = patch->getHighIndex(Patch::XFaceBased,IntVector(0,0,0));
        break;
      case TypeDescription::SFCYVariable:
        patch_high = patch->getHighIndex(Patch::YFaceBased,IntVector(0,0,0));
        break;
      case TypeDescription::SFCZVariable:
        patch_high = patch->getHighIndex(Patch::ZFaceBased,IntVector(0,0,0));
        break;
      default:
        patch_high = patch->getNodeHighIndex();   
      } 
    } else {
//       error("getPatchData::Problem with getting vartype from field");
      return;
    }
  }

#if 0
  LVMesh* lm = sfield->get_typed_mesh().get_rep();

  cerr<<"patch_low = "<<patch_low<<", patch_high = "<<patch_high<<"\n";
  cerr<<"mesh size is "<<lm->get_ni()<<"x"<<lm->get_nj()
      <<"x"<<lm->get_nk()<<"\n";
  cerr<<"offset = "<<offset<<"\n";
#endif
  PatchToFieldThread<T, FIELD> *ptft =
    scinew PatchToFieldThread<T, FIELD>(sfield, *patch_data, offset,
                                      patch_low, patch_high);
  ptft->run();
  delete ptft;
}

// // Similar to build_field, but is called from build_multi_level_field.
template <class T>
   template<class FIELD>
void
FieldExtractorAlgoT<T>::build_patch_field(QueryInfo& qinfo,
                                          const Patch* patch,
                                          IntVector& offset,
                                          FIELD* field,
                                          int remove_boundary)
{
  // Initialize the data
  field->fdata().initialize(T(0));

  map<const Patch*, list<const Patch*> >::iterator oldPatch_it =
    new2OldPatchMap_.find(patch);
  if( oldPatch_it == new2OldPatchMap_.end() ) {
    //  error("No mapping from old patches to new patches.");
    cerr<<"No mapping from old patches to new patches.\n";
    return;
  }
    
  list<const Patch*> oldPatches = (*oldPatch_it).second;
  for(list<const Patch*>::iterator patch_it = oldPatches.begin();
      patch_it != oldPatches.end(); ++patch_it){
    getPatchData(qinfo, offset, field, *patch_it, remove_boundary);
  }
}

// Calls query for a single-level data set.
template <class T>
 template<class FIELD>
void
FieldExtractorAlgoT<T>::build_field(QueryInfo& qinfo, IntVector& offset,
                                    FIELD* field, int remove_boundary)
{

  // Initialize the data
  field->fdata().initialize(T(0));


  //  WallClockTimer my_timer;
  //  my_timer.start();
  for( Level::const_patchIterator patch_it = qinfo.level->patchesBegin();
       patch_it != qinfo.level->patchesEnd(); ++patch_it){
    getPatchData(qinfo, offset, field, *patch_it, remove_boundary);
  //      update_progress(somepercentage, my_timer);
  }
  cerr<<"\n";
  //  timer.add( my_timer.time());
  //  my_timer.stop();
}


// // Creates an MultiLevelField.
template <class T>
FieldHandle
FieldExtractorAlgoT<T>::build_multi_level_field( QueryInfo& qinfo, 
                                                 int basis_order,
                                                 int remove_boundary)
{
    typedef GenericField<LVMesh, ConstantBasis<T>, 
                         FData3d<T, LVMesh> > LVFieldCB;
    typedef GenericField<LVMesh, HexTrilinearLgn<T>, 
                         FData3d<T, LVMesh> > LVFieldLB;

  // Build the minimal patch set.  build_minimal_patch_grid should
  // eventually return the map rather than have it as a member
  // variable to map with all the other parameters that aren't being
  // used by the class.
  IntVector offset;
  GridP grid_minimal = build_minimal_patch_grid( qinfo.grid );
  
  if(basis_order == 0){
    vector<MultiLevelFieldLevel<LVFieldCB>*> levelfields;
    for(int i = 0; i < grid_minimal->numLevels(); i++){
      LevelP level = grid_minimal->getLevel( i );
      vector<LockingHandle< LVFieldCB > > patchfields;
    
      // At this point we should have a mimimal patch set in our
      // grid_minimal, and we want to make a LatVolField for each patch.
      for(Level::const_patchIterator patch_it = level->patchesBegin();
          patch_it != level->patchesEnd(); ++patch_it){
      
        
        IntVector patch_low, patch_high, range;
        BBox pbox;
        if( remove_boundary ==1 ){
          patch_low = (*patch_it)->getInteriorNodeLowIndex();
          patch_high = (*patch_it)->getInteriorNodeHighIndex(); 
          pbox.extend((*patch_it)->getInteriorBox().lower());
          pbox.extend((*patch_it)->getInteriorBox().upper());
        } else {
          patch_low = (*patch_it)->getLowIndex();
          patch_high = (*patch_it)->getHighIndex(); 
          pbox.extend((*patch_it)->getBox().lower());
          pbox.extend((*patch_it)->getBox().upper());
        }
        // ***** This seems like a hack *****
        range = patch_high - patch_low + IntVector(1,1,1); 
        // **********************************


      
//         cerr<<"before mesh update: range is "<<range.x()<<"x"<<
//           range.y()<<"x"<< range.z()<<",  low index is "<<patch_low<<
//           "high index is "<<patch_high<<" , size is  "<<
//           pbox.min()<<", "<<pbox.max()<<" for Patch address "<<
//           (*patch_it)<<"\n";
      
        LVMeshHandle mh = 0;
        FieldExtractor::update_mesh_handle(qinfo.level, patch_high, 
                                           range, pbox, qinfo.type->getType(), 
                                           mh, remove_boundary);
        LVFieldCB *field = scinew LVFieldCB( mh );
        qinfo.level = level;
        set_field_properties(field, qinfo, patch_low);

        build_patch_field(qinfo, (*patch_it), patch_low,
                                  field, remove_boundary);
        patchfields.push_back( field );
      }
      MultiLevelFieldLevel<LVFieldCB> *mrlevel = scinew MultiLevelFieldLevel<LVFieldCB>( patchfields, i );
      levelfields.push_back(mrlevel);
    }
    return scinew MultiLevelField<LVMesh, ConstantBasis<T>, 
                                FData3d<T, LVMesh> >( levelfields );
  } else {
    vector<MultiLevelFieldLevel<LVFieldLB>*> levelfields;
    for(int i = 0; i < grid_minimal->numLevels(); i++){
      LevelP level = grid_minimal->getLevel( i );
      vector<LockingHandle< LVFieldLB > > patchfields;
    
      // At this point we should have a mimimal patch set in our
      // grid_minimal, and we want to make a LatVolField for each patch.
      for(Level::const_patchIterator patch_it = level->patchesBegin();
          patch_it != level->patchesEnd(); ++patch_it){
      
        IntVector patch_low, patch_high, range;
        BBox pbox;
        if( remove_boundary ==1 ){
          patch_low = (*patch_it)->getInteriorNodeLowIndex();
          patch_high = (*patch_it)->getInteriorNodeHighIndex(); 
          pbox.extend((*patch_it)->getInteriorBox().lower());
          pbox.extend((*patch_it)->getInteriorBox().upper());
        } else {
          patch_low = (*patch_it)->getLowIndex();
          patch_high = (*patch_it)->getHighIndex(); 
          pbox.extend((*patch_it)->getBox().lower());
          pbox.extend((*patch_it)->getBox().upper());
        }
        // ***** This seems like a hack *****
        range = patch_high - patch_low + IntVector(1,1,1); 
        // **********************************


      
        //      cerr<<"before mesh update: range is "<<range.x()<<"x"<<
        //      range.y()<<"x"<< range.z()<<",  low index is "<<patch_low<<
        //      "high index is "<<patch_high<<" , size is  "<<
        //      pbox.min()<<", "<<pbox.max()<<"\n";
      
        LVMeshHandle mh = 0;
        FieldExtractor::update_mesh_handle(qinfo.level, patch_high, 
                                           range, pbox, qinfo.type->getType(),
                                           mh, remove_boundary);
        LVFieldLB *field = scinew LVFieldLB( mh );
        qinfo.level = level;
        set_field_properties(field, qinfo, patch_low);

        build_patch_field(qinfo, (*patch_it), patch_low,
                                  field, remove_boundary );
        patchfields.push_back( field );
      }
      MultiLevelFieldLevel<LVFieldLB> *mrlevel = scinew MultiLevelFieldLevel<LVFieldLB>( patchfields, i );
      levelfields.push_back(mrlevel);
    }
    return scinew MultiLevelField<LVMesh, HexTrilinearLgn<T>, 
                                FData3d<T, LVMesh> >( levelfields );
  }

}


// This function makes a switch between building multi-level data or
// single-level data.  Makes a call to either build_field or or
// build_multi_level_field.  The basis_order pertains to whether the
// data is node or cell centerd.  The viariable this is called on should look something
// like CCVariable<T> or NCVariable<T>.
template<class T>
FieldHandle
FieldExtractorAlgoT<T>::getData(QueryInfo& qinfo, IntVector& offset,
                                LVMeshHandle mesh_handle,
                                int remove_boundary, int basis_order)
{
  if (qinfo.get_all_levels) {
    FieldHandle fh = build_multi_level_field(qinfo, 
                                             basis_order, remove_boundary);
    set_field_properties( fh.get_rep(), qinfo, offset);
    fh->set_property( "level", string("all"), false);
    return fh;
  } else {
    typedef GenericField<LVMesh, ConstantBasis<T>, 
                         FData3d<T, LVMesh> > LVFieldCB;
    typedef GenericField<LVMesh, HexTrilinearLgn<T>, 
                         FData3d<T, LVMesh> > LVFieldLB;

    if( basis_order == 0 ){
      LVFieldCB* sf = scinew LVFieldCB(mesh_handle);
      set_field_properties(sf, qinfo, offset);
      build_field(qinfo, offset, sf, remove_boundary);
      return sf;
    } else {
      LVFieldLB* sf = scinew LVFieldLB(mesh_handle);
      set_field_properties(sf, qinfo, offset);
      build_field(qinfo, offset, sf, remove_boundary);
      return sf;
    }
  }
}


} // End namespace Uintah



#endif
