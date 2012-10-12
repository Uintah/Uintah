/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <Core/Math/Matrix3.h> // Must be before include of Constant.h
#include <Core/Basis/Constant.h>               // Must be before include of HexTrilinearLgn.h
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/MultiLevelField.h>

#include <StandAlone/tools/uda2nrrd/build.h>

#include <StandAlone/tools/uda2nrrd/handleVariable.h>
#include <StandAlone/tools/uda2nrrd/update_mesh_handle.h>

#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/LocallyComputedPatchVarMap.h>

#include <map>

using namespace std;
using namespace Uintah;

/////////////////////////////////////////////////////////////////////
// This will gather all the patch data and write it into the single
// contiguous volume.

template <class T, class VarT, class FIELD>
NO_INLINE
void
build_field( QueryInfo &qinfo,
             IntVector& offset,
             T& /* data_type */,
             VarT& /*var*/,
             FIELD *sfield,
             const Args & args )
{
  // TODO: Bigler
  // Not sure I need this yet
  sfield->fdata().initialize(T(0));

  // Loop over each patch and get the data from the data archive.
  for( Level::const_patchIterator patch_it = qinfo.level->patchesBegin();
       patch_it != qinfo.level->patchesEnd(); ++patch_it){
    const Patch* patch = *patch_it;
    // This gets the data
    handlePatchData<T, VarT, FIELD>(qinfo, offset, sfield, patch, args);
  }
}

/////////////////////////////////////////////////////////////////////

map<const Patch*, list<const Patch*> > new2OldPatchMap_;

GridP 
build_minimal_patch_grid( GridP oldGrid )
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
        inLow = Min(inLow, patch->getCellLowIndex());
        inHigh = Max(inHigh, patch->getCellHighIndex());
      }
      
      Patch* newPatch =
        newLevel->addPatch(low, high, inLow, inHigh, newGrid.get_rep());
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
// // Similar to build_field, but is called from build_multi_level_field.

/////////////////////////////////////////////////////////////////////

template<class T, class VarT, class FIELD>
void
build_patch_field(QueryInfo   & qinfo,
                  const Patch * patch,
                  IntVector   & offset,
                  FIELD       * field,
                  const Args  & args )
{
  // Initialize the data
  field->fdata().initialize((typename FIELD::value_type)(0));

  map<const Patch*, list<const Patch*> >::iterator oldPatch_it =
    new2OldPatchMap_.find(patch);
  if( oldPatch_it == new2OldPatchMap_.end() ) {
    //  error("No mapping from old patches to new patches.");
    cerr<<"No mapping from old patches to new patches.\n";
    return;
  }
    
  list<const Patch*> oldPatches = (*oldPatch_it).second;
  for(list<const Patch*>::iterator patch_it = oldPatches.begin(); patch_it != oldPatches.end(); ++patch_it)
    {
      handlePatchData<T, VarT, FIELD>(qinfo, offset, field, *patch_it, args );
    }
}

/////////////////////////////////////////////////////////////////////

template <class T, class VarT, class FIELD>
FieldHandle
build_multi_level_field( QueryInfo &qinfo, const Args & args )
{
  IntVector offset;
  if( args.verbose ) cout<<"Building minimal patch grid.\n";
  GridP grid_minimal = build_minimal_patch_grid( qinfo.grid );
  if( args.verbose ) cout<<"Minimal patch grid built.\n";

  vector<MultiLevelFieldLevel< FIELD >*> levelfields;

    for(int i = 0; i < grid_minimal->numLevels(); i++){
      LevelP level = grid_minimal->getLevel( i );
      vector<LockingHandle< FIELD > > patchfields;
    
      // At this point we should have a mimimal patch set in our
      // grid_minimal, and we want to make a LatVolField for each patch.
      for(Level::const_patchIterator patch_it = level->patchesBegin();
          patch_it != level->patchesEnd(); ++patch_it){
      
        
        IntVector patch_low, patch_high, range;
        BBox pbox;
        if( args.remove_boundary ){
          patch_low = (*patch_it)->getNodeLowIndex();
          patch_high = (*patch_it)->getNodeHighIndex(); 
          pbox.extend((*patch_it)->getBox().lower());
          pbox.extend((*patch_it)->getBox().upper());
        } else {
          patch_low = (*patch_it)->getExtraCellLowIndex();
          patch_high = (*patch_it)->getExtraCellHighIndex(); 
          pbox.extend((*patch_it)->getExtraBox().lower());
          pbox.extend((*patch_it)->getExtraBox().upper());
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
        qinfo.level = level;
        update_mesh_handle(qinfo.level, patch_high, 
                           range, pbox, qinfo.type->getType(), mh, args ); 

        FIELD *field = scinew FIELD( mh );
//         set_field_properties(field, qinfo, patch_low);

        build_patch_field<T, VarT, FIELD>( qinfo, (*patch_it), patch_low, field, args );

        patchfields.push_back( field );
      }
      MultiLevelFieldLevel<FIELD> *mrlevel = 
        scinew MultiLevelFieldLevel<FIELD>( patchfields, i );
      levelfields.push_back(mrlevel);
    }
    return scinew MultiLevelField<typename FIELD::mesh_type, 
                                  typename FIELD::basis_type,
                                  typename FIELD::fdata_type>(levelfields);
}


/////////////////////////////////////////////////////////////////////

template <class T, class VarT, class FIELD, class FLOC>
NO_INLINE
void
build_combined_level_field( QueryInfo &qinfo,
                            IntVector& offset,
                            FIELD *sfield,
                            const Args & args )
{
 // TODO: Bigler
  // Not sure I need this yet
  sfield->fdata().initialize(T(0));

  if( args.verbose ) cout<<"Building Multi-level Field\n";

  FieldHandle fh =  build_multi_level_field<T, VarT, FIELD>( qinfo, args );
  if( args.verbose ) cout<<"Multi-level Field is built\n";

  typedef MultiLevelField<typename FIELD::mesh_type, 
                          typename FIELD::basis_type,
                          typename FIELD::fdata_type> MLField;
  typedef GenericField<typename FIELD::mesh_type, 
                       typename FIELD::basis_type,
                       typename FIELD::fdata_type>  GF;

  MLField *mlfield = (MLField*) fh.get_rep();;

  // print out some field info.

  BBox box;
  //  int nx, ny, nz;
  typename FIELD::mesh_handle_type dst_mh = sfield->get_typed_mesh();
  box = dst_mh->get_bounding_box();
  if( args.verbose ) {
    cout<<"The output field is has grid dimensions of "
        << dst_mh->get_ni() <<"x"<<dst_mh->get_nj()<<"x"<<dst_mh->get_nk()
        <<" and a geometric range from "<< box.min() <<" to "<<box.max()<<"\n";
  }

  typename FIELD::mesh_handle_type src_mh;
  if( args.verbose ) cout<<"The input data has "<<mlfield->nlevels()<<" levels:\n";
  for(int i = 0; i < mlfield->nlevels(); i++){
    const MultiLevelFieldLevel<GF> *lev = mlfield->level(i);
    if( args.verbose ) cout<<"\tLevel "<<i<<" has " <<lev->patches.size()<<" fields:\n";
    for(unsigned int j = 0; j < lev->patches.size(); j++ ){
      src_mh = lev->patches[j]->get_typed_mesh();
      box = src_mh->get_bounding_box();
      if( args.verbose ) {
        cout<<"\t\tField "<<j<<" has dimesions "
            << src_mh->get_ni() <<"x"<<src_mh->get_nj()
            <<"x"<<src_mh->get_nk()
            <<" and a geometric range from "
            << box.min() <<" to "<<box.max()<<"\n";
      }
    }
  }

  // first Map level 0 src field data to the dst field
  for(int i = 0; i < mlfield->nlevels(); i++){
    const MultiLevelFieldLevel<GF> *lev = mlfield->level(i);
    typename FIELD::handle_type src_fh;
    for(unsigned int j = 0; j < lev->patches.size(); j++ ){
      src_fh = lev->patches[j];
      src_mh = src_fh->get_typed_mesh();

      if( !args.quiet ) cerr<<"Filling destination field with level "<<i<<" data ";
      int count = 0;


      typename FLOC::iterator itr, end_itr;
      dst_mh->begin(itr);
      dst_mh->end(end_itr);
      while (itr != end_itr) {
        typename FLOC::array_type locs;
        double weights[MESH_WEIGHT_MAXSIZE];
        Point p;
        dst_mh->get_center(p, *itr);
        bool failed = true;
        const int nw = src_mh->get_weights(p, locs, weights);
        typename FIELD::value_type val(0);
    
        if (nw > 0)	{
          failed = false;
          if (locs.size())
            val = (typename FIELD::value_type)(src_fh->value(locs[0])*weights[0]);
          for (unsigned int k = 1; k < locs.size(); k++) {
            val +=(typename FIELD::value_type)(src_fh->value(locs[k])*weights[k]);
          }
        }
        if (!failed) sfield->set_value(val, *itr);
        ++itr;
        if( !args.quiet ){
          if( count == 100000 ) {
            cerr<<"."; count = 0;
          } else { ++count; }
        }
      }
      if( !args.quiet ) cerr<<" done.\n";
    }
  }
} // end build_combined_level_field()

///////////////////////////////////////////////////////////////////////////////
// Instantiate some of the needed verisons of functions.  The
// following functions are never actually called, but force the
// instantiation of the build* functions that are needed.

template <class T>
void
instHelper()
{
  typedef GenericField<LVMesh, ConstantBasis<T>, FData3d<T, LVMesh> >   LVFieldCB;
  typedef GenericField<LVMesh, HexTrilinearLgn<T>, FData3d<T, LVMesh> > LVFieldLB;
  typedef typename LVFieldCB::mesh_type::Cell FLOC;
  typedef typename LVFieldCB::mesh_type::Node FLON;

  QueryInfo   * qinfo = NULL;
  IntVector     low;
  LVFieldCB   * sf = NULL;
  LVFieldLB   * flb = NULL;
  const Args  * args = NULL;

  SFCXVariable<T> * sfcxv = NULL;
  SFCYVariable<T> * sfcyv = NULL;
  SFCZVariable<T> * sfczv = NULL;
  CCVariable<T>   * ccv = NULL;
  NCVariable<T>   * ncv = NULL;
  T * aT = NULL;

  build_combined_level_field<T, SFCXVariable<T>, LVFieldCB, FLOC>( *qinfo, low, sf, *args );
  build_combined_level_field<T, SFCYVariable<T>, LVFieldCB, FLOC>( *qinfo, low, sf, *args );
  build_combined_level_field<T, SFCZVariable<T>, LVFieldCB, FLOC>( *qinfo, low, sf, *args );
  build_combined_level_field<T, CCVariable<T>,   LVFieldCB, FLOC>( *qinfo, low, sf, *args );
  build_combined_level_field<T, NCVariable<T>,   LVFieldCB, FLOC>( *qinfo, low, sf, *args );

  build_combined_level_field<T, SFCXVariable<T>, LVFieldCB, FLON>( *qinfo, low, sf, *args );
  build_combined_level_field<T, SFCYVariable<T>, LVFieldCB, FLON>( *qinfo, low, sf, *args );
  build_combined_level_field<T, SFCZVariable<T>, LVFieldCB, FLON>( *qinfo, low, sf, *args );
  build_combined_level_field<T, CCVariable<T>,   LVFieldCB, FLON>( *qinfo, low, sf, *args );
  build_combined_level_field<T, NCVariable<T>,   LVFieldCB, FLON>( *qinfo, low, sf, *args );

  build_combined_level_field<T, SFCXVariable<T>, LVFieldLB, FLOC>( *qinfo, low, flb, *args );
  build_combined_level_field<T, SFCYVariable<T>, LVFieldLB, FLOC>( *qinfo, low, flb, *args );
  build_combined_level_field<T, SFCZVariable<T>, LVFieldLB, FLOC>( *qinfo, low, flb, *args );
  build_combined_level_field<T, CCVariable<T>,   LVFieldLB, FLOC>( *qinfo, low, flb, *args );
  build_combined_level_field<T, NCVariable<T>,   LVFieldLB, FLOC>( *qinfo, low, flb, *args );

  build_combined_level_field<T, SFCXVariable<T>, LVFieldLB, FLON>( *qinfo, low, flb, *args );
  build_combined_level_field<T, SFCYVariable<T>, LVFieldLB, FLON>( *qinfo, low, flb, *args );
  build_combined_level_field<T, SFCZVariable<T>, LVFieldLB, FLON>( *qinfo, low, flb, *args );
  build_combined_level_field<T, CCVariable<T>,   LVFieldLB, FLON>( *qinfo, low, flb, *args );
  build_combined_level_field<T, NCVariable<T>,   LVFieldLB, FLON>( *qinfo, low, flb, *args );

  build_field( *qinfo, low, *aT, *sfcxv, sf, *args );
  build_field( *qinfo, low, *aT, *sfcyv, sf, *args );
  build_field( *qinfo, low, *aT, *sfczv, sf, *args );
  build_field( *qinfo, low, *aT, *ccv,   sf, *args );
  build_field( *qinfo, low, *aT, *ncv,   sf, *args );

  build_field( *qinfo, low, *aT, *sfcxv, flb, *args );
  build_field( *qinfo, low, *aT, *sfcyv, flb, *args );
  build_field( *qinfo, low, *aT, *sfczv, flb, *args );
  build_field( *qinfo, low, *aT, *ccv,   flb, *args );
  build_field( *qinfo, low, *aT, *ncv,   flb, *args );
}

void
templateInstantiationForBuildCC()
{
  instHelper<Vector>();
  instHelper<double>();
  instHelper<int>();
  instHelper<float>();
  instHelper<Matrix3>();
}
