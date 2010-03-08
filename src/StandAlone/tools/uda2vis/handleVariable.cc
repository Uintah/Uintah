/*

   The MIT License

   Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
   Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a 
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation 
   the rights to use, copy, modify, merge, publish, distribute, sublicense, 
   and/or sell copies of the Software, and to permit persons to whom the 
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included 
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
   DEALINGS IN THE SOFTWARE.

 */

/////////////////
// Due to template instantiation ordering problems, these 2 includes must be first:
#include <Core/Math/Matrix3.h>
#include <Core/Basis/Constant.h> 
// End template needed .h files
/////////////////

#include <StandAlone/tools/uda2vis/handleVariable.h>

#include <StandAlone/tools/uda2vis/Args.h>
#include <StandAlone/tools/uda2vis/build.h>
#include <StandAlone/tools/uda2vis/wrap_nrrd.h>

#include <StandAlone/tools/uda2nrrd/PatchToField.h>

#include <Core/Geometry/Vector.h>

#include <Core/Containers/FData.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/GenericField.h>

#include <Core/Util/FileUtils.h>

#include <teem/nrrd.h>

///////////////////////////////////////////////////////////////////////////////

typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh;
typedef LVMesh::handle_type LVMeshHandle;

///////////////////////////////////////////////////////////////////////////////


template <class T, class VarT, class FIELD>
void
handlePatchData( QueryInfo& qinfo, IntVector& offset,
                 FIELD* sfield, const Patch* patch,
                 const Args & args )
{
  if( qinfo.materials.size() != 1 ) {
    cout << "ERROR: handlePatchData: number of materials should be one, but it was " 
         << qinfo.materials.size() << "\n";
    return;
  }

  IntVector patch_low, patch_high;
  VarT patch_data;

  IntVector extraCells = patch->getExtraCells();
  
  // necessary check - useful with periodic boundaries
  for (int i = 0; i < 3; i++) {
    if (extraCells(i) == 0) {
      extraCells(i) = 1;
    }
  }

  IntVector noCells = patch->getCellHighIndex() - patch->getCellLowIndex();
     
  static IntVector hi, lo;
  static int currLevel = -1;

  if (currLevel != qinfo.level->getIndex()) {
    currLevel = qinfo.level->getIndex();
    qinfo.level->findNodeIndexRange(lo, hi);
  }

  // remove boundary
  if ( args.remove_boundary ) {
    if(sfield->basis_order() == 0){
      patch_low = patch->getCellLowIndex();
      patch_high = patch->getCellHighIndex();
    } else {
      patch_low = patch->getNodeLowIndex();
      switch (qinfo.type->getType()) {
      case Uintah::TypeDescription::SFCXVariable:
        patch_high = patch->getHighIndex(Patch::XFaceBased);
        break;
      case Uintah::TypeDescription::SFCYVariable:
        patch_high = patch->getHighIndex(Patch::YFaceBased);
        break;
      case Uintah::TypeDescription::SFCZVariable:
        patch_high = patch->getHighIndex(Patch::ZFaceBased);
        break;
      case Uintah::TypeDescription::NCVariable:
        patch_high = patch->getNodeHighIndex();
        break;
      default:
        cerr << "build_field::unknown variable.\n";
        exit(1);
      } 
    }
  }

  // Don't remove the boundary
  else {
    if(sfield->basis_order() == 0){
      patch_low = patch->getCellLowIndex()   - extraCells;
      patch_high = patch->getCellHighIndex() + extraCells;
    } else {
      patch_low = patch->getNodeLowIndex() - extraCells;
      switch (qinfo.type->getType()) {
      case Uintah::TypeDescription::SFCXVariable:
        patch_high = patch_low + noCells;
        patch_high = IntVector(patch_high.x() + 1, patch_high.y(), patch_high.z());
        patch_high = patch_high + extraCells;
        break;
      case Uintah::TypeDescription::SFCYVariable:
        patch_high = patch->getSFCYHighIndex();
        break;
      case Uintah::TypeDescription::SFCZVariable:
        patch_high = patch->getSFCZHighIndex();
        break;
      case Uintah::TypeDescription::NCVariable:
        patch_high = patch->getNodeLowIndex() + noCells + extraCells + IntVector(1, 1, 1);
        break;
      default:
        cerr << "build_field::unknown variable.\n";
        exit(1);
      }
    }
  }

  // necessary check - useful with periodic boundaries
  for (int i = 0; i < 3; i++) {
    if (patch_high(i) > hi(i)) {
      patch_high(i) = hi(i);
    }

    if (patch_low(i) < lo(i)) {
      patch_low(i) = lo(i);
    }
  }

  try {
    int material = *qinfo.materials.begin();
    qinfo.archive->queryRegion(patch_data, qinfo.varname, material, patch->getLevel(), qinfo.timestep, patch_low, patch_high);
  } catch (Exception& e) {
    cerr << "handlePatchData::error in query function\n";
    cerr << e.message()<<"\n";
    return;
  }

  // Rewindow the data if we need only a subset.  This should never
  // get bigger (thus requiring reallocation).
  patch_data.rewindow( patch_low, patch_high );

  PatchToFieldThread<T, FIELD> *worker = 
    scinew PatchToFieldThread<T, FIELD>(sfield, &patch_data, offset,
                                        patch_low, patch_high);
  worker->run();
  delete worker;
}


///////////////////////////////////////////////////////////////////////////////


template<class VarT, class T>
void
handleData( QueryInfo &    qinfo,
            IntVector &    low,
            LVMeshHandle   mesh_handle,
            int            basis_order,
            const Args   & args,
            cellVals& cellValColln,
            bool dataReq, 
            int patchNo )
{

  VarT gridVar;
  T data_T;

  // set the generation and timestep in the field
  if( !args.quiet ) cout << "Building Field from uda data\n";

  // Print out the physical extents
  BBox bbox = mesh_handle->get_bounding_box();
  if( !args.quiet ) cout << "Bounding box: min("<<bbox.min()<<"), max("<<bbox.max()<<")\n";

    
  if( basis_order == 0 ){
    typedef GenericField<LVMesh, ConstantBasis<T>, FData3d<T, LVMesh> > LVFieldCB;
    typedef typename LVFieldCB::mesh_type::Cell FLOC;

    LVFieldCB* sf = scinew LVFieldCB(mesh_handle);
    if (!sf) {
      cerr << "Cannot allocate memory for field\n";
      return;
    }

    if(qinfo.combine_levels){
      // this will only be called for all levels, combined
      build_combined_level_field<T, VarT, LVFieldCB, FLOC>( qinfo, low, sf, args );
    } else {
      build_field( qinfo, low, data_T, gridVar, sf, args, patchNo );
    }

    // Convert the field to a nrrd
    wrap_nrrd( sf, args.matrix_op, args.verbose, cellValColln, dataReq );
    delete sf;
  }

  else {
    typedef GenericField<LVMesh, HexTrilinearLgn<T>, FData3d<T, LVMesh> > LVFieldLB;
    typedef typename LVFieldLB::mesh_type::Node FLOC;

    LVFieldLB* sf = scinew LVFieldLB(mesh_handle);
    if (!sf) {
      cerr << "Cannot allocate memory for field\n";
      return;
    }

    if(qinfo.combine_levels){
      // this will only be called for all levels, combined
      build_combined_level_field<T, VarT, LVFieldLB, FLOC>( qinfo, low, sf, args );
    } else {
      build_field( qinfo, low, data_T, gridVar, sf, args, patchNo );
    }

    // Convert the field to a nrrd
    wrap_nrrd( sf, args.matrix_op, args.verbose, cellValColln, dataReq );
    delete sf;
  }

  return;
}


///////////////////////////////////////////////////////////////////////////////

template<class T>
void
handleVariable( QueryInfo &qinfo, IntVector &low, IntVector& hi,
                IntVector &range, BBox &box,
                const Args & args,
                cellVals& cellVallColln,
                bool dataReq,
                int patchNo )
{
  LVMeshHandle mesh_handle;
  switch( qinfo.type->getType() ) {
  case Uintah::TypeDescription::CCVariable:
    mesh_handle = scinew LVMesh(range.x(), range.y(), range.z(),
                                box.min(), box.max());
    handleData<CCVariable<T>, T>( qinfo, low, mesh_handle, 0, args, cellVallColln, dataReq, patchNo );
    break;
  case Uintah::TypeDescription::NCVariable:
    mesh_handle = scinew LVMesh(range.x(), range.y(), range.z(),
                                box.min(), box.max());
    handleData<NCVariable<T>, T>( qinfo, low, mesh_handle, 1, args, cellVallColln, dataReq, patchNo );
    break;
  case Uintah::TypeDescription::SFCXVariable:
    mesh_handle = scinew LVMesh(range.x(), range.y()-1, range.z()-1,
                                box.min(), box.max());
    handleData<SFCXVariable<T>, T>( qinfo, low, mesh_handle, 1, args, cellVallColln, dataReq, patchNo );
    break;
  case Uintah::TypeDescription::SFCYVariable:
    mesh_handle = scinew LVMesh(range.x()-1, range.y(), range.z()-1,
                                box.min(), box.max());
    handleData<SFCYVariable<T>, T>( qinfo, low, mesh_handle, 1, args, cellVallColln, dataReq, patchNo );
    break;
  case Uintah::TypeDescription::SFCZVariable:
    mesh_handle = scinew LVMesh(range.x()-1, range.y()-1, range.z(),
                                box.min(), box.max());
    handleData<SFCZVariable<T>, T>( qinfo, low, mesh_handle, 1, args, cellVallColln, dataReq, patchNo );
    break;
  default:
    cerr << "Type is unknown.\n";
    return;
    break;

  }
}


///////////////////////////////////////////////////////////////////////////////
// Instantiate some of the needed verisons of functions.

#define INTANTIATE_TEMPLATES_HANDLEVARIABLE_CC(T)                       \
template void handlePatchData<T, SFCXVariable<T>, GenericField<LVMesh, ConstantBasis<T>, FData3d<T, LVMesh> > > (QueryInfo&, IntVector&, GenericField<LVMesh, ConstantBasis<T>, FData3d<T, LVMesh> >*, const Patch*, const Args&); \
template void handlePatchData<T, SFCYVariable<T>, GenericField<LVMesh, ConstantBasis<T>, FData3d<T, LVMesh> > > (QueryInfo&, IntVector&, GenericField<LVMesh, ConstantBasis<T>, FData3d<T, LVMesh> >*, const Patch*, const Args&); \
template void handlePatchData<T, SFCZVariable<T>, GenericField<LVMesh, ConstantBasis<T>, FData3d<T, LVMesh> > > (QueryInfo&, IntVector&, GenericField<LVMesh, ConstantBasis<T>, FData3d<T, LVMesh> >*, const Patch*, const Args&); \
template void handlePatchData<T, CCVariable<T>,   GenericField<LVMesh, ConstantBasis<T>, FData3d<T, LVMesh> > > (QueryInfo&, IntVector&, GenericField<LVMesh, ConstantBasis<T>, FData3d<T, LVMesh> >*, const Patch*, const Args&); \
template void handlePatchData<T, NCVariable<T>,   GenericField<LVMesh, ConstantBasis<T>, FData3d<T, LVMesh> > > (QueryInfo&, IntVector&, GenericField<LVMesh, ConstantBasis<T>, FData3d<T, LVMesh> >*, const Patch*, const Args&); \
                                                                      \
template void handlePatchData<T, SFCXVariable<T>, GenericField<LVMesh, HexTrilinearLgn<T>, FData3d<T, LVMesh> > > (QueryInfo&, IntVector&, GenericField<LVMesh, HexTrilinearLgn<T>, FData3d<T, LVMesh> >*, const Patch*, const Args&); \
template void handlePatchData<T, SFCYVariable<T>, GenericField<LVMesh, HexTrilinearLgn<T>, FData3d<T, LVMesh> > > (QueryInfo&, IntVector&, GenericField<LVMesh, HexTrilinearLgn<T>, FData3d<T, LVMesh> >*, const Patch*, const Args&); \
template void handlePatchData<T, SFCZVariable<T>, GenericField<LVMesh, HexTrilinearLgn<T>, FData3d<T, LVMesh> > > (QueryInfo&, IntVector&, GenericField<LVMesh, HexTrilinearLgn<T>, FData3d<T, LVMesh> >*, const Patch*, const Args&); \
template void handlePatchData<T, CCVariable<T>,   GenericField<LVMesh, HexTrilinearLgn<T>, FData3d<T, LVMesh> > > (QueryInfo&, IntVector&, GenericField<LVMesh, HexTrilinearLgn<T>, FData3d<T, LVMesh> >*, const Patch*, const Args&); \
template void handlePatchData<T, NCVariable<T>,   GenericField<LVMesh, HexTrilinearLgn<T>, FData3d<T, LVMesh> > > (QueryInfo&, IntVector&, GenericField<LVMesh, HexTrilinearLgn<T>, FData3d<T, LVMesh> >*, const Patch*, const Args&); \
                                                                      \
template void handleVariable<T>(QueryInfo&, IntVector&, IntVector&, IntVector&, BBox&, const Args&, cellVals&, bool, int);

INTANTIATE_TEMPLATES_HANDLEVARIABLE_CC(Vector)
INTANTIATE_TEMPLATES_HANDLEVARIABLE_CC(int)
INTANTIATE_TEMPLATES_HANDLEVARIABLE_CC(float)
INTANTIATE_TEMPLATES_HANDLEVARIABLE_CC(double)
INTANTIATE_TEMPLATES_HANDLEVARIABLE_CC(Matrix3)
