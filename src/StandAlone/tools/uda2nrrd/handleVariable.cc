/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
/////////////////
// Due to template instantiation ordering problems, these 2 includes must be first:
#include <Core/Math/Matrix3.h>
#include <Core/Basis/Constant.h> 
// End template needed .h files
/////////////////

#include <StandAlone/tools/uda2nrrd/handleVariable.h>

#include <StandAlone/tools/uda2nrrd/Args.h>
#include <StandAlone/tools/uda2nrrd/build.h>
#include <StandAlone/tools/uda2nrrd/wrap_nrrd.h>

#include <StandAlone/tools/uda2nrrd/PatchToField.h>

#include <Core/Geometry/Vector.h>

#include <Core/Containers/FData.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/GenericField.h>

#include <Core/Util/FileUtils.h>

#include <teem/nrrd.h>

///////////////////////////////////////////////////////////////////////////////

using namespace std;

///////////////////////////////////////////////////////////////////////////////

typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh;
typedef LVMesh::handle_type LVMeshHandle;

///////////////////////////////////////////////////////////////////////////////

template<class VarT, class T>
void
handleData( QueryInfo &    qinfo,
            IntVector &    low,
            LVMeshHandle   mesh_handle,
            int            basis_order,
            const string & filename,
            const Args   & args )
{
  typedef GenericField<LVMesh, ConstantBasis<T>,
                       FData3d<T, LVMesh> > LVFieldCB;
  typedef GenericField<LVMesh, HexTrilinearLgn<T>, 
                       FData3d<T, LVMesh> > LVFieldLB;

  VarT gridVar;
  T data_T;

  // set the generation and timestep in the field
  if( !args.quiet ) cout << "Building Field from uda data\n";
  
  // Print out the psycal extents
  BBox bbox = mesh_handle->get_bounding_box();
  if( !args.quiet ) cout << "Bounding box: min("<<bbox.min()<<"), max("<<bbox.max()<<")\n";

  // Get the nrrd data, and print it out.
  char *err;

  Nrrd * nrrd;
  if( basis_order == 0 ){
    LVFieldCB* sf = scinew LVFieldCB(mesh_handle);
    typedef typename LVFieldCB::mesh_type::Cell FLOC;
    if (!sf) {
      cerr << "Cannot allocate memory for field\n";
      return;
    }
    if(qinfo.combine_levels){
      build_combined_level_field<T, VarT, LVFieldCB, FLOC>( qinfo, low, sf, args );
    } else {
      build_field( qinfo, low, data_T, gridVar, sf, args );
    }
    // Convert the field to a nrrd
    nrrd = wrap_nrrd( sf, args.matrix_op, args.verbose );
    // Clean up our memory
    delete sf;
  } else {
    LVFieldLB* sf = scinew LVFieldLB(mesh_handle);
    typedef typename LVFieldLB::mesh_type::Node FLOC;
    if (!sf) {
      cerr << "Cannot allocate memory for field\n";
      return;
    }
    if(qinfo.combine_levels){
      build_combined_level_field<T, VarT, LVFieldLB, FLOC>( qinfo, low, sf, args );
    } else {
      build_field( qinfo, low, data_T, gridVar, sf, args );
    }
    // Convert the field to a nrrd
    nrrd = wrap_nrrd( sf, args.matrix_op, args.verbose );
    // Clean up our memory
    delete sf;
  }

#if 0
  Piostream *fieldstrm =
    scinew BinaryPiostream(string(filename + ".fld").c_str(),
                           Piostream::Write);
  if (fieldstrm->error()) {
    cerr << "Could not open test.fld for writing.\n";
    exit(1);
  } else {
    Pio(*fieldstrm, *source_field);
    delete fieldstrm;
  }
#endif

  if( nrrd ) { // Save the NRRD to a file.
    string filetype = args.attached_header ? ".nrrd": ".nhdr";

    if( !args.quiet ) cout << "Writing nrrd file: " << filename + filetype << "\n";

    char answer = 'y';
    if( !args.force_overwrite && validFile( filename + filetype ) ) {
      cout << "\nWARNING: File already exists... overwrite? [y/n]\n";
      cin >> answer;
    }
      
    if( answer != 'y' ) {
        cout << "Aborting save...\n";
    }
    else {

      BBox lbox;
      qinfo.level->getSpatialRange( lbox );

      ostringstream extentsString, levelString, timeString;
      extentsString << lbox;
      timeString << qinfo.time;

      if( qinfo.combine_levels ) {
        levelString << "all (" << qinfo.grid->numLevels() << ")";
      } 
      else {
        levelString << qinfo.level->getIndex();
      }

      nrrdKeyValueAdd( nrrd, "time",    timeString.str().c_str() );
      nrrdKeyValueAdd( nrrd, "extents", extentsString.str().c_str() );
      nrrdKeyValueAdd( nrrd, "level",   levelString.str().c_str() );

      if( nrrdSave(string(filename + filetype).c_str(), nrrd, 0) ) {
        // There was a problem
        err = biffGetDone(NRRD);
        cerr << "Error writing nrrd:\n" << err <<"\n";
      } else {
        if( !args.quiet ) cout << "Done writing nrrd file\n";
      }
    }
    // nrrdNuke deletes the nrrd and the data inside the nrrd
    nrrdNuke( nrrd );
  } else {
    // There was a problem
    err = biffGetDone(NRRD);
    cerr << "Error wrapping nrrd: "<<err<<"\n";
  }  
  return;
} // end getData

///////////////////////////////////////////////////////////////////////////////

template<class T>
void
handleVariable( QueryInfo &qinfo, IntVector &low, IntVector& hi,
                IntVector &range, BBox &box,
                const string &filename,
                const Args & args )
{
  LVMeshHandle mesh_handle;
  switch( qinfo.type->getType() ) {
  case Uintah::TypeDescription::CCVariable:
    mesh_handle = scinew LVMesh(range.x(), range.y(),
                                 range.z(), box.min(),
                                 box.max());
    handleData<CCVariable<T>, T>( qinfo, low, mesh_handle, 0, filename, args );
    break;
  case Uintah::TypeDescription::NCVariable:
    mesh_handle = scinew LVMesh(range.x(), range.y(),
                                 range.z(), box.min(),
                                 box.max());
    handleData<NCVariable<T>, T>( qinfo, low, mesh_handle, 1, filename, args );
    break;
  case Uintah::TypeDescription::SFCXVariable:
    mesh_handle = scinew LVMesh(range.x(), range.y()-1,
                                 range.z()-1, box.min(),
                                 box.max());
    handleData<SFCXVariable<T>, T>( qinfo, low, mesh_handle, 1, filename, args );
    break;
  case Uintah::TypeDescription::SFCYVariable:
    mesh_handle = scinew LVMesh(range.x()-1, range.y(),
                                 range.z()-1, box.min(),
                                 box.max());
    handleData<SFCYVariable<T>, T>( qinfo, low, mesh_handle, 1, filename, args );
    break;
  case Uintah::TypeDescription::SFCZVariable:
    mesh_handle = scinew LVMesh(range.x()-1, range.y()-1,
                                 range.z(), box.min(),
                                 box.max());
    handleData<SFCZVariable<T>, T>( qinfo, low, mesh_handle, 1, filename, args );
    break;
  default:
    cerr << "Type is unknown.\n";
    return;
    break;
  
  }
} // end handleVariable()

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
  try {

    int material = *qinfo.materials.begin();

    if( !args.quiet ) { 
      int patchNum = patch->getLevelIndex();
      int modLevel = 1;
      int numPatches = patch->getLevel()->numPatches();

      if( numPatches > 100 )  { modLevel *= 10; } 
      if( numPatches > 1000 ) { modLevel *= 10; } 

      if( patchNum % modLevel == 0 ) {
        cout << "  Extracting data for material " << material
             << ". (Patch: " << patchNum+1 << "/" << patch->getLevel()->numPatches() << ")\n"; 
      }
    }

    qinfo.archive->query(patch_data, qinfo.varname, material, patch,
                         qinfo.timestep);
  } catch (Exception& e) {
    //     error("query caused an exception: " + string(e.message()));
    cerr << "handlePatchData::error in query function\n";
    cerr << e.message()<<"\n";
    return;
  }

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
      default:
        patch_high = patch->getExtraNodeHighIndex();   
      } 
    }
      
  } else { // Don't remove the boundary
    if( sfield->basis_order() == 0){
      patch_low = patch->getExtraCellLowIndex();
      patch_high = patch->getExtraCellHighIndex();
    } else {
      patch_low = patch->getExtraNodeLowIndex();
      switch (qinfo.type->getType()) {
      case Uintah::TypeDescription::SFCXVariable:
        patch_high = patch->getSFCXHighIndex();
        break;
      case Uintah::TypeDescription::SFCYVariable:
        patch_high = patch->getSFCYHighIndex();
        break;
      case Uintah::TypeDescription::SFCZVariable:
        patch_high = patch->getSFCZHighIndex();
        break;
      case Uintah::TypeDescription::NCVariable:
        patch_high = patch->getNodeHighIndex();
        break;
      default:
        cerr << "build_field::unknown variable.\n";
        exit(1);
      }
    }
  } // if (remove_boundary)

    // Rewindow the data if we need only a subset.  This should never
    // get bigger (thus requiring reallocation).  rewindow returns
    // true iff no reallocation is needed.
  patch_data.rewindow( patch_low, patch_high );
    
  PatchToFieldThread<T, FIELD> *worker = 
    scinew PatchToFieldThread<T, FIELD>(sfield, &patch_data, offset,
                                        patch_low, patch_high);
  worker->run();
  delete worker;

} // end handlePatchData()

///////////////////////////////////////////////////////////////////////////////
// Instantiate some of the needed verisons of functions.  These
// functions are never called, but force the compiler to instantiate the
// handleVariable<Vector> function that is needed.


typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh_template;

#define INTANTIATE_TEMPLATES_HANDLEVARIABLE_CC(T) \
template void handlePatchData<T, SFCXVariable<T>, GenericField<LVMesh_template, ConstantBasis<T>, FData3d<T, LVMesh_template> > > (QueryInfo&, IntVector&, GenericField<LVMesh_template, ConstantBasis<T>, FData3d<T, LVMesh_template> >*, const Patch*, const Args&);\
template void handlePatchData<T, SFCYVariable<T>, GenericField<LVMesh_template, ConstantBasis<T>, FData3d<T, LVMesh_template> > > (QueryInfo&, IntVector&, GenericField<LVMesh_template, ConstantBasis<T>, FData3d<T, LVMesh_template> >*, const Patch*, const Args&);\
template void handlePatchData<T, SFCZVariable<T>, GenericField<LVMesh_template, ConstantBasis<T>, FData3d<T, LVMesh_template> > > (QueryInfo&, IntVector&, GenericField<LVMesh_template, ConstantBasis<T>, FData3d<T, LVMesh_template> >*, const Patch*, const Args&);\
template void handlePatchData<T, CCVariable<T>,   GenericField<LVMesh_template, ConstantBasis<T>, FData3d<T, LVMesh_template> > > (QueryInfo&, IntVector&, GenericField<LVMesh_template, ConstantBasis<T>, FData3d<T, LVMesh_template> >*, const Patch*, const Args&);\
template void handlePatchData<T, NCVariable<T>,   GenericField<LVMesh_template, ConstantBasis<T>, FData3d<T, LVMesh_template> > > (QueryInfo&, IntVector&, GenericField<LVMesh_template, ConstantBasis<T>, FData3d<T, LVMesh_template> >*, const Patch*, const Args&);\
\
template void handlePatchData<T, SFCXVariable<T>, GenericField<LVMesh_template, HexTrilinearLgn<T>, FData3d<T, LVMesh_template> > > (QueryInfo&, IntVector&, GenericField<LVMesh_template, HexTrilinearLgn<T>, FData3d<T, LVMesh_template> >*, const Patch*, const Args&);\
template void handlePatchData<T, SFCYVariable<T>, GenericField<LVMesh_template, HexTrilinearLgn<T>, FData3d<T, LVMesh_template> > > (QueryInfo&, IntVector&, GenericField<LVMesh_template, HexTrilinearLgn<T>, FData3d<T, LVMesh_template> >*, const Patch*, const Args&);\
template void handlePatchData<T, SFCZVariable<T>, GenericField<LVMesh_template, HexTrilinearLgn<T>, FData3d<T, LVMesh_template> > > (QueryInfo&, IntVector&, GenericField<LVMesh_template, HexTrilinearLgn<T>, FData3d<T, LVMesh_template> >*, const Patch*, const Args&);\
template void handlePatchData<T, CCVariable<T>,   GenericField<LVMesh_template, HexTrilinearLgn<T>, FData3d<T, LVMesh_template> > > (QueryInfo&, IntVector&, GenericField<LVMesh_template, HexTrilinearLgn<T>, FData3d<T, LVMesh_template> >*, const Patch*, const Args&);\
template void handlePatchData<T, NCVariable<T>,   GenericField<LVMesh_template, HexTrilinearLgn<T>, FData3d<T, LVMesh_template> > > (QueryInfo&, IntVector&, GenericField<LVMesh_template, HexTrilinearLgn<T>, FData3d<T, LVMesh_template> >*, const Patch*, const Args&);\
\
template void handleVariable<T>(QueryInfo&, IntVector&, IntVector&, IntVector&, BBox&, const string&, const Args&);

INTANTIATE_TEMPLATES_HANDLEVARIABLE_CC(Vector)
INTANTIATE_TEMPLATES_HANDLEVARIABLE_CC(int)
INTANTIATE_TEMPLATES_HANDLEVARIABLE_CC(float)
INTANTIATE_TEMPLATES_HANDLEVARIABLE_CC(double)
INTANTIATE_TEMPLATES_HANDLEVARIABLE_CC(Matrix3)
