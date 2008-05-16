
/////////////////
// Due to template instantiation ordering problems, these 2 includes must be first:
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Core/Basis/Constant.h> 
// End template needed .h files
/////////////////

#include <Packages/Uintah/StandAlone/tools/uda2nrrd/handleVariable.h>

#include <Packages/Uintah/StandAlone/tools/uda2nrrd/Args.h>
#include <Packages/Uintah/StandAlone/tools/uda2nrrd/build.h>
#include <Packages/Uintah/StandAlone/tools/uda2nrrd/wrap_nrrd.h>

#include <Packages/Uintah/Dataflow/Modules/Selectors/PatchToField.h>

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

template<class VarT, class T>
void
handleData( QueryInfo &    qinfo,
            IntVector &    low,
            LVMeshHandle   mesh_handle,
            int            basis_order,
            const string & filename,
            const Args   & args,
			cellVals& cellValColln,
			bool dataReq )
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
  // char *err;

  // Nrrd * nrrd;
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
    wrap_nrrd( sf, args.matrix_op, args.verbose, cellValColln, dataReq );
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
    wrap_nrrd( sf, args.matrix_op, args.verbose, cellValColln, dataReq );
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

  // if( nrrd ) { // Save the NRRD to a file.
    // No more needed 
    /*string filetype = args.attached_header ? ".nrrd": ".nhdr";

    if( !args.quiet ) cout << "Writing nrrd file: " << filename + filetype << "\n";

    char answer = 'y';
    if( validFile( filename + filetype ) ) {
      cout << "\nWARNING: File already exists... overwrite? [y/n]\n";
      cin >> answer;
    }
      
    if( answer != 'y' ) {
        cout << "Aborting save...\n";
    }
    else {
      if( nrrdSave(string(filename + filetype).c_str(), nrrd, 0) ) {
        // There was a problem
        err = biffGetDone(NRRD);
        cerr << "Error writing nrrd:\n" << err <<"\n";
      } else {
        if( !args.quiet ) cout << "Done writing nrrd file\n";
      }
    }*/

    // Coying the values from the nrrd directly. This at some time should be changed
	// to avoid the double doing and data should be copied directy from the source.

	/*unsigned int dim = cellValColln.dim = nrrd->dim;
	
	cout << "Nrrd has " << dim << " dimensions\n";
	
	vector<int> dimension;
	unsigned int numComponents = 1;
	
	for (unsigned int i = 0; i < dim; i++) {
	  numComponents *= nrrd->axis[i].size;
	}  
	
	if (dim == 5) { // Tensors (rank 2): max dimension
	  cellValColln.x = nrrd->axis[2].size;
	  cellValColln.y = nrrd->axis[3].size;
	  cellValColln.z = nrrd->axis[4].size;
	}
	else if (dim == 4) { // Vectors
	  cellValColln.x = nrrd->axis[1].size;
	  cellValColln.y = nrrd->axis[2].size;
	  cellValColln.z = nrrd->axis[3].size;
	}
	else if (dim == 3) { // Scalars + Tensors (rank 0)
	  cellValColln.x = nrrd->axis[0].size;
	  cellValColln.y = nrrd->axis[1].size;
	  cellValColln.z = nrrd->axis[2].size;
	}*/

	// unsigned int x = cellValColln.x = nrrd->axis[0].size;
	// unsigned int y = cellValColln.y = nrrd->axis[1].size;
	// unsigned int z = cellValColln.z = nrrd->axis[2].size;
	// unsigned int w = nrrd->axis[3].size; // may be present in the case of vectors

	// cout << cellValColln.x << "|" << cellValColln.y << "|" << cellValColln.z << "\n";
	
	// unsigned int numComponents;
	
	/*if (w == 0)
	  numComponents = x * y * z;
	else { 
	  cellValColln.x = y;
	  cellValColln.y = z;
	  cellValColln.z = w;
	  numComponents = x * y * z * w;
	}*/  
	  
	/*if (dataReq) {
	  typeDouble* cellValVecPtr = new typeDouble();
	  double* data = (double*)nrrd->data;   
	  double* testPtr = NULL;
	  for( unsigned int cnt = 0; cnt < numComponents; cnt++ ) {
	    if ((testPtr = (data + cnt)) == NULL ) {
		  cout << "NULL encountered, pushed zero\n";
		  cellValVecPtr->push_back(0);
		}  
		else  
		  cellValVecPtr->push_back(data[cnt]);
	  }*/
	  
	  /*for (unsigned int i = 0; i < x; ++i) {
	    double* rowData = (double*)(nrrd->data) + i * y * z;
		// if (rowData == NULL) { cout << "rowData: NULL"; exit(1); }
	    for (unsigned int j = 0; j < y; ++j) {
	      double* cellData = rowData + j * z;
		  // cout << "cellData: " << cellData << "\n";
	      // if (cellData == NULL) { cout << "cellData: NULL"; exit(1); } 
		  for (unsigned int k = 0 ; k < z; ++k) {
		    // if( i > 150 && j > 160 && k > 10 ) {
		    // cout << &cellData[k] << ": " << i << " " << j << " " << k << " ";
			// cout.flush(); 
			// cout << "cellData[k]: " << cellData[k] << "\n";
			// }
		    cellValVecPtr->push_back(cellData[k]);
		  }
	    }
	  }*/
	  
	  /*cout << "Data in cellValVecPtr pushed\n";
	  cout << "Assigning cellValVecPtr ptr\n";
	  cellValColln.cellValVec = cellValVecPtr;
	  cout << "Assignment successful\n";		          
	}*/
		
    // nrrdNuke deletes the nrrd and the data inside the nrrd
    // nrrdNuke( nrrd );
  // } else {
    // There was a problem
    // err = biffGetDone(NRRD);
    // cerr << "Error wrapping nrrd: "<<err<<"\n";
  // }  
  return;
} // end getData

///////////////////////////////////////////////////////////////////////////////

template<class T>
void
handleVariable( QueryInfo &qinfo, IntVector &low, IntVector& hi,
                IntVector &range, BBox &box,
                const string &filename,
                const Args & args,
				cellVals& cellVallColln,
				bool dataReq )
{
  LVMeshHandle mesh_handle;
  switch( qinfo.type->getType() ) {
  case Uintah::TypeDescription::CCVariable:
    mesh_handle = scinew LVMesh(range.x(), range.y(),
                                 range.z(), box.min(),
                                 box.max());
    handleData<CCVariable<T>, T>( qinfo, low, mesh_handle, 0, filename, args, cellVallColln, dataReq );
    break;
  case Uintah::TypeDescription::NCVariable:
    mesh_handle = scinew LVMesh(range.x(), range.y(),
                                 range.z(), box.min(),
                                 box.max());
    handleData<NCVariable<T>, T>( qinfo, low, mesh_handle, 1, filename, args, cellVallColln, dataReq );
    break;
  case Uintah::TypeDescription::SFCXVariable:
    mesh_handle = scinew LVMesh(range.x(), range.y()-1,
                                 range.z()-1, box.min(),
                                 box.max());
    handleData<SFCXVariable<T>, T>( qinfo, low, mesh_handle, 1, filename, args, cellVallColln, dataReq );
    break;
  case Uintah::TypeDescription::SFCYVariable:
    mesh_handle = scinew LVMesh(range.x()-1, range.y(),
                                 range.z()-1, box.min(),
                                 box.max());
    handleData<SFCYVariable<T>, T>( qinfo, low, mesh_handle, 1, filename, args, cellVallColln, dataReq );
    break;
  case Uintah::TypeDescription::SFCZVariable:
    mesh_handle = scinew LVMesh(range.x()-1, range.y()-1,
                                 range.z(), box.min(),
                                 box.max());
    handleData<SFCZVariable<T>, T>( qinfo, low, mesh_handle, 1, filename, args, cellVallColln, dataReq );
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

    if( !args.quiet ) { cout << "  Extracting data for material " << material << ".\n"; }

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
      patch_low = patch->getInteriorCellLowIndex();
      patch_high = patch->getInteriorCellHighIndex();
    } else {
      patch_low = patch->getInteriorNodeLowIndex();
      switch (qinfo.type->getType()) {
      case Uintah::TypeDescription::SFCXVariable:
        patch_high = patch->getInteriorHighIndex(Patch::XFaceBased);
        break;
      case Uintah::TypeDescription::SFCYVariable:
        patch_high = patch->getInteriorHighIndex(Patch::YFaceBased);
        break;
      case Uintah::TypeDescription::SFCZVariable:
        patch_high = patch->getInteriorHighIndex(Patch::ZFaceBased);
        break;
      default:
        patch_high = patch->getInteriorNodeHighIndex();   
      } 
    }
      
  } else { // Don't remove the boundary
    if( sfield->basis_order() == 0){
      patch_low = patch->getCellLowIndex();
      patch_high = patch->getCellHighIndex();
    } else {
      patch_low = patch->getNodeLowIndex();
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
  ASSERT(patch_data.rewindow( patch_low, patch_high ));
    
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

template <class T>
void
templateInstantiationForGetCCHelper()
{
  typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh;
  typedef GenericField<LVMesh, ConstantBasis<T>, FData3d<T, LVMesh> >   LVFieldCB;
  typedef GenericField<LVMesh, HexTrilinearLgn<T>, FData3d<T, LVMesh> > LVFieldLB;

  LVFieldCB   * gfcb = NULL;
  LVFieldLB   * gfhb = NULL;
  QueryInfo   * qinfo = NULL;
  IntVector     hi;
  const Patch * patch = NULL;
  const Args  * args = NULL;

  handlePatchData<T, SFCXVariable<T>, LVFieldCB> ( *qinfo, hi, gfcb, patch, *args );
  handlePatchData<T, SFCYVariable<T>, LVFieldCB> ( *qinfo, hi, gfcb, patch, *args );
  handlePatchData<T, SFCZVariable<T>, LVFieldCB> ( *qinfo, hi, gfcb, patch, *args );
  handlePatchData<T, CCVariable<T>,   LVFieldCB> ( *qinfo, hi, gfcb, patch, *args );
  handlePatchData<T, NCVariable<T>,   LVFieldCB> ( *qinfo, hi, gfcb, patch, *args );

  handlePatchData<T, SFCXVariable<T>, LVFieldLB> ( *qinfo, hi, gfhb, patch, *args );
  handlePatchData<T, SFCYVariable<T>, LVFieldLB> ( *qinfo, hi, gfhb, patch, *args );
  handlePatchData<T, SFCZVariable<T>, LVFieldLB> ( *qinfo, hi, gfhb, patch, *args );
  handlePatchData<T, CCVariable<T>,   LVFieldLB> ( *qinfo, hi, gfhb, patch, *args );
  handlePatchData<T, NCVariable<T>,   LVFieldLB> ( *qinfo, hi, gfhb, patch, *args );
}

void
templateInstantiationForGetCC()
{
  IntVector    hi, low, range;
  BBox         box;
  QueryInfo  * qinfo = NULL;
  const Args * args = NULL;
  LVMeshHandle mesh_handle;
  string       filename;
  cellVals     cellVallColln;
  bool		   dataReq;	

  handleVariable<Vector> ( *qinfo, low, hi, range, box, "", *args, cellVallColln, dataReq );
  handleVariable<double> ( *qinfo, low, hi, range, box, "", *args, cellVallColln, dataReq );
  handleVariable<int>    ( *qinfo, low, hi, range, box, "", *args, cellVallColln, dataReq );
  handleVariable<float>  ( *qinfo, low, hi, range, box, "", *args, cellVallColln, dataReq );
  handleVariable<Matrix3>( *qinfo, low, hi, range, box, "", *args, cellVallColln, dataReq );

  templateInstantiationForGetCCHelper<Vector>();
  templateInstantiationForGetCCHelper<int>();
  templateInstantiationForGetCCHelper<float>();
  templateInstantiationForGetCCHelper<double>();
  templateInstantiationForGetCCHelper<Matrix3>();
}

