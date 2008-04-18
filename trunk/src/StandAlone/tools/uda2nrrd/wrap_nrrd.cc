// Allocates memory for dest, then copies all the data to dest from
// source.

////// 
// For templates instantiations to be found, these includes must remain in this order:
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Core/Basis/Constant.h> 
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
// End in-order templates.
//////

#include <Packages/Uintah/StandAlone/tools/uda2nrrd/wrap_nrrd.h>

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Util/TypeDescription.h>

using namespace SCIRun;
using namespace Uintah;

////////////////////////////////////////////////////////////////////////////////////
// Special nrrd functions

template <class T> unsigned int get_nrrd_type();

template <> unsigned int get_nrrd_type<char>()               { return nrrdTypeChar; }
template <> unsigned int get_nrrd_type<unsigned char>()      { return nrrdTypeUChar; }
template <> unsigned int get_nrrd_type<short>()              { return nrrdTypeShort; }
template <> unsigned int get_nrrd_type<unsigned short>()     { return nrrdTypeUShort; }
template <> unsigned int get_nrrd_type<int>()                { return nrrdTypeInt; }
template <> unsigned int get_nrrd_type<unsigned int>()       { return nrrdTypeUInt; }
template <> unsigned int get_nrrd_type<long long>()          { return nrrdTypeLLong; }
template <> unsigned int get_nrrd_type<unsigned long long>() { return nrrdTypeULLong; }
template <> unsigned int get_nrrd_type<float>()              { return nrrdTypeFloat; }

template <class T>
unsigned int get_nrrd_type() {
  return nrrdTypeDouble;
}

////////////////////////////////////////////////////////////////////////////////////
// Helper functions for wrap_nrrd

template <class T>
bool 
wrap_copy( T* fdata, double*& datap, unsigned int size, Matrix_Op matrix_op ){
  cerr<<"Should not be called for scalar data, no copy required!";
  return false;
}

// Vector version
template <>
bool
wrap_copy( Vector* fdata, double*& datap, unsigned int size, Matrix_Op matrix_op ){

  // Copy the data
  for(unsigned int i = 0; i < size; i++) {
    *datap++ = fdata->x();
    *datap++ = fdata->y();
    *datap++ = fdata->z();
    fdata++;
  }
  return true;
}

// Matrix3 version

template <>
bool
wrap_copy( Matrix3* fdata, double*& datap, unsigned int size, Matrix_Op matrix_op ){

  switch (matrix_op) {
  case None:
    for(unsigned int i = 0; i < size; i++) {
      for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
          *datap++ = (*fdata)(i,j);
      fdata++;
    }
    break;
  case Det:
    for(unsigned int i = 0; i < size; i++) {
      *datap++ = fdata->Determinant();
      fdata++;
    }
    break;
  case Trace:
    for(unsigned int i = 0; i < size; i++) {
      *datap++ = fdata->Trace();
      fdata++;
    }
    break;
  case Norm:
    for(unsigned int i = 0; i < size; i++) {
      *datap++ = fdata->Norm();
      fdata++;
    }
    break;
  default:
    cerr << "Unknown matrix operation\n";
    return false;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////

template<class FIELD>
Nrrd*
wrap_nrrd( FIELD * source, Matrix_Op matrix_op, bool verbose )
{
  Nrrd *out = nrrdNew();
  int dim = -1;
  size_t size[5];
  
  const SCIRun::TypeDescription *td = source->get_type_description( Field::FDATA_TD_E );

  if( td->get_name().find( "Vector") != string::npos ) {  // Vectors
    dim = 4;
    size[0] = 3;
    size[1] = source->fdata().dim3();
    size[2] = source->fdata().dim2();
    size[3] = source->fdata().dim1();

    cout << size[0] << " " << size[1] << " " << size[2] << endl;   
  
    unsigned int num_vec = source->fdata().size();
	cout << num_vec << endl;
    double *data = new double[num_vec*3];
    if (!data) {
      cerr << "Cannot allocate memory ("<<num_vec*3*sizeof(double)<<" byptes) for temp storage of vectors\n";
      nrrdNix(out);
      return 0;
    }
    double *datap = data;
    typename FIELD::value_type *vec_data = &(source->fdata()(0,0,0));
    

    // Copy the data
    wrap_copy( vec_data, datap, num_vec, matrix_op );
    
    if (nrrdWrap_nva(out, data, nrrdTypeDouble, dim, size) == 0) {
      return out;
    } else {
      nrrdNix(out);
      delete data;
      return 0;
    }
  } else if (td->get_name().find( "Matrix3") != string::npos ) { // Matrix3
    dim = (matrix_op == None) ? 5 : 3;
    if (matrix_op == None) {
      size[0] = 3;
      size[1] = 3;
      size[2] = source->fdata().dim3();
      size[3] = source->fdata().dim2();
      size[4] = source->fdata().dim1();
    } else {
      size[0] = source->fdata().dim3();
      size[1] = source->fdata().dim2();
      size[2] = source->fdata().dim1();
    }
    unsigned int num_mat = source->fdata().size();
    int elem_size = (matrix_op == None) ? 9 : 1;
    double *data = new double[num_mat*elem_size];
    if (!data) {
      cerr << "Cannot allocate memory ("<<num_mat*elem_size*sizeof(double)<<" byptes) for temp storage of vectors\n";
      nrrdNix(out);
      return 0;
    }
    double *datap = data;
    typename FIELD::value_type *mat_data = &(source->fdata()(0,0,0));
    // Copy the data
    if( !wrap_copy( mat_data, datap, num_mat, matrix_op ) ){
      nrrdNix(out);
      delete data;
      return 0;
    }

    if (nrrdWrap_nva(out, data, nrrdTypeDouble, dim, size) == 0) {
      return out;
    } else {
      nrrdNix(out);
      delete data;
      return 0;
    }
  } else { // Scalars
    dim = 3;
    size[0] = source->fdata().dim3();
    size[1] = source->fdata().dim2();
    size[2] = source->fdata().dim1();

    // We don't need to copy data, so just get the pointer to the data
    size_t field_size = (source->fdata().size() *
                         sizeof(typename FIELD::value_type));
    void* data = malloc(field_size);
    if (!data) {
      cerr << "Cannot allocate memory ("<<field_size<<" byptes) for scalar nrrd copy.\n";
      nrrdNix(out);
      return 0;
    }
    memcpy(data, (void*)&(source->fdata()(0,0,0)), field_size);

    if (nrrdWrap_nva(out, data, get_nrrd_type< typename FIELD::value_type>(), 
                     dim, size) == 0) {
      return out;
    } else {
      nrrdNix(out);
      free(data);
      return 0;
    }
  }

  if (verbose) for(int i = 0; i < dim; i++) cout << "size["<<i<<"] = "<<size[i]<<endl;

} // end wrap_nrrd()

///////////////////////////////////////////////////////////////////////////////
// Instantiate some of the needed verisons of functions.  This
// function is never called, but forces the instantiation of the
// wrap_nrrd functions that are needed.

template <class T>
static
void
instHelper()
{
  typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh;

  typedef GenericField<LVMesh, ConstantBasis<T>, FData3d<T, LVMesh> >   LVFieldCB;
  typedef GenericField<LVMesh, HexTrilinearLgn<T>, FData3d<T, LVMesh> > LVFieldLB;

  LVFieldCB   * sf = NULL;
  LVFieldLB   * flb = NULL;

  wrap_nrrd( sf,  (Matrix_Op)0, false );
  wrap_nrrd( flb, (Matrix_Op)0, false );
}

void
templateInstantiationForWrapNrrdCC()
{
  instHelper<double>();
  instHelper<float>();
  instHelper<int>();
  instHelper<Matrix3>();
  instHelper<Vector>();
}
