/*

   The MIT License

   Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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

#if 0


// Allocates memory for dest, then copies all the data to dest from
// source.

////// 
// For templates instantiations to be found, these includes must remain in this order:
#include <Core/Math/Matrix3.h>
#include <Core/Basis/Constant.h> 
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
// End in-order templates.
//////

#include <StandAlone/tools/uda2vis/wrap_nrrd.h>

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
wrap_copy( T* fdata, typeDouble* cellValVecPtr, unsigned int size, Matrix_Op matrix_op ){
  cerr<<"Should not be called for scalar data, no copy required!";
  return false;
}

// Vector version
template <>
bool
wrap_copy( Vector* fdata, typeDouble* cellValVecPtr, unsigned int size, Matrix_Op matrix_op ){

  for(unsigned int i = 0; i < size; i++) {
    cellValVecPtr->push_back(fdata->x());
    cellValVecPtr->push_back(fdata->y());
    cellValVecPtr->push_back(fdata->z());
    fdata++;
  }
  return true;
}

// Matrix3 version
template <>
bool
wrap_copy( Matrix3* fdata, typeDouble* cellValVecPtr, unsigned int size, Matrix_Op matrix_op ){

  switch (matrix_op) {
  case None:
    for(unsigned int i = 0; i < size; i++) {
      for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++) {
          cellValVecPtr->push_back((*fdata)(i,j));
        }  
      fdata++;
    }
    break;
  case Det:
    for(unsigned int i = 0; i < size; i++) {
      cellValVecPtr->push_back(fdata->Determinant());
      fdata++;
    }
    break;
  case Trace:
    for(unsigned int i = 0; i < size; i++) {
      cellValVecPtr->push_back(fdata->Trace());
      fdata++;
    }
    break;
  case Norm:
    for(unsigned int i = 0; i < size; i++) {
      cellValVecPtr->push_back(fdata->Norm());
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
void
wrap_nrrd( FIELD * source, Matrix_Op matrix_op, bool verbose, cellVals& cellValColln, bool dataReq )
{
  typeDouble* cellValVecPtr = NULL;
  int dim = -1;
  size_t size[5];

  const SCIRun::TypeDescription *td = source->get_type_description( Field::FDATA_TD_E );

  // Vectors
  if( td->get_name().find( "Vector") != string::npos ) {
    dim = 4;
    size[0] = 3;
    size[1] = source->fdata().dim3();
    size[2] = source->fdata().dim2();
    size[3] = source->fdata().dim1();

    cellValColln.dim = dim;
    cellValColln.x = size[1];
    cellValColln.y = size[2];
    cellValColln.z = size[3];

    unsigned int num_vec = source->fdata().size();

    if (dataReq) {
      cellValVecPtr = new typeDouble();
      cellValVecPtr->reserve(num_vec * 3);

      typename FIELD::value_type *vec_data = &(source->fdata()(0,0,0));

      // Copy the data
      wrap_copy( vec_data, cellValVecPtr, num_vec, matrix_op );
    }  
  }

  // Matrix3
  else if (td->get_name().find( "Matrix3") != string::npos ) { 
    dim = (matrix_op == None) ? 5 : 3;
    if (matrix_op == None) {
      size[0] = 3;
      size[1] = 3;
      size[2] = source->fdata().dim3();
      size[3] = source->fdata().dim2();
      size[4] = source->fdata().dim1();

      cellValColln.x = size[2];
      cellValColln.y = size[3];
      cellValColln.z = size[4];	
    } else {
      size[0] = source->fdata().dim3();
      size[1] = source->fdata().dim2();
      size[2] = source->fdata().dim1();

      cellValColln.x = size[0];
      cellValColln.y = size[1];
      cellValColln.z = size[2];
    }

    cellValColln.dim = dim;

    unsigned int num_mat = source->fdata().size();
    int elem_size = (matrix_op == None) ? 9 : 1;

    if (dataReq) {
      cellValVecPtr = new typeDouble();
      cellValVecPtr->reserve(num_mat*elem_size);

      typename FIELD::value_type *mat_data = &(source->fdata()(0,0,0));

      // Copy the data
      wrap_copy( mat_data, cellValVecPtr, num_mat, matrix_op );
    }
  }

  // Scalars
  else {
    dim = 3;
    size[0] = source->fdata().dim3();
    size[1] = source->fdata().dim2();
    size[2] = source->fdata().dim1();

    cellValColln.dim = dim;
    cellValColln.x = size[0];
    cellValColln.y = size[1];
    cellValColln.z = size[2];

    // We don't need to copy data, so just get the pointer to the data
    size_t field_size = (source->fdata().size() *
                         sizeof(typename FIELD::value_type));
    void* data = malloc(field_size);
    memcpy(data, (void*)&(source->fdata()(0,0,0)), field_size);

    unsigned int num_elements = source->fdata().size();

    if (dataReq) {
      cellValVecPtr = new typeDouble();
      cellValVecPtr->reserve(num_elements);
      
      for (unsigned int i = 0; i < num_elements; i++) {
        if (get_nrrd_type<typename FIELD::value_type>() == nrrdTypeDouble) {
          cellValVecPtr->push_back(*((double*)data + i));
        }  
        else if (get_nrrd_type<typename FIELD::value_type>() == nrrdTypeFloat) {
          cellValVecPtr->push_back(*((float*)data + i)); 
        }  
        else if (get_nrrd_type<typename FIELD::value_type>() == nrrdTypeInt) {
          cellValVecPtr->push_back(*((int*)data + i));
        }
        else if (get_nrrd_type<typename FIELD::value_type>() == nrrdTypeChar) {
          cellValVecPtr->push_back(*((char*)data + i));
        }
        else if (get_nrrd_type<typename FIELD::value_type>() == nrrdTypeUChar) {
          cellValVecPtr->push_back(*((unsigned char*)data + i));
        }
        else if (get_nrrd_type<typename FIELD::value_type>() == nrrdTypeShort) {
          cellValVecPtr->push_back(*((short*)data + i));
        }
        else if (get_nrrd_type<typename FIELD::value_type>() == nrrdTypeUShort) {
          cellValVecPtr->push_back(*((unsigned short*)data + i));
        }
        else if (get_nrrd_type<typename FIELD::value_type>() == nrrdTypeUInt) {
          cellValVecPtr->push_back(*((unsigned int*)data + i));
        }
        else if (get_nrrd_type<typename FIELD::value_type>() == nrrdTypeLLong) {
          cellValVecPtr->push_back(*((long long*)data + i));
        }
        else if (get_nrrd_type<typename FIELD::value_type>() == nrrdTypeULLong) {
          cellValVecPtr->push_back(*((unsigned long long*)data + i));
        }
      }
    }

    free(data);
  }

  if (dataReq)
    cellValColln.cellValVec = cellValVecPtr; 	  

  if (verbose) for(int i = 0; i < dim; i++) cout << "size["<<i<<"] = "<<size[i]<<endl;
}



///////////////////////////////////////////////////////////////////////////////
// Instantiate some of the needed verisons of functions.

typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh_template;

#define INTANTIATE_TEMPLATES_WRAP_NRRD_CC(T)                            \
  template void wrap_nrrd< GenericField< LVMesh_template, ConstantBasis<T>,   FData3d<T, LVMesh_template > > >(GenericField< LVMesh_template, ConstantBasis<T>,   FData3d<T, LVMesh_template > >*, Matrix_Op, bool, cellVals&, bool); \
  template void wrap_nrrd< GenericField< LVMesh_template, HexTrilinearLgn<T>, FData3d<T, LVMesh_template > > >(GenericField< LVMesh_template, HexTrilinearLgn<T>, FData3d<T, LVMesh_template > >*, Matrix_Op, bool, cellVals&, bool);

INTANTIATE_TEMPLATES_WRAP_NRRD_CC(double)
  INTANTIATE_TEMPLATES_WRAP_NRRD_CC(float)
  INTANTIATE_TEMPLATES_WRAP_NRRD_CC(int)
  INTANTIATE_TEMPLATES_WRAP_NRRD_CC(Matrix3)
  INTANTIATE_TEMPLATES_WRAP_NRRD_CC(Vector)


#endif
