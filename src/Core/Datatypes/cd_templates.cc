/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 * Manual template instantiations
 */


/*
 * These aren't used by Datatypes directly, but since they are used in
 * a lot of different modules, we instantiate them here to avoid bloat
 *
 * Find the bloaters with:
find . -name "*.ii" -print | xargs cat | sort | uniq -c | sort -nr | more
 */

#include <Core/Containers/LockingHandle.h>
#include <Core/Malloc/Allocator.h>



using namespace SCIRun;
#ifdef __sgi
#pragma set woff 1468
#endif

#include <Core/Geometry/Tensor.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/MaskedTetVol.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/MaskedLatticeVol.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/Datatypes/ContourField.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/Datatypes/ImageField.h>
#include <Core/Datatypes/ScanlineField.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Datatypes/PropertyManager.h>



template class LockingHandle<ColumnMatrix>;
template class LockingHandle<Matrix>;

template class MaskedTetVol<Tensor>;
template class MaskedTetVol<Vector>;
template class MaskedTetVol<double>;
template class MaskedTetVol<int>;
template class MaskedTetVol<short>;
template class MaskedTetVol<unsigned char>;

const TypeDescription* get_type_description(MaskedTetVol<Tensor>*);
const TypeDescription* get_type_description(MaskedTetVol<Vector>*);
const TypeDescription* get_type_description(MaskedTetVol<double>*);
const TypeDescription* get_type_description(MaskedTetVol<int>*);
const TypeDescription* get_type_description(MaskedTetVol<short>*);
const TypeDescription* get_type_description(MaskedTetVol<unsigned char>*);

template class TetVol<Tensor>;
template class TetVol<Vector>;
template class TetVol<double>;
template class TetVol<int>;
template class TetVol<short>;
template class TetVol<unsigned char>;
template class GenericField<TetVolMesh, vector<Tensor> >;
template class GenericField<TetVolMesh, vector<Vector> >;
template class GenericField<TetVolMesh, vector<double> >;
template class GenericField<TetVolMesh, vector<int> >;
template class GenericField<TetVolMesh, vector<short> >;
template class GenericField<TetVolMesh, vector<unsigned char> >;

const TypeDescription* get_type_description(TetVol<Tensor>*);
const TypeDescription* get_type_description(TetVol<Vector>*);
const TypeDescription* get_type_description(TetVol<double>*);
const TypeDescription* get_type_description(TetVol<int>*);
const TypeDescription* get_type_description(TetVol<short>*);
const TypeDescription* get_type_description(TetVol<unsigned char>*);

//Index types
const TypeDescription* get_type_description(NodeIndex<int>*);
const TypeDescription* get_type_description(EdgeIndex<int>*);
const TypeDescription* get_type_description(FaceIndex<int>*);
const TypeDescription* get_type_description(CellIndex<int>*);

const TypeDescription* get_type_description(vector<NodeIndex<int> >*);
const TypeDescription* get_type_description(vector<EdgeIndex<int> >*);
const TypeDescription* get_type_description(vector<FaceIndex<int> >*);
const TypeDescription* get_type_description(vector<CellIndex<int> >*);

template class MaskedLatticeVol<Tensor>;
template class MaskedLatticeVol<Vector>;
template class MaskedLatticeVol<double>;
template class MaskedLatticeVol<int>;
template class MaskedLatticeVol<short>;
template class MaskedLatticeVol<unsigned char>;

const TypeDescription* get_type_description(MaskedLatticeVol<Tensor>*);
const TypeDescription* get_type_description(MaskedLatticeVol<Vector>*);
const TypeDescription* get_type_description(MaskedLatticeVol<double>*);
const TypeDescription* get_type_description(MaskedLatticeVol<int>*);
const TypeDescription* get_type_description(MaskedLatticeVol<short>*);
const TypeDescription* get_type_description(MaskedLatticeVol<unsigned char>*);

template class LatticeVol<Tensor>;
template class LatticeVol<Vector>;
template class LatticeVol<double>;
template class LatticeVol<int>;
template class LatticeVol<short>;
template class LatticeVol<unsigned char>;
template class GenericField<LatVolMesh, FData3d<Tensor> >;
template class GenericField<LatVolMesh, FData3d<Vector> >;
template class GenericField<LatVolMesh, FData3d<double> >;
template class GenericField<LatVolMesh, FData3d<int> >;
template class GenericField<LatVolMesh, FData3d<short> >;
template class GenericField<LatVolMesh, FData3d<unsigned char> >;
template class FData3d<Tensor>;
template class FData3d<Vector>;
template class FData3d<double>;
template class FData3d<int>;
template class FData3d<short>;
template class FData3d<unsigned char>;

const TypeDescription* get_type_description(LatticeVol<Tensor>*);
const TypeDescription* get_type_description(LatticeVol<Vector>*);
const TypeDescription* get_type_description(LatticeVol<double>*);
const TypeDescription* get_type_description(LatticeVol<int>*);
const TypeDescription* get_type_description(LatticeVol<short>*);
const TypeDescription* get_type_description(LatticeVol<unsigned char>*);


template class TriSurf<Tensor>;
template class TriSurf<Vector>;
template class TriSurf<double>;
template class TriSurf<int>;
template class TriSurf<short>;
template class TriSurf<unsigned char>;
template class GenericField<TriSurfMesh, vector<Tensor> >;
template class GenericField<TriSurfMesh, vector<Vector> >;
template class GenericField<TriSurfMesh, vector<double> >;
template class GenericField<TriSurfMesh, vector<int> >;
template class GenericField<TriSurfMesh, vector<short> >;
template class GenericField<TriSurfMesh, vector<unsigned char> >;

const TypeDescription* get_type_description(TriSurf<Tensor>*);
const TypeDescription* get_type_description(TriSurf<Vector>*);
const TypeDescription* get_type_description(TriSurf<double>*);
const TypeDescription* get_type_description(TriSurf<int>*);
const TypeDescription* get_type_description(TriSurf<short>*);
const TypeDescription* get_type_description(TriSurf<unsigned char>*);

template class ImageField<Tensor>;
template class ImageField<Vector>;
template class ImageField<double>;
template class ImageField<int>;
template class ImageField<short>;
template class ImageField<unsigned char>;
template class GenericField<ImageMesh, FData2d<Tensor> >;
template class GenericField<ImageMesh, FData2d<Vector> >;
template class GenericField<ImageMesh, FData2d<double> >;
template class GenericField<ImageMesh, FData2d<int> >;
template class GenericField<ImageMesh, FData2d<short> >;
template class GenericField<ImageMesh, FData2d<unsigned char> >;
template class FData2d<Tensor>;
template class FData2d<Vector>;
template class FData2d<double>;
template class FData2d<int>;
template class FData2d<short>;
template class FData2d<unsigned char>;

const TypeDescription* get_type_description(ImageField<Tensor>*);
const TypeDescription* get_type_description(ImageField<Vector>*);
const TypeDescription* get_type_description(ImageField<double>*);
const TypeDescription* get_type_description(ImageField<int>*);
const TypeDescription* get_type_description(ImageField<short>*);
const TypeDescription* get_type_description(ImageField<unsigned char>*);

template class ContourField<Tensor>;
template class ContourField<Vector>;
template class ContourField<double>;
template class ContourField<int>;
template class ContourField<short>;
template class ContourField<unsigned char>;
template class GenericField<ContourMesh, vector<Tensor> >;
template class GenericField<ContourMesh, vector<Vector> >;
template class GenericField<ContourMesh, vector<double> >;
template class GenericField<ContourMesh, vector<int> >;
template class GenericField<ContourMesh, vector<short> >;
template class GenericField<ContourMesh, vector<unsigned char> >;

const TypeDescription* get_type_description(ContourField<Tensor>*);
const TypeDescription* get_type_description(ContourField<Vector>*);
const TypeDescription* get_type_description(ContourField<double>*);
const TypeDescription* get_type_description(ContourField<int>*);
const TypeDescription* get_type_description(ContourField<short>*);
const TypeDescription* get_type_description(ContourField<unsigned char>*);

template class ScanlineField<Tensor>;
template class ScanlineField<Vector>;
template class ScanlineField<double>;
template class ScanlineField<int>;
template class ScanlineField<short>;
template class ScanlineField<unsigned char>;
template class GenericField<ScanlineMesh, vector<Tensor> >;
template class GenericField<ScanlineMesh, vector<Vector> >;
template class GenericField<ScanlineMesh, vector<double> >;
template class GenericField<ScanlineMesh, vector<int> >;
template class GenericField<ScanlineMesh, vector<short> >;
template class GenericField<ScanlineMesh, vector<unsigned char> >;

const TypeDescription* get_type_description(ScanlineField<Tensor>*);
const TypeDescription* get_type_description(ScanlineField<Vector>*);
const TypeDescription* get_type_description(ScanlineField<double>*);
const TypeDescription* get_type_description(ScanlineField<int>*);
const TypeDescription* get_type_description(ScanlineField<short>*);
const TypeDescription* get_type_description(ScanlineField<unsigned char>*);

template class PointCloud<Tensor>;
template class PointCloud<Vector>;
template class PointCloud<double>;
template class PointCloud<int>;
template class PointCloud<short>;
template class PointCloud<unsigned char>;
template class GenericField<PointCloudMesh, vector<Tensor> >;
template class GenericField<PointCloudMesh, vector<Vector> >;
template class GenericField<PointCloudMesh, vector<double> >;
template class GenericField<PointCloudMesh, vector<int> >;
template class GenericField<PointCloudMesh, vector<short> >;
template class GenericField<PointCloudMesh, vector<unsigned char> >;

const TypeDescription* get_type_description(PointCloud<Tensor>*);
const TypeDescription* get_type_description(PointCloud<Vector>*);
const TypeDescription* get_type_description(PointCloud<double>*);
const TypeDescription* get_type_description(PointCloud<int>*);
const TypeDescription* get_type_description(PointCloud<short>*);
const TypeDescription* get_type_description(PointCloud<unsigned char>*);

template class Property<string>;
template class Property<Array1<double> >;
template class Property<Array1<Tensor> >;
template class Property<pair<int,double> >;
template class Property<pair<double,double> >;
template class Property<pair<float,float> >;
template class Property<pair<unsigned int,unsigned int> >;
template class Property<pair<int,int> >;
template class Property<pair<unsigned short,unsigned short> >;
template class Property<pair<short,short> >;
template class Property<pair<unsigned char,unsigned char> >;
template class Property<pair<char,char> >;

//! Instantiate the specialized query_scalar_interface methods just once.
template <>
ScalarFieldInterface *
TetVol<double>::query_scalar_interface() const
{
  return scinew SFInterface<TetVol<double> >(this);
}

template <>
ScalarFieldInterface *
TetVol<float>::query_scalar_interface() const
{
  return scinew SFInterface<TetVol<float> >(this);
}

template <>
ScalarFieldInterface *
TetVol<int>::query_scalar_interface() const
{
  return scinew SFInterface<TetVol<int> >(this);
}

template <>
ScalarFieldInterface *
TetVol<short>::query_scalar_interface() const
{
  return scinew SFInterface<TetVol<short> >(this);
}

template <>
ScalarFieldInterface *
TetVol<unsigned char>::query_scalar_interface() const
{
  return scinew SFInterface<TetVol<unsigned char> >(this);
}


template <>
VectorFieldInterface *
TetVol<Vector>::query_vector_interface() const
{
  return scinew VFInterface<TetVol<Vector> >(this);
}

template <>
TensorFieldInterface *
TetVol<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<TetVol<Tensor> >(this);
}


// ---

template <>
ScalarFieldInterface *
LatticeVol<double>::query_scalar_interface() const
{
  return scinew SFInterface<LatticeVol<double> >(this);
}

template <>
ScalarFieldInterface *
LatticeVol<float>::query_scalar_interface() const
{
  return scinew SFInterface<LatticeVol<float> >(this);
}

template <>
ScalarFieldInterface *
LatticeVol<int>::query_scalar_interface() const
{
  return scinew SFInterface<LatticeVol<int> >(this);
}

template <>
ScalarFieldInterface *
LatticeVol<short>::query_scalar_interface() const
{
  return scinew SFInterface<LatticeVol<short> >(this);
}

template <>
ScalarFieldInterface *
LatticeVol<unsigned char>::query_scalar_interface() const
{
  return scinew SFInterface<LatticeVol<unsigned char> >(this);
}


template <>
VectorFieldInterface *
LatticeVol<Vector>::query_vector_interface() const
{
  return scinew VFInterface<LatticeVol<Vector> >(this);
}


template <>
TensorFieldInterface *
LatticeVol<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<LatticeVol<Tensor> >(this);
}

// ---

template <>
ScalarFieldInterface *
TriSurf<double>::query_scalar_interface() const
{
  return scinew SFInterface<TriSurf<double> >(this);
}

template <>
ScalarFieldInterface *
TriSurf<float>::query_scalar_interface() const
{
  return scinew SFInterface<TriSurf<float> >(this);
}

template <>
ScalarFieldInterface *
TriSurf<int>::query_scalar_interface() const
{
  return scinew SFInterface<TriSurf<int> >(this);
}

template <>
ScalarFieldInterface *
TriSurf<short>::query_scalar_interface() const
{
  return scinew SFInterface<TriSurf<short> >(this);
}

template <>
ScalarFieldInterface *
TriSurf<unsigned char>::query_scalar_interface() const
{
  return scinew SFInterface<TriSurf<unsigned char> >(this);
}


template <>
VectorFieldInterface *
TriSurf<Vector>::query_vector_interface() const
{
  return scinew VFInterface<TriSurf<Vector> >(this);
}

template <>
TensorFieldInterface *
TriSurf<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<TriSurf<Tensor> >(this);
}

// ---

template <>
ScalarFieldInterface *
ImageField<double>::query_scalar_interface() const
{
  return scinew SFInterface<ImageField<double> >(this);
}

template <>
ScalarFieldInterface *
ImageField<int>::query_scalar_interface() const
{
  return scinew SFInterface<ImageField<int> >(this);
}

template <>
ScalarFieldInterface *
ImageField<short>::query_scalar_interface() const
{
  return scinew SFInterface<ImageField<short> >(this);
}

template <>
ScalarFieldInterface *
ImageField<unsigned char>::query_scalar_interface() const
{
  return scinew SFInterface<ImageField<unsigned char> >(this);
}


template <>
VectorFieldInterface *
ImageField<Vector>::query_vector_interface() const
{
  return scinew VFInterface<ImageField<Vector> >(this);
}


template <>
TensorFieldInterface *
ImageField<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<ImageField<Tensor> >(this);
}

// ---

template <>
ScalarFieldInterface *
ContourField<double>::query_scalar_interface() const
{
  return scinew SFInterface<ContourField<double> >(this);
}

template <>
ScalarFieldInterface *
ContourField<int>::query_scalar_interface() const
{
  return scinew SFInterface<ContourField<int> >(this);
}

template <>
ScalarFieldInterface *
ContourField<short>::query_scalar_interface() const
{
  return scinew SFInterface<ContourField<short> >(this);
}

template <>
ScalarFieldInterface *
ContourField<unsigned char>::query_scalar_interface() const
{
  return scinew SFInterface<ContourField<unsigned char> >(this);
}


template <>
VectorFieldInterface *
ContourField<Vector>::query_vector_interface() const
{
  return scinew VFInterface<ContourField<Vector> >(this);
}


template <>
TensorFieldInterface *
ContourField<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<ContourField<Tensor> >(this);
}

// ---

template <>
ScalarFieldInterface *
ScanlineField<double>::query_scalar_interface() const
{
  return scinew SFInterface<ScanlineField<double> >(this);
}

template <>
ScalarFieldInterface *
ScanlineField<int>::query_scalar_interface() const
{
  return scinew SFInterface<ScanlineField<int> >(this);
}

template <>
ScalarFieldInterface *
ScanlineField<short>::query_scalar_interface() const
{
  return scinew SFInterface<ScanlineField<short> >(this);
}

template <>
ScalarFieldInterface *
ScanlineField<unsigned char>::query_scalar_interface() const
{
  return scinew SFInterface<ScanlineField<unsigned char> >(this);
}


template <>
VectorFieldInterface *
ScanlineField<Vector>::query_vector_interface() const
{
  return scinew VFInterface<ScanlineField<Vector> >(this);
}


template <>
TensorFieldInterface *
ScanlineField<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<ScanlineField<Tensor> >(this);
}

// ---

template <>
ScalarFieldInterface *
PointCloud<double>::query_scalar_interface() const
{
  return scinew SFInterface<PointCloud<double> >(this);
}

template <>
ScalarFieldInterface *
PointCloud<int>::query_scalar_interface() const
{
  return scinew SFInterface<PointCloud<int> >(this);
}

template <>
ScalarFieldInterface *
PointCloud<short>::query_scalar_interface() const
{
  return scinew SFInterface<PointCloud<short> >(this);
}

template <>
ScalarFieldInterface *
PointCloud<unsigned char>::query_scalar_interface() const
{
  return scinew SFInterface<PointCloud<unsigned char> >(this);
}


template <>
VectorFieldInterface *
PointCloud<Vector>::query_vector_interface() const
{
  return scinew VFInterface<PointCloud<Vector> >(this);
}


template <>
TensorFieldInterface *
PointCloud<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<PointCloud<Tensor> >(this);
}


//! Compute the gradient g in cell ci.
template <>
Vector TetVol<Vector>::cell_gradient(TetVolMesh::Cell::index_type /*ci*/)
{
  ASSERT(type_name(1) != "Vector");  // redundant, useful error message
  return Vector(0, 0, 0);
}


template <>
Vector TetVol<Tensor>::cell_gradient(TetVolMesh::Cell::index_type /*ci*/)
{
  ASSERT(type_name(1) != "Tensor");  // redundant, useful error message
  return Vector(0, 0, 0);
}

template <> bool LatticeVol<Tensor>::get_gradient(Vector &, const Point &/*p*/)
{
  ASSERT(type_name(1) != "Tensor");  // redundant, useful error message
  return false;
}

template <> bool LatticeVol<Vector>::get_gradient(Vector &, const Point &/*p*/)
{
  ASSERT(type_name(1) != "Tensor");  // redundant, useful error message
  return false;
}


#ifdef __sgi
#pragma reset woff 1468
#endif










