#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/ScanlineField.h>
#include <Core/Datatypes/PointCloud.h>


using namespace SCIRun;

template class GenericField<ScanlineMesh, vector<Tensor> >;
template class GenericField<ScanlineMesh, vector<Vector> >;
template class GenericField<ScanlineMesh, vector<double> >;
template class GenericField<ScanlineMesh, vector<float> >;
template class GenericField<ScanlineMesh, vector<int> >;
template class GenericField<ScanlineMesh, vector<short> >;
template class GenericField<ScanlineMesh, vector<unsigned char> >;

template class GenericField<PointCloudMesh, vector<Tensor> >;
template class GenericField<PointCloudMesh, vector<Vector> >;
template class GenericField<PointCloudMesh, vector<double> >;
template class GenericField<PointCloudMesh, vector<float> >;
template class GenericField<PointCloudMesh, vector<int> >;
template class GenericField<PointCloudMesh, vector<short> >;
template class GenericField<PointCloudMesh, vector<unsigned char> >;

template class ScanlineField<Tensor>;
template class ScanlineField<Vector>;
template class ScanlineField<double>;
template class ScanlineField<float>;
template class ScanlineField<int>;
template class ScanlineField<short>;
template class ScanlineField<unsigned char>;

const TypeDescription* get_type_description(ScanlineField<Tensor> *);
const TypeDescription* get_type_description(ScanlineField<Vector> *);
const TypeDescription* get_type_description(ScanlineField<double> *);
const TypeDescription* get_type_description(ScanlineField<float> *);
const TypeDescription* get_type_description(ScanlineField<int> *);
const TypeDescription* get_type_description(ScanlineField<short> *);
const TypeDescription* get_type_description(ScanlineField<unsigned char> *);

template class PointCloud<Tensor>;
template class PointCloud<Vector>;
template class PointCloud<double>;
template class PointCloud<float>;
template class PointCloud<int>;
template class PointCloud<short>;
template class PointCloud<unsigned char>;

const TypeDescription* get_type_description(PointCloud<Tensor> *);
const TypeDescription* get_type_description(PointCloud<Vector> *);
const TypeDescription* get_type_description(PointCloud<double> *);
const TypeDescription* get_type_description(PointCloud<float> *);
const TypeDescription* get_type_description(PointCloud<int> *);
const TypeDescription* get_type_description(PointCloud<short> *);
const TypeDescription* get_type_description(PointCloud<unsigned char> *);


template <>
TensorFieldInterface *
ScanlineField<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<ScanlineField<Tensor> >(this);
}


template <>
VectorFieldInterface *
ScanlineField<Vector>::query_vector_interface() const
{
  return scinew VFInterface<ScanlineField<Vector> >(this);
}


template <>
ScalarFieldInterface *
ScanlineField<double>::query_scalar_interface() const
{
  return scinew SFInterface<ScanlineField<double> >(this);
}

template <>
ScalarFieldInterface *
ScanlineField<float>::query_scalar_interface() const
{
  return scinew SFInterface<ScanlineField<float> >(this);
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
TensorFieldInterface *
PointCloud<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<PointCloud<Tensor> >(this);
}


template <>
VectorFieldInterface *
PointCloud<Vector>::query_vector_interface() const
{
  return scinew VFInterface<PointCloud<Vector> >(this);
}


template <>
ScalarFieldInterface *
PointCloud<double>::query_scalar_interface() const
{
  return scinew SFInterface<PointCloud<double> >(this);
}

template <>
ScalarFieldInterface *
PointCloud<float>::query_scalar_interface() const
{
  return scinew SFInterface<PointCloud<float> >(this);
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
