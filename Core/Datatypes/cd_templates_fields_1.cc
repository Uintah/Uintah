#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/ImageField.h>


using namespace SCIRun;

template class FData2d<Tensor>;
template class FData2d<Vector>;
template class FData2d<double>;
template class FData2d<float>;
template class FData2d<int>;
template class FData2d<short>;
template class FData2d<unsigned char>;

template class GenericField<ImageMesh, FData2d<Tensor> >;
template class GenericField<ImageMesh, FData2d<Vector> >;
template class GenericField<ImageMesh, FData2d<double> >;
template class GenericField<ImageMesh, FData2d<float> >;
template class GenericField<ImageMesh, FData2d<int> >;
template class GenericField<ImageMesh, FData2d<short> >;
template class GenericField<ImageMesh, FData2d<unsigned char> >;

template class ImageField<Tensor>;
template class ImageField<Vector>;
template class ImageField<double>;
template class ImageField<float>;
template class ImageField<int>;
template class ImageField<short>;
template class ImageField<unsigned char>;

const TypeDescription* get_type_description(ImageField<Tensor> *);
const TypeDescription* get_type_description(ImageField<Vector> *);
const TypeDescription* get_type_description(ImageField<double> *);
const TypeDescription* get_type_description(ImageField<float> *);
const TypeDescription* get_type_description(ImageField<int> *);
const TypeDescription* get_type_description(ImageField<short> *);
const TypeDescription* get_type_description(ImageField<unsigned char> *);


template <>
TensorFieldInterface *
ImageField<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<ImageField<Tensor> >(this);
}


template <>
VectorFieldInterface *
ImageField<Vector>::query_vector_interface() const
{
  return scinew VFInterface<ImageField<Vector> >(this);
}


template <>
ScalarFieldInterface *
ImageField<double>::query_scalar_interface() const
{
  return scinew SFInterface<ImageField<double> >(this);
}

template <>
ScalarFieldInterface *
ImageField<float>::query_scalar_interface() const
{
  return scinew SFInterface<ImageField<float> >(this);
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


