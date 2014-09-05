#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/ImageField.h>
#include <Core/Datatypes/QuadSurf.h>

using namespace SCIRun;

template class FData2d<Tensor>;
template class FData2d<Vector>;
template class FData2d<double>;
template class FData2d<float>;
template class FData2d<int>;
template class FData2d<short>;
template class FData2d<char>;
template class FData2d<unsigned int>;
template class FData2d<unsigned short>;
template class FData2d<unsigned char>;

template class GenericField<ImageMesh, FData2d<Tensor> >;
template class GenericField<ImageMesh, FData2d<Vector> >;
template class GenericField<ImageMesh, FData2d<double> >;
template class GenericField<ImageMesh, FData2d<float> >;
template class GenericField<ImageMesh, FData2d<int> >;
template class GenericField<ImageMesh, FData2d<short> >;
template class GenericField<ImageMesh, FData2d<char> >;
template class GenericField<ImageMesh, FData2d<unsigned int> >;
template class GenericField<ImageMesh, FData2d<unsigned short> >;
template class GenericField<ImageMesh, FData2d<unsigned char> >;

template class ImageField<Tensor>;
template class ImageField<Vector>;
template class ImageField<double>;
template class ImageField<float>;
template class ImageField<int>;
template class ImageField<short>;
template class ImageField<char>;
template class ImageField<unsigned int>;
template class ImageField<unsigned short>;
template class ImageField<unsigned char>;

const TypeDescription* get_type_description(ImageField<Tensor> *);
const TypeDescription* get_type_description(ImageField<Vector> *);
const TypeDescription* get_type_description(ImageField<double> *);
const TypeDescription* get_type_description(ImageField<float> *);
const TypeDescription* get_type_description(ImageField<int> *);
const TypeDescription* get_type_description(ImageField<short> *);
const TypeDescription* get_type_description(ImageField<char> *);
const TypeDescription* get_type_description(ImageField<unsigned int> *);
const TypeDescription* get_type_description(ImageField<unsigned short> *);
const TypeDescription* get_type_description(ImageField<unsigned char> *);


template class GenericField<QuadSurfMesh, vector<Tensor> >;
template class GenericField<QuadSurfMesh, vector<Vector> >;
template class GenericField<QuadSurfMesh, vector<double> >;
template class GenericField<QuadSurfMesh, vector<float> >;
template class GenericField<QuadSurfMesh, vector<int> >;
template class GenericField<QuadSurfMesh, vector<short> >;
template class GenericField<QuadSurfMesh, vector<char> >;
template class GenericField<QuadSurfMesh, vector<unsigned int> >;
template class GenericField<QuadSurfMesh, vector<unsigned short> >;
template class GenericField<QuadSurfMesh, vector<unsigned char> >;

template class QuadSurf<Tensor>;
template class QuadSurf<Vector>;
template class QuadSurf<double>;
template class QuadSurf<float>;
template class QuadSurf<int>;
template class QuadSurf<short>;
template class QuadSurf<char>;
template class QuadSurf<unsigned int>;
template class QuadSurf<unsigned short>;
template class QuadSurf<unsigned char>;

const TypeDescription* get_type_description(QuadSurf<Tensor> *);
const TypeDescription* get_type_description(QuadSurf<Vector> *);
const TypeDescription* get_type_description(QuadSurf<double> *);
const TypeDescription* get_type_description(QuadSurf<float> *);
const TypeDescription* get_type_description(QuadSurf<int> *);
const TypeDescription* get_type_description(QuadSurf<short> *);
const TypeDescription* get_type_description(QuadSurf<char> *);
const TypeDescription* get_type_description(QuadSurf<unsigned int> *);
const TypeDescription* get_type_description(QuadSurf<unsigned short> *);
const TypeDescription* get_type_description(QuadSurf<unsigned char> *);


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
ImageField<char>::query_scalar_interface() const
{
  return scinew SFInterface<ImageField<char> >(this);
}

template <>
ScalarFieldInterface *
ImageField<unsigned int>::query_scalar_interface() const
{
  return scinew SFInterface<ImageField<unsigned int> >(this);
}

template <>
ScalarFieldInterface *
ImageField<unsigned short>::query_scalar_interface() const
{
  return scinew SFInterface<ImageField<unsigned short> >(this);
}

template <>
ScalarFieldInterface *
ImageField<unsigned char>::query_scalar_interface() const
{
  return scinew SFInterface<ImageField<unsigned char> >(this);
}




template <>
TensorFieldInterface *
QuadSurf<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<QuadSurf<Tensor> >(this);
}


template <>
VectorFieldInterface *
QuadSurf<Vector>::query_vector_interface() const
{
  return scinew VFInterface<QuadSurf<Vector> >(this);
}


template <>
ScalarFieldInterface *
QuadSurf<double>::query_scalar_interface() const
{
  return scinew SFInterface<QuadSurf<double> >(this);
}

template <>
ScalarFieldInterface *
QuadSurf<float>::query_scalar_interface() const
{
  return scinew SFInterface<QuadSurf<float> >(this);
}

template <>
ScalarFieldInterface *
QuadSurf<int>::query_scalar_interface() const
{
  return scinew SFInterface<QuadSurf<int> >(this);
}

template <>
ScalarFieldInterface *
QuadSurf<short>::query_scalar_interface() const
{
  return scinew SFInterface<QuadSurf<short> >(this);
}

template <>
ScalarFieldInterface *
QuadSurf<char>::query_scalar_interface() const
{
  return scinew SFInterface<QuadSurf<char> >(this);
}

template <>
ScalarFieldInterface *
QuadSurf<unsigned int>::query_scalar_interface() const
{
  return scinew SFInterface<QuadSurf<unsigned int> >(this);
}

template <>
ScalarFieldInterface *
QuadSurf<unsigned short>::query_scalar_interface() const
{
  return scinew SFInterface<QuadSurf<unsigned short> >(this);
}

template <>
ScalarFieldInterface *
QuadSurf<unsigned char>::query_scalar_interface() const
{
  return scinew SFInterface<QuadSurf<unsigned char> >(this);
}


