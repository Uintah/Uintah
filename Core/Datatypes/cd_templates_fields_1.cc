#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/ImageField.h>
#include <Core/Datatypes/QuadSurfField.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
/*
cc-1468 CC: REMARK File = ../src/Core/Datatypes/cd_templates_fields_0.cc, Line = 11
  Inline function "SCIRun::FData3d<SCIRun::Tensor>::end" cannot be explicitly
          instantiated.
*/
#pragma set woff 1468
#endif

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

template class QuadSurfField<Tensor>;
template class QuadSurfField<Vector>;
template class QuadSurfField<double>;
template class QuadSurfField<float>;
template class QuadSurfField<int>;
template class QuadSurfField<short>;
template class QuadSurfField<char>;
template class QuadSurfField<unsigned int>;
template class QuadSurfField<unsigned short>;
template class QuadSurfField<unsigned char>;

const TypeDescription* get_type_description(QuadSurfField<Tensor> *);
const TypeDescription* get_type_description(QuadSurfField<Vector> *);
const TypeDescription* get_type_description(QuadSurfField<double> *);
const TypeDescription* get_type_description(QuadSurfField<float> *);
const TypeDescription* get_type_description(QuadSurfField<int> *);
const TypeDescription* get_type_description(QuadSurfField<short> *);
const TypeDescription* get_type_description(QuadSurfField<char> *);
const TypeDescription* get_type_description(QuadSurfField<unsigned int> *);
const TypeDescription* get_type_description(QuadSurfField<unsigned short> *);
const TypeDescription* get_type_description(QuadSurfField<unsigned char> *);


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
QuadSurfField<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<QuadSurfField<Tensor> >(this);
}


template <>
VectorFieldInterface *
QuadSurfField<Vector>::query_vector_interface() const
{
  return scinew VFInterface<QuadSurfField<Vector> >(this);
}


template <>
ScalarFieldInterface *
QuadSurfField<double>::query_scalar_interface() const
{
  return scinew SFInterface<QuadSurfField<double> >(this);
}

template <>
ScalarFieldInterface *
QuadSurfField<float>::query_scalar_interface() const
{
  return scinew SFInterface<QuadSurfField<float> >(this);
}

template <>
ScalarFieldInterface *
QuadSurfField<int>::query_scalar_interface() const
{
  return scinew SFInterface<QuadSurfField<int> >(this);
}

template <>
ScalarFieldInterface *
QuadSurfField<short>::query_scalar_interface() const
{
  return scinew SFInterface<QuadSurfField<short> >(this);
}

template <>
ScalarFieldInterface *
QuadSurfField<char>::query_scalar_interface() const
{
  return scinew SFInterface<QuadSurfField<char> >(this);
}

template <>
ScalarFieldInterface *
QuadSurfField<unsigned int>::query_scalar_interface() const
{
  return scinew SFInterface<QuadSurfField<unsigned int> >(this);
}

template <>
ScalarFieldInterface *
QuadSurfField<unsigned short>::query_scalar_interface() const
{
  return scinew SFInterface<QuadSurfField<unsigned short> >(this);
}

template <>
ScalarFieldInterface *
QuadSurfField<unsigned char>::query_scalar_interface() const
{
  return scinew SFInterface<QuadSurfField<unsigned char> >(this);
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif

