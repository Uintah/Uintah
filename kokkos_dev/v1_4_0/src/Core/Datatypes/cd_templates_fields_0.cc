#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/MaskedLatVolField.h>


using namespace SCIRun;

template class FData3d<Tensor>;
template class FData3d<Vector>;
template class FData3d<double>;
template class FData3d<float>;
template class FData3d<int>;
template class FData3d<short>;
template class FData3d<char>;
template class FData3d<unsigned int>;
template class FData3d<unsigned short>;
template class FData3d<unsigned char>;

template class GenericField<LatVolMesh, FData3d<Tensor> >;
template class GenericField<LatVolMesh, FData3d<Vector> >;
template class GenericField<LatVolMesh, FData3d<double> >;
template class GenericField<LatVolMesh, FData3d<float> >;
template class GenericField<LatVolMesh, FData3d<int> >;
template class GenericField<LatVolMesh, FData3d<short> >;
template class GenericField<LatVolMesh, FData3d<char> >;
template class GenericField<LatVolMesh, FData3d<unsigned int> >;
template class GenericField<LatVolMesh, FData3d<unsigned short> >;
template class GenericField<LatVolMesh, FData3d<unsigned char> >;

template class LatVolField<Tensor>;
template class LatVolField<Vector>;
template class LatVolField<double>;
template class LatVolField<float>;
template class LatVolField<int>;
template class LatVolField<short>;
template class LatVolField<char>;
template class LatVolField<unsigned int>;
template class LatVolField<unsigned short>;
template class LatVolField<unsigned char>;

const TypeDescription* get_type_description(LatVolField<Tensor> *);
const TypeDescription* get_type_description(LatVolField<Vector> *);
const TypeDescription* get_type_description(LatVolField<double> *);
const TypeDescription* get_type_description(LatVolField<float> *);
const TypeDescription* get_type_description(LatVolField<int> *);
const TypeDescription* get_type_description(LatVolField<short> *);
const TypeDescription* get_type_description(LatVolField<char> *);
const TypeDescription* get_type_description(LatVolField<unsigned int> *);
const TypeDescription* get_type_description(LatVolField<unsigned short> *);
const TypeDescription* get_type_description(LatVolField<unsigned char> *);

template class MaskedLatVolField<Tensor>;
template class MaskedLatVolField<Vector>;
template class MaskedLatVolField<double>;
template class MaskedLatVolField<float>;
template class MaskedLatVolField<int>;
template class MaskedLatVolField<short>;
template class MaskedLatVolField<char>;
template class MaskedLatVolField<unsigned int>;
template class MaskedLatVolField<unsigned short>;
template class MaskedLatVolField<unsigned char>;

const TypeDescription* get_type_description(MaskedLatVolField<Tensor> *);
const TypeDescription* get_type_description(MaskedLatVolField<Vector> *);
const TypeDescription* get_type_description(MaskedLatVolField<double> *);
const TypeDescription* get_type_description(MaskedLatVolField<float> *);
const TypeDescription* get_type_description(MaskedLatVolField<int> *);
const TypeDescription* get_type_description(MaskedLatVolField<short> *);
const TypeDescription* get_type_description(MaskedLatVolField<char> *);
const TypeDescription* get_type_description(MaskedLatVolField<unsigned int> *);
const TypeDescription* get_type_description(MaskedLatVolField<unsigned short> *);
const TypeDescription* get_type_description(MaskedLatVolField<unsigned char> *);


template <>
TensorFieldInterface *
LatVolField<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<LatVolField<Tensor> >(this);
}


template <>
VectorFieldInterface *
LatVolField<Vector>::query_vector_interface() const
{
  return scinew VFInterface<LatVolField<Vector> >(this);
}


template <>
ScalarFieldInterface *
LatVolField<double>::query_scalar_interface() const
{
  return scinew SFInterface<LatVolField<double> >(this);
}

template <>
ScalarFieldInterface *
LatVolField<float>::query_scalar_interface() const
{
  return scinew SFInterface<LatVolField<float> >(this);
}

template <>
ScalarFieldInterface *
LatVolField<int>::query_scalar_interface() const
{
  return scinew SFInterface<LatVolField<int> >(this);
}

template <>
ScalarFieldInterface *
LatVolField<short>::query_scalar_interface() const
{
  return scinew SFInterface<LatVolField<short> >(this);
}

template <>
ScalarFieldInterface *
LatVolField<char>::query_scalar_interface() const
{
  return scinew SFInterface<LatVolField<char> >(this);
}

template <>
ScalarFieldInterface *
LatVolField<unsigned int>::query_scalar_interface() const
{
  return scinew SFInterface<LatVolField<unsigned int> >(this);
}

template <>
ScalarFieldInterface *
LatVolField<unsigned short>::query_scalar_interface() const
{
  return scinew SFInterface<LatVolField<unsigned short> >(this);
}

template <>
ScalarFieldInterface *
LatVolField<unsigned char>::query_scalar_interface() const
{
  return scinew SFInterface<LatVolField<unsigned char> >(this);
}
