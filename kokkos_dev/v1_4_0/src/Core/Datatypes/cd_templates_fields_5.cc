#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/MaskedHexVolField.h>


using namespace SCIRun;


template class GenericField<HexVolMesh, vector<Tensor> >;
template class GenericField<HexVolMesh, vector<Vector> >;
template class GenericField<HexVolMesh, vector<double> >;
template class GenericField<HexVolMesh, vector<float> >;
template class GenericField<HexVolMesh, vector<int> >;
template class GenericField<HexVolMesh, vector<short> >;
template class GenericField<HexVolMesh, vector<char> >;
template class GenericField<HexVolMesh, vector<unsigned int> >;
template class GenericField<HexVolMesh, vector<unsigned short> >;
template class GenericField<HexVolMesh, vector<unsigned char> >;

template class HexVolField<Tensor>;
template class HexVolField<Vector>;
template class HexVolField<double>;
template class HexVolField<float>;
template class HexVolField<int>;
template class HexVolField<short>;
template class HexVolField<char>;
template class HexVolField<unsigned int>;
template class HexVolField<unsigned short>;
template class HexVolField<unsigned char>;

const TypeDescription* get_type_description(HexVolField<Tensor> *);
const TypeDescription* get_type_description(HexVolField<Vector> *);
const TypeDescription* get_type_description(HexVolField<double> *);
const TypeDescription* get_type_description(HexVolField<float> *);
const TypeDescription* get_type_description(HexVolField<int> *);
const TypeDescription* get_type_description(HexVolField<short> *);
const TypeDescription* get_type_description(HexVolField<char> *);
const TypeDescription* get_type_description(HexVolField<unsigned int> *);
const TypeDescription* get_type_description(HexVolField<unsigned short> *);
const TypeDescription* get_type_description(HexVolField<unsigned char> *);


template class MaskedHexVolField<Tensor>;
template class MaskedHexVolField<Vector>;
template class MaskedHexVolField<double>;
template class MaskedHexVolField<float>;
template class MaskedHexVolField<int>;
template class MaskedHexVolField<short>;
template class MaskedHexVolField<char>;
template class MaskedHexVolField<unsigned int>;
template class MaskedHexVolField<unsigned short>;
template class MaskedHexVolField<unsigned char>;

const TypeDescription* get_type_description(MaskedHexVolField<Tensor> *);
const TypeDescription* get_type_description(MaskedHexVolField<Vector> *);
const TypeDescription* get_type_description(MaskedHexVolField<double> *);
const TypeDescription* get_type_description(MaskedHexVolField<float> *);
const TypeDescription* get_type_description(MaskedHexVolField<int> *);
const TypeDescription* get_type_description(MaskedHexVolField<short> *);
const TypeDescription* get_type_description(MaskedHexVolField<char> *);
const TypeDescription* get_type_description(MaskedHexVolField<unsigned int> *);
const TypeDescription* get_type_description(MaskedHexVolField<unsigned short> *);
const TypeDescription* get_type_description(MaskedHexVolField<unsigned char> *);








template <>
TensorFieldInterface *
HexVolField<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<HexVolField<Tensor> >(this);
}


template <>
VectorFieldInterface *
HexVolField<Vector>::query_vector_interface() const
{
  return scinew VFInterface<HexVolField<Vector> >(this);
}


template <>
ScalarFieldInterface *
HexVolField<double>::query_scalar_interface() const
{
  return scinew SFInterface<HexVolField<double> >(this);
}

template <>
ScalarFieldInterface *
HexVolField<float>::query_scalar_interface() const
{
  return scinew SFInterface<HexVolField<float> >(this);
}

template <>
ScalarFieldInterface *
HexVolField<int>::query_scalar_interface() const
{
  return scinew SFInterface<HexVolField<int> >(this);
}

template <>
ScalarFieldInterface *
HexVolField<short>::query_scalar_interface() const
{
  return scinew SFInterface<HexVolField<short> >(this);
}

template <>
ScalarFieldInterface *
HexVolField<char>::query_scalar_interface() const
{
  return scinew SFInterface<HexVolField<char> >(this);
}

template <>
ScalarFieldInterface *
HexVolField<unsigned int>::query_scalar_interface() const
{
  return scinew SFInterface<HexVolField<unsigned int> >(this);
}

template <>
ScalarFieldInterface *
HexVolField<unsigned short>::query_scalar_interface() const
{
  return scinew SFInterface<HexVolField<unsigned short> >(this);
}

template <>
ScalarFieldInterface *
HexVolField<unsigned char>::query_scalar_interface() const
{
  return scinew SFInterface<HexVolField<unsigned char> >(this);
}
