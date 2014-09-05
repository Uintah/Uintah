#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/HexVol.h>
#include <Core/Datatypes/MaskedHexVol.h>


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

template class HexVol<Tensor>;
template class HexVol<Vector>;
template class HexVol<double>;
template class HexVol<float>;
template class HexVol<int>;
template class HexVol<short>;
template class HexVol<char>;
template class HexVol<unsigned int>;
template class HexVol<unsigned short>;
template class HexVol<unsigned char>;

const TypeDescription* get_type_description(HexVol<Tensor> *);
const TypeDescription* get_type_description(HexVol<Vector> *);
const TypeDescription* get_type_description(HexVol<double> *);
const TypeDescription* get_type_description(HexVol<float> *);
const TypeDescription* get_type_description(HexVol<int> *);
const TypeDescription* get_type_description(HexVol<short> *);
const TypeDescription* get_type_description(HexVol<char> *);
const TypeDescription* get_type_description(HexVol<unsigned int> *);
const TypeDescription* get_type_description(HexVol<unsigned short> *);
const TypeDescription* get_type_description(HexVol<unsigned char> *);


template class MaskedHexVol<Tensor>;
template class MaskedHexVol<Vector>;
template class MaskedHexVol<double>;
template class MaskedHexVol<float>;
template class MaskedHexVol<int>;
template class MaskedHexVol<short>;
template class MaskedHexVol<char>;
template class MaskedHexVol<unsigned int>;
template class MaskedHexVol<unsigned short>;
template class MaskedHexVol<unsigned char>;

const TypeDescription* get_type_description(MaskedHexVol<Tensor> *);
const TypeDescription* get_type_description(MaskedHexVol<Vector> *);
const TypeDescription* get_type_description(MaskedHexVol<double> *);
const TypeDescription* get_type_description(MaskedHexVol<float> *);
const TypeDescription* get_type_description(MaskedHexVol<int> *);
const TypeDescription* get_type_description(MaskedHexVol<short> *);
const TypeDescription* get_type_description(MaskedHexVol<char> *);
const TypeDescription* get_type_description(MaskedHexVol<unsigned int> *);
const TypeDescription* get_type_description(MaskedHexVol<unsigned short> *);
const TypeDescription* get_type_description(MaskedHexVol<unsigned char> *);








template <>
TensorFieldInterface *
HexVol<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<HexVol<Tensor> >(this);
}


template <>
VectorFieldInterface *
HexVol<Vector>::query_vector_interface() const
{
  return scinew VFInterface<HexVol<Vector> >(this);
}


template <>
ScalarFieldInterface *
HexVol<double>::query_scalar_interface() const
{
  return scinew SFInterface<HexVol<double> >(this);
}

template <>
ScalarFieldInterface *
HexVol<float>::query_scalar_interface() const
{
  return scinew SFInterface<HexVol<float> >(this);
}

template <>
ScalarFieldInterface *
HexVol<int>::query_scalar_interface() const
{
  return scinew SFInterface<HexVol<int> >(this);
}

template <>
ScalarFieldInterface *
HexVol<short>::query_scalar_interface() const
{
  return scinew SFInterface<HexVol<short> >(this);
}

template <>
ScalarFieldInterface *
HexVol<char>::query_scalar_interface() const
{
  return scinew SFInterface<HexVol<char> >(this);
}

template <>
ScalarFieldInterface *
HexVol<unsigned int>::query_scalar_interface() const
{
  return scinew SFInterface<HexVol<unsigned int> >(this);
}

template <>
ScalarFieldInterface *
HexVol<unsigned short>::query_scalar_interface() const
{
  return scinew SFInterface<HexVol<unsigned short> >(this);
}

template <>
ScalarFieldInterface *
HexVol<unsigned char>::query_scalar_interface() const
{
  return scinew SFInterface<HexVol<unsigned char> >(this);
}
