#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/MaskedTetVol.h>
#include <Core/Datatypes/QuadraticTetVol.h>

using namespace SCIRun;

template class GenericField<TetVolMesh, vector<Tensor> >;
template class GenericField<TetVolMesh, vector<Vector> >;
template class GenericField<TetVolMesh, vector<double> >;
template class GenericField<TetVolMesh, vector<float> >;
template class GenericField<TetVolMesh, vector<int> >;
template class GenericField<TetVolMesh, vector<short> >;
template class GenericField<TetVolMesh, vector<char> >;
template class GenericField<TetVolMesh, vector<unsigned int> >;
template class GenericField<TetVolMesh, vector<unsigned short> >;
template class GenericField<TetVolMesh, vector<unsigned char> >;

template class TetVol<Tensor>;
template class TetVol<Vector>;
template class TetVol<double>;
template class TetVol<float>;
template class TetVol<int>;
template class TetVol<short>;
template class TetVol<char>;
template class TetVol<unsigned int>;
template class TetVol<unsigned short>;
template class TetVol<unsigned char>;

const TypeDescription* get_type_description(TetVol<Tensor> *);
const TypeDescription* get_type_description(TetVol<Vector> *);
const TypeDescription* get_type_description(TetVol<double> *);
const TypeDescription* get_type_description(TetVol<float> *);
const TypeDescription* get_type_description(TetVol<int> *);
const TypeDescription* get_type_description(TetVol<short> *);
const TypeDescription* get_type_description(TetVol<char> *);
const TypeDescription* get_type_description(TetVol<unsigned int> *);
const TypeDescription* get_type_description(TetVol<unsigned short> *);
const TypeDescription* get_type_description(TetVol<unsigned char> *);

template class MaskedTetVol<Tensor>;
template class MaskedTetVol<Vector>;
template class MaskedTetVol<double>;
template class MaskedTetVol<float>;
template class MaskedTetVol<int>;
template class MaskedTetVol<short>;
template class MaskedTetVol<char>;
template class MaskedTetVol<unsigned int>;
template class MaskedTetVol<unsigned short>;
template class MaskedTetVol<unsigned char>;

const TypeDescription* get_type_description(MaskedTetVol<Tensor> *);
const TypeDescription* get_type_description(MaskedTetVol<Vector> *);
const TypeDescription* get_type_description(MaskedTetVol<double> *);
const TypeDescription* get_type_description(MaskedTetVol<float> *);
const TypeDescription* get_type_description(MaskedTetVol<int> *);
const TypeDescription* get_type_description(MaskedTetVol<short> *);
const TypeDescription* get_type_description(MaskedTetVol<char> *);
const TypeDescription* get_type_description(MaskedTetVol<unsigned int> *);
const TypeDescription* get_type_description(MaskedTetVol<unsigned short> *);
const TypeDescription* get_type_description(MaskedTetVol<unsigned char> *);

template class QuadraticTetVol<Tensor>;
template class QuadraticTetVol<Vector>;
template class QuadraticTetVol<double>;
template class QuadraticTetVol<float>;
template class QuadraticTetVol<int>;
template class QuadraticTetVol<short>;
template class QuadraticTetVol<char>;
template class QuadraticTetVol<unsigned int>;
template class QuadraticTetVol<unsigned short>;
template class QuadraticTetVol<unsigned char>;

const TypeDescription* get_type_description(QuadraticTetVol<Tensor> *);
const TypeDescription* get_type_description(QuadraticTetVol<Vector> *);
const TypeDescription* get_type_description(QuadraticTetVol<double> *);
const TypeDescription* get_type_description(QuadraticTetVol<float> *);
const TypeDescription* get_type_description(QuadraticTetVol<int> *);
const TypeDescription* get_type_description(QuadraticTetVol<short> *);
const TypeDescription* get_type_description(QuadraticTetVol<char> *);
const TypeDescription* get_type_description(QuadraticTetVol<unsigned int> *);
const TypeDescription* get_type_description(QuadraticTetVol<unsigned short> *);
const TypeDescription* get_type_description(QuadraticTetVol<unsigned char> *);


template <>
TensorFieldInterface *
TetVol<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<TetVol<Tensor> >(this);
}


template <>
VectorFieldInterface *
TetVol<Vector>::query_vector_interface() const
{
  return scinew VFInterface<TetVol<Vector> >(this);
}


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
TetVol<char>::query_scalar_interface() const
{
  return scinew SFInterface<TetVol<char> >(this);
}

template <>
ScalarFieldInterface *
TetVol<unsigned int>::query_scalar_interface() const
{
  return scinew SFInterface<TetVol<unsigned int> >(this);
}

template <>
ScalarFieldInterface *
TetVol<unsigned short>::query_scalar_interface() const
{
  return scinew SFInterface<TetVol<unsigned short> >(this);
}

template <>
ScalarFieldInterface *
TetVol<unsigned char>::query_scalar_interface() const
{
  return scinew SFInterface<TetVol<unsigned char> >(this);
}

//-------

template <>
TensorFieldInterface *
QuadraticTetVol<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<QuadraticTetVol<Tensor> >(this);
}


template <>
VectorFieldInterface *
QuadraticTetVol<Vector>::query_vector_interface() const
{
  return scinew VFInterface<QuadraticTetVol<Vector> >(this);
}


template <>
ScalarFieldInterface *
QuadraticTetVol<double>::query_scalar_interface() const
{
  return scinew SFInterface<QuadraticTetVol<double> >(this);
}

template <>
ScalarFieldInterface *
QuadraticTetVol<float>::query_scalar_interface() const
{
  return scinew SFInterface<QuadraticTetVol<float> >(this);
}

template <>
ScalarFieldInterface *
QuadraticTetVol<int>::query_scalar_interface() const
{
  return scinew SFInterface<QuadraticTetVol<int> >(this);
}

template <>
ScalarFieldInterface *
QuadraticTetVol<short>::query_scalar_interface() const
{
  return scinew SFInterface<QuadraticTetVol<short> >(this);
}

template <>
ScalarFieldInterface *
QuadraticTetVol<char>::query_scalar_interface() const
{
  return scinew SFInterface<QuadraticTetVol<char> >(this);
}

template <>
ScalarFieldInterface *
QuadraticTetVol<unsigned int>::query_scalar_interface() const
{
  return scinew SFInterface<QuadraticTetVol<unsigned int> >(this);
}

template <>
ScalarFieldInterface *
QuadraticTetVol<unsigned short>::query_scalar_interface() const
{
  return scinew SFInterface<QuadraticTetVol<unsigned short> >(this);
}

template <>
ScalarFieldInterface *
QuadraticTetVol<unsigned char>::query_scalar_interface() const
{
  return scinew SFInterface<QuadraticTetVol<unsigned char> >(this);
}


