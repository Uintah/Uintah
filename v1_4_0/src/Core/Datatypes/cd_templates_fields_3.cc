#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/MaskedTetVolField.h>
#include <Core/Datatypes/QuadraticTetVolField.h>

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

template class TetVolField<Tensor>;
template class TetVolField<Vector>;
template class TetVolField<double>;
template class TetVolField<float>;
template class TetVolField<int>;
template class TetVolField<short>;
template class TetVolField<char>;
template class TetVolField<unsigned int>;
template class TetVolField<unsigned short>;
template class TetVolField<unsigned char>;

const TypeDescription* get_type_description(TetVolField<Tensor> *);
const TypeDescription* get_type_description(TetVolField<Vector> *);
const TypeDescription* get_type_description(TetVolField<double> *);
const TypeDescription* get_type_description(TetVolField<float> *);
const TypeDescription* get_type_description(TetVolField<int> *);
const TypeDescription* get_type_description(TetVolField<short> *);
const TypeDescription* get_type_description(TetVolField<char> *);
const TypeDescription* get_type_description(TetVolField<unsigned int> *);
const TypeDescription* get_type_description(TetVolField<unsigned short> *);
const TypeDescription* get_type_description(TetVolField<unsigned char> *);

template class MaskedTetVolField<Tensor>;
template class MaskedTetVolField<Vector>;
template class MaskedTetVolField<double>;
template class MaskedTetVolField<float>;
template class MaskedTetVolField<int>;
template class MaskedTetVolField<short>;
template class MaskedTetVolField<char>;
template class MaskedTetVolField<unsigned int>;
template class MaskedTetVolField<unsigned short>;
template class MaskedTetVolField<unsigned char>;

const TypeDescription* get_type_description(MaskedTetVolField<Tensor> *);
const TypeDescription* get_type_description(MaskedTetVolField<Vector> *);
const TypeDescription* get_type_description(MaskedTetVolField<double> *);
const TypeDescription* get_type_description(MaskedTetVolField<float> *);
const TypeDescription* get_type_description(MaskedTetVolField<int> *);
const TypeDescription* get_type_description(MaskedTetVolField<short> *);
const TypeDescription* get_type_description(MaskedTetVolField<char> *);
const TypeDescription* get_type_description(MaskedTetVolField<unsigned int> *);
const TypeDescription* get_type_description(MaskedTetVolField<unsigned short> *);
const TypeDescription* get_type_description(MaskedTetVolField<unsigned char> *);

template class QuadraticTetVolField<Tensor>;
template class QuadraticTetVolField<Vector>;
template class QuadraticTetVolField<double>;
template class QuadraticTetVolField<float>;
template class QuadraticTetVolField<int>;
template class QuadraticTetVolField<short>;
template class QuadraticTetVolField<char>;
template class QuadraticTetVolField<unsigned int>;
template class QuadraticTetVolField<unsigned short>;
template class QuadraticTetVolField<unsigned char>;

const TypeDescription* get_type_description(QuadraticTetVolField<Tensor> *);
const TypeDescription* get_type_description(QuadraticTetVolField<Vector> *);
const TypeDescription* get_type_description(QuadraticTetVolField<double> *);
const TypeDescription* get_type_description(QuadraticTetVolField<float> *);
const TypeDescription* get_type_description(QuadraticTetVolField<int> *);
const TypeDescription* get_type_description(QuadraticTetVolField<short> *);
const TypeDescription* get_type_description(QuadraticTetVolField<char> *);
const TypeDescription* get_type_description(QuadraticTetVolField<unsigned int> *);
const TypeDescription* get_type_description(QuadraticTetVolField<unsigned short> *);
const TypeDescription* get_type_description(QuadraticTetVolField<unsigned char> *);


template <>
TensorFieldInterface *
TetVolField<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<TetVolField<Tensor> >(this);
}


template <>
VectorFieldInterface *
TetVolField<Vector>::query_vector_interface() const
{
  return scinew VFInterface<TetVolField<Vector> >(this);
}


template <>
ScalarFieldInterface *
TetVolField<double>::query_scalar_interface() const
{
  return scinew SFInterface<TetVolField<double> >(this);
}

template <>
ScalarFieldInterface *
TetVolField<float>::query_scalar_interface() const
{
  return scinew SFInterface<TetVolField<float> >(this);
}

template <>
ScalarFieldInterface *
TetVolField<int>::query_scalar_interface() const
{
  return scinew SFInterface<TetVolField<int> >(this);
}

template <>
ScalarFieldInterface *
TetVolField<short>::query_scalar_interface() const
{
  return scinew SFInterface<TetVolField<short> >(this);
}

template <>
ScalarFieldInterface *
TetVolField<char>::query_scalar_interface() const
{
  return scinew SFInterface<TetVolField<char> >(this);
}

template <>
ScalarFieldInterface *
TetVolField<unsigned int>::query_scalar_interface() const
{
  return scinew SFInterface<TetVolField<unsigned int> >(this);
}

template <>
ScalarFieldInterface *
TetVolField<unsigned short>::query_scalar_interface() const
{
  return scinew SFInterface<TetVolField<unsigned short> >(this);
}

template <>
ScalarFieldInterface *
TetVolField<unsigned char>::query_scalar_interface() const
{
  return scinew SFInterface<TetVolField<unsigned char> >(this);
}

//-------

template <>
TensorFieldInterface *
QuadraticTetVolField<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<QuadraticTetVolField<Tensor> >(this);
}


template <>
VectorFieldInterface *
QuadraticTetVolField<Vector>::query_vector_interface() const
{
  return scinew VFInterface<QuadraticTetVolField<Vector> >(this);
}


template <>
ScalarFieldInterface *
QuadraticTetVolField<double>::query_scalar_interface() const
{
  return scinew SFInterface<QuadraticTetVolField<double> >(this);
}

template <>
ScalarFieldInterface *
QuadraticTetVolField<float>::query_scalar_interface() const
{
  return scinew SFInterface<QuadraticTetVolField<float> >(this);
}

template <>
ScalarFieldInterface *
QuadraticTetVolField<int>::query_scalar_interface() const
{
  return scinew SFInterface<QuadraticTetVolField<int> >(this);
}

template <>
ScalarFieldInterface *
QuadraticTetVolField<short>::query_scalar_interface() const
{
  return scinew SFInterface<QuadraticTetVolField<short> >(this);
}

template <>
ScalarFieldInterface *
QuadraticTetVolField<char>::query_scalar_interface() const
{
  return scinew SFInterface<QuadraticTetVolField<char> >(this);
}

template <>
ScalarFieldInterface *
QuadraticTetVolField<unsigned int>::query_scalar_interface() const
{
  return scinew SFInterface<QuadraticTetVolField<unsigned int> >(this);
}

template <>
ScalarFieldInterface *
QuadraticTetVolField<unsigned short>::query_scalar_interface() const
{
  return scinew SFInterface<QuadraticTetVolField<unsigned short> >(this);
}

template <>
ScalarFieldInterface *
QuadraticTetVolField<unsigned char>::query_scalar_interface() const
{
  return scinew SFInterface<QuadraticTetVolField<unsigned char> >(this);
}


