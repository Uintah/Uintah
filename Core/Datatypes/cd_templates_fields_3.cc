#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/MaskedTetVol.h>


using namespace SCIRun;

template class GenericField<TetVolMesh, vector<Tensor> >;
template class GenericField<TetVolMesh, vector<Vector> >;
template class GenericField<TetVolMesh, vector<double> >;
template class GenericField<TetVolMesh, vector<float> >;
template class GenericField<TetVolMesh, vector<int> >;
template class GenericField<TetVolMesh, vector<short> >;
template class GenericField<TetVolMesh, vector<unsigned char> >;

template class TetVol<Tensor>;
template class TetVol<Vector>;
template class TetVol<double>;
template class TetVol<float>;
template class TetVol<int>;
template class TetVol<short>;
template class TetVol<unsigned char>;

const TypeDescription* get_type_description(TetVol<Tensor> *);
const TypeDescription* get_type_description(TetVol<Vector> *);
const TypeDescription* get_type_description(TetVol<double> *);
const TypeDescription* get_type_description(TetVol<float> *);
const TypeDescription* get_type_description(TetVol<int> *);
const TypeDescription* get_type_description(TetVol<short> *);
const TypeDescription* get_type_description(TetVol<unsigned char> *);

template class MaskedTetVol<Tensor>;
template class MaskedTetVol<Vector>;
template class MaskedTetVol<double>;
template class MaskedTetVol<float>;
template class MaskedTetVol<int>;
template class MaskedTetVol<short>;
template class MaskedTetVol<unsigned char>;

const TypeDescription* get_type_description(MaskedTetVol<Tensor> *);
const TypeDescription* get_type_description(MaskedTetVol<Vector> *);
const TypeDescription* get_type_description(MaskedTetVol<double> *);
const TypeDescription* get_type_description(MaskedTetVol<float> *);
const TypeDescription* get_type_description(MaskedTetVol<int> *);
const TypeDescription* get_type_description(MaskedTetVol<short> *);
const TypeDescription* get_type_description(MaskedTetVol<unsigned char> *);



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
TetVol<unsigned char>::query_scalar_interface() const
{
  return scinew SFInterface<TetVol<unsigned char> >(this);
}



