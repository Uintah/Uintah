#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/Datatypes/ContourField.h>
#include <Core/Datatypes/MaskedTriSurf.h>


using namespace SCIRun;

template class GenericField<TriSurfMesh, vector<Tensor> >;
template class GenericField<TriSurfMesh, vector<Vector> >;
template class GenericField<TriSurfMesh, vector<double> >;
template class GenericField<TriSurfMesh, vector<float> >;
template class GenericField<TriSurfMesh, vector<int> >;
template class GenericField<TriSurfMesh, vector<short> >;
template class GenericField<TriSurfMesh, vector<unsigned char> >;

template class GenericField<ContourMesh, vector<Tensor> >;
template class GenericField<ContourMesh, vector<Vector> >;
template class GenericField<ContourMesh, vector<double> >;
template class GenericField<ContourMesh, vector<float> >;
template class GenericField<ContourMesh, vector<int> >;
template class GenericField<ContourMesh, vector<short> >;
template class GenericField<ContourMesh, vector<unsigned char> >;

template class TriSurf<Tensor>;
template class TriSurf<Vector>;
template class TriSurf<double>;
template class TriSurf<float>;
template class TriSurf<int>;
template class TriSurf<short>;
template class TriSurf<unsigned char>;

const TypeDescription* get_type_description(TriSurf<Tensor> *);
const TypeDescription* get_type_description(TriSurf<Vector> *);
const TypeDescription* get_type_description(TriSurf<double> *);
const TypeDescription* get_type_description(TriSurf<float> *);
const TypeDescription* get_type_description(TriSurf<int> *);
const TypeDescription* get_type_description(TriSurf<short> *);
const TypeDescription* get_type_description(TriSurf<unsigned char> *);

template class ContourField<Tensor>;
template class ContourField<Vector>;
template class ContourField<double>;
template class ContourField<float>;
template class ContourField<int>;
template class ContourField<short>;
template class ContourField<unsigned char>;

const TypeDescription* get_type_description(ContourField<Tensor> *);
const TypeDescription* get_type_description(ContourField<Vector> *);
const TypeDescription* get_type_description(ContourField<double> *);
const TypeDescription* get_type_description(ContourField<float> *);
const TypeDescription* get_type_description(ContourField<int> *);
const TypeDescription* get_type_description(ContourField<short> *);
const TypeDescription* get_type_description(ContourField<unsigned char> *);

template class MaskedTriSurf<Tensor>;
template class MaskedTriSurf<Vector>;
template class MaskedTriSurf<double>;
template class MaskedTriSurf<float>;
template class MaskedTriSurf<int>;
template class MaskedTriSurf<short>;
template class MaskedTriSurf<unsigned char>;

const TypeDescription* get_type_description(MaskedTriSurf<Tensor> *);
const TypeDescription* get_type_description(MaskedTriSurf<Vector> *);
const TypeDescription* get_type_description(MaskedTriSurf<double> *);
const TypeDescription* get_type_description(MaskedTriSurf<float> *);
const TypeDescription* get_type_description(MaskedTriSurf<int> *);
const TypeDescription* get_type_description(MaskedTriSurf<short> *);
const TypeDescription* get_type_description(MaskedTriSurf<unsigned char> *);


template <>
TensorFieldInterface *
TriSurf<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<TriSurf<Tensor> >(this);
}


template <>
VectorFieldInterface *
TriSurf<Vector>::query_vector_interface() const
{
  return scinew VFInterface<TriSurf<Vector> >(this);
}


template <>
ScalarFieldInterface *
TriSurf<double>::query_scalar_interface() const
{
  return scinew SFInterface<TriSurf<double> >(this);
}

template <>
ScalarFieldInterface *
TriSurf<float>::query_scalar_interface() const
{
  return scinew SFInterface<TriSurf<float> >(this);
}

template <>
ScalarFieldInterface *
TriSurf<int>::query_scalar_interface() const
{
  return scinew SFInterface<TriSurf<int> >(this);
}

template <>
ScalarFieldInterface *
TriSurf<short>::query_scalar_interface() const
{
  return scinew SFInterface<TriSurf<short> >(this);
}

template <>
ScalarFieldInterface *
TriSurf<unsigned char>::query_scalar_interface() const
{
  return scinew SFInterface<TriSurf<unsigned char> >(this);
}


template <>
TensorFieldInterface *
ContourField<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<ContourField<Tensor> >(this);
}


template <>
VectorFieldInterface *
ContourField<Vector>::query_vector_interface() const
{
  return scinew VFInterface<ContourField<Vector> >(this);
}


template <>
ScalarFieldInterface *
ContourField<double>::query_scalar_interface() const
{
  return scinew SFInterface<ContourField<double> >(this);
}

template <>
ScalarFieldInterface *
ContourField<float>::query_scalar_interface() const
{
  return scinew SFInterface<ContourField<float> >(this);
}

template <>
ScalarFieldInterface *
ContourField<int>::query_scalar_interface() const
{
  return scinew SFInterface<ContourField<int> >(this);
}

template <>
ScalarFieldInterface *
ContourField<short>::query_scalar_interface() const
{
  return scinew SFInterface<ContourField<short> >(this);
}

template <>
ScalarFieldInterface *
ContourField<unsigned char>::query_scalar_interface() const
{
  return scinew SFInterface<ContourField<unsigned char> >(this);
}
