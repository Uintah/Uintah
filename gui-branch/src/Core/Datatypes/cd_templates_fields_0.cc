#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/MaskedLatticeVol.h>


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

template class LatticeVol<Tensor>;
template class LatticeVol<Vector>;
template class LatticeVol<double>;
template class LatticeVol<float>;
template class LatticeVol<int>;
template class LatticeVol<short>;
template class LatticeVol<char>;
template class LatticeVol<unsigned int>;
template class LatticeVol<unsigned short>;
template class LatticeVol<unsigned char>;

const TypeDescription* get_type_description(LatticeVol<Tensor> *);
const TypeDescription* get_type_description(LatticeVol<Vector> *);
const TypeDescription* get_type_description(LatticeVol<double> *);
const TypeDescription* get_type_description(LatticeVol<float> *);
const TypeDescription* get_type_description(LatticeVol<int> *);
const TypeDescription* get_type_description(LatticeVol<short> *);
const TypeDescription* get_type_description(LatticeVol<char> *);
const TypeDescription* get_type_description(LatticeVol<unsigned int> *);
const TypeDescription* get_type_description(LatticeVol<unsigned short> *);
const TypeDescription* get_type_description(LatticeVol<unsigned char> *);

template class MaskedLatticeVol<Tensor>;
template class MaskedLatticeVol<Vector>;
template class MaskedLatticeVol<double>;
template class MaskedLatticeVol<float>;
template class MaskedLatticeVol<int>;
template class MaskedLatticeVol<short>;
template class MaskedLatticeVol<char>;
template class MaskedLatticeVol<unsigned int>;
template class MaskedLatticeVol<unsigned short>;
template class MaskedLatticeVol<unsigned char>;

const TypeDescription* get_type_description(MaskedLatticeVol<Tensor> *);
const TypeDescription* get_type_description(MaskedLatticeVol<Vector> *);
const TypeDescription* get_type_description(MaskedLatticeVol<double> *);
const TypeDescription* get_type_description(MaskedLatticeVol<float> *);
const TypeDescription* get_type_description(MaskedLatticeVol<int> *);
const TypeDescription* get_type_description(MaskedLatticeVol<short> *);
const TypeDescription* get_type_description(MaskedLatticeVol<char> *);
const TypeDescription* get_type_description(MaskedLatticeVol<unsigned int> *);
const TypeDescription* get_type_description(MaskedLatticeVol<unsigned short> *);
const TypeDescription* get_type_description(MaskedLatticeVol<unsigned char> *);


template <>
TensorFieldInterface *
LatticeVol<Tensor>::query_tensor_interface() const
{
  return scinew TFInterface<LatticeVol<Tensor> >(this);
}


template <>
VectorFieldInterface *
LatticeVol<Vector>::query_vector_interface() const
{
  return scinew VFInterface<LatticeVol<Vector> >(this);
}


template <>
ScalarFieldInterface *
LatticeVol<double>::query_scalar_interface() const
{
  return scinew SFInterface<LatticeVol<double> >(this);
}

template <>
ScalarFieldInterface *
LatticeVol<float>::query_scalar_interface() const
{
  return scinew SFInterface<LatticeVol<float> >(this);
}

template <>
ScalarFieldInterface *
LatticeVol<int>::query_scalar_interface() const
{
  return scinew SFInterface<LatticeVol<int> >(this);
}

template <>
ScalarFieldInterface *
LatticeVol<short>::query_scalar_interface() const
{
  return scinew SFInterface<LatticeVol<short> >(this);
}

template <>
ScalarFieldInterface *
LatticeVol<char>::query_scalar_interface() const
{
  return scinew SFInterface<LatticeVol<char> >(this);
}

template <>
ScalarFieldInterface *
LatticeVol<unsigned int>::query_scalar_interface() const
{
  return scinew SFInterface<LatticeVol<unsigned int> >(this);
}

template <>
ScalarFieldInterface *
LatticeVol<unsigned short>::query_scalar_interface() const
{
  return scinew SFInterface<LatticeVol<unsigned short> >(this);
}

template <>
ScalarFieldInterface *
LatticeVol<unsigned char>::query_scalar_interface() const
{
  return scinew SFInterface<LatticeVol<unsigned char> >(this);
}
