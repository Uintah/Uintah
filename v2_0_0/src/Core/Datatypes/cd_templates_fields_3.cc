#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/PrismVolField.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/MaskedTetVolField.h>
#include <Core/Datatypes/QuadraticTetVolField.h>
#include <Core/Datatypes/QuadraticLatVolField.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
/*
cc-1468 CC: REMARK File = ../src/Core/Datatypes/cd_templates_fields_0.cc, Line = 11
  Inline function "SCIRun::FData3d<SCIRun::Tensor>::end" cannot be explicitly
          instantiated.
*/
#pragma set woff 1468
#endif

using namespace SCIRun;

template class GenericField<PrismVolMesh, vector<Tensor> >;
template class GenericField<PrismVolMesh, vector<Vector> >;
template class GenericField<PrismVolMesh, vector<double> >;
template class GenericField<PrismVolMesh, vector<float> >;
template class GenericField<PrismVolMesh, vector<int> >;
template class GenericField<PrismVolMesh, vector<short> >;
template class GenericField<PrismVolMesh, vector<char> >;
template class GenericField<PrismVolMesh, vector<unsigned int> >;
template class GenericField<PrismVolMesh, vector<unsigned short> >;
template class GenericField<PrismVolMesh, vector<unsigned char> >;

template class PrismVolField<Tensor>;
template class PrismVolField<Vector>;
template class PrismVolField<double>;
template class PrismVolField<float>;
template class PrismVolField<int>;
template class PrismVolField<short>;
template class PrismVolField<char>;
template class PrismVolField<unsigned int>;
template class PrismVolField<unsigned short>;
template class PrismVolField<unsigned char>;

const TypeDescription* get_type_description(PrismVolField<Tensor> *);
const TypeDescription* get_type_description(PrismVolField<Vector> *);
const TypeDescription* get_type_description(PrismVolField<double> *);
const TypeDescription* get_type_description(PrismVolField<float> *);
const TypeDescription* get_type_description(PrismVolField<int> *);
const TypeDescription* get_type_description(PrismVolField<short> *);
const TypeDescription* get_type_description(PrismVolField<char> *);
const TypeDescription* get_type_description(PrismVolField<unsigned int> *);
const TypeDescription* get_type_description(PrismVolField<unsigned short> *);
const TypeDescription* get_type_description(PrismVolField<unsigned char> *);

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

template class GenericField<QuadraticTetVolMesh, vector<Tensor> >;
template class GenericField<QuadraticTetVolMesh, vector<Vector> >;
template class GenericField<QuadraticTetVolMesh, vector<double> >;
template class GenericField<QuadraticTetVolMesh, vector<float> >;
template class GenericField<QuadraticTetVolMesh, vector<int> >;
template class GenericField<QuadraticTetVolMesh, vector<short> >;
template class GenericField<QuadraticTetVolMesh, vector<char> >;
template class GenericField<QuadraticTetVolMesh, vector<unsigned int> >;
template class GenericField<QuadraticTetVolMesh, vector<unsigned short> >;
template class GenericField<QuadraticTetVolMesh, vector<unsigned char> >;

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

template class GenericField<QuadraticLatVolMesh, vector<Tensor> >;
template class GenericField<QuadraticLatVolMesh, vector<Vector> >;
template class GenericField<QuadraticLatVolMesh, vector<double> >;
template class GenericField<QuadraticLatVolMesh, vector<float> >;
template class GenericField<QuadraticLatVolMesh, vector<int> >;
template class GenericField<QuadraticLatVolMesh, vector<short> >;
template class GenericField<QuadraticLatVolMesh, vector<char> >;
template class GenericField<QuadraticLatVolMesh, vector<unsigned int> >;
template class GenericField<QuadraticLatVolMesh, vector<unsigned short> >;
template class GenericField<QuadraticLatVolMesh, vector<unsigned char> >;

template class QuadraticLatVolField<Tensor>;
template class QuadraticLatVolField<Vector>;
template class QuadraticLatVolField<double>;
template class QuadraticLatVolField<float>;
template class QuadraticLatVolField<int>;
template class QuadraticLatVolField<short>;
template class QuadraticLatVolField<char>;
template class QuadraticLatVolField<unsigned int>;
template class QuadraticLatVolField<unsigned short>;
template class QuadraticLatVolField<unsigned char>;

const TypeDescription* get_type_description(QuadraticLatVolField<Tensor> *);
const TypeDescription* get_type_description(QuadraticLatVolField<Vector> *);
const TypeDescription* get_type_description(QuadraticLatVolField<double> *);
const TypeDescription* get_type_description(QuadraticLatVolField<float> *);
const TypeDescription* get_type_description(QuadraticLatVolField<int> *);
const TypeDescription* get_type_description(QuadraticLatVolField<short> *);
const TypeDescription* get_type_description(QuadraticLatVolField<char> *);
const TypeDescription* get_type_description(QuadraticLatVolField<unsigned int> *);
const TypeDescription* get_type_description(QuadraticLatVolField<unsigned short> *);
const TypeDescription* get_type_description(QuadraticLatVolField<unsigned char> *);

const TypeDescription* get_type_description(QuadraticLatVolMesh::Node::index_type *);

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif
