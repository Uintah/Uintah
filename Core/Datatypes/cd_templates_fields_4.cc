#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/CurveField.h>
#include <Core/Datatypes/MaskedTriSurfField.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
/*
cc-1468 CC: REMARK File = ../src/Core/Datatypes/cd_templates_fields_0.cc, Line = 11
  Inline function "SCIRun::FData3d<SCIRun::Tensor>::end" cannot be explicitly
          instantiated.
*/
#pragma set woff 1468
#endif

using namespace SCIRun;

template class GenericField<TriSurfMesh, vector<Tensor> >;
template class GenericField<TriSurfMesh, vector<Vector> >;
template class GenericField<TriSurfMesh, vector<double> >;
template class GenericField<TriSurfMesh, vector<float> >;
template class GenericField<TriSurfMesh, vector<int> >;
template class GenericField<TriSurfMesh, vector<short> >;
template class GenericField<TriSurfMesh, vector<char> >;
template class GenericField<TriSurfMesh, vector<unsigned int> >;
template class GenericField<TriSurfMesh, vector<unsigned short> >;
template class GenericField<TriSurfMesh, vector<unsigned char> >;
template class GenericField<TriSurfMesh, vector<vector<pair<NodeIndex<int>,double> > > >;

template class GenericField<CurveMesh, vector<Tensor> >;
template class GenericField<CurveMesh, vector<Vector> >;
template class GenericField<CurveMesh, vector<double> >;
template class GenericField<CurveMesh, vector<float> >;
template class GenericField<CurveMesh, vector<int> >;
template class GenericField<CurveMesh, vector<short> >;
template class GenericField<CurveMesh, vector<char> >;
template class GenericField<CurveMesh, vector<unsigned int> >;
template class GenericField<CurveMesh, vector<unsigned short> >;
template class GenericField<CurveMesh, vector<unsigned char> >;

template class TriSurfField<Tensor>;
template class TriSurfField<Vector>;
template class TriSurfField<double>;
template class TriSurfField<float>;
template class TriSurfField<int>;
template class TriSurfField<short>;
template class TriSurfField<char>;
template class TriSurfField<unsigned int>;
template class TriSurfField<unsigned short>;
template class TriSurfField<unsigned char>;
template class TriSurfField<vector<pair<NodeIndex<int>,double> > >;

const TypeDescription* get_type_description(TriSurfField<Tensor> *);
const TypeDescription* get_type_description(TriSurfField<Vector> *);
const TypeDescription* get_type_description(TriSurfField<double> *);
const TypeDescription* get_type_description(TriSurfField<float> *);
const TypeDescription* get_type_description(TriSurfField<int> *);
const TypeDescription* get_type_description(TriSurfField<short> *);
const TypeDescription* get_type_description(TriSurfField<char> *);
const TypeDescription* get_type_description(TriSurfField<unsigned int> *);
const TypeDescription* get_type_description(TriSurfField<unsigned short> *);
const TypeDescription* get_type_description(TriSurfField<unsigned char> *);
const TypeDescription* get_type_description(TriSurfField<vector<pair<NodeIndex<int>,double> > > *);

template class CurveField<Tensor>;
template class CurveField<Vector>;
template class CurveField<double>;
template class CurveField<float>;
template class CurveField<int>;
template class CurveField<short>;
template class CurveField<char>;
template class CurveField<unsigned int>;
template class CurveField<unsigned short>;
template class CurveField<unsigned char>;

const TypeDescription* get_type_description(CurveField<Tensor> *);
const TypeDescription* get_type_description(CurveField<Vector> *);
const TypeDescription* get_type_description(CurveField<double> *);
const TypeDescription* get_type_description(CurveField<float> *);
const TypeDescription* get_type_description(CurveField<int> *);
const TypeDescription* get_type_description(CurveField<short> *);
const TypeDescription* get_type_description(CurveField<char> *);
const TypeDescription* get_type_description(CurveField<unsigned int> *);
const TypeDescription* get_type_description(CurveField<unsigned short> *);
const TypeDescription* get_type_description(CurveField<unsigned char> *);

template class MaskedTriSurfField<Tensor>;
template class MaskedTriSurfField<Vector>;
template class MaskedTriSurfField<double>;
template class MaskedTriSurfField<float>;
template class MaskedTriSurfField<int>;
template class MaskedTriSurfField<short>;
template class MaskedTriSurfField<char>;
template class MaskedTriSurfField<unsigned int>;
template class MaskedTriSurfField<unsigned short>;
template class MaskedTriSurfField<unsigned char>;

const TypeDescription* get_type_description(MaskedTriSurfField<Tensor> *);
const TypeDescription* get_type_description(MaskedTriSurfField<Vector> *);
const TypeDescription* get_type_description(MaskedTriSurfField<double> *);
const TypeDescription* get_type_description(MaskedTriSurfField<float> *);
const TypeDescription* get_type_description(MaskedTriSurfField<int> *);
const TypeDescription* get_type_description(MaskedTriSurfField<short> *);
const TypeDescription* get_type_description(MaskedTriSurfField<char> *);
const TypeDescription* get_type_description(MaskedTriSurfField<unsigned int> *);
const TypeDescription* get_type_description(MaskedTriSurfField<unsigned short> *);
const TypeDescription* get_type_description(MaskedTriSurfField<unsigned char> *);


#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif
