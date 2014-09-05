#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/MaskedHexVolField.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
/*
cc-1468 CC: REMARK File = ../src/Core/Datatypes/cd_templates_fields_0.cc, Line = 11
  Inline function "SCIRun::FData3d<SCIRun::Tensor>::end" cannot be explicitly
          instantiated.
*/
#pragma set woff 1468
#endif

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


#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif
