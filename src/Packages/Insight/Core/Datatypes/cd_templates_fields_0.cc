#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GenericField.h>
#include <Packages/Insight/Core/Datatypes/ITKLatVolField.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
/*
cc-1468 CC: REMARK File = ../src/Core/Datatypes/cd_templates_fields_0.cc, Line = 11
  Inline function "SCIRun::FData3d<SCIRun::Tensor>::end" cannot be explicitly
          instantiated.
*/
#pragma set woff 1468
#endif

using namespace SCIRun;
using namespace Insight;

template class ITKFData3d<Tensor>;
template class ITKFData3d<Vector>;
template class ITKFData3d<double>;
template class ITKFData3d<float>;
template class ITKFData3d<int>;
template class ITKFData3d<short>;
template class ITKFData3d<char>;
template class ITKFData3d<unsigned int>;
template class ITKFData3d<unsigned short>;
template class ITKFData3d<unsigned char>;

template class GenericField<LatVolMesh, ITKFData3d<Tensor> >;
template class GenericField<LatVolMesh, ITKFData3d<Vector> >;
template class GenericField<LatVolMesh, ITKFData3d<double> >;
template class GenericField<LatVolMesh, ITKFData3d<float> >;
template class GenericField<LatVolMesh, ITKFData3d<int> >;
template class GenericField<LatVolMesh, ITKFData3d<short> >;
template class GenericField<LatVolMesh, ITKFData3d<char> >;
template class GenericField<LatVolMesh, ITKFData3d<unsigned int> >;
template class GenericField<LatVolMesh, ITKFData3d<unsigned short> >;
template class GenericField<LatVolMesh, ITKFData3d<unsigned char> >;

template class ITKLatVolField<Tensor>;
template class ITKLatVolField<Vector>;
template class ITKLatVolField<double>;
template class ITKLatVolField<float>;
template class ITKLatVolField<int>;
template class ITKLatVolField<short>;
template class ITKLatVolField<char>;
template class ITKLatVolField<unsigned int>;
template class ITKLatVolField<unsigned short>;
template class ITKLatVolField<unsigned char>;

const TypeDescription* get_type_description(ITKLatVolField<Tensor> *);
const TypeDescription* get_type_description(ITKLatVolField<Vector> *);
const TypeDescription* get_type_description(ITKLatVolField<double> *);
const TypeDescription* get_type_description(ITKLatVolField<float> *);
const TypeDescription* get_type_description(ITKLatVolField<int> *);
const TypeDescription* get_type_description(ITKLatVolField<short> *);
const TypeDescription* get_type_description(ITKLatVolField<char> *);
const TypeDescription* get_type_description(ITKLatVolField<unsigned int> *);
const TypeDescription* get_type_description(ITKLatVolField<unsigned short> *);
const TypeDescription* get_type_description(ITKLatVolField<unsigned char> *);

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif
