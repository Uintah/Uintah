#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GenericField.h>
#include <Packages/Insight/Core/Datatypes/ITKImageField.h>

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

template class ITKFData2d<Tensor>;
template class ITKFData2d<Vector>;
template class ITKFData2d<double>;
template class ITKFData2d<float>;
template class ITKFData2d<int>;
template class ITKFData2d<short>;
template class ITKFData2d<char>;
template class ITKFData2d<unsigned int>;
template class ITKFData2d<unsigned short>;
template class ITKFData2d<unsigned char>;
template class ITKFData2d<unsigned long>;

template class GenericField<ImageMesh, ITKFData2d<Tensor> >;
template class GenericField<ImageMesh, ITKFData2d<Vector> >;
template class GenericField<ImageMesh, ITKFData2d<double> >;
template class GenericField<ImageMesh, ITKFData2d<float> >;
template class GenericField<ImageMesh, ITKFData2d<int> >;
template class GenericField<ImageMesh, ITKFData2d<short> >;
template class GenericField<ImageMesh, ITKFData2d<char> >;
template class GenericField<ImageMesh, ITKFData2d<unsigned int> >;
template class GenericField<ImageMesh, ITKFData2d<unsigned short> >;
template class GenericField<ImageMesh, ITKFData2d<unsigned char> >;
template class GenericField<ImageMesh, ITKFData2d<unsigned long> >;

template class ITKImageField<Tensor>;
template class ITKImageField<Vector>;
template class ITKImageField<double>;
template class ITKImageField<float>;
template class ITKImageField<int>;
template class ITKImageField<short>;
template class ITKImageField<char>;
template class ITKImageField<unsigned int>;
template class ITKImageField<unsigned short>;
template class ITKImageField<unsigned char>;
template class ITKImageField<unsigned long>;

const TypeDescription* get_type_description(ITKImageField<Tensor> *);
const TypeDescription* get_type_description(ITKImageField<Vector> *);
const TypeDescription* get_type_description(ITKImageField<double> *);
const TypeDescription* get_type_description(ITKImageField<float> *);
const TypeDescription* get_type_description(ITKImageField<int> *);
const TypeDescription* get_type_description(ITKImageField<short> *);
const TypeDescription* get_type_description(ITKImageField<char> *);
const TypeDescription* get_type_description(ITKImageField<unsigned int> *);
const TypeDescription* get_type_description(ITKImageField<unsigned short> *);
const TypeDescription* get_type_description(ITKImageField<unsigned char> *);
const TypeDescription* get_type_description(ITKImageField<unsigned long> *);

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif









