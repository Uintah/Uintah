/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

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
template class ITKFData3d<unsigned long>;

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
template class GenericField<LatVolMesh, ITKFData3d<unsigned long> >;

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
template class ITKLatVolField<unsigned long>;

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
const TypeDescription* get_type_description(ITKLatVolField<unsigned long> *);

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif
