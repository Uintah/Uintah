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
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/MaskedLatVolField.h>
#include <Core/Datatypes/MRLatVolField.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
/*
cc-1468 CC: REMARK File = ../src/Core/Datatypes/cd_templates_fields_0.cc, Line = 11
  Inline function "SCIRun::FData3d<SCIRun::Tensor>::end" cannot be explicitly
          instantiated.
*/
#pragma set woff 1468
#endif

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
template class FData3d<unsigned long>;

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
template class GenericField<LatVolMesh, FData3d<unsigned long> >;

template class LatVolField<Tensor>;
template class LatVolField<Vector>;
template class LatVolField<double>;
template class LatVolField<float>;
template class LatVolField<int>;
template class LatVolField<short>;
template class LatVolField<char>;
template class LatVolField<unsigned int>;
template class LatVolField<unsigned short>;
template class LatVolField<unsigned char>;
template class LatVolField<unsigned long>;

const TypeDescription* get_type_description(LatVolField<Tensor> *);
const TypeDescription* get_type_description(LatVolField<Vector> *);
const TypeDescription* get_type_description(LatVolField<double> *);
const TypeDescription* get_type_description(LatVolField<float> *);
const TypeDescription* get_type_description(LatVolField<int> *);
const TypeDescription* get_type_description(LatVolField<short> *);
const TypeDescription* get_type_description(LatVolField<char> *);
const TypeDescription* get_type_description(LatVolField<unsigned int> *);
const TypeDescription* get_type_description(LatVolField<unsigned short> *);
const TypeDescription* get_type_description(LatVolField<unsigned char> *);
const TypeDescription* get_type_description(LatVolField<unsigned long> *);

template class MaskedLatVolField<Tensor>;
template class MaskedLatVolField<Vector>;
template class MaskedLatVolField<double>;
template class MaskedLatVolField<float>;
template class MaskedLatVolField<int>;
template class MaskedLatVolField<short>;
template class MaskedLatVolField<char>;
template class MaskedLatVolField<unsigned int>;
template class MaskedLatVolField<unsigned short>;
template class MaskedLatVolField<unsigned char>;

const TypeDescription* get_type_description(MaskedLatVolField<Tensor> *);
const TypeDescription* get_type_description(MaskedLatVolField<Vector> *);
const TypeDescription* get_type_description(MaskedLatVolField<double> *);
const TypeDescription* get_type_description(MaskedLatVolField<float> *);
const TypeDescription* get_type_description(MaskedLatVolField<int> *);
const TypeDescription* get_type_description(MaskedLatVolField<short> *);
const TypeDescription* get_type_description(MaskedLatVolField<char> *);
const TypeDescription* get_type_description(MaskedLatVolField<unsigned int> *);
const TypeDescription* get_type_description(MaskedLatVolField<unsigned short> *);
const TypeDescription* get_type_description(MaskedLatVolField<unsigned char> *);

template class MRLatVolField<Tensor>;
template class MRLatVolField<Vector>;
template class MRLatVolField<double>;
template class MRLatVolField<float>;
template class MRLatVolField<int>;
template class MRLatVolField<short>;
template class MRLatVolField<char>;
template class MRLatVolField<unsigned int>;
template class MRLatVolField<unsigned short>;
template class MRLatVolField<unsigned char>;

const TypeDescription* get_type_description(MRLatVolField<Tensor> *);
const TypeDescription* get_type_description(MRLatVolField<Vector> *);
const TypeDescription* get_type_description(MRLatVolField<double> *);
const TypeDescription* get_type_description(MRLatVolField<float> *);
const TypeDescription* get_type_description(MRLatVolField<int> *);
const TypeDescription* get_type_description(MRLatVolField<short> *);
const TypeDescription* get_type_description(MRLatVolField<char> *);
const TypeDescription* get_type_description(MRLatVolField<unsigned int> *);
const TypeDescription* get_type_description(MRLatVolField<unsigned short> *);
const TypeDescription* get_type_description(MRLatVolField<unsigned char> *);


#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif
