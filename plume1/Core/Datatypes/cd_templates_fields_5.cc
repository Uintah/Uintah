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
#include <Core/Datatypes/MaskedHexVolField.h>
#include <Core/Datatypes/HexVolField.h>

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
