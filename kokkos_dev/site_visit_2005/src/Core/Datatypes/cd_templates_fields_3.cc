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
#include <Core/Datatypes/PrismVolField.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/MaskedTetVolField.h>

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

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif
