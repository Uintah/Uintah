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
#include <Core/Datatypes/QuadraticLatVolField.h>
#include <Core/Datatypes/QuadraticTetVolField.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
/*
cc-1468 CC: REMARK File = ../src/Core/Datatypes/cd_templates_fields_0.cc, Line = 11
  Inline function "SCIRun::FData3d<SCIRun::Tensor>::end" cannot be explicitly
          instantiated.
*/
#pragma set woff 1468
#endif

using namespace SCIRun;

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
