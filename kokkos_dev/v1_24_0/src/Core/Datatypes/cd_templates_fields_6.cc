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


/*
 * Manual template instantiations
 */


/*
 * These aren't used by Datatypes directly, but since they are used in
 * a lot of different modules, we instantiate them here to avoid bloat
 *
 * Find the bloaters with:
find . -name "*.ii" -print | xargs cat | sort | uniq -c | sort -nr | more
 */

#include <Core/Containers/LockingHandle.h>
#include <Core/Malloc/Allocator.h>


using namespace SCIRun;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1468
#endif

#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/StructCurveField.h>
#include <Core/Datatypes/StructQuadSurfField.h>
#include <Core/Datatypes/StructHexVolField.h>

template <>
Vector StructHexVolField<Vector>::cell_gradient(StructHexVolMesh::Cell::index_type /*ci*/)
{
  ASSERT(type_name(1) != "Vector");  // redundant, useful error message
  return Vector(0, 0, 0);
}


template <>
Vector StructHexVolField<Tensor>::cell_gradient(StructHexVolMesh::Cell::index_type /*ci*/)
{
  ASSERT(type_name(1) != "Tensor");  // redundant, useful error message
  return Vector(0, 0, 0);
}

template class GenericField<StructCurveMesh, vector<Tensor> >;
template class GenericField<StructCurveMesh, vector<Vector> >;
template class GenericField<StructCurveMesh, vector<double> >;
template class GenericField<StructCurveMesh, vector<float> >;
template class GenericField<StructCurveMesh, vector<int> >;
template class GenericField<StructCurveMesh, vector<short> >;
template class GenericField<StructCurveMesh, vector<char> >;
template class GenericField<StructCurveMesh, vector<unsigned int> >;
template class GenericField<StructCurveMesh, vector<unsigned short> >;
template class GenericField<StructCurveMesh, vector<unsigned char> >;

template class GenericField<StructQuadSurfMesh, FData2d<Tensor> >;
template class GenericField<StructQuadSurfMesh, FData2d<Vector> >;
template class GenericField<StructQuadSurfMesh, FData2d<double> >;
template class GenericField<StructQuadSurfMesh, FData2d<float> >;
template class GenericField<StructQuadSurfMesh, FData2d<int> >;
template class GenericField<StructQuadSurfMesh, FData2d<short> >;
template class GenericField<StructQuadSurfMesh, FData2d<char> >;
template class GenericField<StructQuadSurfMesh, FData2d<unsigned int> >;
template class GenericField<StructQuadSurfMesh, FData2d<unsigned short> >;
template class GenericField<StructQuadSurfMesh, FData2d<unsigned char> >;

template class GenericField<StructHexVolMesh, FData3d<Tensor> >;
template class GenericField<StructHexVolMesh, FData3d<Vector> >;
template class GenericField<StructHexVolMesh, FData3d<double> >;
template class GenericField<StructHexVolMesh, FData3d<float> >;
template class GenericField<StructHexVolMesh, FData3d<int> >;
template class GenericField<StructHexVolMesh, FData3d<short> >;
template class GenericField<StructHexVolMesh, FData3d<char> >;
template class GenericField<StructHexVolMesh, FData3d<unsigned int> >;
template class GenericField<StructHexVolMesh, FData3d<unsigned short> >;
template class GenericField<StructHexVolMesh, FData3d<unsigned char> >;

template class StructCurveField<Tensor>;
template class StructCurveField<Vector>;
template class StructCurveField<double>;
template class StructCurveField<float>;
template class StructCurveField<int>;
template class StructCurveField<short>;
template class StructCurveField<char>;
template class StructCurveField<unsigned int>;
template class StructCurveField<unsigned short>;
template class StructCurveField<unsigned char>;

template class StructQuadSurfField<Tensor>;
template class StructQuadSurfField<Vector>;
template class StructQuadSurfField<double>;
template class StructQuadSurfField<float>;
template class StructQuadSurfField<int>;
template class StructQuadSurfField<short>;
template class StructQuadSurfField<char>;
template class StructQuadSurfField<unsigned int>;
template class StructQuadSurfField<unsigned short>;
template class StructQuadSurfField<unsigned char>;

template class StructHexVolField<Tensor>;
template class StructHexVolField<Vector>;
template class StructHexVolField<double>;
template class StructHexVolField<float>;
template class StructHexVolField<int>;
template class StructHexVolField<short>;
template class StructHexVolField<char>;
template class StructHexVolField<unsigned int>;
template class StructHexVolField<unsigned short>;
template class StructHexVolField<unsigned char>;

const TypeDescription* get_type_description(StructCurveField<Tensor> *);
const TypeDescription* get_type_description(StructCurveField<Vector> *);
const TypeDescription* get_type_description(StructCurveField<double> *);
const TypeDescription* get_type_description(StructCurveField<float> *);
const TypeDescription* get_type_description(StructCurveField<int> *);
const TypeDescription* get_type_description(StructCurveField<short> *);
const TypeDescription* get_type_description(StructCurveField<char> *);
const TypeDescription* get_type_description(StructCurveField<unsigned int> *);
const TypeDescription* get_type_description(StructCurveField<unsigned short> *);
const TypeDescription* get_type_description(StructCurveField<unsigned char> *);

const TypeDescription* get_type_description(StructQuadSurfField<Tensor> *);
const TypeDescription* get_type_description(StructQuadSurfField<Vector> *);
const TypeDescription* get_type_description(StructQuadSurfField<double> *);
const TypeDescription* get_type_description(StructQuadSurfField<float> *);
const TypeDescription* get_type_description(StructQuadSurfField<int> *);
const TypeDescription* get_type_description(StructQuadSurfField<short> *);
const TypeDescription* get_type_description(StructQuadSurfField<char> *);
const TypeDescription* get_type_description(StructQuadSurfField<unsigned int> *);
const TypeDescription* get_type_description(StructQuadSurfField<unsigned short> *);
const TypeDescription* get_type_description(StructQuadSurfField<unsigned char> *);

const TypeDescription* get_type_description(StructHexVolField<Tensor> *);
const TypeDescription* get_type_description(StructHexVolField<Vector> *);
const TypeDescription* get_type_description(StructHexVolField<double> *);
const TypeDescription* get_type_description(StructHexVolField<float> *);
const TypeDescription* get_type_description(StructHexVolField<int> *);
const TypeDescription* get_type_description(StructHexVolField<short> *);
const TypeDescription* get_type_description(StructHexVolField<char> *);
const TypeDescription* get_type_description(StructHexVolField<unsigned int> *);
const TypeDescription* get_type_description(StructHexVolField<unsigned short> *);
const TypeDescription* get_type_description(StructHexVolField<unsigned char> *);

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif


