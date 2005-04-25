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
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/NrrdData.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/QuadraticTetVolField.h>
#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/MaskedTetVolField.h>
#include <Core/Datatypes/MaskedHexVolField.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/MaskedLatVolField.h>
#include <Core/Datatypes/MaskedTriSurfField.h>
#include <Core/Datatypes/PrismVolField.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/CurveField.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/ImageField.h>
#include <Core/Datatypes/ScanlineField.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Datatypes/PropertyManager.h>

#if !defined(__sgi)
// Needed for optimized linux build only
#if 0
template void Pio<char, char>(Piostream&, pair<char, char>&);
template void Pio<int, int>(Piostream&, pair<int, int>&);
template void Pio<float, float>(Piostream&, pair<float, float>&);
template void Pio<int, double>(Piostream&, pair<int, double>&);
template void Pio<double, double>(Piostream&, pair<double, double>&);
template void Pio<short, short>(Piostream&, pair<short, short>&);
template void Pio<unsigned char, unsigned char>(Piostream&, pair<unsigned char,
		  unsigned char>&);
template void Pio<unsigned int, unsigned int>(Piostream&, pair<unsigned int,
		  unsigned int>&);
template void Pio<unsigned short, unsigned short>(Piostream&, pair<unsigned short,
		  unsigned short>&);
#endif
#endif

template class LockingHandle<ColumnMatrix>;
template class LockingHandle<Matrix>;

//Index types
const TypeDescription* get_type_description(NodeIndex<int>*);
const TypeDescription* get_type_description(EdgeIndex<int>*);
const TypeDescription* get_type_description(FaceIndex<int>*);
const TypeDescription* get_type_description(CellIndex<int>*);

const TypeDescription* get_type_description(vector<NodeIndex<int> >*);
const TypeDescription* get_type_description(vector<EdgeIndex<int> >*);
const TypeDescription* get_type_description(vector<FaceIndex<int> >*);
const TypeDescription* get_type_description(vector<CellIndex<int> >*);

// Property types
template class Property<int>;
template class Property<string>;
template class Property<double>;
template class Property<float>;
template class Property<Array1<double> >;
template class Property<Array1<Tensor> >;
template class Property<pair<int,double> >;
template class Property<pair<double,double> >;
template class Property<pair<float,float> >;
template class Property<pair<unsigned int,unsigned int> >;
template class Property<pair<int,int> >;
template class Property<pair<unsigned short,unsigned short> >;
template class Property<pair<short,short> >;
template class Property<pair<unsigned char,unsigned char> >;
template class Property<pair<char,char> >;
template class Property<vector<pair<string,Tensor> > >;
template class Property<vector<pair<int,double> > >;
template class Property<LockingHandle<Matrix> >;
template class Property<LockingHandle<NrrdData> >;


//! Compute the gradient g in cell ci.
template <>
Vector PrismVolField<Vector>::cell_gradient(PrismVolMesh::Cell::index_type /*ci*/)
{
  ASSERT(type_name(1) != "Vector");  // redundant, useful error message
  return Vector(0, 0, 0);
}

template <>
Vector PrismVolField<Tensor>::cell_gradient(PrismVolMesh::Cell::index_type /*ci*/)
{
  ASSERT(type_name(1) != "Tensor");  // redundant, useful error message
  return Vector(0, 0, 0);
}


template <>
Vector TetVolField<Vector>::cell_gradient(TetVolMesh::Cell::index_type /*ci*/)
{
  ASSERT(type_name(1) != "Vector");  // redundant, useful error message
  return Vector(0, 0, 0);
}

template <>
Vector TetVolField<Tensor>::cell_gradient(TetVolMesh::Cell::index_type /*ci*/)
{
  ASSERT(type_name(1) != "Tensor");  // redundant, useful error message
  return Vector(0, 0, 0);
}


template <>
Vector QuadraticTetVolField<Vector>::cell_gradient(TetVolMesh::Cell::index_type)
{
  ASSERT(type_name(1) != "Vector");  // redundant, useful error message
  return Vector(0, 0, 0);
}

template <>
Vector QuadraticTetVolField<Tensor>::cell_gradient(TetVolMesh::Cell::index_type)
{
  ASSERT(type_name(1) != "Tensor");  // redundant, useful error message
  return Vector(0, 0, 0);
}


template <>
Vector HexVolField<Vector>::cell_gradient(HexVolMesh::Cell::index_type /*ci*/)
{
  ASSERT(type_name(1) != "Vector");  // redundant, useful error message
  return Vector(0, 0, 0);
}

template <>
Vector HexVolField<Tensor>::cell_gradient(HexVolMesh::Cell::index_type /*ci*/)
{
  ASSERT(type_name(1) != "Tensor");  // redundant, useful error message
  return Vector(0, 0, 0);
}


template <> bool LatVolField<Vector>::get_gradient(Vector &, const Point &/*p*/)
{
  ASSERT(type_name(1) != "Vector");  // redundant, useful error message
  return false;
}

template <> bool LatVolField<Tensor>::get_gradient(Vector &, const Point &/*p*/)
{
  ASSERT(type_name(1) != "Tensor");  // redundant, useful error message
  return false;
}

template <>
Vector LatVolField<Vector>::cell_gradient(const LatVolMesh::Cell::index_type &/*ci*/) const
{
  ASSERT(type_name(1) != "Vector");  // redundant, useful error message
  return Vector(0, 0, 0);
}

template <>
Vector LatVolField<Tensor>::cell_gradient(const LatVolMesh::Cell::index_type &/*ci*/) const
{
  ASSERT(type_name(1) != "Tensor");  // redundant, useful error message
  return Vector(0, 0, 0);
}


#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif
