/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
#ifdef __sgi
#pragma set woff 1468
#endif

#include <Core/Geometry/Tensor.h>

#include <Core/Datatypes/ColumnMatrix.h>
template class LockingHandle<ColumnMatrix>;

#include <Core/Datatypes/Matrix.h>
template class LockingHandle<Matrix>;

#include <Core/Datatypes/TetVol.h>
template class TetVol<Tensor>;
template class TetVol<Vector>;
template class TetVol<double>;
template class TetVol<float>;
template class TetVol<int>;
template class TetVol<unsigned int>;
template class TetVol<short>;
template class TetVol<unsigned short>;
template class TetVol<char>;
template class TetVol<unsigned char>;
template class TetVol<bool>;
template class GenericField<TetVolMesh, vector<Tensor> >;
template class GenericField<TetVolMesh, vector<Vector> >;
template class GenericField<TetVolMesh, vector<double> >;
template class GenericField<TetVolMesh, vector<float> >;
template class GenericField<TetVolMesh, vector<int> >;
template class GenericField<TetVolMesh, vector<unsigned int> >;
template class GenericField<TetVolMesh, vector<short> >;
template class GenericField<TetVolMesh, vector<unsigned short> >;
template class GenericField<TetVolMesh, vector<char> >;
template class GenericField<TetVolMesh, vector<unsigned char> >;
template class GenericField<TetVolMesh, vector<bool> >;


#include <Core/Datatypes/LatticeVol.h>
template class LatticeVol<Tensor>;
template class LatticeVol<Vector>;
template class LatticeVol<double>;
template class LatticeVol<float>;
template class LatticeVol<int>;
template class LatticeVol<unsigned int>;
template class LatticeVol<short>;
template class LatticeVol<unsigned short>;
template class LatticeVol<char>;
template class LatticeVol<unsigned char>;
template class LatticeVol<bool>;
template class GenericField<LatVolMesh, FData3d<Tensor> >;
template class GenericField<LatVolMesh, FData3d<Vector> >;
template class GenericField<LatVolMesh, FData3d<double> >;
template class GenericField<LatVolMesh, FData3d<float> >;
template class GenericField<LatVolMesh, FData3d<int> >;
template class GenericField<LatVolMesh, FData3d<unsigned int> >;
template class GenericField<LatVolMesh, FData3d<short> >;
template class GenericField<LatVolMesh, FData3d<unsigned short> >;
template class GenericField<LatVolMesh, FData3d<char> >;
template class GenericField<LatVolMesh, FData3d<unsigned char> >;
template class GenericField<LatVolMesh, FData3d<bool> >;

#include <Core/Datatypes/TriSurf.h>
template class TriSurf<Tensor>;
template class TriSurf<Vector>;
template class TriSurf<double>;
template class TriSurf<float>;
template class TriSurf<int>;
template class TriSurf<unsigned int>;
template class TriSurf<short>;
template class TriSurf<unsigned short>;
template class TriSurf<char>;
template class TriSurf<unsigned char>;
template class TriSurf<bool>;
template class GenericField<TriSurfMesh, vector<Tensor> >;
template class GenericField<TriSurfMesh, vector<Vector> >;
template class GenericField<TriSurfMesh, vector<double> >;
template class GenericField<TriSurfMesh, vector<float> >;
template class GenericField<TriSurfMesh, vector<int> >;
template class GenericField<TriSurfMesh, vector<unsigned int> >;
template class GenericField<TriSurfMesh, vector<short> >;
template class GenericField<TriSurfMesh, vector<unsigned short> >;
template class GenericField<TriSurfMesh, vector<char> >;
template class GenericField<TriSurfMesh, vector<unsigned char> >;
template class GenericField<TriSurfMesh, vector<bool> >;

#include <Core/Datatypes/ContourField.h>
template class ContourField<double>;
template class GenericField<ContourMesh, vector<double> >;

#include <Core/Datatypes/GenericField.h>
#include <Core/Persistent/PersistentSTL.h>

#include <Core/Datatypes/PropertyManager.h>
template class Property<string>;
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


//! Compute the gradient g in cell ci.
template <>
Vector TetVol<Vector>::cell_gradient(TetVolMesh::cell_index /*ci*/)
{
  ASSERT(type_name(1) != "Vector");  // redundant, useful error message
  return Vector(0, 0, 0);
}


template <>
Vector TetVol<Tensor>::cell_gradient(TetVolMesh::cell_index /*ci*/)
{
  ASSERT(type_name(1) != "Tensor");  // redundant, useful error message
  return Vector(0, 0, 0);
}

template <> bool LatticeVol<Tensor>::get_gradient(Vector &, Point &/*p*/)
{
  ASSERT(type_name(1) != "Tensor");  // redundant, useful error message
  return false;
}

template <> bool LatticeVol<Vector>::get_gradient(Vector &, Point &/*p*/)
{
  ASSERT(type_name(1) != "Tensor");  // redundant, useful error message
  return false;
}


#ifdef __sgi
#pragma reset woff 1468
#endif










