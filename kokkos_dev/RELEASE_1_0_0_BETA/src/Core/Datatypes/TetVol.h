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
 *  TetVol.h
 *
 *  Written by:
 *   Martin Cole
 *   School of Computing
 *   University of Utah
 *
 *  Copyright (C) 2001 SCI Institute
 */

#ifndef Datatypes_TetVol_h
#define Datatypes_TetVol_h

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Util/Assert.h>
#include <vector>

namespace SCIRun {

template <class T> 
class TetVol : public GenericField<TetVolMesh, vector<T> > {
public:
  TetVol() : 
    GenericField<TetVolMesh, vector<T> >() {}
  TetVol(Field::data_location data_at) : 
    GenericField<TetVolMesh, vector<T> >(data_at) {}
  TetVol(TetVolMeshHandle mesh, Field::data_location data_at) : 
    GenericField<TetVolMesh, vector<T> >(mesh, data_at) {}

  virtual ~TetVol() {};

  /*! Ask mesh to compute edges and faces. Does nothing if mesh 
    is already finished. */
  void finish_mesh() { get_typed_mesh()->finish(); }

  bool get_gradient(Vector &, Point &);
  Vector cell_gradient(TetVolMesh::cell_index);

  //! Persistent IO
  void    io(Piostream &stream);
  static  PersistentTypeID type_id;


  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

private:
  static Persistent *maker();
};

// Pio defs.
const int TET_VOL_VERSION = 1;

template <class T>
Persistent*
TetVol<T>::maker()
{
  return scinew TetVol<T>;
}

template <class T>
PersistentTypeID 
TetVol<T>::type_id(type_name(), 
		   GenericField<TetVolMesh, vector<T> >::type_name(),
		   maker);


template <class T>
void 
TetVol<T>::io(Piostream& stream)
{
  stream.begin_class(type_name().c_str(), TET_VOL_VERSION);
  GenericField<TetVolMesh, vector<T> >::io(stream);
  stream.end_class();
}

template <class T> 
const string 
TetVol<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "TetVol";
  }
  else
  {
    return find_type_name((T *)0);
  }
}

//! compute the gradient g, at point p
template <class T>
bool TetVol<T>::get_gradient(Vector &g, Point &p) {
  TetVolMesh::cell_index ci;
  if (get_typed_mesh()->locate(ci, p)) {
    g = cell_gradient(ci);
    return true;
  } else {
    return false;
  }
}

//! Compute the gradient g in cell ci.
template <>
Vector TetVol<Vector>::cell_gradient(TetVolMesh::cell_index ci);

template <>
Vector TetVol<Tensor>::cell_gradient(TetVolMesh::cell_index ci);

template <class T>
Vector TetVol<T>::cell_gradient(TetVolMesh::cell_index ci)
{
  // for now we only know how to do this for field with doubles at the nodes
  ASSERT(data_at() == Field::NODE);

  // load up the indices of the nodes for this cell
  TetVolMesh::node_array nodes;
  get_typed_mesh()->get_nodes(nodes, ci);
  Vector gb0, gb1, gb2, gb3;
  get_typed_mesh()->get_gradient_basis(ci, gb0, gb1, gb2, gb3);

  // we really want this for all scalars... 
  //  but for now, we'll just make doubles work
  return Vector(gb0 * value(nodes[0]) + gb1 * value(nodes[1]) + 
		gb2 * value(nodes[2]) + gb3 * value(nodes[3]));
}


} // end namespace SCIRun

#endif // Datatypes_TetVol_h
