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
 *  HexVolField.h
 *
 *  Written by:
 *   Martin Cole
 *   School of Computing
 *   University of Utah
 *
 *  Copyright (C) 2001 SCI Institute
 */

#ifndef Datatypes_HexVolField_h
#define Datatypes_HexVolField_h

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Util/Assert.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

template <class T> 
class HexVolField : public GenericField<HexVolMesh, vector<T> >
{
public:
  HexVolField();
  HexVolField(Field::data_location data_at);
  HexVolField(HexVolMeshHandle mesh, Field::data_location data_at);
  virtual HexVolField<T> *clone() const;
  virtual ~HexVolField();

  //! Persistent IO
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);

  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

  // HexVolField specific methods.
  bool get_gradient(Vector &, Point &);
  Vector cell_gradient(HexVolMesh::Cell::index_type);

private:
  static Persistent *maker();
};

template <class T>
HexVolField<T>::HexVolField()
  : GenericField<HexVolMesh, vector<T> >()
{
}

template <class T>
HexVolField<T>::HexVolField(Field::data_location data_at)
  : GenericField<HexVolMesh, vector<T> >(data_at)
{
}

template <class T>
HexVolField<T>::HexVolField(HexVolMeshHandle mesh, Field::data_location data_at)
  : GenericField<HexVolMesh, vector<T> >(mesh, data_at)
{
}

template <class T>
HexVolField<T> *
HexVolField<T>::clone() const
{
  return new HexVolField(*this);
}

template <class T>
HexVolField<T>::~HexVolField()
{
}

template <class T>
Persistent*
HexVolField<T>::maker()
{
  return scinew HexVolField<T>;
}


template <class T>
PersistentTypeID 
HexVolField<T>::type_id(type_name(-1), 
		   GenericField<HexVolMesh, vector<T> >::type_name(-1),
		   maker);


// Pio defs.
const int HEX_VOL_FIELD_VERSION = 1;

template <class T>
void 
HexVolField<T>::io(Piostream& stream)
{
  /*int version=*/stream.begin_class(type_name(-1), HEX_VOL_FIELD_VERSION);
  GenericField<HexVolMesh, vector<T> >::io(stream);
  stream.end_class();
}

template <class T> 
const string 
HexVolField<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "HexVolField";
  }
  else
  {
    return find_type_name((T *)0);
  }
}

template <class T> 
const TypeDescription*
HexVolField<T>::get_type_description(int n) const
{
  ASSERT((n >= -1) && n <= 1);

  TypeDescription* td = 0;
  static string name( type_name(0) );
  static string namesp("SCIRun");
  static string path(__FILE__);

  if(!td){
    if (n == -1) {
      const TypeDescription *sub = SCIRun::get_type_description((T*)0);
      TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
      (*subs)[0] = sub;
      td = scinew TypeDescription(name, subs, path, namesp);
    }
    else if(n == 0) {
      td = scinew TypeDescription(name, 0, path, namesp);
    }
    else {
      td = (TypeDescription *) SCIRun::get_type_description((T*)0);
    }
  }
  return td;
}

//! compute the gradient g, at point p
template <class T>
bool HexVolField<T>::get_gradient(Vector &g, Point &p) {
  HexVolMesh::Cell::index_type ci;
  if (get_typed_mesh()->locate(ci, p)) {
    g = cell_gradient(ci);
    return true;
  } else {
    return false;
  }
}

//! Compute the gradient g in cell ci.
template <>
Vector HexVolField<Vector>::cell_gradient(HexVolMesh::Cell::index_type ci);

template <>
Vector HexVolField<Tensor>::cell_gradient(HexVolMesh::Cell::index_type ci);

template <class T>
Vector HexVolField<T>::cell_gradient(HexVolMesh::Cell::index_type ci)
{
  // for now we only know how to do this for field with doubles at the nodes
  ASSERT(data_at() == Field::NODE);

  // load up the indices of the nodes for this cell
  HexVolMesh::Node::array_type nodes;
  get_typed_mesh()->get_nodes(nodes, ci);
  Vector gb0, gb1, gb2, gb3, gb4, gb5, gb6, gb7;
  get_typed_mesh()->get_gradient_basis(ci, gb0, gb1, gb2, gb3, gb4, gb5, gb6, gb7);

  // we really want this for all scalars... 
  //  but for now, we'll just make doubles work
  return Vector(gb0 * value(nodes[0]) + gb1 * value(nodes[1]) + 
		gb2 * value(nodes[2]) + gb3 * value(nodes[3]) +
		gb4 * value(nodes[4]) + gb5 * value(nodes[5]) +
		gb6 * value(nodes[6]) + gb7 * value(nodes[7]));
}


} // end namespace SCIRun

#endif // Datatypes_HexVolField_h
