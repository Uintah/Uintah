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
 *  TetVolField.h
 *
 *  Written by:
 *   Martin Cole
 *   School of Computing
 *   University of Utah
 *
 *  Copyright (C) 2001 SCI Institute
 */

#ifndef Datatypes_TetVolField_h
#define Datatypes_TetVolField_h

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TetVolMesh.h>
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
class TetVolField : public GenericField<TetVolMesh, vector<T> >
{
public:
  TetVolField();
  TetVolField(Field::data_location data_at);
  TetVolField(TetVolMeshHandle mesh, Field::data_location data_at);
  virtual TetVolField<T> *clone() const;
  virtual ~TetVolField();

  //! Persistent IO
  virtual void io(Piostream &stream);
  static  PersistentTypeID type_id;

  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

  // TetVolField specific methods.
  bool get_gradient(Vector &, Point &);
  Vector cell_gradient(TetVolMesh::Cell::index_type);

private:
  static Persistent *maker();
};

template <class T>
TetVolField<T>::TetVolField()
  : GenericField<TetVolMesh, vector<T> >()
{
}

template <class T>
TetVolField<T>::TetVolField(Field::data_location data_at)
  : GenericField<TetVolMesh, vector<T> >(data_at)
{
  ASSERTMSG(data_at != Field::EDGE, 
	    "TetVolField does NOT currently support data at edges."); 
  ASSERTMSG(data_at != Field::FACE, 
	    "TetVolField does NOT currently support data at faces."); 

}

template <class T>
TetVolField<T>::TetVolField(TetVolMeshHandle mesh, Field::data_location data_at)
  : GenericField<TetVolMesh, vector<T> >(mesh, data_at)
{
  ASSERTMSG(data_at != Field::EDGE, 
	    "TetVolField does NOT currently support data at edges."); 
  ASSERTMSG(data_at != Field::FACE, 
	    "TetVolField does NOT currently support data at faces."); 
}

template <class T>
TetVolField<T> *
TetVolField<T>::clone() const
{
  return new TetVolField(*this);
}

template <class T>
TetVolField<T>::~TetVolField()
{
}


template <class T>
Persistent*
TetVolField<T>::maker()
{
  return scinew TetVolField<T>;
}


template <class T>
PersistentTypeID 
TetVolField<T>::type_id(type_name(-1), 
		   GenericField<TetVolMesh, vector<T> >::type_name(-1),
		   maker);


// Pio defs.
const int TET_VOL_FIELD_VERSION = 1;

template <class T>
void 
TetVolField<T>::io(Piostream& stream)
{
  /* int version=*/stream.begin_class(type_name(-1), TET_VOL_FIELD_VERSION);
  GenericField<TetVolMesh, vector<T> >::io(stream);
  stream.end_class();
}

template <class T> 
const string 
TetVolField<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    return "TetVolField";
  }
  else
  {
    return find_type_name((T *)0);
  }
}

template <class T> 
const TypeDescription*
TetVolField<T>::get_type_description(int n) const
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
bool TetVolField<T>::get_gradient(Vector &g, Point &p) {
  TetVolMesh::Cell::index_type ci;
  if (get_typed_mesh()->locate(ci, p)) {
    g = cell_gradient(ci);
    return true;
  } else {
    return false;
  }
}

//! Compute the gradient g in cell ci.
template <>
Vector TetVolField<Vector>::cell_gradient(TetVolMesh::Cell::index_type ci);

template <>
Vector TetVolField<Tensor>::cell_gradient(TetVolMesh::Cell::index_type ci);

template <class T>
Vector TetVolField<T>::cell_gradient(TetVolMesh::Cell::index_type ci)
{
  // for now we only know how to do this for field with doubles at the nodes
  ASSERT(data_at() == Field::NODE);

  // load up the indices of the nodes for this cell
  TetVolMesh::Node::array_type nodes;
  get_typed_mesh()->get_nodes(nodes, ci);
  Vector gb0, gb1, gb2, gb3;
  get_typed_mesh()->get_gradient_basis(ci, gb0, gb1, gb2, gb3);

  // we really want this for all scalars... 
  //  but for now, we'll just make doubles work
  return Vector(gb0 * value(nodes[0]) + gb1 * value(nodes[1]) + 
		gb2 * value(nodes[2]) + gb3 * value(nodes[3]));
}


} // end namespace SCIRun

#endif // Datatypes_TetVolField_h
