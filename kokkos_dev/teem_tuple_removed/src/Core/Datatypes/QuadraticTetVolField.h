//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : QuadraticTetVolField.h
//    Author : Martin Cole
//    Date   : Sun Feb 24 13:47:31 2002

#ifndef Datatypes_QuadraticTetVolField_h
#define Datatypes_QuadraticTetVolField_h

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/QuadraticTetVolMesh.h>
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
class QuadraticTetVolField : public GenericField<QuadraticTetVolMesh, vector<T> >
{
public:
  // Avoids a warning with g++ 3.1
  // ../src/Core/Datatypes/QuadraticTetVolField.h:95: warning: `typename 
  // SCIRun::QuadraticTetVolField<T>::mesh_handle_type' is implicitly a typename
  // ../src/Core/Datatypes/QuadraticTetVolField.h:95: warning: implicit typename is 
  // deprecated, please see the documentation for details
  typedef typename GenericField<QuadraticTetVolMesh, vector<T> >::mesh_handle_type mesh_handle_type;

  QuadraticTetVolField();
  QuadraticTetVolField(Field::data_location data_at);
  QuadraticTetVolField(QuadraticTetVolMeshHandle mesh, 
		       Field::data_location data_at);

  static QuadraticTetVolField<T>* create_from(const TetVolField<T> &);
  virtual QuadraticTetVolField<T> *clone() const;
  virtual ~QuadraticTetVolField();

  //! Persistent IO
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);
  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

  // QuadraticTetVolField specific methods.
  bool get_gradient(Vector &, Point &);
  Vector cell_gradient(QuadraticTetVolMesh::Cell::index_type);

private:
  static Persistent *maker();
};

template <class T>
QuadraticTetVolField<T>::QuadraticTetVolField() : 
  GenericField<QuadraticTetVolMesh, vector<T> >()
{
}

template <class T>
QuadraticTetVolField<T>::QuadraticTetVolField(Field::data_location data_at) : 
  GenericField<QuadraticTetVolMesh, vector<T> >(data_at)
{
}

template <class T>
QuadraticTetVolField<T>::QuadraticTetVolField(QuadraticTetVolMeshHandle mesh, 
					      Field::data_location data_at) : 
  GenericField<QuadraticTetVolMesh, vector<T> >(mesh, data_at)
{
}

// will end up with no data...
template <class T>
QuadraticTetVolField<T> *
QuadraticTetVolField<T>::create_from(const TetVolField<T> &tv) 
{
  QuadraticTetVolMesh *m = 
    scinew QuadraticTetVolMesh(*tv.get_typed_mesh().get_rep());

  mesh_handle_type mh(m);
  QuadraticTetVolField<T> *rval = scinew QuadraticTetVolField(mh, 
							      tv.data_at());
  rval->fdata()=tv.fdata();
  rval->copy_properties(&tv);
  rval->freeze();
  return rval;
}

template <class T>
QuadraticTetVolField<T> *
QuadraticTetVolField<T>::clone() const
{
  return new QuadraticTetVolField(*this);
}

template <class T>
QuadraticTetVolField<T>::~QuadraticTetVolField()
{
}

template <class T>
Persistent*
QuadraticTetVolField<T>::maker()
{
  return scinew QuadraticTetVolField<T>;
}


template <class T>
PersistentTypeID 
QuadraticTetVolField<T>::type_id(QuadraticTetVolField<T>::type_name(-1), 
			    GenericField<QuadraticTetVolMesh, vector<T> >::type_name(-1),
			    maker);


// Pio defs.
const int QUADRATIC_TET_VOL_FIELD_VERSION = 1;

template <class T>
void 
QuadraticTetVolField<T>::io(Piostream& stream)
{
  /*int version=*/stream.begin_class(type_name(-1), 
				     QUADRATIC_TET_VOL_FIELD_VERSION);
  GenericField<QuadraticTetVolMesh, vector<T> >::io(stream);
  stream.end_class();
}


template <class T> 
const string 
QuadraticTetVolField<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "QuadraticTetVolField";
  }
  else
  {
    return find_type_name((T *)0);
  }
}

template <class T> 
const TypeDescription*
QuadraticTetVolField<T>::get_type_description(int n) const
{
  ASSERT((n >= -1) && n <= 1);

  TypeDescription* td = 0;
  static string name( type_name(0) );
  static string namesp("SCIRun");
  static string path(__FILE__);

  if (n == -1) {
    static TypeDescription* tdn1 = 0;
    if (tdn1 == 0) {
      const TypeDescription *sub = SCIRun::get_type_description((T*)0);
      TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
      (*subs)[0] = sub;
      tdn1 = scinew TypeDescription(name, subs, path, namesp);
    } 
    td = tdn1;
  }
  else if(n == 0) {
    static TypeDescription* tdn0 = 0;
    if (tdn0 == 0) {
      tdn0 = scinew TypeDescription(name, 0, path, namesp);
    }
    td = tdn0;
  }
  else {
    static TypeDescription* tdnn = 0;
    if (tdnn == 0) {
      tdnn = (TypeDescription *) SCIRun::get_type_description((T*)0);
    }
    td = tdnn;
  }
  return td;
}

//! compute the gradient g, at point p
template <class T>
bool QuadraticTetVolField<T>::get_gradient(Vector &g, Point &p) {
  QuadraticTetVolMesh::Cell::index_type ci;
  if (get_typed_mesh()->locate(ci, p)) {
    g = cell_gradient(ci);
    return true;
  } else {
    return false;
  }
}

//! Compute the gradient g in cell ci.
template <>
Vector QuadraticTetVolField<Vector>::cell_gradient(QuadraticTetVolMesh::Cell::index_type ci);

template <>
Vector QuadraticTetVolField<Tensor>::cell_gradient(QuadraticTetVolMesh::Cell::index_type ci);

template <class T>
Vector QuadraticTetVolField<T>::cell_gradient(QuadraticTetVolMesh::Cell::index_type ci)
{
  // for now we only know how to do this for field with doubles at the nodes
  ASSERT(data_at() == Field::NODE);

  // load up the indices of the nodes for this cell
  QuadraticTetVolMesh::Node::array_type nodes;
  get_typed_mesh()->get_nodes(nodes, ci);
  Vector gb0, gb1, gb2, gb3, gb4, gb5, gb6, gb7, gb8, gb9;

  // get basis at the cell center...
  Point center;
  get_typed_mesh()->get_center(center, ci);
  get_typed_mesh()->get_gradient_basis(ci, center, gb0, gb1, gb2, gb3, gb4, 
				       gb5, gb6, gb7, gb8, gb9);

  // we really want this for all scalars... 
  //  but for now, we'll just make doubles work
  return Vector(gb0 * value(nodes[0]) + gb1 * value(nodes[1]) + 
		gb2 * value(nodes[2]) + gb3 * value(nodes[3]) +
		gb4 * value(nodes[4]) + gb5 * value(nodes[5]) +
		gb6 * value(nodes[6]) + gb7 * value(nodes[7]) +
		gb8 * value(nodes[8]) + gb9 * value(nodes[9]));
}


} // end namespace SCIRun

#endif // Datatypes_QuadraticTetVolField_h
