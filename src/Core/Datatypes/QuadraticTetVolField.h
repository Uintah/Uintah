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
#include <vector>

namespace SCIRun {

template <class T> 
class QuadraticTetVolField : public GenericField<QuadraticTetVolMesh, vector<T> >
{
public:
  QuadraticTetVolField();
  QuadraticTetVolField(Field::data_location data_at);
  QuadraticTetVolField(QuadraticTetVolMeshHandle mesh, 
		       Field::data_location data_at);

  static QuadraticTetVolField<T>* create_from(const TetVolField<T> &);
  virtual QuadraticTetVolField<T> *clone() const;
  virtual ~QuadraticTetVolField();

  virtual ScalarFieldInterface* query_scalar_interface() const;
  virtual VectorFieldInterface* query_vector_interface() const;
  virtual TensorFieldInterface* query_tensor_interface() const;

  //! Persistent IO
  void    io(Piostream &stream);
  static  PersistentTypeID type_id;
  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const;
  virtual const TypeDescription* get_type_description() const;

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
  PropertyManager *pm = rval;
  *pm = tv;
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

template <> ScalarFieldInterface *
QuadraticTetVolField<double>::query_scalar_interface() const;

template <> ScalarFieldInterface *
QuadraticTetVolField<float>::query_scalar_interface() const;

template <> ScalarFieldInterface *
QuadraticTetVolField<int>::query_scalar_interface() const;

template <> ScalarFieldInterface*
QuadraticTetVolField<short>::query_scalar_interface() const;

template <> ScalarFieldInterface*
QuadraticTetVolField<char>::query_scalar_interface() const;

template <> ScalarFieldInterface *
QuadraticTetVolField<unsigned int>::query_scalar_interface() const;

template <> ScalarFieldInterface*
QuadraticTetVolField<unsigned short>::query_scalar_interface() const;

template <> ScalarFieldInterface*
QuadraticTetVolField<unsigned char>::query_scalar_interface() const;

template <class T>
ScalarFieldInterface*
QuadraticTetVolField<T>::query_scalar_interface() const 
{
  return 0;
}

template <>
VectorFieldInterface*
QuadraticTetVolField<Vector>::query_vector_interface() const;

template <class T>
VectorFieldInterface*
QuadraticTetVolField<T>::query_vector_interface() const
{
  return 0;
}

template <>
TensorFieldInterface*
QuadraticTetVolField<Tensor>::query_tensor_interface() const;

template <class T>
TensorFieldInterface*
QuadraticTetVolField<T>::query_tensor_interface() const
{
  return 0;
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
const string
QuadraticTetVolField<T>::get_type_name(int n) const
{
  return type_name(n);
}

template <class T>
const TypeDescription* 
get_type_description(QuadraticTetVolField<T>*)
{
  static TypeDescription* td = 0;
  static string name("QuadraticTetVolField");
  static string namesp("SCIRun");
  static string path(__FILE__);
  if(!td){
    const TypeDescription *sub = SCIRun::get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription(name, subs, path, namesp);
  }
  return td;
}

template <class T>
const TypeDescription* 
QuadraticTetVolField<T>::get_type_description() const 
{
  return SCIRun::get_type_description((QuadraticTetVolField<T>*)0);
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
