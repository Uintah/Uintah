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
class TetVol : public GenericField<TetVolMesh, vector<T> >
{
public:
  TetVol();
  TetVol(Field::data_location data_at);
  TetVol(TetVolMeshHandle mesh, Field::data_location data_at);
  virtual TetVol<T> *clone() const;
  virtual ~TetVol();

  virtual ScalarFieldInterface* query_scalar_interface() const;
  virtual VectorFieldInterface* query_vector_interface() const;
  virtual TensorFieldInterface* query_tensor_interface() const;

  //! Persistent IO
  void    io(Piostream &stream);
  static  PersistentTypeID type_id;
  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const;
  virtual const TypeDescription* get_type_description() const;

  // TetVol specific methods.
  bool get_gradient(Vector &, Point &);
  Vector cell_gradient(TetVolMesh::Cell::index_type);

private:
  static Persistent *maker();
};

template <class T>
TetVol<T>::TetVol()
  : GenericField<TetVolMesh, vector<T> >()
{
}

template <class T>
TetVol<T>::TetVol(Field::data_location data_at)
  : GenericField<TetVolMesh, vector<T> >(data_at)
{
}

template <class T>
TetVol<T>::TetVol(TetVolMeshHandle mesh, Field::data_location data_at)
  : GenericField<TetVolMesh, vector<T> >(mesh, data_at)
{
}

template <class T>
TetVol<T> *
TetVol<T>::clone() const
{
  return new TetVol(*this);
}

template <class T>
TetVol<T>::~TetVol()
{
}

template <> ScalarFieldInterface *
TetVol<double>::query_scalar_interface() const;

template <> ScalarFieldInterface *
TetVol<float>::query_scalar_interface() const;

template <> ScalarFieldInterface *
TetVol<int>::query_scalar_interface() const;

template <> ScalarFieldInterface*
TetVol<short>::query_scalar_interface() const;

template <> ScalarFieldInterface*
TetVol<unsigned char>::query_scalar_interface() const;

template <class T>
ScalarFieldInterface*
TetVol<T>::query_scalar_interface() const 
{
  return 0;
}

template <>
VectorFieldInterface*
TetVol<Vector>::query_vector_interface() const;

template <class T>
VectorFieldInterface*
TetVol<T>::query_vector_interface() const
{
  return 0;
}

template <>
TensorFieldInterface*
TetVol<Tensor>::query_tensor_interface() const;

template <class T>
TensorFieldInterface*
TetVol<T>::query_tensor_interface() const
{
  return 0;
}


template <class T>
Persistent*
TetVol<T>::maker()
{
  return scinew TetVol<T>;
}


template <class T>
PersistentTypeID 
TetVol<T>::type_id(type_name(-1), 
		   GenericField<TetVolMesh, vector<T> >::type_name(-1),
		   maker);


// Pio defs.
const int TET_VOL_VERSION = 1;

template <class T>
void 
TetVol<T>::io(Piostream& stream)
{
  stream.begin_class(type_name(-1), TET_VOL_VERSION);
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


template <class T> 
const string
TetVol<T>::get_type_name(int n) const
{
  return type_name(n);
}

template <class T>
const TypeDescription* 
get_type_description(TetVol<T>*)
{
  static TypeDescription* td = 0;
  static string name("TetVol");
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
TetVol<T>::get_type_description() const 
{
  return SCIRun::get_type_description((TetVol<T>*)0);
}

//! compute the gradient g, at point p
template <class T>
bool TetVol<T>::get_gradient(Vector &g, Point &p) {
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
Vector TetVol<Vector>::cell_gradient(TetVolMesh::Cell::index_type ci);

template <>
Vector TetVol<Tensor>::cell_gradient(TetVolMesh::Cell::index_type ci);

template <class T>
Vector TetVol<T>::cell_gradient(TetVolMesh::Cell::index_type ci)
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

#endif // Datatypes_TetVol_h
