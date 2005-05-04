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

//    File   : QuadraticTetVolField.h
//    Author : Martin Cole
//    Date   : Sun Feb 24 13:47:31 2002

#ifndef Datatypes_QuadraticTetVolField_h
#define Datatypes_QuadraticTetVolField_h

#include <Core/Datatypes/QuadraticTetVolMesh.h>
#include <Core/Datatypes/TetVolField.h>
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
  QuadraticTetVolField(int order);
  QuadraticTetVolField(QuadraticTetVolMeshHandle mesh, 
		       int order);

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
QuadraticTetVolField<T>::QuadraticTetVolField(int order) : 
  GenericField<QuadraticTetVolMesh, vector<T> >(order)
{
}

template <class T>
QuadraticTetVolField<T>::QuadraticTetVolField(QuadraticTetVolMeshHandle mesh, 
					      int order) : 
  GenericField<QuadraticTetVolMesh, vector<T> >(mesh, order)
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
							      tv.basis_order());
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
  if (this->mesh_->locate(ci, p)) {
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
  ASSERT(this->basis_order() == 1);

  // load up the indices of the nodes for this cell
  QuadraticTetVolMesh::Node::array_type nodes;
  this->mesh_->get_nodes(nodes, ci);
  Vector gb0, gb1, gb2, gb3, gb4, gb5, gb6, gb7, gb8, gb9;

  // get basis at the cell center...
  Point center;
  this->mesh_->get_center(center, ci);
  this->mesh_->get_gradient_basis(ci, center, gb0, gb1, gb2, gb3, gb4, 
				  gb5, gb6, gb7, gb8, gb9);

  // we really want this for all scalars... 
  //  but for now, we'll just make doubles work
  return Vector(gb0 * this->value(nodes[0]) + gb1 * this->value(nodes[1]) + 
		gb2 * this->value(nodes[2]) + gb3 * this->value(nodes[3]) +
		gb4 * this->value(nodes[4]) + gb5 * this->value(nodes[5]) +
		gb6 * this->value(nodes[6]) + gb7 * this->value(nodes[7]) +
		gb8 * this->value(nodes[8]) + gb9 * this->value(nodes[9]));
}


} // end namespace SCIRun

#endif // Datatypes_QuadraticTetVolField_h
