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
 *  PrismVolField.h
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computing
 *   University of Utah
 *
 *  Copyright (C) 2003 SCI Institute
 */

#ifndef Datatypes_PrismVolField_h
#define Datatypes_PrismVolField_h

#include <Core/Datatypes/PrismVolMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Util/Assert.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

template <class T> 
class PrismVolField : public GenericField<PrismVolMesh, vector<T> >
{
public:
  PrismVolField();
  PrismVolField(int order);
  PrismVolField(PrismVolMeshHandle mesh, int order);
  virtual PrismVolField<T> *clone() const;
  virtual ~PrismVolField();

  //! Persistent IO
  virtual void io(Piostream &stream);
  static  PersistentTypeID type_id;

  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

  // PrismVolField specific methods.
  bool get_gradient(Vector &, const Point &);
  Vector cell_gradient(PrismVolMesh::Cell::index_type);

private:
  static Persistent *maker();
};

template <class T>
PrismVolField<T>::PrismVolField()
  : GenericField<PrismVolMesh, vector<T> >()
{
}

template <class T>
PrismVolField<T>::PrismVolField(int order)
  : GenericField<PrismVolMesh, vector<T> >(order)
{
  ASSERTMSG((! (order == 0 && this->mesh_->dimensionality() == 1)), 
	    "PrismVolField does NOT currently support data at edges."); 
  ASSERTMSG((! (order == 0 && this->mesh_->dimensionality() == 2)), 
	    "PrismVolField does NOT currently support data at faces."); 
}

template <class T>
PrismVolField<T>::PrismVolField(PrismVolMeshHandle mesh, int order)
  : GenericField<PrismVolMesh, vector<T> >(mesh, order)
{
  ASSERTMSG((! (order == 0 && this->mesh_->dimensionality() == 1)), 
	    "PrismVolField does NOT currently support data at edges."); 
  ASSERTMSG((! (order == 0 && this->mesh_->dimensionality() == 2)), 
	    "PrismVolField does NOT currently support data at faces."); 
}

template <class T>
PrismVolField<T> *
PrismVolField<T>::clone() const
{
  return new PrismVolField(*this);
}

template <class T>
PrismVolField<T>::~PrismVolField()
{
}


template <class T>
Persistent*
PrismVolField<T>::maker()
{
  return scinew PrismVolField<T>;
}


template <class T>
PersistentTypeID 
PrismVolField<T>::type_id(type_name(-1), 
		   GenericField<PrismVolMesh, vector<T> >::type_name(-1),
		   maker);


// Pio defs.
const int PRISM_VOL_FIELD_VERSION = 1;

template <class T>
void 
PrismVolField<T>::io(Piostream& stream)
{
  /* int version=*/stream.begin_class(type_name(-1), PRISM_VOL_FIELD_VERSION);
  GenericField<PrismVolMesh, vector<T> >::io(stream);
  stream.end_class();
}

template <class T> 
const string 
PrismVolField<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    return "PrismVolField";
  }
  else
  {
    return find_type_name((T *)0);
  }
}

template <class T> 
const TypeDescription*
PrismVolField<T>::get_type_description(int n) const
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
bool PrismVolField<T>::get_gradient(Vector &g, const Point &p) {
  PrismVolMesh::Cell::index_type ci;
  if (this->mesh_->locate(ci, p)) {
    g = cell_gradient(ci);
    return true;
  } else {
    return false;
  }
}

//! Compute the gradient g in cell ci.
template <>
Vector PrismVolField<Vector>::cell_gradient(PrismVolMesh::Cell::index_type ci);

template <>
Vector PrismVolField<Tensor>::cell_gradient(PrismVolMesh::Cell::index_type ci);

template <class T>
Vector PrismVolField<T>::cell_gradient(PrismVolMesh::Cell::index_type ci)
{
  // for now we only know how to do this for field with doubles at the nodes
  ASSERT(this->basis_order() == 1);

  // load up the indices of the nodes for this cell
  PrismVolMesh::Node::array_type nodes;
  this->mesh_->get_nodes(nodes, ci);
  Vector gb0, gb1, gb2, gb3, gb4, gb5;
  this->mesh_->get_gradient_basis(ci, gb0, gb1, gb2, gb3, gb4, gb5);

  // we really want this for all scalars... 
  //  but for now, we'll just make doubles work
  return Vector(gb0 * this->value(nodes[0]) + gb1 * this->value(nodes[1]) + 
		gb2 * this->value(nodes[2]) + gb3 * this->value(nodes[3]) + 
		gb4 * this->value(nodes[4]) + gb5 * this->value(nodes[5]));
}


} // end namespace SCIRun

#endif // Datatypes_PrismVolField_h
