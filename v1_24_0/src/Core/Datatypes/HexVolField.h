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

#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/GenericField.h>
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
  HexVolField(int order);
  HexVolField(HexVolMeshHandle mesh, int order);
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
HexVolField<T>::HexVolField(int order)
  : GenericField<HexVolMesh, vector<T> >(order)
{
}

template <class T>
HexVolField<T>::HexVolField(HexVolMeshHandle mesh, int order)
  : GenericField<HexVolMesh, vector<T> >(mesh, order)
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
bool HexVolField<T>::get_gradient(Vector &g, Point &p) {
  HexVolMesh::Cell::index_type ci;
  if (this->mesh_->locate(ci, p)) {
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
  ASSERT(this->basis_order() == 1);

  // load up the indices of the nodes for this cell
  HexVolMesh::Node::array_type nodes;
  this->mesh_->get_nodes(nodes, ci);
  Vector gb0, gb1, gb2, gb3, gb4, gb5, gb6, gb7;
  this->mesh_->get_gradient_basis(ci, gb0, gb1, gb2, gb3, gb4, gb5, gb6, gb7);

  // we really want this for all scalars... 
  //  but for now, we'll just make doubles work
  return Vector(gb0 * this->value(nodes[0]) + gb1 * this->value(nodes[1]) + 
		gb2 * this->value(nodes[2]) + gb3 * this->value(nodes[3]) +
		gb4 * this->value(nodes[4]) + gb5 * this->value(nodes[5]) +
		gb6 * this->value(nodes[6]) + gb7 * this->value(nodes[7]));
}


} // end namespace SCIRun

#endif // Datatypes_HexVolField_h
