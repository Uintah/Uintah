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

#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

template <class T> 
class TetVolField : public GenericField<TetVolMesh, vector<T> >
{
public:
  TetVolField();
  TetVolField(int order);
  TetVolField(TetVolMeshHandle mesh, int order);
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
  TetVolMesh::Node::index_type insert_node_watson(Point &);

private:
  static Persistent *maker();
};

template <class T>
TetVolField<T>::TetVolField()
  : GenericField<TetVolMesh, vector<T> >()
{
}

template <class T>
TetVolField<T>::TetVolField(int order)
  : GenericField<TetVolMesh, vector<T> >(order)
{
  ASSERTMSG((! (order == 0 && this->mesh_->dimensionality() == 1)), 
	    "TetVolField does NOT currently support data at edges."); 
  ASSERTMSG((! (order == 0 && this->mesh_->dimensionality() == 2)), 
	    "TetVolField does NOT currently support data at faces."); 
}

template <class T>
TetVolField<T>::TetVolField(TetVolMeshHandle mesh, int order)
  : GenericField<TetVolMesh, vector<T> >(mesh, order)
{
  ASSERTMSG((! (order == 0 && this->mesh_->dimensionality() == 1)), 
	    "TetVolField does NOT currently support data at edges."); 
  ASSERTMSG((! (order == 0 && this->mesh_->dimensionality() == 2)), 
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
bool TetVolField<T>::get_gradient(Vector &g, Point &p) {
  TetVolMesh::Cell::index_type ci;
  if (this->mesh_->locate(ci, p)) {
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
  ASSERT(this->basis_order() == 1);

  // load up the indices of the nodes for this cell
  TetVolMesh::Node::array_type nodes;
  this->mesh_->get_nodes(nodes, ci);
  Vector gb0, gb1, gb2, gb3;
  this->mesh_->get_gradient_basis(ci, gb0, gb1, gb2, gb3);

  // we really want this for all scalars... 
  //  but for now, we'll just make doubles work
  return Vector(gb0 * this->value(nodes[0]) + gb1 * this->value(nodes[1]) + 
		gb2 * this->value(nodes[2]) + gb3 * this->value(nodes[3]));
}




template <>
TetVolMesh::Node::index_type
TetVolField<Tensor>::insert_node_watson(Point &);

template <class T>
TetVolMesh::Node::index_type
TetVolField<T>::insert_node_watson(Point &p) 
{
  TetVolMesh::Node::index_type ret_val;

  if (this->basis_order() == 0) {  
    TetVolMesh::Cell::array_type new_cells, mod_cells;
    ret_val = this->mesh_->insert_node_watson(p,&new_cells,&mod_cells);
    this->resize_fdata();
    
    T tot = this->value(mod_cells[0]);
    const unsigned int num_mod = mod_cells.size();
    for (unsigned int c = 1; c < num_mod; ++c)
      tot += this->value(mod_cells[c]);
    tot /= num_mod;
    for (unsigned int c = 0; c < num_mod; ++c)
      set_value(tot,mod_cells[c]);
    for (unsigned int c = 0; c < new_cells.size(); ++c)
      set_value(tot,new_cells[c]);

  } else if (this->basis_order() == 1) {

    ret_val = this->mesh_->insert_node_watson(p);
    this->resize_fdata();

    vector<TetVolMesh::Node::index_type> nodes;
    this->mesh_->synchronize(Mesh::NODE_NEIGHBORS_E);
    this->mesh_->get_neighbors(nodes, ret_val);

    T tot = this->value(nodes[0]);
    const unsigned int num_nodes = nodes.size();
    for (unsigned int n = 1; n < num_nodes; ++n)
      tot += this->value(nodes[n]);
    tot /= num_nodes;
    set_value(tot, ret_val);

  } else {

    ret_val = this->mesh_->insert_node_watson(p);
    this->resize_fdata();    
  }
  return ret_val;
}

} // end namespace SCIRun

#endif // Datatypes_TetVolField_h
