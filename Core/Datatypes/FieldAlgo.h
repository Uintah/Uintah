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
 *  FieldAlgo.h
 *
 *  Written by:
 *   Martin Cole
 *   School of Computing
 *   University of Utah
 *
 *  Copyright (C) 2001 SCI Institute
 */

#ifndef Datatypes_FieldAlgo_h
#define Datatypes_FieldAlgo_h

#include <iostream>
#include <utility>
#include <map>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace SCIRun {
using std::cout;
using std::cerr;
using std::endl;
using std::pair;

//! Instead of calling this with a node_index just call get_point yourself.
template <class Mesh, class Index>
void
calc_weights(const Mesh *mesh, Index i, const Point &p, 
	     typename Mesh::weight_array &weights) {
  
  typename Mesh::node_array nodes;
  mesh->get_nodes(nodes, i);

  weights.resize(nodes.size()); //clear and size correctly.
  typename Mesh::weight_array::iterator witer = weights.begin();

  Point np;

  typename Mesh::node_array::iterator iter = nodes.begin();
  while (iter != nodes.end()) {
    typename Mesh::node_index ni = *iter;
    ++iter;
    mesh->get_point(np, ni);
    // Calculate the weight, and store it.
    *witer = (p - np).length();
    ++witer;
  }
}

template<class Field, class T>
bool
field_minmax( Field &field, pair<T,T>& minmax )
{
  typedef typename Field::value_type value_type;
  pair<value_type,value_type> local_minmax;
  
  if ( !field.get( "minmax", local_minmax ) ) {
    // compute minmax
    typename Field::fdata_type::iterator i = field.fdata().begin();
    if ( i == field.fdata().end() ) 
      return false;// error! empty field

    local_minmax.first = local_minmax.second = *i;
    for (++i; i != field.fdata().end(); i++ ) {
      value_type v = *i;
      if ( v < local_minmax.first ) local_minmax.first = v;
      else if ( v > local_minmax.second ) local_minmax.second = v;
    }

    // cache in the field properties
    field.store( "minmax", local_minmax );
  }

  minmax.first = T(local_minmax.first);
  minmax.second = T(local_minmax.second);

  return true;
}
    

template <class Field, class Functor>
bool
interpolate(const Field &fld, const Point &p, Functor &f) {
  typedef typename Field::mesh_type Mesh;
  
  typename Mesh::cell_index ci;
  const typename Field::mesh_handle_type &mesh = fld.get_typed_mesh();
  if (! mesh->locate(ci, p)) return false;

  calc_weights(mesh.get_rep(), ci, p, f.weights_);

  switch (fld.data_at()) {
  case Field::NODE :
    {
      int i = 0;
      typename Mesh::node_array nodes;
      mesh->get_nodes(nodes, ci);
      typename Mesh::node_array::iterator iter = nodes.begin();
      while (iter != nodes.end()) {
	f(fld, *iter, i);
	++iter; ++i;
      }
    }
  break;
  case Field::EDGE:
    {
    }
    break;
  case Field::FACE:
    {
    }
    break;
  case Field::CELL:
    {
    }
    break;
  case Field::NONE:
    cerr << "Error: Field data at location NONE!!" << endl;
    return false;
  } 
  return true;
} 

template <class Field>
struct InterpFunctor {
  typedef Field field_type;
  typedef typename Field::value_type value_type;
  typedef typename Field::mesh_type::weight_array weight_array;
  InterpFunctor() :
    result_(0) {}

  virtual ~InterpFunctor() {}

  weight_array    weights_;
  value_type      result_;
};

//! Linear Interpolation functor.
template <class Field, class Index>
struct LinearInterp : public InterpFunctor<Field> {
  
  LinearInterp() :
    InterpFunctor<Field>() {}

  void operator()(const Field &field, Index idx, int widx)
  {
    // TODO: This looses precision, bad math.
    result_ += (typename InterpFunctor<Field>::value_type)
      (field.value(idx) * weights_[widx]);
  }
};

} // end namespace SCIRun
#endif //Datatypes_FieldAlgo_h
