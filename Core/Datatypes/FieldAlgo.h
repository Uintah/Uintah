/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 *  FieldAlgo.h
 *
 *  Written by:
 *   Martin Cole
 *   School of Computing
 *   University of Utah
 *
 */

#ifndef Datatypes_FieldAlgo_h
#define Datatypes_FieldAlgo_h

#include <utility>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace SCIRun {
using std::pair;

//! Instead of calling this with a Node::index_type just call get_point yourself.
template <class Mesh, class Index>
void
calc_weights(const Mesh *mesh, Index i, const Point &p, 
	     vector<double> &weights) {
  
  typename Mesh::Node::array_type nodes;
  mesh->get_nodes(nodes, i);

  weights.resize(nodes.size()); //clear and size correctly.
  vector<double>::iterator witer = weights.begin();

  Point np;

  typename Mesh::Node::array_type::iterator iter = nodes.begin();
  while (iter != nodes.end()) {
    typename Mesh::Node::index_type ni = *iter;
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

  pair<value_type,value_type> tmp;
  
  if ( !field.get_property( "minmax", tmp ) ) {
    // compute minmax
    typename Field::fdata_type::iterator i = field.fdata().begin();

    if ( i == field.fdata().end() ) 
      return false;// error! empty field

    tmp.first = tmp.second = *i;
    for (++i; i != field.fdata().end(); i++ ) {
      const value_type v = *i;
      if (v < tmp.first ) tmp.first = v;
      else if ( v > tmp.second ) tmp.second = v;
    }

    // cache in the field properties
    field.set_property("minmax", tmp, true);
  }
  
  minmax.first = T(tmp.first);
  minmax.second = T(tmp.second);

  return true;
}


} // end namespace SCIRun
#endif //Datatypes_FieldAlgo_h
