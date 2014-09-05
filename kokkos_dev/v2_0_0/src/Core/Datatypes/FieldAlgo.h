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

#include <sgi_stl_warnings_off.h>
#include <utility>
#include <sgi_stl_warnings_on.h>

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
