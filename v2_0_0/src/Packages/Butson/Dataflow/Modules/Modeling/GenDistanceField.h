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

//    File   : GenDistanceField.h
//    Author : Michael Callahan
//    Date   : June 2001

#if !defined(GenDistanceField_h)
#define GenDistanceField_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

namespace SCIRun {

//! GenDistanceFieldAlgo supports the dynamically loadable algorithm
//! concept.  When dynamically loaded the user will dynamically cast
//! to a GenDistanceFieldBaseBase from the DynamicAlgoBase they will
//! have a pointer to.
class GenDistanceFieldAlgo : public DynamicAlgoBase
{
public:
  double distance_to_line2(const Point &p,
			   const Point &a, const Point &b) const;

  virtual double get_dist(const Point &location, FieldHandle skel) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc);
};


template <class FSRC>
class GenDistanceFieldAlgoT : public GenDistanceFieldAlgo
{
public:
  //! virtual interface. 
  virtual double get_dist(const Point &location, FieldHandle skel);
};


template <class FSRC>
double
GenDistanceFieldAlgoT<FSRC>::get_dist(const Point &location,
				      FieldHandle field_h)
{
  FSRC *fsrc = dynamic_cast<FSRC *>(field_h.get_rep());

  typename FSRC::mesh_handle_type msrc = fsrc->get_typed_mesh();

  double dist = 1.0e6;

  // Compute shorted weighted distance.
  typename FSRC::mesh_type::Edge::iterator citr, ecitr;
  msrc->begin(citr);
  msrc->end(ecitr);
  while (citr != ecitr)
  {
    typename FSRC::value_type val;
    fsrc->value(val, *citr);
    val = val * val;
    if (val < 1.0e-6) { val = 1.0e-6; }

    typename FSRC::mesh_type::Node::array_type array;

    msrc->get_nodes(array, *citr);

    Point cloc1;
    Point cloc2;
    msrc->get_center(cloc1, array[0]);
    msrc->get_center(cloc2, array[1]);

    dist = min(distance_to_line2(location, cloc1, cloc2) / val, dist);
    ++citr;
  }

  return sqrt(dist);
}

 

} // end namespace SCIRun

#endif // GenDistanceField_h
