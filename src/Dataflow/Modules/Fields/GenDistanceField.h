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

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/DynamicLoader.h>

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


  virtual void execute_node(FieldHandle fdst, FieldHandle fsrc,
			    bool accumulate_p) = 0;

  virtual void execute_edge(FieldHandle fdst, FieldHandle fsrc,
			    bool accumulate_p) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *fdst,
				       const TypeDescription *ldst,
				       const TypeDescription *fsrc);
};


template <class FDST, class LDST, class FSRC>
class GenDistanceFieldAlgoT : public GenDistanceFieldAlgo
{
public:
  //! virtual interface. 
  virtual void execute_node(FieldHandle fdst, FieldHandle fsrc,
			    bool accumulate_p);

  virtual void execute_edge(FieldHandle fdst, FieldHandle fsrc,
			    bool accumulate_p);
};


template <class FDST, class LDST, class FSRC>
void
GenDistanceFieldAlgoT<FDST, LDST, FSRC>::execute_node(FieldHandle fdsthandle,
						      FieldHandle fsrchandle,
						      bool accumulate_p)
{
  FDST *fdst = dynamic_cast<FDST *>(fdsthandle.get_rep());
  FSRC *fsrc = dynamic_cast<FSRC *>(fsrchandle.get_rep());

  typename FDST::mesh_handle_type mdst = fdst->get_typed_mesh();
  typename FSRC::mesh_handle_type msrc = fsrc->get_typed_mesh();

  typename LDST::iterator itr, eitr;
  mdst->begin(itr);
  mdst->end(eitr);
  while (itr != eitr)
  {
    Point location;
    mdst->get_center(location, *itr);

    double dist;
    if (accumulate_p)
    {
      typename FDST::value_type val;
      fdst->value(val, *itr);
      dist = val * val;
    }
    else
    {
      dist = 1.0e6;
    }

    // Compute shorted weighted distance.
    typename FSRC::mesh_type::Node::iterator citr, ecitr;
    msrc->begin(citr);
    msrc->end(ecitr);
    while (citr != ecitr)
    {
      typename FSRC::value_type val;
      fsrc->value(val, *citr);
      val = val * val;
      if (val < 1.0e-3) { val = 1.0e-3; }

      Point cloc;
      msrc->get_center(cloc, *citr);

      dist = min((location - cloc).length2() * val, dist);
      ++citr;
    }

    dist = sqrt(dist);
    fdst->set_value(dist, *itr);

    ++itr;
  }
}


template <class FDST, class LDST, class FSRC>
void
GenDistanceFieldAlgoT<FDST, LDST, FSRC>::execute_edge(FieldHandle fdsthandle,
						      FieldHandle fsrchandle,
						      bool accumulate_p)
{
  FDST *fdst = dynamic_cast<FDST *>(fdsthandle.get_rep());
  FSRC *fsrc = dynamic_cast<FSRC *>(fsrchandle.get_rep());

  typename FDST::mesh_handle_type mdst = fdst->get_typed_mesh();
  typename FSRC::mesh_handle_type msrc = fsrc->get_typed_mesh();

  typename LDST::iterator itr, eitr;
  mdst->begin(itr);
  mdst->end(eitr);
  while (itr != eitr)
  {
    Point location;
    mdst->get_center(location, *itr);

    double dist;
    if (accumulate_p)
    {
      typename FDST::value_type val;
      fdst->value(val, *itr);
      dist = val * val;
    }
    else
    {
      dist = 1.0e6;
    }

    // Compute shorted weighted distance.
    typename FSRC::mesh_type::Edge::iterator citr, ecitr;
    msrc->begin(citr);
    msrc->end(ecitr);
    while (citr != ecitr)
    {
      typename FSRC::value_type val;
      fsrc->value(val, *citr);
      val = val * val;
      if (val < 1.0e-3) { val = 1.0e-3; }

      typename FSRC::mesh_type::Node::array_type array;

      msrc->get_nodes(array, *citr);

      Point cloc1;
      Point cloc2;
      msrc->get_center(cloc1, array[0]);
      msrc->get_center(cloc2, array[1]);

      dist = min(distance_to_line2(location, cloc1, cloc2) * val, dist);
      ++citr;
    }

    dist = sqrt(dist);
    fdst->set_value(dist, *itr);

    ++itr;
  }
}


} // end namespace SCIRun

#endif // GenDistanceField_h
