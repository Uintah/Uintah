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

//    File   : MapDistanceField.h
//    Author : Michael Callahan
//    Date   : June 2001

#if !defined(MapDistanceField_h)
#define MapDistanceField_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

namespace SCIRun {

class MapDistanceFieldAlgo : public DynamicAlgoBase
{
public:
  virtual pair<FieldHandle, FieldHandle> execute(FieldHandle fsrcH,
						 MeshHandle dstH,
						 Field::data_location l) = 0;

  double distance_to_line2(const Point &p,
			   const Point &a, const Point &b) const;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const TypeDescription *lsrc,
					    const TypeDescription *mdst,
					    const TypeDescription *ldst,
					    const TypeDescription *fdst);
};


template <class FSRC, class LSRC, class MDST, class LDST,
  class FOUTS, class FOUTD>
class MapDistanceFieldAlgoT : public MapDistanceFieldAlgo
{
public:
  //! virtual interface. 
  virtual pair<FieldHandle, FieldHandle> execute(FieldHandle fsrcH,
						 MeshHandle dstH,
						 Field::data_location ldst);
};



template <class FSRC, class LSRC, class MDST, class LDST,
  class FOUTS, class FOUTD>
pair<FieldHandle, FieldHandle>
MapDistanceFieldAlgoT<FSRC, LSRC, MDST, LDST, FOUTS, FOUTD>::
execute(FieldHandle fsrcH, MeshHandle mdstH, Field::data_location loc_dst)
{
  FSRC *fsrc = dynamic_cast<FSRC *>(fsrcH.get_rep());
  MDST *mdst = dynamic_cast<MDST *>(mdstH.get_rep());
  typename FSRC::mesh_handle_type msrc = fsrc->get_typed_mesh();

  FOUTD *foutdst = scinew FOUTD(mdst, loc_dst);
  FOUTS *foutsrc = scinew FOUTS(msrc, fsrc->data_at());

  typename LDST::iterator itr, end_itr;
  mdst->begin(itr);
  mdst->end(end_itr);

  
  typename LSRC::size_type msrcsize;
  msrc->size(msrcsize);
  vector<int> edgecounts((unsigned int)msrcsize, 0);

  while (itr != end_itr)
  {
    Point location;
    mdst->get_center(location, *itr);
    
    double dist = 1.0e6;

    typename LSRC::iterator citr, citr_end;
    msrc->begin(citr);
    msrc->end(citr_end);
    typename LSRC::index_type edge = *citr;
    while (citr != citr_end)
    {
      typename FSRC::value_type val;
      fsrc->value(val, *citr);
      val = val * val;
      if (val < 1.0e-3) { val = 1.0e-3; }

      typename FSRC::mesh_type::Node::array_type array;
      msrc->get_nodes(array, *citr);

      Point cloc1, cloc2;
      msrc->get_center(cloc1, array[0]);
      msrc->get_center(cloc2, array[1]);
//      const double tmp = distance_to_line2(location, cloc1, cloc2) / val;
      const double tmp = distance_to_line2(location, cloc1, cloc2);
      if (tmp < dist)
      {
	dist = tmp;
	edge = *citr;
      }
      ++citr;
    }
    vector<pair<typename LSRC::index_type, double> > svec;
    svec.push_back(pair<typename LSRC::index_type, double>(edge, 1.0));
    foutdst->set_value(svec, *itr);

    edgecounts[(unsigned int)edge]++;

    // push face onto edge in foutsrc
    vector<pair<typename LDST::index_type, double> > dvec;
    foutsrc->value(dvec, edge);
    dvec.push_back(pair<typename LDST::index_type, double>(*itr, 1.0));
    foutsrc->set_value(dvec, edge);

    ++itr;
  }

  // Normalize the values at the edges
  typename LSRC::iterator citr, citr_end;
  msrc->begin(citr);
  msrc->end(citr_end);
  while (citr != citr_end)
  {
    vector<pair<typename LDST::index_type, double> > dvec;
    foutsrc->value(dvec, *citr);
    if (dvec.size() == 0)
    {
// This doesn't work for some reason.  crb 6/15/02
//      module.error("Edge " + to_string((unsigned int)(*citr)) +
//		   " is mapped to zero surface elements.");
    }
    for (unsigned int i=0; i<dvec.size(); i++)
    {
      dvec[i].second = 1.0 / (double)dvec.size();
    }
    foutsrc->set_value(dvec, *citr);

    ++citr;
  }

  // Normalize the values at the edges
  mdst->begin(itr);
  mdst->end(end_itr);
  while (itr != end_itr)
  {
    vector<pair<typename LSRC::index_type, double> > svec;
    foutdst->value(svec, *itr);
    svec[0].second = 1.0 / edgecounts[(unsigned int)(svec[0].first)];
    foutdst->set_value(svec, *itr);

    ++itr;
  }

  return pair<FieldHandle, FieldHandle>(foutsrc, foutdst);
}


} // end namespace SCIRun

#endif // MapDistanceField_h
