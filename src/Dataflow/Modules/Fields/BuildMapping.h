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

//    File   : BuildMapping.h
//    Author : Michael Callahan
//    Date   : June 2001

#if !defined(BuildMapping_h)
#define BuildMapping_h

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/DynamicLoader.h>

namespace SCIRun {

#define BIA_MAX_DISTANCE (1.0e6)

class BuildMappingAlgo : public DynamicAlgoBase
{
public:
  virtual pair<FieldHandle, FieldHandle> execute(FieldHandle fsrcH,
						 MeshHandle dstH,
						 Field::data_location l) = 0;

  double distance_to_line2(const Point &p,
			   const Point &a, const Point &b) const;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *fsrc,
				       const TypeDescription *lsrc,
				       const TypeDescription *mdst,
				       const TypeDescription *ldst,
				       const TypeDescription *fdst);
};


template <class FSRC, class LSRC, class MDST, class LDST,
  class FOUTS, class FOUTD>
class BuildMappingAlgoT : public BuildMappingAlgo
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
BuildMappingAlgoT<FSRC, LSRC, MDST, LDST, FOUTS, FOUTD>::
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

  while (itr != end_itr)
  {
    Point location;
    mdst->get_center(location, *itr);
    
    double dist = 1.0e6;

    typename LSRC::iterator citr, citr_end;
    msrc->begin(citr);
    msrc->end(citr_end);
    typename LSRC::index_type edge= *citr;
    while (citr != citr_end)
    {
      typename FSRC::value_type val;
      fsrc->value(val, *citr);
      val = val * val;
      if (val < 1.0e-3) { val = 1.0e-3; }

      typename LSRC::array_type array;
      msrc->get_nodes(array, *citr);

      Point cloc1, cloc2;
      msrc->get_center(cloc1, array[0]);
      msrc->get_center(cloc2, array[1]);
      const double tmp = distance_to_line2(location, cloc1, cloc2) * val;
      if (tmp < dist)
      {
	dist = tmp;
	edge = *citr;
      }
      ++citr;
    }
    vector<pair<typename LSRC::index_type, double> > svec(1);
    svec.push_back(pair<typename LSRC::index_type, double>(edge, 1.0));
    foutdst->set_value(svec, *itr);

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
    for (unsigned int i=0; i<dvec.size(); i++)
    {
      dvec[i].second = 1.0 / dvec.size();
    }
    foutsrc->set_value(dvec, *citr);

    ++citr;
  }

  return pair<FieldHandle, FieldHandle>(foutsrc, foutdst);
}


} // end namespace SCIRun

#endif // BuildMapping_h
