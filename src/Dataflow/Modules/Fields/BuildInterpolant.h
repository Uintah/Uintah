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

//    File   : BuildInterpolant.h
//    Author : Michael Callahan
//    Date   : June 2001

#if !defined(BuildInterpolant_h)
#define BuildInterpolant_h

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/DynamicLoader.h>

namespace SCIRun {

#define BIA_MAX_DISTANCE (1.0e6)

class BuildInterpAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(MeshHandle src, MeshHandle dst,
			      Field::data_location loc) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *msrc,
				       const TypeDescription *lsrc,
				       const TypeDescription *mdst,
				       const TypeDescription *ldst,
				       const TypeDescription *fdst);
};


template <class MSRC, class LSRC, class MDST, class LDST, class FOUT>
class BuildInterpAlgoT : public BuildInterpAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(MeshHandle src, MeshHandle dst,
			      Field::data_location loc);

private:
  double find_closest(typename LSRC::index_type &index,
		      MSRC *mesh, const Point &p);
};



template <class MSRC, class LSRC, class MDST, class LDST, class FOUT>
double
BuildInterpAlgoT<MSRC, LSRC, MDST, LDST, FOUT>::find_closest(typename LSRC::index_type &index, MSRC *mesh, const Point &p)
{
  typename LSRC::iterator itr, eitr;
  mesh->begin(itr);
  mesh->end(eitr);
  double mindist = BIA_MAX_DISTANCE;
  while (itr != eitr)
  {
    Point c;
    mesh->get_center(c, *itr);
    const double dist = (p - c).length();
    if (dist < mindist)
    {
      mindist = dist;
      index = *itr;
    }
    ++itr;
  }
  return mindist;
}


template <class MSRC, class LSRC, class MDST, class LDST, class FOUT>
FieldHandle
BuildInterpAlgoT<MSRC, LSRC, MDST, LDST, FOUT>::execute(MeshHandle src_meshH, MeshHandle dst_meshH, Field::data_location loc)
{
  MSRC *src_mesh = dynamic_cast<MSRC *>(src_meshH.get_rep());
  MDST *dst_mesh = dynamic_cast<MDST *>(dst_meshH.get_rep());
  FOUT *ofield = scinew FOUT(dst_mesh, loc);

  typename LDST::iterator itr, end_itr;
  dst_mesh->begin(itr);
  dst_mesh->end(end_itr);

  while (itr != end_itr)
  {
    typename LSRC::array_type locs;
    vector<double> weights;
    Point p;

    dst_mesh->get_center(p, *itr);

    src_mesh->get_weights(p, locs, weights);

    vector<pair<typename LSRC::index_type, double> > v;
    if (weights.size() > 0)
    {
      for (unsigned int i = 0; i < locs.size(); i++)
      {
	v.push_back(pair<typename LSRC::index_type, double>
		    (locs[i], weights[i]));
      }
    }
    else
    {
      typename LSRC::index_type index;
      if (find_closest(index, src_mesh, p) < BIA_MAX_DISTANCE)
      {
	v.push_back(pair<typename LSRC::index_type, double>(index, 1.0));
      }
    }

    ofield->set_value(v, *itr);
    ++itr;
  }

  FieldHandle fh(ofield);
  return fh;
}


} // end namespace SCIRun

#endif // BuildInterpolant_h
