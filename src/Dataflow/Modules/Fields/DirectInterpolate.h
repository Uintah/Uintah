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

//    File   : DirectInterpolate.h
//    Author : Michael Callahan
//    Date   : June 2001

#if !defined(DirectInterpolate_h)
#define DirectInterpolate_h

#include <Core/Thread/Barrier.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <sci_hash_map.h>
#include <float.h> // for DBL_MAX

namespace SCIRun {

class DirectInterpAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src, MeshHandle dst,
			      Field::data_location loc,
			      const string &basis, bool source_to_single_dest,
			      bool exhaustive_search, double dist, int np) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const TypeDescription *lsrc,
					    const TypeDescription *fdst,
					    const TypeDescription *ldst);
};


template <class FSRC, class LSRC, class FOUT, class LDST>
class DirectInterpAlgoT : public DirectInterpAlgo
{
  typedef pair<typename LDST::index_type, 
    vector<typename LSRC::index_type> > dst_src_pair;
#ifdef HAVE_HASH_MAP
  typedef hash_map<unsigned int,
    dst_src_pair,
    hash<unsigned int>, 
    equal_to<unsigned int> > hash_type;
#else
  typedef map<unsigned int,
    dst_src_pair,
    equal_to<unsigned int> > hash_type;
#endif

  typedef struct _DIData {
    FieldHandle src_fieldH;
    MeshHandle dst_meshH;
    Field::data_location loc;
    string basis;
    bool source_to_single_dest;
    bool exhaustive_search;
    double dist;
    int np;
    FieldHandle out_fieldH;
    hash_type dstmap;
    Barrier barrier;
    Mutex maplock;
    
    _DIData() : barrier("DirectInterpolate Barrier"), maplock("DirectInterpolate Map Lock") {}
  } DIData;

public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src, MeshHandle dst,
			      Field::data_location loc,
			      const string &basis, bool source_to_single_dest,
			      bool exhaustive_search, double dist, int np);

private:
  double find_closest_src_loc(typename LSRC::index_type &index,
			      typename FSRC::mesh_type *mesh, 
			      const Point &p) const;
  double find_closest_dst_loc(typename LDST::index_type &index,
			      typename FOUT::mesh_type *mesh, 
			      const Point &p) const;
  void parallel_execute(int proc, DIData *d);
};

template <class FSRC, class LSRC, class FOUT, class LDST>
double
DirectInterpAlgoT<FSRC, LSRC, FOUT, LDST>::find_closest_src_loc(typename LSRC::index_type &index, typename FSRC::mesh_type *mesh, const Point &p) const
{
  double mindist = DBL_MAX;

  typename LSRC::iterator itr, eitr;
  mesh->begin(itr);
  mesh->end(eitr);
  while (itr != eitr)
  {
    Point c;
    mesh->get_center(c, *itr);
    const double dist = (p - c).length2();
    if (dist < mindist)
    {
      mindist = dist;
      index = *itr;
    }
    ++itr;
  }
  return mindist;
}

template <class FSRC, class LSRC, class FOUT, class LDST>
double
DirectInterpAlgoT<FSRC, LSRC, FOUT, LDST>::find_closest_dst_loc(typename LDST::index_type &index, typename FOUT::mesh_type *mesh, const Point &p) const
{
  double mindist = DBL_MAX;
  
  typename LDST::iterator itr, eitr;
  mesh->begin(itr);
  mesh->end(eitr);
  while (itr != eitr)
  {
    Point c;
    mesh->get_center(c, *itr);
    const double dist = (p - c).length2();
    if (dist < mindist)
    {
      mindist = dist;
      index = *itr;
    }
    ++itr;
  }
  return mindist;
}

template <class FSRC, class LSRC, class FOUT, class LDST>
FieldHandle
DirectInterpAlgoT<FSRC, LSRC, FOUT, LDST>::execute(FieldHandle src_fieldH, MeshHandle dst_meshH, Field::data_location loc, const string &basis, bool source_to_single_dest, bool exhaustive_search, double dist, int np)
{
  DIData d;
  d.src_fieldH=src_fieldH;
  d.dst_meshH=dst_meshH;
  d.loc=loc;
  d.basis=basis;
  d.source_to_single_dest=source_to_single_dest;
  d.exhaustive_search=exhaustive_search;
  d.dist=dist;
  d.np=np;
  typename FOUT::mesh_type *dst_mesh = 
    dynamic_cast<typename FOUT::mesh_type *>(dst_meshH.get_rep());
  FOUT *out_field = scinew FOUT(dst_mesh, loc);  
  d.out_fieldH = out_field;

  if (np==1)
    parallel_execute(0, &d);
  else
    Thread::parallel(this, 
	     &DirectInterpAlgoT<FSRC, LSRC, FOUT, LDST>::parallel_execute,
	     np, true, &d);
  return out_field;
}

template <class FSRC, class LSRC, class FOUT, class LDST>
void
DirectInterpAlgoT<FSRC, LSRC, FOUT, LDST>::parallel_execute(int proc,
							    DIData *d) {
  FieldHandle src_fieldH = d->src_fieldH;
  MeshHandle dst_meshH = d->dst_meshH;
  Field::data_location loc = d->loc;
  const string& basis = d->basis;
  bool source_to_single_dest = d->source_to_single_dest;
  bool exhaustive_search = d->exhaustive_search;
  double dist = d->dist;
  int np = d->np;
  FieldHandle out_fieldH = d->out_fieldH;
  
  FSRC *src_field = dynamic_cast<FSRC *>(src_fieldH.get_rep());
  typename FSRC::mesh_type *src_mesh = src_field->get_typed_mesh().get_rep();
  typename FOUT::mesh_type *dst_mesh = 
    dynamic_cast<typename FOUT::mesh_type *>(dst_meshH.get_rep());
  FOUT *out_field = dynamic_cast<FOUT *>(out_fieldH.get_rep());

  if (proc == 0) {
    if (loc == Field::NODE) src_mesh->synchronize(Mesh::NODES_E);
    else if (loc == Field::CELL) src_mesh->synchronize(Mesh::CELLS_E);
    else if (loc == Field::FACE) src_mesh->synchronize(Mesh::FACES_E);
    else if (loc == Field::EDGE) src_mesh->synchronize(Mesh::EDGES_E);
  }
  d->barrier.wait(np);
  int count=0;

  if ((basis == "constant") && source_to_single_dest) {
    // For each source location, we will map it to a single destination
    //   location.  This is different from our other interpolation
    //   methods in that here, many destination locations are likely to
    //   have no source location mapped to them at all.
    // Note: it is possible that multiple source will be mapped to the
    //   same destination, which is fine -- but we will flag it as a
    //   remark, just to let the user know.  
    // Also: if a source is outside of the destination volume,
    //   and the "exhaustive search" option is not selected,
    //   that source will not be mapped to any destination.
    if (proc == 0) {
      typename LDST::iterator itr, end_itr;
      dst_mesh->begin(itr);
      dst_mesh->end(end_itr);
      typename FOUT::value_type val(0);
      while (itr != end_itr) {
	out_field->set_value(val, *itr);
	++itr;
      }
    }
    d->barrier.wait(np);

    typename LSRC::iterator itr, end_itr;
    src_mesh->begin(itr);
    src_mesh->end(end_itr);
    typename LDST::size_type sz;
    dst_mesh->size(sz);

    while (itr != end_itr) {
      if (count%np != proc) {
	++itr;
	++count;
	continue;
      }
      typename LDST::array_type locs;
      vector<double> weights;
      Point p;
      src_mesh->get_center(p, *itr);
      bool failed = true;
      dst_mesh->get_weights(p, locs, weights);
      if (weights.size() > 0) {
	failed = false;
	double max_weight=weights[0];
	int max_idx=0;
	for (unsigned int i=1; i<locs.size(); i++) {
	  if (weights[i] > max_weight) {
	    max_idx = i;
	    max_weight = weights[i];
	  }
	}
	unsigned int uint_idx = (unsigned int) locs[max_idx];
	d->maplock.lock();
	typename hash_type::iterator dst_iter = d->dstmap.find(uint_idx);
	if (dst_iter != d->dstmap.end()) {
	  dst_iter->second.second.push_back(*itr);
	} else {
	  vector<typename LSRC::index_type> v;
	  v.push_back(*itr);
	  d->dstmap[uint_idx] = dst_src_pair(locs[max_idx], v);
	}
	d->maplock.unlock();
      }
      if (exhaustive_search && failed) {
	typename LDST::index_type index;
	double dd=find_closest_dst_loc(index, dst_mesh, p);
	if (dist<=0 || dd<dist) {
	  unsigned int uint_idx = (unsigned int) index;
	  d->maplock.lock();
	  typename hash_type::iterator dst_iter = d->dstmap.find(uint_idx);
	  if (dst_iter != d->dstmap.end()) {
	    dst_iter->second.second.push_back(*itr);
	  } else {
	    vector<typename LSRC::index_type> v;
	    v.push_back(*itr);
	    d->dstmap[uint_idx] = dst_src_pair(index, v);
	  }
	  d->maplock.unlock();
	}
      }
      ++itr;
      ++count;
    }
    d->barrier.wait(np);
    typename hash_type::iterator dst_iter = d->dstmap.begin();
    count=0;
    while (dst_iter != d->dstmap.end()) {
      if (count%np != proc) {
	++dst_iter;
	++count;
	continue;
      }
      vector<pair<typename LSRC::index_type, double> > v;
      unsigned long n=dst_iter->second.second.size();
      double invn=1./((double)n);
      if (n) {
	typename FOUT::value_type val =
	  (typename FOUT::value_type)(src_field->value(dst_iter->second.second[0])*invn);
	for (unsigned int i=1; i<n; i++) {
	  val += 
	    (typename FOUT::value_type)(src_field->value(dst_iter->second.second[i])*invn);
	}
	out_field->set_value(val, dst_iter->second.first);
      }
      ++dst_iter;
      ++count;
    }
  } else { // linear (or constant, with each src mapping to many dests)
    typename LDST::iterator itr, end_itr;
    dst_mesh->begin(itr);
    dst_mesh->end(end_itr);
    bool linear(basis == "linear");
    while (itr != end_itr) {
      if (count%np != proc) {
	++itr;
	++count;
	continue;
      }
      typename LSRC::array_type locs;
      vector<double> weights;
      Point p;
      dst_mesh->get_center(p, *itr);
      bool failed = true;
      src_mesh->get_weights(p, locs, weights);
      typename FOUT::value_type val;
      if (weights.size() > 0)	{
	failed = false;
	if (linear) {
	  if (locs.size())
	    val = (typename FOUT::value_type)(src_field->value(locs[0])*weights[0]);
	  for (unsigned int i = 1; i < locs.size(); i++) {
	    val +=(typename FOUT::value_type)(src_field->value(locs[i])*weights[i]);
	  }
	} else {
	  double max_weight=weights[0];
	  int max_idx=0;
	  for (unsigned int i=1; i<locs.size(); i++) {
	    if (weights[i] > max_weight) {
	      max_idx = i;
	      max_weight = weights[i];
	    }
	  }
	  val = (typename FOUT::value_type)(src_field->value(locs[max_idx]));
	}
      }
      if (exhaustive_search && failed) {
	typename LSRC::index_type index;
	double d=find_closest_src_loc(index, src_mesh, p);
	if (dist<=0 || d<dist)
	{
	  failed = false;
	  val = (typename FOUT::value_type)(src_field->value(index));
	}
      }
      if (failed) val = typename FOUT::value_type(0);
      out_field->set_value(val, *itr);
      ++itr;
      ++count;
    }
  }
}


} // end namespace SCIRun

#endif // DirectInterpolate_h
