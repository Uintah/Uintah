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


//    File   : BuildInterpolant.h
//    Author : Michael Callahan
//    Date   : June 2001

#if !defined(BuildInterpolant_h)
#define BuildInterpolant_h

#include <Core/Thread/Barrier.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <sci_hash_map.h>
#include <float.h> // for DBL_MAX

namespace SCIRun {

class BuildInterpAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(MeshHandle src, MeshHandle dst,
			      int basis_order,
			      const string &basis, bool source_to_single_dest,
			      bool exhaustive_search, double dist, int np) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *msrc,
					    const TypeDescription *lsrc,
					    const TypeDescription *mdst,
					    const TypeDescription *ldst,
					    const TypeDescription *fdst);
};


template <class MSRC, class LSRC, class MDST, class LDST, class FOUT>
class BuildInterpAlgoT : public BuildInterpAlgo
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
	    less<unsigned int> > hash_type;
#endif

typedef struct _BIData {
  MeshHandle src_meshH;
  MeshHandle dst_meshH;
  int basis_order;
  string basis;
  bool source_to_single_dest;
  bool exhaustive_search;
  double dist;
  int np;
  FieldHandle out_fieldH;
  hash_type dstmap;
  Barrier barrier;
  Mutex maplock;
  
  _BIData() : barrier("BuildInterpolant Barrier"), maplock("BuildInterp Map Lock") {}
} BIData;

public:
  //! virtual interface. 
  virtual FieldHandle execute(MeshHandle src, MeshHandle dst,
			      int basis_order,
			      const string &basis, bool source_to_single_dest,
			      bool exhaustive_search, double dist, int np);

private:
  double find_closest_src_loc(typename LSRC::index_type &index,
			      MSRC *mesh, const Point &p) const;
  double find_closest_dst_loc(typename LDST::index_type &index,
			      MDST *mesh, const Point &p) const;
  void parallel_execute(int proc, BIData *d);
};



template <class MSRC, class LSRC, class MDST, class LDST, class FOUT>
double
BuildInterpAlgoT<MSRC, LSRC, MDST, LDST, FOUT>::find_closest_src_loc(typename LSRC::index_type &index, MSRC *mesh, const Point &p) const
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

template <class MSRC, class LSRC, class MDST, class LDST, class FOUT>
double
BuildInterpAlgoT<MSRC, LSRC, MDST, LDST, FOUT>::find_closest_dst_loc(typename LDST::index_type &index, MDST *mesh, const Point &p) const
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

template <class MSRC, class LSRC, class MDST, class LDST, class FOUT>
FieldHandle
BuildInterpAlgoT<MSRC, LSRC, MDST, 
		 LDST, FOUT>::execute(MeshHandle src_meshH, 
				      MeshHandle dst_meshH, 
				      int basis_order, 
				      const string &basis, 
				      bool source_to_single_dest, 
				      bool exhaustive_search, 
				      double dist, 
				      int np)
{
  BIData d;
  d.src_meshH=src_meshH;
  d.dst_meshH=dst_meshH;
  d.basis_order=basis_order;
  d.basis=basis;
  d.source_to_single_dest=source_to_single_dest;
  d.exhaustive_search=exhaustive_search;
  d.dist=dist;
  d.np=np;
  MDST *dst_mesh = dynamic_cast<MDST *>(dst_meshH.get_rep());
  FOUT *out_field = scinew FOUT(dst_mesh, basis_order);  
  d.out_fieldH = out_field;

  if (np==1)
    parallel_execute(0, &d);
  else
    Thread::parallel(this, 
       &BuildInterpAlgoT<MSRC, LSRC, MDST, LDST, FOUT>::parallel_execute,
       np, true, &d);

  if (out_field)
  {
    MSRC *mesh = dynamic_cast<MSRC *>(src_meshH.get_rep());
    if (mesh)
    {
      typename LSRC::size_type size;
      mesh->size(size);
      unsigned int range = (unsigned int)size;
      out_field->set_property("interp-source-range", range, false);
    }
  }

  return out_field;
}

template <class MSRC, class LSRC, class MDST, class LDST, class FOUT>
void
BuildInterpAlgoT<MSRC, LSRC, MDST, LDST, FOUT>::parallel_execute(int proc,
								 BIData *d) {
  MeshHandle src_meshH = d->src_meshH;
  MeshHandle dst_meshH = d->dst_meshH;
  int basis_order = d->basis_order;
  const string& basis = d->basis;
  bool source_to_single_dest = d->source_to_single_dest;
  bool exhaustive_search = d->exhaustive_search;
  double dist = d->dist;
  int np = d->np;
  FieldHandle out_fieldH = d->out_fieldH;
  
  MSRC *src_mesh = dynamic_cast<MSRC *>(src_meshH.get_rep());
  MDST *dst_mesh = dynamic_cast<MDST *>(dst_meshH.get_rep());
  FOUT *out_field = dynamic_cast<FOUT *>(out_fieldH.get_rep());

  if (proc == 0) {
    if (basis_order > 0) {
      src_mesh->synchronize(Mesh::NODES_E);
    }
    else if (basis_order == 0 && src_mesh->dimensionality() == 3) {
      src_mesh->synchronize(Mesh::CELLS_E);
    } else if (basis_order == 0 && src_mesh->dimensionality() == 2) {
      src_mesh->synchronize(Mesh::FACES_E);
    } else if (basis_order == 0 && src_mesh->dimensionality() == 1) {
      src_mesh->synchronize(Mesh::EDGES_E);
    }
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
      double weights[MESH_WEIGHT_MAXSIZE];
      Point p;
      src_mesh->get_center(p, *itr);
      bool failed = true;
      const int nw = dst_mesh->get_weights(p, locs, weights);
      if (nw > 0) {
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
      for (unsigned int i=0; i<n; i++) {
	pair<typename LSRC::index_type, double> p(dst_iter->second.second[i], 
						  1./(double)n);
	v.push_back(p);
      }
      out_field->set_value(v, dst_iter->second.first);
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
      double weights[MESH_WEIGHT_MAXSIZE];
      Point p;
      dst_mesh->get_center(p, *itr);
      vector<pair<typename LSRC::index_type, double> > v;
      bool failed = true;
      const int nw = src_mesh->get_weights(p, locs, weights);
      if (nw > 0)	{
	failed = false;
	if (linear) {
	  for (unsigned int i = 0; i < locs.size(); i++) {
	    v.push_back(pair<typename LSRC::index_type, double>
			(locs[i], weights[i]));
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
	  v.push_back(pair<typename LSRC::index_type, double>
		      (locs[max_idx], 1.0));
	}
      }
      if (exhaustive_search && failed) {
	typename LSRC::index_type index;
	double d=find_closest_src_loc(index, src_mesh, p);
	if (dist<=0 || d<dist)
	{
	  v.push_back(pair<typename LSRC::index_type, double>(index, 1.0));
	}
      }
      out_field->set_value(v, *itr);
      ++itr;
      ++count;
    }
  }
}


} // end namespace SCIRun

#endif // BuildInterpolant_h
