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

//    File   : StreamLines.h
//    Author : Michael Callahan
//    Date   : July 2001

#if !defined(StreamLines_h)
#define StreamLines_h

#include <Core/Thread/Thread.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/CurveField.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Containers/Array1.h>
#include <algorithm>

namespace SCIRun {

typedef struct _SLData {
  Array1<Array1<Point> >pts;
  Array1<Array1<double> >vals;
  Array1<Array1<pair<int,int> > >edges;
  MeshHandle seed_mesh_h;
  VectorFieldInterface *vfi;
  double tolerance;
  double stepsize;
  int maxsteps;
  int direction;
  int color;
  bool rcp;
  int met;
  int np;
} SLData;

class StreamLinesAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(MeshHandle seed_mesh_h,
			      VectorFieldInterface *vfi,
			      double tolerance,
			      double stepsize,
			      int maxsteps,
			      int direction,
			      int color,
			      bool remove_colinear_p,
			      int method, 
			      int np) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *smesh,
					    const TypeDescription *sloc);
protected:

  //! This particular implementation uses Runge-Kutta-Fehlberg.
  void FindNodes(vector<Point>&, Point, double, double, int, 
		 VectorFieldInterface *, bool remove_colinear_p, int method);
};


template <class SFLD, class SLOC>
class StreamLinesAlgoT : public StreamLinesAlgo
{
public:
  //! virtual interface. 
  void parallel_generate(int proc, SLData *d);

  virtual FieldHandle execute(MeshHandle seed_mesh_h,
			      VectorFieldInterface *vfi,
			      double tolerance,
			      double stepsize,
			      int maxsteps,
			      int direction,
			      int color,
			      bool remove_colinear_p,
			      int method,
			      int np);
};

template <class SMESH, class SLOC>
void
StreamLinesAlgoT<SMESH, SLOC>::parallel_generate( int proc, SLData *d) {
  SMESH *smesh = dynamic_cast<SMESH *>(d->seed_mesh_h.get_rep());

  const double tolerance2 = d->tolerance * d->tolerance;

  Point seed;
  Vector test;
  vector<Point> nodes;

  // Try to find the streamline for each seed point.
  typename SLOC::iterator seed_iter, seed_iter_end;
  int count=0;
  smesh->begin(seed_iter);
  smesh->end(seed_iter_end);
  while (seed_iter != seed_iter_end)
  {
    // if this seed doesn't "belong" to this parallel thread,
    //   ignore it and continue on the next seed...
    if (count%d->np != proc) {
      ++seed_iter;
      ++count;
      continue;
    }

    // this is one of our seeds -- generate its streamline
    smesh->get_point(seed, *seed_iter);

    // Is the seed point inside the field?
    if (!d->vfi->interpolate(test, seed))
    {
      ++seed_iter;
      ++count;
      continue;
    }

    nodes.clear();
    nodes.push_back(seed);

    int cc = 0;

    // Find the negative streamlines.
    if( d->direction <= 1 )
    {
      FindNodes(nodes, seed, tolerance2, -d->stepsize, d->maxsteps, 
		d->vfi, d->rcp, d->met);
      if ( d->direction == 1 )
      {
	std::reverse(nodes.begin(), nodes.end());
	cc = nodes.size();
	cc = -(cc - 1);
      }
    }
    // Append the positive streamlines.
    if( d->direction >= 1 )
    {
      FindNodes(nodes, seed, tolerance2, d->stepsize, d->maxsteps, 
		d->vfi, d->rcp, d->met);
    }

    for (int i=0; i<nodes.size(); i++) {
      int pt_idx=d->pts[proc].size();
      d->pts[proc].add(nodes[i]);
      if (d->color) {
	d->vals[proc].add((double)abs(cc+i));
      } else {
	d->vals[proc].add((double)(*seed_iter));
      }
      if (i!=nodes.size()-1) {
	d->edges[proc].add(pair<int,int>(pt_idx,pt_idx+1));
      }
    }
    ++seed_iter;
    ++count;
  }
}
						  
template <class SMESH, class SLOC>
FieldHandle
StreamLinesAlgoT<SMESH, SLOC>::execute(MeshHandle seed_mesh_h,
				       VectorFieldInterface *vfi,
				       double tolerance,
				       double stepsize,
				       int maxsteps,
				       int direction,
				       int color,
				       bool rcp,
				       int met,
				       int np)
{
  SLData d;
  d.pts.resize(np);
  d.vals.resize(np);
  d.edges.resize(np);
  d.seed_mesh_h=seed_mesh_h;
  d.vfi=vfi;
  d.tolerance=tolerance;
  d.stepsize=stepsize;
  d.maxsteps=maxsteps;
  d.direction=direction;
  d.color=color;
  d.rcp=rcp;
  d.met=met;
  d.np=np;

  //cerr << "starting up "<<np<<" threads.\n";
  Thread::parallel (this,
		    &StreamLinesAlgoT<SMESH, SLOC>::parallel_generate,
//		    np, true, (SLData &)d);
		    np, true, &d);
  //cerr << "threads made it out ok\n";

  CurveMeshHandle cmesh = scinew CurveMesh();
  int count=0;
  Array1<int> offsets(np);
  CurveMesh::Node::index_type n1, n2;
  Array1<CurveMesh::Node::index_type> nodemap;

  //cerr << "Adding nodes...\n";
  int i,j;
  for (i=0; i<np; i++) {
    offsets[i]=count;
    count+=d.pts[i].size();
    for (j=0; j<d.pts[i].size(); j++) {
      nodemap.add(cmesh->add_node(d.pts[i][j]));
    }
  }
  //cerr << "Adding edges...\n";
  for (i=0; i<np; i++) {
    for (j=0; j<d.edges[i].size(); j++) {
      cmesh->add_edge(nodemap[d.edges[i][j].first+offsets[i]],
		      nodemap[d.edges[i][j].second+offsets[i]]);
    }
  }
  CurveField<double> *cf = scinew CurveField<double>(cmesh, Field::NODE);
  //cerr << "Adding data...\n";
  int ctr=0;
  for (i=0; i<np; i++) {
    for (j=0; j<d.vals[i].size(); j++, ctr++) {
      cf->fdata()[ctr]=d.vals[i][j];
    }
  }
  //cerr << "Done!\n";
  cf->freeze();

  if (count == 0)
  {
    delete cf;
    return 0;
  }
  else
  {
    return cf;
  }
}


} // end namespace SCIRun

#endif // StreamLines_h
