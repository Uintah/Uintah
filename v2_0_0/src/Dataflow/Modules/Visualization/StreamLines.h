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
  CurveField<double> *cf;
  Mutex lock;
  MeshHandle seed_mesh_h;
  VectorFieldInterfaceHandle vfi;
  double tolerance;
  double stepsize;
  int maxsteps;
  int direction;
  int color;
  bool rcp;
  int met;
  int np;

  _SLData() : lock("StreamLines Lock") {}
} SLData;

class StreamLinesAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(MeshHandle seed_mesh_h,
			      VectorFieldInterfaceHandle vfi,
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
		 VectorFieldInterfaceHandle, bool remove_colinear_p, int method);
};


template <class SFLD, class SLOC>
class StreamLinesAlgoT : public StreamLinesAlgo
{
public:
  //! virtual interface. 
  void parallel_generate(int proc, SLData *d);

  virtual FieldHandle execute(MeshHandle seed_mesh_h,
			      VectorFieldInterfaceHandle vfi,
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
StreamLinesAlgoT<SMESH, SLOC>::parallel_generate( int proc, SLData *d)
						  
{
  SMESH *smesh = dynamic_cast<SMESH *>(d->seed_mesh_h.get_rep());

  const double tolerance2 = d->tolerance * d->tolerance;

  //CurveMeshHandle cmesh = scinew CurveMesh();
  //CurveField<double> *cf = scinew CurveField<double>(cmesh, Field::NODE);

  Point seed;
  Vector test;
  vector<Point> nodes;
  nodes.reserve(d->maxsteps);

  vector<Point>::iterator node_iter;
  CurveMesh::Node::index_type n1, n2;

  // Try to find the streamline for each seed point.
  typename SLOC::iterator seed_iter, seed_iter_end;
  smesh->begin(seed_iter);
  smesh->end(seed_iter_end);
  int count = 0;
  while (seed_iter != seed_iter_end)
  {
    // If this seed doesn't "belong" to this parallel thread,
    // ignore it and continue on the next seed.
    if (count%d->np != proc) {
      ++seed_iter;
      ++count;
      continue;
    }

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

    node_iter = nodes.begin();

    if (node_iter != nodes.end())
    {
      d->lock.lock();
      n1 = d->cf->get_typed_mesh()->add_node(*node_iter);
      d->cf->resize_fdata();
      if( d->color )
	d->cf->set_value((double)abs(cc), n1);
      else
	d->cf->set_value((double)(*seed_iter), n1);

      ++node_iter;

      cc++;

      while (node_iter != nodes.end())
      {
	n2 = d->cf->get_typed_mesh()->add_node(*node_iter);
	d->cf->resize_fdata();

	if( d->color )
	  d->cf->set_value((double)abs(cc), n2);
	else
	  d->cf->set_value((double)(*seed_iter), n2);

	d->cf->get_typed_mesh()->add_edge(n1, n2);

	n1 = n2;
	++node_iter;

	cc++;
      }
      d->lock.unlock();
    }

    ++seed_iter;
    ++count;
  }
}


template <class SMESH, class SLOC>
FieldHandle
StreamLinesAlgoT<SMESH, SLOC>::execute(MeshHandle seed_mesh_h,
				       VectorFieldInterfaceHandle vfi,
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

  CurveMeshHandle cmesh = scinew CurveMesh();
  CurveField<double> *cf = scinew CurveField<double>(cmesh, Field::NODE);
  
  d.cf = cf;

  if (np > 1)
  {
    Thread::parallel (this,
		      &StreamLinesAlgoT<SMESH, SLOC>::parallel_generate,
		      np, true, &d);
  }
  else
  {
    parallel_generate(0, &d);
  }

  cf->freeze();

  return cf;

  CurveMesh::Node::size_type count;
  cf->get_typed_mesh()->size(count);
  if (((unsigned int)count) == 0)
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
