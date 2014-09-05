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

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/DynamicLoader.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/ContourMesh.h>

namespace SCIRun {

class StreamLinesAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(MeshHandle vmesh,
		       MeshHandle smesh,
		       VectorFieldInterface *vfi,
		       double tolerance,
		       double stepsize,
		       int maxsteps,
		       ContourMeshHandle cmesh) = 0;


  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *vmesh,
				       const TypeDescription *smesh,
				       const TypeDescription *sloc);
protected:
  bool interpolate(VectorFieldInterface *vfi, const Point &p, Vector &v);

  //! This particular implementation uses Runge-Kutta-Fehlberg.
  void FindStreamLineNodes(vector<Point>&, Point, double, double, int, 
			   VectorFieldInterface *);

  //! Compute the inner terms of the RKF formula.
  bool ComputeRKFTerms(vector<Vector> &, const Point&, double,
		       VectorFieldInterface *);

};


template <class VMESH, class SMESH, class SLOC>
class StreamLinesAlgoT : public StreamLinesAlgo
{
public:
  //! virtual interface. 
  virtual void execute(MeshHandle vmesh,
		       MeshHandle smesh,
		       VectorFieldInterface *vfi,
		       double tolerance,
		       double stepsize,
		       int maxsteps,
		       ContourMeshHandle cmesh);
};



template <class VMESH, class SMESH, class SLOC>
void 
StreamLinesAlgoT<VMESH, SMESH, SLOC>::execute(MeshHandle vmesh_h,
					      MeshHandle smesh_h,
					      VectorFieldInterface *vfi,
					      double tolerance,
					      double stepsize,
					      int maxsteps,
					      ContourMeshHandle cmesh)
{
  VMESH *vmesh = dynamic_cast<VMESH *>(vmesh_h.get_rep());
  SMESH *smesh = dynamic_cast<SMESH *>(smesh_h.get_rep());

  Point seed;
  Vector test;
  vector<Point> nodes;
  vector<Point>::iterator node_iter;
  ContourMesh::Node::index_type n1, n2;

  // Try to find the streamline for each seed point.
  typename SLOC::iterator seed_iter, seed_iter_end;
  smesh->begin(seed_iter);
  smesh->end(seed_iter_end);
  while (seed_iter != seed_iter_end)
  {
    smesh->get_point(seed, *seed_iter);

    // Is the seed point inside the field?
    if (!interpolate(vfi, seed, test))
    {
      cout << "StreamLines: WARNING: seed point was not inside the field.\n";
      ++seed_iter;
      continue;
    }

    // Find the positive streamlines.
    nodes.clear();
    FindStreamLineNodes(nodes, seed, tolerance, stepsize, maxsteps, vfi);

    node_iter = nodes.begin();
    if (node_iter != nodes.end())
    {
      n1 = cmesh->add_node(*node_iter);
      ++node_iter;
      while (node_iter != nodes.end())
      {
	n2 = cmesh->add_node(*node_iter);
	cmesh->add_edge(n1, n2);
	n1 = n2;
	++node_iter;
      }
    }

    // Find the negative streamlines.
    nodes.clear();
    FindStreamLineNodes(nodes, seed, tolerance, -stepsize, maxsteps, vfi);

    node_iter = nodes.begin();
    if (node_iter != nodes.end())
    {
      n1 = cmesh->add_node(*node_iter);
      ++node_iter;
      while (node_iter != nodes.end())
      {
	n2 = cmesh->add_node(*node_iter);
	cmesh->add_edge(n1, n2);
	n1 = n2;
	++node_iter;
      }
    }

    ++seed_iter;
  }
}


} // end namespace SCIRun

#endif // StreamLines_h
