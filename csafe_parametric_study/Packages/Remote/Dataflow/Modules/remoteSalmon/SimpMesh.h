//////////////////////////////////////////////////////////////////////
// SimpMesh.h - Simplifies the given model using quadric error metric.
// David K. McAllister, August 1999.
// Implements a simplifiable mesh, which is a subclass of Mesh.

#ifndef _simpmesh_h
#define _simpmesh_h

#include <Packages/Remote/Tools/Model/Mesh.h>

namespace Remote {
using namespace Remote::Tools;
class SimpMesh : public Mesh
{
  vector<Edge *> Collapses;
  int TargetTris;
  double BoundaryScale;

				// Compute each vertex's quadric.
  void ComputeQuadrics();

				// Compute the target for each edge
				// collapse.
  void ComputeEdgeCollapses();

				// Perform the edge collapses.
  void PerformEdgeCollapses();

				// Fill in the point the edge will
				// collapse to.
  void FindCollapseTarget(Edge *);

				// Remove this vertex from the mesh.
  void DoCollapse(Edge *Xe);
  
				// Add a face perp. to a face of a
				// boundary edge.
  void FixBoundary(Edge *E);

  ////////////////////////////////////////
  // The edge collapse heap.

  void HeapDump();

  inline void SwapInHeap(int a, int b)
  {
    ASSERT1M(Collapses[a]->Heap == a && Collapses[b]->Heap == b,
      "Inconsistent heap indices.");
    Collapses[a]->Heap = b;
    Collapses[b]->Heap = a;
    Edge *T = Collapses[a];
    Collapses[a] = Collapses[b];
    Collapses[b] = T;
  }

  inline void Heapify(int i)
  {
    int l = i<<1;
    int r = l|1;

    int best;
    if(l<Collapses.size() && Collapses[l]->Cost < Collapses[i]->Cost)
      best = l;
    else
      best = i;

    if(r<Collapses.size() && Collapses[r]->Cost < Collapses[best]->Cost)
      best = r;
    if(best != i)
      {
	SwapInHeap(i, best);
	Heapify(best);
      }
  }

  inline void MakeHeap()
  {
    for(int i=Collapses.size()/2; i >= 0; i--)
	Heapify(i);
  }

  inline void HeapValueChanged(Edge *E)
  {
    int i = E->Heap;
    ASSERT1M(i>=0, "Messing with deleted edge.");
    int p = i>>1;
    
    if(Collapses[p]->Cost > E->Cost) // False for root.
      {
				// Need to bubble me up.
	while(i && Collapses[p]->Cost > E->Cost)
	  SwapInHeap(i, p);
      }
    else
      Heapify(i);
  }

				// True to remove dead edges and
				// verts.
  void DeleteFace(Face *F, const bool RemoveDead = false);
  
  void DeleteVertex(Vertex *V);
  void DeleteEdge(Edge *E);
  void TransferFaces(Edge *X, Edge *K);

public:
  inline SimpMesh() {
    TargetTris = 0; BoundaryScale = 0;
  }

  inline SimpMesh(const Object &M) : Mesh(M) {
    TargetTris = 0; BoundaryScale = 0;
  }

				// Returns the number of tris in the
				// new model.  Scale up the weight on
				// the boundary edges.
  int Simplify(int target_tris, double _BoundaryScale = 0);


				// transform this mesh using the
				// matrix m
  void Transform(Matrix44 m) {
    Vertex* v = Verts;
    while (v) {
      v->V = m.Project(v->V);
      v = v->next;
    }
  }

};
} // End namespace Remote


#endif
