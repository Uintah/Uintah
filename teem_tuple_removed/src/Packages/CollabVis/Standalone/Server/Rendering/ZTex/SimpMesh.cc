//////////////////////////////////////////////////////////////////////
// SimpMesh.cpp - Simplifies the given model using quadric error metric.
//
// David K. McAllister, August 1999.

#include <stdio.h>
#include <Rendering/ZTex/SimpMesh.h>

namespace SemotusVisum {
namespace Rendering {

void SimpMesh::DeleteFace(Face *F, bool RemoveDead)
{
  // cerr << "F";
  if(F == Faces)
    Faces = F->next;
      
  F->ListRemove(F->v0->Faces);
  F->ListRemove(F->v1->Faces);
  F->ListRemove(F->v2->Faces);
  F->ListRemove(F->e0->Faces);
  F->ListRemove(F->e1->Faces);
  F->ListRemove(F->e2->Faces);

  if(RemoveDead)
    {
      if(F->v0->Faces.size() == 0) DeleteVertex(F->v0);
      if(F->v1->Faces.size() == 0) DeleteVertex(F->v1);
      if(F->v2->Faces.size() == 0) DeleteVertex(F->v2);
      if(F->e0->Faces.size() == 0) DeleteEdge(F->e0);
      if(F->e1->Faces.size() == 0) DeleteEdge(F->e1);
      if(F->e2->Faces.size() == 0) DeleteEdge(F->e2);
    }

  delete F;
  FaceCount--;
}

void SimpMesh::DeleteVertex(Vertex *V)
{
  // cerr << "V";
  if(V == Verts)
    Verts = V->next;

  ASSERT1(V);

  // XXX No recursion?
  ASSERT1M(V->Faces.size() == 0, "VWhy you still got faces?");
  ASSERT1M(V->Edges.size() == 0, "VWhy you still got edges?");

  delete V;
  VertexCount--;
}

void SimpMesh::DeleteEdge(Edge *E)
{
  ASSERT1(E);

  // cerr << "E";
  if(E == Edges)
    Edges = E->next;

  int i = E->Heap;
  Collapses[i] = Collapses.back();
  Collapses[i]->Heap = i;
  Collapses.pop_back();
  Heapify(i);
  E->Heap = -1;
  
  ASSERT1M(E->Faces.size() == 0, "EWhy you still got faces?");

  E->ListRemove(E->v0->Edges);
  E->ListRemove(E->v1->Edges);
    
  delete E;
  EdgeCount--;
}

// Returns the number of tris in the new model.
int SimpMesh::Simplify(int target_tris, double _BoundaryScale)
{
  cerr << "Beginning simplification.\n";

  TargetTris = target_tris;
  BoundaryScale = _BoundaryScale;

  // CheckIntegrity();
  if(FaceCount <= TargetTris)
    return FaceCount;

  ComputeQuadrics();
  ComputeEdgeCollapses();
  PerformEdgeCollapses();

  // CheckIntegrity(true);

  return FaceCount;
}

void SimpMesh::FixBoundary(Edge *E)
{
  // Make a plane perp. to the face containing the edge.
  Vector P0, P1, P2;
  if(E->Faces[0]->v1 != E->v0 && E->Faces[0]->v1 != E->v1)
    {
      P0 = E->Faces[0]->v1->V;
      P1 = E->Faces[0]->v0->V;
      P2 = E->Faces[0]->v2->V;
    }
  else if(E->Faces[0]->v2 != E->v0 && E->Faces[0]->v2 != E->v1)
    {
      P0 = E->Faces[0]->v2->V;
      P1 = E->Faces[0]->v1->V;
      P2 = E->Faces[0]->v0->V;
    }
  else
    {
      ASSERT1(E->Faces[0]->v0 != E->v0 && E->Faces[0]->v0 != E->v1);
      P0 = E->Faces[0]->v0->V;
      P1 = E->Faces[0]->v1->V;
      P2 = E->Faces[0]->v2->V;
    }

  // The edge.
  Vector En = P2 - P1;
  En.normalize();
  Vector E2 = P0 - P1;
	      
  Vector F = En * Dot(E2, En);
  Vector N = E2 - F;
  N.normalize();
  double D = -Dot(N, P1);
	      
  Quadric3 Q;
  Q.DoSym(N, D);

  Q.Scale(BoundaryScale);
  E->v0->Q += Q;
  E->v1->Q += Q;
}

// Compute each vertex's quadric.
// Assumes the vertex quadrics are all zero.
void SimpMesh::ComputeQuadrics()
{
  cerr << "Computing planes.\n";

  // Compute the plane equation for each face.
  for(Face *F = Faces; F; F = F->next)
    {
      Vector N;
      double D;

      ComputePlane(F->v0->V, F->v1->V, F->v2->V, N, D);

      Quadric3 Q;
      Q.DoSym(N, D);

      // Accumulate the face quadric into this vertex.
      F->v0->Q += Q;
      F->v1->Q += Q;
      F->v2->Q += Q;
      
      // If any edge of the face has no other faces, add a perp. plane
      // to the vertices of the edge. This is a cheap way to preserve
      // boundary edges.
      if(BoundaryScale != 0)
	{
	  if(F->e0->Faces.size() == 1) FixBoundary(F->e0);
	  if(F->e1->Faces.size() == 1) FixBoundary(F->e1);
	  if(F->e2->Faces.size() == 1) FixBoundary(F->e2);
	}
    }
}

// Compute the location to collapse this edge to and store the answer
// and the cost in the edge.
void SimpMesh::FindCollapseTarget(Edge *E)
{
  bool FoundMin = E->Q.FindMin(E->V);

  if(FoundMin)
    {
      ASSERT1M(!(isnan(E->V.x), isnan(E->V.y), isnan(E->V.z)), "Not a number");
      E->Cost = E->Q.MulPt(E->V);
    }
  else
    {
      // cerr << "Noninvertible.\n";
  
      double e0 = E->Q.MulPt(E->v0->V);
      double e1 = E->Q.MulPt(E->v1->V);
      Vector M = (E->v0->V + E->v1->V) * 0.5;
      double em = E->Q.MulPt(M);
	  
      if(e0 < e1)
	{
	  if(e0 < em)
	    {
	      E->Cost = e0;
	      E->V = E->v0->V;
	    }
	  else
	    {
	      E->Cost = em;
	      E->V = M;
	    }
	}
      else
	{
	  if(e1 < em)
	    {
	      E->Cost = e1;
	      E->V = E->v1->V;
	    }
	  else
	    {
	      E->Cost = em;
	      E->V = M;
	    }
	}
    }

  //double L0 = (E->v1->V - E->v0->V).length();
  //double L1 = (E->V - E->v0->V).length();

  // if(L1 > L0*2) cerr << "V0=" << E->v0->V << " V1=" << E->v1->V << " V=" << E->V << E->Cost << endl;

  if(HasColor || HasTexCoords)
    {
      // Compute the weights of v0 and v1 to apply to color and texcoords.
      // XXX This assumes that V is between v0 and v1.
      double L0 = (E->v1->V - E->v0->V).length();
      double L1 = (E->V - E->v0->V).length();
      double w = L1 / L0;

      if(HasColor)
	  E->Col = E->v0->Col + (E->v1->Col - E->v0->Col) * w;

      if(HasTexCoords)
	  E->Tex = E->v0->Tex + (E->v1->Tex - E->v0->Tex) * w;
    }
}

// Compute the target for each edge collapse and make the heap.
void SimpMesh::ComputeEdgeCollapses()
{
  cerr << "Computing Edge Collapses.\n";

  for(Edge *E = Edges; E; E = E->next)
    {
      // Compute the quadric for after this edge collapse.
      E->Q = E->v0->Q + E->v1->Q;

      // Find the collapse target.
      FindCollapseTarget(E);

      E->Heap = Collapses.size();
      Collapses.push_back(E);
    }

  // Make a heap of edge collapses sorted by Cost.
  MakeHeap();
}

void SimpMesh::TransferFaces(Edge *Xef, Edge *Kef)
{
  // cerr << "Transfering faces.\n";
  while(Xef->Faces.size())
    {
      Face *Kf = Xef->Faces.back();

#ifdef SCI_DEBUG
      // XXX Remove this.
      // Make sure not to make a duplicate face.
      int DupFace = 0;
      for(int k=0; k<Kef->Faces.size(); k++)
	{
	  Face *fkf = Kef->Faces[k];

	  ASSERT1(fkf != Kf);

	  if(fkf->e0 == Kf->e0 || fkf->e0 == Kf->e1 || fkf->e0 == Kf->e2 || 
	     fkf->e1 == Kf->e0 || fkf->e1 == Kf->e1 || fkf->e1 == Kf->e2 || 
	     fkf->e2 == Kf->e0 || fkf->e2 == Kf->e1 || fkf->e2 == Kf->e2)
	    DupFace++;
	  else if((fkf->v0 == Kf->v0) + (fkf->v0 == Kf->v1) + (fkf->v0 == Kf->v2) + 
		  (fkf->v1 == Kf->v0) + (fkf->v1 == Kf->v1) + (fkf->v1 == Kf->v2) + 
		  (fkf->v2 == Kf->v0) + (fkf->v2 == Kf->v1) + (fkf->v2 == Kf->v2) > 1)
	    DupFace++;
	}
      if(DupFace > 1)
	cerr << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& Going to make a duplicate face.\n";
#endif

      // Change edge from Xef to Kef.
      if(Kf->e0 == Xef) Kf->e0 = Kef;
      else if(Kf->e1 == Xef) Kf->e1 = Kef;
      else
	{
	  ASSERT1(Kf->e2 == Xef);
	  Kf->e2 = Kef;
	}

      // Add this face to Kef.
      Kef->Faces.push_back(Kf);
      Xef->Faces.pop_back();
    }
}

// Collapse an edge, removing one vertex. Replace the other vertex
// with the one I stored. Remove one edge per dead face.
void SimpMesh::DoCollapse(Edge *Xe)
{
  ASSERT1(Xe);
  // fprintf(stderr, "Collapsing edge 0x%08lx of cost %.19f\n", long(Xe), Xe->Cost);

  Vertex *Xv = Xe->v0, *Kv = Xe->v1;

  ASSERT1M(Xv != Kv, "Edge has same vertex at both ends.");
  ASSERT1M(Xe->next || Xe->prev, "Edge already deleted.");

  while(Xe->Faces.size())
    {
      // cerr << ".";
      Face *Xf = Xe->Faces.back();

      // Find which edge is Xef and which is Kef.
      Edge *Xef, *Kef;
      if(Xf->e0 == Xe)
	{
	  if(Xf->e1->v0 == Xv || Xf->e1->v1 == Xv)
	    {Xef = Xf->e1; Kef = Xf->e2;}
	  else
	    {Xef = Xf->e2; Kef = Xf->e1;}
	}
      else if(Xf->e1 == Xe)
	{
	  if(Xf->e0->v0 == Xv || Xf->e0->v1 == Xv)
	    {Xef = Xf->e0; Kef = Xf->e2;}
	  else
	    {Xef = Xf->e2; Kef = Xf->e0;}
	}
      else
	{
	  ASSERT1(Xf->e2 == Xe);
	  if(Xf->e0->v0 == Xv || Xf->e0->v1 == Xv)
	    {Xef = Xf->e0; Kef = Xf->e1;}
	  else
	    {Xef = Xf->e1; Kef = Xf->e0;}
	}

      Xf->ListRemove(Xef->Faces);
      TransferFaces(Xef, Kef);

      // At this point:
      // Xef has no faces.
      // Kef has all Xef's faces.
      // Xv still has all of Xef's faces (incl. Xf).

      DeleteFace(Xf); // Implicitly removes the face from Xe->Faces.
      DeleteEdge(Xef);
      
      if(Kef->Faces.size() == 0)
	{
	  // cerr << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n";

	  // Find which vertex is Vf.
	  Vertex *Vf;
	  if(Kef->v0 != Kv) Vf = Kef->v0;
	  else
	    {
	      ASSERT1(Kef->v1 != Kv);
	      Vf = Kef->v1;
	    }

	  DeleteEdge(Kef);
      
	  if(Vf->Edges.size() < 2 || Vf->Faces.size() == 0)
	    {
	      // cerr << Vf->Edges.size() << " " << Vf->Faces.size() << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
	      DeleteVertex(Vf);
	    }
	}
      // cerr << "yo";
    }

  // At this point:
  // All Xef are deleted and not pointed to.
  // All Xf are deleted and not pointed to.
  // All Xef's faces are in their Kef.
  // All Xef's faces are still in Xv.

  // Replace Kv with the new vertex.
  Kv->Q = Xe->Q;
  Kv->V = Xe->V;
  Kv->Col = Xe->Col;
  Kv->Tex = Xe->Tex;

  ASSERT1M(Xe->Faces.size() == 0, "Why you still got faces?");
  DeleteEdge(Xe);

  // Move faces to new vertex.
  while(Xv->Faces.size())
    {
      Face *Kf = Xv->Faces.back();
      // cerr << "f";

      if(Kf->v0 == Xv) Kf->v0 = Kv;
      else if(Kf->v1 == Xv) Kf->v1 = Kv;
      else
	{
	  ASSERT1(Kf->v2 == Xv);
	  Kf->v2 = Kv;
	}

      // for(int i=0; i<Kv->Faces.size(); i++) {ASSERT1M(Kv->Faces[i] != Kf, "Face already in Kv");}

      Kv->Faces.push_back(Kf);
      Xv->Faces.pop_back();
    }

  // Move edges to new vertex.
  while(Xv->Edges.size())
    {
      // cerr << "e";
      Edge *Ke = Xv->Edges.back();

      ASSERT1M(Ke != Xe, "You should be dead by now.");
      ASSERT1M(Ke->Heap >= 0, "Deleted edge");

      if(Ke->v0 == Xv) Ke->v0 = Kv;
      else
	{
	  ASSERT1(Ke->v1 == Xv);
	  Ke->v1 = Kv;
	}

      // XXX for(int i=0; i<Kv->Edges.size(); i++) {ASSERT1M(Kv->Edges[i] != Ke, "Edge already in Kv");}
      
      Kv->Edges.push_back(Ke);
      Xv->Edges.pop_back();
    }

  // Remove Xv.
  DeleteVertex(Xv);

  int i;
  if(Kv->Edges.size() <= 1 || Kv->Faces.size() == 0)
    {
      ASSERT1M(Kv->Edges.size() == 0, "Mustn't have edges");
      ASSERT1M(Kv->Faces.size() == 0, "Mustn't have faces");
      // cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";

      DeleteVertex(Kv);
    }
  else
    {
      // See if I have any duplicate faces and combine them.
      for(i=0; (unsigned)i<Kv->Faces.size(); i++)
	{
	  Face *F0 = Kv->Faces[i];
	  ASSERT1M(F0->v0, "Vertex points to dead face");
	  
	  // Make sure there is no duplicate face.
	  for(int j=i+1; (unsigned)j<Kv->Faces.size(); j++)
	    {
	      Face *F1 = Kv->Faces[j];
	      if((F0->v0 == F1->v0 && ((F0->v1 == F1->v1 && F0->v2 == F1->v2) || (F0->v1 == F1->v2 && F0->v2 == F1->v1))) ||
		 (F0->v0 == F1->v1 && ((F0->v1 == F1->v0 && F0->v2 == F1->v2) || (F0->v1 == F1->v2 && F0->v2 == F1->v0))) ||
		 (F0->v0 == F1->v2 && ((F0->v1 == F1->v0 && F0->v2 == F1->v1) || (F0->v1 == F1->v1 && F0->v2 == F1->v0))) ||
		 (F0->e0 == F1->e0 && (F0->e1 == F1->e1 || F0->e1 == F1->e2 || F0->e2 == F1->e1 || F0->e2 == F1->e2)) || 
		 (F0->e0 == F1->e1 && (F0->e1 == F1->e0 || F0->e1 == F1->e2 || F0->e2 == F1->e0 || F0->e2 == F1->e2)) || 
		 (F0->e0 == F1->e2 && (F0->e1 == F1->e1 || F0->e1 == F1->e0 || F0->e2 == F1->e1 || F0->e2 == F1->e0)))
		{
		  // cerr << "Duplicate face\n";
		  DeleteFace(F1, true);
		  j--;
		}
	    }
	}

      // See if I have any duplicate edges and combine them.
      for(int k=0; (unsigned)k<Kv->Edges.size(); k++)
	{
	  Edge *E0 = Kv->Edges[k];
	  for(int l=k+1; (unsigned)l<Kv->Edges.size(); l++)
	    {
	      Edge *E1 = Kv->Edges[l];
	      if((E0->v0 == E1->v0 && E0->v1 == E1->v1) || 
		 (E0->v0 == E1->v1 && E0->v1 == E1->v0))
		{
		  // cerr << "Duplicate edge\n";
		  TransferFaces(E1, E0);
		  DeleteEdge(E1);
		  l--;
		}
	      else
		ASSERT1M(E1->Faces.size() > 0, "Faceless edge\n");
	    }
	}

      // Compute new error quadrics for edges.
      for(i=0; (unsigned)i<Kv->Edges.size(); i++)
	{
	  Edge *Ke = Kv->Edges[i];
	  ASSERT1M(Ke != Xe, "What you doin' here?");
	  ASSERT1M(Ke->Heap >= 0, "Deleted edge");
	
	  Ke->Q = Ke->v0->Q + Ke->v1->Q;
	
	  FindCollapseTarget(Ke);
	
	  HeapValueChanged(Ke);
	}
    }

  // cerr << " Finished collapse.\n";
}

void SimpMesh::HeapDump()
{
  for(int i=0; (unsigned)i<Collapses.size(); i++)
    {
      Edge *Ed = Collapses[i];
      fprintf(stderr, "%d\t%d %d\t%.34f\t0x%lx\n", i, Ed->Faces.size(), Ed->Heap, Ed->Cost, long(Ed));
      ASSERT1M(i == Ed->Heap, "Heap index bad");
    }

  ASSERT1M(EdgeCount == Collapses.size(), "Wrong edge count");
}

// Perform the edge collapses.
void SimpMesh::PerformEdgeCollapses()
{
  cerr << "PerformEdgeCollapses\n";

  while(FaceCount > TargetTris)
    {
      Edge *Xe = Collapses[0];

      DoCollapse(Xe);

      // HeapDump();
      // Dump();
      // CheckIntegrity(FaceCount % 300 == 0);
    }
}

} // namespace Tools {y
} // namespace SemotusVisum {

