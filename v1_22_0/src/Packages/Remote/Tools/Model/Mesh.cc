//////////////////////////////////////////////////////////////////////
// Mesh.cpp - Represent a model as a mesh.
// David K. McAllister, August 1999.

#include <stdio.h>
#include <Packages/Remote/Tools/Model/Mesh.h>

namespace Remote {
//----------------------------------------------------------------------
// Returns the index of this vertex. Creates it if necessary.
Vertex *Mesh::FindVertex(const Vector &V, bool &AddedVert, const Vector *Color, const Vector *TexCoord)
{
  Vertex TmpV;
  TmpV.V = V;
  // This is so that TmpV can be properly destroyed.
  TmpV.next = TmpV.prev = NULL;
  KDVertex TmpK(&TmpV);

  KDVertex Res;
  // if(VertTree.FindNearEnough(TmpK, Res, MeshMaxDist))
  if(VertTree.Find(TmpK, Res))
    return Res.Vert;
  
  Vertex *N = AddVertex(V, Color, TexCoord);
  VertTree.Insert(KDVertex(N));
  AddedVert = true;

  return N;
}

// Returns the index of this edge. Creates it if necessary.
Edge *Mesh::FindEdge(Vertex *v0, Vertex *v1)
{
  ASSERT1(v0 != v1);

  for(int i=0; i<v0->Edges.size(); i++)
    {
      Edge *Ed = v0->Edges[i];
      if((Ed->v0 == v0 && Ed->v1 == v1) || (Ed->v0 == v1 && Ed->v1 == v0))
	return Ed;
    }

  Edge *N = AddEdge(v0, v1);

  return N;
}

Vertex *Mesh::FindVertexInEdgeList(const vector<Edge *> &Edg, const Vector &V, Edge * &e) const
{
  for(int j=0; j<Edg.size(); j++)
    if(VecEq(Edg[j]->v0->V, V, Sqr(MeshMaxDist)))
      {e = Edg[j]; return Edg[j]->v0;}
    else if(VecEq(Edg[j]->v1->V, V, Sqr(MeshMaxDist)))
      {e = Edg[j]; return Edg[j]->v1;}

  e = NULL;
  return NULL;
}

//----------------------------------------------------------------------
// Convert the incoming model into a mesh structure.
void Mesh::ImportObject(const Object &Ob)
{
  int p, i;
  
  cerr << "Converting Object to Mesh.\n";

  MeshMaxDist = (Ob.Box.MaxV - Ob.Box.MinV).length() * DIST_FACTOR;

  ASSERT1M(Ob.verts.size() % 3 == 0,
    "Must have a multiple of three vertices in Object.");

  bool HasOneColor = Ob.dcolors.size() == 1;
  bool HasOneTexCoords = Ob.texcoords.size() == 1;

  HasColor = Ob.verts.size() == Ob.dcolors.size() || HasOneColor;
  HasTexCoords = Ob.verts.size() == Ob.texcoords.size() || HasOneTexCoords;


  if (HasColor) cerr << "Has color.\n";
  if (HasTexCoords) cerr << "Has texcoords.\n";

  // cerr << (Ob.verts.size() / 3) << " triangles.\n";
  for (i=0; i<Ob.verts.size(); i+=3) {
    bool AddedVert = false;
    Edge *e0 = NULL, *e1 = NULL, *e2 = NULL;
    
    Vertex *v0 = FindVertex(Ob.verts[i], AddedVert,
      HasOneColor ? &Ob.dcolors[0] :
        (HasColor ? &Ob.dcolors[i] : NULL),
      HasOneTexCoords ? &Ob.texcoords[0] :
        (HasTexCoords ? &Ob.texcoords[i] : NULL));
    
    Vertex *v1 = FindVertexInEdgeList(v0->Edges, Ob.verts[i+1], e0);

				// Degenerate triangle.
    if (v0 == v1) continue;
    
    if (v1 == NULL) {
      v1 = FindVertex(Ob.verts[i+1], AddedVert,
	HasOneColor ? &Ob.dcolors[0] :
	  (HasColor ? &Ob.dcolors[i+1] : NULL),
	HasOneTexCoords ? &Ob.texcoords[0] :
	  (HasTexCoords ? &Ob.texcoords[i+1] : NULL));

				// Degenerate triangle.
      if(v0 == v1) continue;
      
				// Need to add e01 to all lists.
      e0 = FindEdge(v0, v1);
    }
    
    Vertex *v2 = FindVertexInEdgeList(v0->Edges, Ob.verts[i+2], e1);
    Vertex *vt = FindVertexInEdgeList(v1->Edges, Ob.verts[i+2], e2);
    
    if (v2 == NULL) v2 = vt;
    
    if (v2 == NULL)
      v2 = FindVertex(Ob.verts[i+2], AddedVert,
	HasOneColor ? &Ob.dcolors[0] : 
	  (HasColor ? &Ob.dcolors[i+2] : NULL),
	HasOneTexCoords ? &Ob.texcoords[0] :
	  (HasTexCoords ? &Ob.texcoords[i+2] : NULL));

				// Degenerate triangle.
    if (v2 == v1 || v2 == v0) continue;

				// All three vertices now exist. e1
				// and e2 might not exist yet.
    
				// Need to add e02 to all lists.
    if (e1 == NULL) e1 = FindEdge(v0, v2);

				// Need to add e12 to all lists.
    if (e2 == NULL) e2 = FindEdge(v1, v2);

    ASSERT1M(v0 != v1 && v1 != v2 && v0 != v2,
      "Degenerate triangle vertices.");
    ASSERT1M(e0 != e1 && e1 != e2 && e0 != e2,
      "Degenerate triangle edges.");

				// Need some check for unique face.
    if (!AddedVert) {
				// Probably a duplicate face. Check it
				// out, boys.
      // cerr << "Checking for duplicate face.\n";
      for(p = 0; p<v0->Faces.size(); p++) {
	Face *F = v0->Faces[p];
	if((F->v0 == v0 || F->v0 == v1 || F->v0 == v2) &&
	  (F->v1 == v0 || F->v1 == v1 || F->v1 == v2) &&
	  (F->v2 == v0 || F->v2 == v1 || F->v2 == v2))
	  {
	    // cerr << "Duplicate face!\n";
	    break;
	  }
      }
      if(p < v0->Faces.size()) continue;
    }
    
    AddFace(v0, v1, v2, e0, e1, e2);
  }
  
#ifdef SCI_DEBUG
  CheckIntegrity(true);
#else
  CheckIntegrity();
#endif
}

//----------------------------------------------------------------------
void Mesh::CheckIntegrity(const bool Slow)
{
  // cerr << "Checking mesh integrity.\n";

  int vc = 0, fc = 0, ec = 0;
  int i, j;
  Edge* E, * E1;
  Face* F, * F0, * F1;
  Vertex* V;
  
  // Edges
  for(E = Edges; E; E = E->next, ec++)
    {
      if(E != Edges)
	ASSERT1M(E->prev, "No prev pointer");

      ASSERT1M(E->Faces.size() > 0, "Edge must have at least one face!");
      if(E->Faces.size() < 1 || E->Faces.size() > 2)
	fprintf(stderr, "Nonmanifold Edge: %d 0x%08x\n", E->Faces.size(), long(E));

      ASSERT1M(E->v0 != E->v1, "Edge with the same vertices!");
      ASSERT1M(E->v0 && E->v1, "Edge vertex pointer is NULL!");
      for(i=0; i<E->Faces.size(); i++)
	ASSERT1M(E->Faces[i], "Edge face pointer is NULL!");

      for(i=0; i<E->v0->Edges.size(); i++)
	if(E->v0->Edges[i] == E)
	  break;
      ASSERT1M(i < E->v0->Edges.size(), "Vertex0 doesn't point to its Edge!");

      for(i=0; i<E->v1->Edges.size(); i++)
	if(E->v1->Edges[i] == E)
	  break;
      ASSERT1M(i < E->v1->Edges.size(), "Vertex1 doesn't point to its Edge!");

      for(i=0; i<E->Faces.size(); i++)
	{
	  F = E->Faces[i];
	  ASSERT1M(F->e0 == E || F->e1 == E || F->e2 == E, "Face doesn't point to its Edge!");
	}

      for(i=0; i<E->Faces.size(); i++)
	{
	  F0 = E->Faces[i];
	  ASSERT1M(F0->v0, "Edge points to dead face");
	  
	  // Make sure there is no duplicate face.
	  for(j=i+1; j<E->Faces.size(); j++)
	    {
	      F1 = E->Faces[j];
	      ASSERT1M(F0 != F1, "Same face exists twice in an edge");

	      ASSERT1M(F0->e0 != F1->e0 || F0->e1 != F1->e1 || F0->e2 != F1->e2, "EDup face 0");
	      ASSERT1M(F0->e0 != F1->e0 || F0->e1 != F1->e2 || F0->e2 != F1->e1, "EDup face 0");
	      ASSERT1M(F0->e0 != F1->e1 || F0->e1 != F1->e0 || F0->e2 != F1->e2, "EDup face 1");
	      ASSERT1M(F0->e0 != F1->e1 || F0->e1 != F1->e2 || F0->e2 != F1->e0, "EDup face 1");
	      ASSERT1M(F0->e0 != F1->e2 || F0->e1 != F1->e0 || F0->e2 != F1->e1, "EDup face 2");
	      ASSERT1M(F0->e0 != F1->e2 || F0->e1 != F1->e1 || F0->e2 != F1->e0, "EDup face 2");
	    }
	}

      // Only check for a duplicate edge every so often.
      if(Slow)
	for(E1 = E->next; E1; E1 = E1->next)
	  {
	    // Make sure there's no duplicate edge.
	    ASSERT1M(E1->v0 != E->v0 || E1->v1 != E->v1, "Duplicate edge");
	    ASSERT1M(E1->v0 != E->v1 || E1->v1 != E->v0, "Duplicate edge");
	  }

      ASSERT1M((E->v0->next || E->v0->prev) && (E->v1->next || E->v1->prev), "Edge points to dead vertex");
    }

  // Faces
  for(F = Faces; F; F = F->next, fc++)
    {
      if(F != Faces)
	ASSERT1M(F->prev, "No prev pointer");

      ASSERT1M(F->v0 != F->v1 && F->v1 != F->v2 && F->v0 != F->v2, "Face with the same vertices!");
      ASSERT1M(F->e0 != F->e1 && F->e1 != F->e2 && F->e0 != F->e2, "Face with the same edges!");
      ASSERT1M(F->e0 && F->e1 && F->e2, "Face has NULL edge pointer!");
      ASSERT1M(F->v0 && F->v1 && F->v2, "Face has NULL vertex pointer!");

      for(i=0; i<F->e0->Faces.size(); i++)
	if(F->e0->Faces[i] == F)
	  break;
      ASSERT1M(i < F->e0->Faces.size(), "Edge0 doesn't point to its face!");

      for(i=0; i<F->e1->Faces.size(); i++)
	if(F->e1->Faces[i] == F)
	  break;
      ASSERT1M(i < F->e1->Faces.size(), "Edge1 doesn't point to its face!");

      for(i=0; i<F->e2->Faces.size(); i++)
	if(F->e2->Faces[i] == F)
	  break;
      ASSERT1M(i < F->e2->Faces.size(), "Edge2 doesn't point to its face!");

      for(i=0; i<F->v0->Faces.size(); i++)
	if(F->v0->Faces[i] == F)
	  break;
      ASSERT1M(i < F->v0->Faces.size(), "Vertex0 doesn't point to its face!");

      for(i=0; i<F->v1->Faces.size(); i++)
	if(F->v1->Faces[i] == F)
	  break;
      ASSERT1M(i < F->v1->Faces.size(), "Vertex1 doesn't point to its face!");

      for(i=0; i<F->v2->Faces.size(); i++)
	if(F->v2->Faces[i] == F)
	  break;
      ASSERT1M(i < F->v2->Faces.size(), "Vertex2 doesn't point to its face!");

      // Make sure I don't point to any dead vertices or edges.
      ASSERT1M((F->v0->next || F->v0->prev) && (F->v1->next || F->v1->prev) && (F->v2->next || F->v2->prev),
		"Face points to dead vertex");

      ASSERT1M(F->e0->v0 && F->e1->v0 && F->e2->v0, "Face points to dead edge");

      if(Slow)
	for(F1 = F->next; F1; F1 = F1->next)
	  {
	    ASSERT1M(F != F1, "Same face exists twice in the list");

	    ASSERT1M(F->v0 != F1->v0 || F->v1 != F1->v1 || F->v2 != F1->v2, "Dup face 0");
	    ASSERT1M(F->v0 != F1->v0 || F->v1 != F1->v2 || F->v2 != F1->v1, "Dup face 0");
	    ASSERT1M(F->v0 != F1->v1 || F->v1 != F1->v0 || F->v2 != F1->v2, "Dup face 1");
	    ASSERT1M(F->v0 != F1->v1 || F->v1 != F1->v2 || F->v2 != F1->v0, "Dup face 1");
	    ASSERT1M(F->v0 != F1->v2 || F->v1 != F1->v0 || F->v2 != F1->v1, "Dup face 2");
	    ASSERT1M(F->v0 != F1->v2 || F->v1 != F1->v1 || F->v2 != F1->v0, "Dup face 2");
	  }
    }

  // Vertices
  for(V = Verts; V; V = V->next, vc++)
    {
      if(V != Verts)
	{
	  ASSERT1M(V->prev, "No prev pointer");
	  ASSERT1M(V->prev->next == V, "Broken vertex link");
	}

      ASSERT1M(V->Edges.size() >= 2, "Vertex must be part of at least two edges!");
      ASSERT1M(V->Faces.size() >= 1, "Vertex must be part of at least one face!");

      for(i=0; i<V->Edges.size(); i++)
	if(V->Edges[i]->v0 == V || V->Edges[i]->v1 == V)
	  break;
      ASSERT1M(i < V->Edges.size(), "Edge doesn't point to its vertex!");

      for(i=0; i<V->Faces.size(); i++)
	if(V->Faces[i]->v0 == V || V->Faces[i]->v1 == V || V->Faces[i]->v2 == V)
	  break;
      ASSERT1M(i < V->Faces.size(), "Face doesn't point to its vertex!");

      // Make sure I don't point to any dead faces or edges.
      for(i=0; i<V->Edges.size(); i++)
	{
	  // fprintf(stderr, "0x%08x\n", long(V->Edges[i]));
	  ASSERT1M(V->Edges[i]->Faces.size(), "Vertex points to dead edge");

	  for(j=i+1; j<V->Edges.size(); j++)
	    ASSERT1M(V->Edges[i] != V->Edges[j], "Bad medicine");
	}

      // Make sure I don't have any duplicate faces
      for(i=0; i<V->Faces.size(); i++)
	{
	  // cerr << "f"<<V->Faces.size();
	  Face *F0 = V->Faces[i];
	  ASSERT1M(F0->v0, "Vertex points to dead face");
	  
	  // Make sure there is no duplicate face.
	  for(j=i+1; j<V->Faces.size(); j++)
	    {
	      F1 = V->Faces[j];
	      ASSERT1M(F0 != F1, "Same face exists twice in a vertex");

	      ASSERT1M(F0->v0 != F1->v0 || F0->v1 != F1->v1 || F0->v2 != F1->v2, "Dup face 0");
	      ASSERT1M(F0->v0 != F1->v0 || F0->v1 != F1->v2 || F0->v2 != F1->v1, "Dup face 0");
	      ASSERT1M(F0->v0 != F1->v1 || F0->v1 != F1->v0 || F0->v2 != F1->v2, "Dup face 1");
	      ASSERT1M(F0->v0 != F1->v1 || F0->v1 != F1->v2 || F0->v2 != F1->v0, "Dup face 1");
	      ASSERT1M(F0->v0 != F1->v2 || F0->v1 != F1->v0 || F0->v2 != F1->v1, "Dup face 2");
	      ASSERT1M(F0->v0 != F1->v2 || F0->v1 != F1->v1 || F0->v2 != F1->v0, "Dup face 2");
	    }

	  // Make sure V points to two edges of the face.
	  int cnt = 0;
	  for(j=0; j<V->Edges.size(); j++)
	    if(V->Edges[j] == F0->e0 || V->Edges[j] == F0->e1 || V->Edges[j] == F0->e2)
	      cnt++;
	  // cerr << cnt << endl;
	  ASSERT1M(cnt == 2, "Vertex must have 2 edges of each of its faces.");
	}
    }

  //Dump();
  ASSERT1M(VertexCount == vc, "Bad vertex count");
  ASSERT1M(FaceCount == fc, "Bad face count");
  ASSERT1M(EdgeCount == ec, "Bad edge count");
}

//----------------------------------------------------------------------
void Mesh::Dump()
{
  cerr << "Mesh vert count: " << VertexCount << " edge count: "
       << EdgeCount << " face count: " << FaceCount << endl;
}

//----------------------------------------------------------------------
// Convert the mesh back to a model and return it.
// We could put unconnected components in separate Objects.
Object Mesh::ExportObject()
{
  Object Ob;
  Face* F;

  for(F = Faces; F; F = F->next) {
    Ob.verts.push_back(F->v0->V);
    Ob.Box += F->v0->V;
    Ob.verts.push_back(F->v1->V);
    Ob.Box += F->v1->V;
    Ob.verts.push_back(F->v2->V);
    Ob.Box += F->v2->V;
    
				// XXX Output color and texcoord here.
    if(HasColor) {
      Ob.dcolors.push_back(F->v0->Col);
      Ob.dcolors.push_back(F->v1->Col);
      Ob.dcolors.push_back(F->v2->Col);
    }
    
    if(HasTexCoords) {
      Ob.texcoords.push_back(F->v0->Tex);
      Ob.texcoords.push_back(F->v1->Tex);
      Ob.texcoords.push_back(F->v2->Tex);
    }
  }
  
  //Ob.GenNormals(0);
  
  return Ob;
}

//----------------------------------------------------------------------
// This is recursive.
void Mesh::FlipMe(Face *F, Vertex *v0, Vertex *v1, Vertex *v2)
{
  int i;
  if(F->visited)
    return;

  F->visited = true;

  ASSERT1M(((F->v0 == v0) + (F->v1 == v0) + (F->v2 == v0) + (F->v0 == v1) + (F->v1 == v1) + (F->v2 == v1) + (F->v0 == v2) + (F->v1 == v2) + (F->v2 == v2)) >= 2, "Faces should share exactly two vertices.");

  if((F->v0 == v0 && (F->v1 != v1 && F->v2 != v2)) || 
     (F->v1 == v0 && (F->v2 != v1 && F->v0 != v2)) || 
     (F->v2 == v0 && (F->v0 != v1 && F->v1 != v2)) ||

     (F->v0 == v1 && (F->v1 != v2 && F->v2 != v0)) || 
     (F->v1 == v1 && (F->v2 != v2 && F->v0 != v0)) || 
     (F->v2 == v1 && (F->v0 != v2 && F->v1 != v0)) ||

     (F->v0 == v2 && (F->v1 != v0 && F->v2 != v1)) || 
     (F->v1 == v2 && (F->v2 != v0 && F->v0 != v1)) || 
     (F->v2 == v2 && (F->v0 != v0 && F->v1 != v1)))
    {
      // Swap the winding.
      // cerr << "Swapping.\n";
      Vertex *T = F->v1;
      F->v1 = F->v2;
      F->v2 = T;
    }

  // Fix my neighbors.
  for(i=0; i<F->e0->Faces.size(); i++)
    if(F->e0->Faces[i] != F)
      FlipMe(F->e0->Faces[i], F->v0, F->v2, F->v1);

  for(i=0; i<F->e1->Faces.size(); i++)
    if(F->e1->Faces[i] != F)
      FlipMe(F->e1->Faces[i], F->v0, F->v2, F->v1);

  for(i=0; i<F->e2->Faces.size(); i++)
    if(F->e2->Faces[i] != F)
      FlipMe(F->e2->Faces[i], F->v0, F->v2, F->v1);
}

//----------------------------------------------------------------------
// Give all faces in each manifold the same winding.
void Mesh::FixFacing()
{
  Face* F;
  
  // Clear their visited flag.
  for(F = Faces; F; F = F->next)
    F->visited = false;

  for(F = Faces; F; F = F->next)
    FlipMe(F, F->v0, F->v1, F->v2);
}
} // End namespace Remote


