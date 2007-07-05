//////////////////////////////////////////////////////////////////////
// Mesh.h - Represent a model as a mesh.
//
// David K. McAllister, August 1999.
//
// This represents arbitrary non-manifold meshes of triangles, edges,
// and faces. It can import and export a Model. This mesh includes
// things used only for simplifying meshes. Using it for other
// purposes won't be slow, but it will take more memory, since every
// Vertex, Edge, and Face has a Quadric in it.

#ifndef mesh_h
#define mesh_h

#include <Malloc/Allocator.h>
#include <Rendering/ZTex/Model.h>
#include <Rendering/ZTex/Quadric.h>

namespace SemotusVisum {
namespace Rendering {

#define DIST_FACTOR 0.000

// Returns true if the points are within D of eachother.
inline bool VecEq(const Vector &V0, const Vector &V1, const double &DSqr = 0.0)
{
  return (V0 - V1).length2() <= DSqr;
}

struct Edge;
struct Face;

struct Vertex
{
  Quadric3 Q;  // Used for simplifying meshes.
  Vector V;
  Vector Col;
  Vector Tex;
  Vertex *next, *prev;

  vector<Edge *> Edges;
  vector<Face *> Faces;

  inline Vertex()
  {
    // XXX This isn't necessary in general, but for simplification it speeds things up.
    Q.zero();
  }

  inline ~Vertex()
  {
    if(next)
      next->prev = prev;
    if(prev)
      prev->next = next;

#ifdef SCI_MESH_DEBUG
    next = prev = NULL;
#endif
  }

  inline void ListRemove(vector<Vertex *> &Ll)
  {
    for(unsigned i=0; i<Ll.size(); )
      if(this == Ll[i])
	{
	  Ll[i] = Ll.back();
	  Ll.pop_back();
	}
      else
	i++;
  }
};

struct Edge
{
  Quadric3 Q; // This is the resulting error when the edge is collapsed.
  Vector V; // The vertex result after the edge is collapsed.
  Vector Col; // Results of collapse.
  Vector Tex; // Results of collapse.
  double Cost;
  Edge *next, *prev; // For the linked list of all edges.

  int Heap; // Index into the heap vector.

  vector<Face *> Faces; // Should be 1 or 2 if manifold.
  Vertex *v0, *v1;

  inline ~Edge()
  {
    if(next)
      next->prev = prev;
    if(prev)
      prev->next = next;

#ifdef SCI_MESH_DEBUG
    v0 = NULL;
    next = prev = NULL;
#endif
  }

  inline void ListRemove(vector<Edge *> &Ll)
  {
    for(int i=0; i<(int)Ll.size(); )
      if(this == Ll[i])
	{
	  Ll[i] = Ll.back();
	  Ll.pop_back();
	}
      else
	i++;
  }
};

struct Face
{
  Face *next, *prev;

  Vertex *v0, *v1, *v2;
  Edge *e0, *e1, *e2;
  bool visited;

  inline ~Face()
  {
    if(next)
      next->prev = prev;
    if(prev)
      prev->next = next;

#ifdef SCI_MESH_DEBUG
    v0 = NULL;
    next = prev = NULL;
#endif
  }

  inline void ListRemove(vector<Face *> &Ll)
  {
    for(int i=0; i<(int)Ll.size(); )
      if(this == Ll[i])
	{
	  Ll[i] = Ll.back();
	  Ll.pop_back();
	}
      else
	i++;
  }
};

} // namespace Tools
} // namespace Remote

#include <Rendering/ZTex/VertexTree.h>

namespace SemotusVisum {
namespace Rendering {

//----------------------------------------------------------------------
struct Mesh
{
  Vertex *Verts;
  Edge *Edges;
  Face *Faces;

  double MeshMaxDist; // When making mesh, this tells when two verts are the same.
  int FaceCount, VertexCount, EdgeCount;
  bool HasColor, HasTexCoords;

  KDTree<KDVertex> VertTree;

  inline Mesh()
  {
    MeshMaxDist = 0;
    Verts = NULL;
    Edges = NULL;
    Faces = NULL;
    EdgeCount = VertexCount = FaceCount = 0;
    HasColor = HasTexCoords = false;
  }

  inline Mesh(const Object &M)
  {
    Verts = NULL;
    Edges = NULL;
    Faces = NULL;
    EdgeCount = VertexCount = FaceCount = 0;
    HasColor = HasTexCoords = false;
    ImportObject(M);
  }

  // ADDING DESTRUCTOR 
  inline ~Mesh() {

    // Delete vertices
    Vertex * vertex = Verts;
    Vertex * vtmp = NULL;
    while ( vertex != NULL ) {
      vtmp = vertex->next;
      delete vertex;
      vertex = vtmp;
    }

    // Delete edges
    Edge *edge = Edges;
    Edge *etmp = NULL;
    while ( edge != NULL ) {
      etmp = edge->next;
      delete edge;
      edge = etmp;
    }

    // Delete faces
    Face *face = Faces;
    Face *ftmp = NULL;
    while ( face != NULL ) {
      ftmp = face->next;
      delete face;
      face = ftmp;
    }
  }
  
  // Add the vertex without seeing if it already exists.
  // Doesn't make anything point to this vertex.
  inline Vertex *AddVertex(const Vector &Vec, const Vector *Color, const Vector *TexCoord)
  {
    Vertex *V = scinew Vertex;
    V->V = Vec;
    if(HasColor)
      V->Col = *Color;
    if(HasTexCoords)
      V->Tex = *TexCoord;
    V->next = Verts;
    V->prev = NULL;
    if(Verts)
      Verts->prev = V;
    Verts = V;
    VertexCount++;
    
    return V;
  }

  // Add the vertex without seeing if it already exists.
  // Doesn't make anything point to this vertex.
  // This is faster.
  inline Vertex *AddVertex(const Vector &Vec)
  {
    Vertex *V = scinew Vertex; // HERE
    V->V = Vec;
    V->next = Verts;
    V->prev = NULL;
    if(Verts)
      Verts->prev = V;
    Verts = V;
    VertexCount++;
    
    return V;
  }

  // Makes the vertices point to this edge.
  inline Edge *AddEdge(Vertex *v0, Vertex *v1)
  {
    // Add the edge to the front.
    Edge *E = scinew Edge; // HERE
    E->v0 = v0;
    E->v1 = v1;
    E->next = Edges;
    E->prev = NULL;
    if(Edges)
      Edges->prev = E;
    Edges = E;
    EdgeCount++;

    v0->Edges.push_back(E);
    v1->Edges.push_back(E);
    
    return E;
  }

  // Makes the edges and vertices point to this face.
  inline Face *AddFace(Vertex *v0, Vertex *v1, Vertex *v2, Edge *e0, Edge *e1, Edge *e2)
  {
    // Create the face.
    Face *F = scinew Face; // HERE
    F->next = Faces;
    F->prev = NULL;
    if(Faces)
      Faces->prev = F;
    Faces = F;
    F->v0 = v0;
    F->v1 = v1;
    F->v2 = v2;
    F->e0 = e0;
    F->e1 = e1;
    F->e2 = e2;
    
    // Add the face index to the vertices and edges.
    v0->Faces.push_back(F);
    v1->Faces.push_back(F);
    v2->Faces.push_back(F);
    e0->Faces.push_back(F);
    e1->Faces.push_back(F);
    e2->Faces.push_back(F);
    
    FaceCount++;

    return F;
  }

  // Add the incoming object to the mesh.
  void ImportObject(const Object &M);
  
  // Return an object made from the mesh.
  Object ExportObject();

  // Makes sure the mesh is proper and counts everything, too.
  void CheckIntegrity(const bool Slow = false);

  void Dump();

  void FixFacing();

  void FlipMe(Face *F, Vertex *v0, Vertex *v1, Vertex *v2);

  // AddedVert will be true if it added.
  Vertex *FindVertex(const Vector &V, bool &AddedVert, const Vector *Color = NULL, const Vector *TexCoord = NULL);
  Vertex *FindVertexInEdgeList(const vector<Edge *> &Edg, const Vector &V, Edge * &e) const;

  Edge *FindEdge(Vertex *v0, Vertex *v1);

  
};


} // namespace Tools
} // namespace Remote

#endif
