#ifndef NAUTILUS_ADJPRIMS_INCLUDED // -*- C++ -*-
#define NAUTILUS_ADJPRIMS_INCLUDED

/************************************************************************

  Primitive entities for adjacency models (AdjModel).
  $Id$

  Adapted from:
     mlab: (Id: primitives.h,v 1.7 1997/02/06 16:30:14 garland Exp)
  via
     QSlim: (Id: primitives.h,v 1.3 1997/02/18 21:11:29 garland Exp)

 ************************************************************************/

#include "Nautilus.h"
#include "NPrim.h"
#include <gfx/tools/Buffer.h>
#include <gfx/math/Vec3.h>
#include <gfx/geom/3D.h>

class Vertex;
class Edge;
class Face;

typedef buffer<Vertex *> vert_buffer;
typedef buffer<Edge *> edge_buffer;
typedef buffer<Face *> face_buffer;

extern void untagFaceLoop(Vertex *v);
extern void collectFaceLoop(Vertex *v, face_buffer& faces);

#define EDGE_BOGUS 0
#define EDGE_BORDER 1
#define EDGE_MANIFOLD 2
#define EDGE_NONMANIFOLD 3
extern int classifyEdge(Edge *);

#define VERTEX_INTERIOR 0
#define VERTEX_BORDER 1
#define VERTEX_BORDER_ONLY 2
extern int classifyVertex(Vertex *);

////////////////////////////////////////////////////////////////////////
//
// The actual class definitions
//

class VProp
{
public:
    Vec3 color;
};

class FProp
{
public:
    Vec3 color;
};

class Vertex : public Vec3, public NTaggedPrim
{
    edge_buffer edge_uses;
    
public:
#ifdef SUPPORT_VCOLOR
    VProp *props;
#endif

    Vertex(real x, real y, real z) : Vec3(x, y, z), edge_uses(6) {
#ifdef SUPPORT_VCOLOR
	props = NULL;
#endif
    }
    //
    // Standard methods for all objects
    //
    void kill();
    edge_buffer& edgeUses() { return edge_uses; }

    //
    // Basic primitives for manipulating model topology
    //
    void linkEdge(Edge *);
    void unlinkEdge(Edge *);
    void remapTo(Vertex *v);
};


class Edge : public NPrim
{
private:

    Vertex *v1;

    face_buffer *face_uses;
    Edge *twin;

    Edge(Edge *twin, Vertex *org); // the twin constructor

public:
    Edge(Vertex *, Vertex *);
    ~Edge();

    //
    // Fundamental Edge accessors
    //
    Vertex *org()  { return v1;       }
    Vertex *dest() { return twin->v1; }
    Edge *sym()    { return twin;     }

    //
    // Standard methods for all objects
    //
    void kill();
    face_buffer& faceUses() { return *face_uses; }

    //
    // Basic primitives for manipulating model topology
    //
    void linkFace(Face *);
    void unlinkFace(Face *);
    void remapEndpoint(Vertex *from, Vertex *to);
    void remapTo(Edge *);
};

class Face : public Face3, public NTaggedPrim
{
    Edge *edges[3];


public:
#ifdef SUPPORT_FCOLOR
    FProp *props;
#endif

    Face(Edge *, Edge *, Edge *);

    //
    // Basic Face accessors
    //
    const Vec3& vertexPos(int i) const { return *edges[i]->org(); }
    void vertexPos(int, const Vec3&) {
	fatal_error("Face: can't directly set vertex position.");
    }

    Vertex *vertex(int i) { return edges[i]->org(); }
    const Vertex *vertex(int i) const { return edges[i]->org(); }
    Edge *edge(int i)               { return edges[i];        }


    //
    // Standard methods for all objects
    //
    void kill();

    void remapEdge(Edge *from, Edge *to);
};


// NAUTILUS_ADJPRIMS_INCLUDED
#endif
