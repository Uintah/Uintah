/************************************************************************

  Primitive entities for adjacency models (AdjModel).
  $Id$

  Adapted from:
     mlab: (Id: primitives.cc,v 1.7 1997/02/06 16:32:45 garland Exp)
  via
     QSlim: (Id: primitives.cc,v 1.2 1997/02/18 21:11:27 garland Exp)

 ************************************************************************/

#include "AdjPrims.h"


void Vertex::kill()
{
#ifdef SAFETY
    assert( edge_uses.length() == 0 );
#endif
    markInvalid();
    edge_uses.reset();
}

void Vertex::linkEdge(Edge *e)
{
    edge_uses.add(e);
}

void Vertex::unlinkEdge(Edge *e)
{
    int index = edge_uses.find(e);

#ifdef SAFETY
    assert( index >= 0 );
#endif
    edge_uses.remove(index);
    if( edge_uses.length() <= 0 )
	kill();
}

void Vertex::remapTo(Vertex *v)
{
    if( v != this )
    {
	for(int i=0; i<edge_uses.length(); i++)
	{
	    assert( edge_uses(i)->org() == this );
	    edge_uses(i)->remapEndpoint(this, v);
	}

	kill();
    }
}



Edge::Edge(Vertex *a, Vertex *b)
{
    v1 = a;
    v1->linkEdge(this);

    face_uses = new buffer<Face *>(2);

    twin = new Edge(this, b);
}

Edge::Edge(Edge *sibling, Vertex *org)
{
    v1 = org;
    v1->linkEdge(this);

    face_uses = sibling->face_uses;
    twin = sibling;
}

Edge::~Edge()
{
    if( twin )
    {
	face_uses->free();
	delete face_uses;

	twin->twin = NULL;
	delete twin;
    }
}

void Edge::kill()
{
#ifdef SAFETY
    assert( face_uses->length() == 0 );
#endif
    if( isValid() )
    {
	org()->unlinkEdge(this);
	dest()->unlinkEdge(sym());
	markInvalid();
	twin->markInvalid();
	face_uses->reset();
    }
}

void Edge::linkFace(Face *face)
{
    face_uses->add(face);
}

void Edge::unlinkFace(Face *face)
{
    int index = face_uses->find(face);

#ifdef SAFETY
    assert( index>=0 );
#endif
    face_uses->remove(index);
    if( face_uses->length() == 0 )
	kill();
}

void Edge::remapTo(Edge *e)
{
    if( e != this )
    {
	for(int i=0; i<face_uses->length(); i++)
	{
	    (*face_uses)(i)->remapEdge(this, e);
	}
    
	// Disconnect from all faces and vertices
	kill();
    }
}

void Edge::remapEndpoint(Vertex *from, Vertex *to)
{
    if( org()==from )
    {
	v1 = to;
	to->linkEdge(this);
    }
    else if( dest()==from )
    {
	twin->v1 = to;
	to->linkEdge(twin);
    }
    else
    {
	cerr << "WARNING remapEndpoint: Illegal endpoint." << endl;
    }

    //
    // The cached Plane equations for the faces attached to us may
    // no longer be valid (in general, chances are pretty slim that they're OK)
    for(int i=0; i<face_uses->length(); i++)
    {
	face_uses->ref(i)->invalidatePlane();
    }
}



Face::Face(Edge *e0, Edge *e1, Edge *e2)
    : Face3(*e0->org(), *e1->org(), *e2->org())
{
    edges[0] = e0;
    edges[1] = e1;
    edges[2] = e2;

    edges[0]->linkFace(this);
    edges[1]->linkFace(this);
    edges[2]->linkFace(this);

#ifdef SUPPORT_FCOLOR
    props = NULL;
#endif
}

void Face::kill()
{
    if( isValid() )
    {
	if( edge(0)->isValid() )
	    edge(0)->unlinkFace(this);

	if( edge(1)->isValid() )
	    edge(1)->unlinkFace(this);

	if( edge(2)->isValid() )
	    edge(2)->unlinkFace(this);

	markInvalid();
    }
}

void Face::remapEdge(Edge *from, Edge *to)
{
    for(int i=0; i<3; i++)
    {
	if( edges[i] == from )
	{
	    edges[i] = to;
	    to->linkFace(this);
	}
	else if( edges[i] == from->sym() )
	{
	    edges[i] = to->sym();
	    to->sym()->linkFace(this);
	}
    }

    invalidatePlane();
}



void untagFaceLoop(Vertex *v)
{
    edge_buffer& edges = v->edgeUses();

    for(int j=0; j<edges.length(); j++)
    {	    
	face_buffer& faces = edges(j)->faceUses();
	for(int k=0; k<faces.length(); k++)
	    faces(k)->untag();
    }
}

void collectFaceLoop(Vertex *v, face_buffer& loop)
{
    edge_buffer& edges = v->edgeUses();

    for(int j=0; j<edges.length(); j++)
    {	    
	face_buffer& faces = edges(j)->faceUses();
	for(int k=0; k<faces.length(); k++)
	    if( !faces(k)->isTagged() )
	    {
		loop.add(faces(k));
		faces(k)->tag();
	    }
    }
}

int classifyEdge(Edge *e)
{
    int cls = e->faceUses().length();

    if( cls>3 ) cls=3;

    return cls;
}

int classifyVertex(Vertex *v)
{
    int border_count = 0;
    const edge_buffer& edges = v->edgeUses();

    for(int i=0; i<edges.length(); i++)
	if( classifyEdge(edges(i)) == 1 )
	    border_count++;

    if( border_count == edges.length() )
	return VERTEX_BORDER_ONLY;
    else 
	return (border_count > 0);
}
