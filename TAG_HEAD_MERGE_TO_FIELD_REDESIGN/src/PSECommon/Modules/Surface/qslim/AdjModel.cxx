/************************************************************************

  Adjacency model representation.
  $Id$

  Adapted from:
     mlab: (Id: polymodel.cc,v 1.21 1997/02/06 16:34:14 garland Exp)
  via
     QSlim: (Id: polymodel.cc,v 1.5 1997/02/25 20:36:36 garland Exp)

 ************************************************************************/

#include "AdjModel.h"

int Model::in_Vertex(const Vec3& p)
{
    Vertex *v = newVertex(p[X], p[Y], p[Z]);
    bounds.addPoint(p);
    return vertCount() - 1;
}

int Model::in_Face(int a, int b, int c)
{
    Vertex *v1 = vertices(a-1);
    Vertex *v2 = vertices(b-1);
    Vertex *v3 = vertices(c-1);
	
    Face *t = newFace(v1, v2, v3);

    return faceCount() - 1;
}

#ifdef SUPPORT_FCOLOR
int Model::in_FColor(const Vec3& c)
{
    Face *f = faces(faces.length()-1);

    if( !f->props )
	f->props = new FProp;

    f->props->color = c;
    return 0;
}
#endif

#ifdef SUPPORT_VCOLOR
int Model::in_VColor(const Vec3& c)
{
    Vertex *v = vertices(vertices.length()-1);

    if( !v->props )
	v->props = new VProp;

    v->props->color = c;
    return 0;
}
#endif

Vec3 Model::synthesizeNormal(Vertex *v)
{
    Vec3 n(0,0,0);
    int n_count = 0;

    edge_buffer& edges = v->edgeUses();
    for(int i=0; i<edges.length(); i++)
    {
	face_buffer& faces = edges(i)->faceUses();

	for(int j=0; j<faces.length(); j++)
	{
	    n += faces(j)->plane().normal();
	    n_count++;
	}
    }

    if( n_count )
	n /= (real)n_count;
    else
    {
	cerr << "Vertex with no normals!!: " << v->uniqID;
	cerr << " / " << v->tempID << endl;
    }
    return n;
}



////////////////////////////////////////////////////////////////////////
//
// Model Transmogrification:
//
// These routines are the basic model munging procedures.
// All model simplification is implemented in terms of these
// transmogrification primitives.
////////////////////////////////////////////////////////////////////////


Vertex *Model::newVertex(real x, real y, real z)
{
    Vertex *v = new Vertex(x, y, z);

    v->uniqID = vertices.add(v);

    if( logfile && selected_output&OUTPUT_MODEL_DEFN )
	*logfile << "v " << x << " " << y << " " << z << endl;

    validVertCount++;

    return v;
}

Edge *Model::newEdge(Vertex *a, Vertex *b)
{
    Edge *e = new Edge(a, b);

    e->uniqID = edges.add(e);
    e->sym()->uniqID = e->uniqID;

    validEdgeCount++;

    return e;
}

static Edge *get_edge(Model *m, Vertex *org, Vertex *v)
{
    edge_buffer& edge_uses = org->edgeUses();

    for(int i=0; i<edge_uses.length(); i++)
	if( edge_uses(i)->dest() == v )
	    return edge_uses(i);

    Edge *e = m->newEdge(org, v);

    return e;
}


Face *Model::newFace(Vertex *v1, Vertex *v2, Vertex *v3)
{
    Edge *e0 = get_edge(this, v1, v2);   // v1->edgeTo(m, v2);
    Edge *e1 = get_edge(this, v2, v3);   // v2->edgeTo(m, v3);
    Edge *e2 = get_edge(this, v3, v1);   // v3->edgeTo(m, v1);
  
    Face *t = new Face(e0, e1, e2);

    t->uniqID = faces.add(t);

    if( logfile && selected_output&OUTPUT_MODEL_DEFN )
	*logfile << "f " << v1->uniqID+1 << " "
		 << v2->uniqID+1 << " " << v3->uniqID+1 << endl;

    validFaceCount++;

    return t;
}


void Model::killVertex(Vertex *v)
{
    if( v->isValid() )
    {
	if( logfile && selected_output&OUTPUT_CONTRACTIONS )
	    *logfile << "v- " << v->uniqID+1 << endl;

	v->kill();
	validVertCount--;
    }
}

void Model::killEdge(Edge *e)
{
    if( e->isValid() )
    {
	e->kill();
	validEdgeCount--;
    }
}

void Model::killFace(Face *f)
{
    if( f->isValid() )
    {
	if( logfile && selected_output&OUTPUT_CONTRACTIONS )
	    *logfile << "f- " << f->uniqID+1 << endl;

	f->kill();
	validFaceCount--;
    }
}

void Model::reshapeVertex(Vertex *v, real x, real y, real z)
{
    v->set(x, y, z);

#ifdef LOG_LOWLEVEL_OPS
    if( logfile )
	*logfile << "v! " << v->uniqID+1 << " "
		 << x << " " << y << " " << z << endl;
#endif
}

void Model::remapVertex(Vertex *from, Vertex *to)
{
    from->remapTo(to);

#ifdef LOG_LOWLEVEL_OPS
    if( logfile )
	*logfile << "v> " << from->uniqID+1 << " " << to->uniqID+1 << endl;
#endif
}





void Model::contract(Vertex *v1,
		     const vert_buffer& rest,
		     const Vec3& to,
		     face_buffer& changed)
{
    int i;

    if( logfile && selected_output&OUTPUT_CONTRACTIONS )
    {
	*logfile << "v% (" << v1->uniqID+1;
	for(i=0; i<rest.length(); i++)
	    *logfile << " " << rest(i)->uniqID+1;
	
	*logfile << ") " << to[X] << " " << to[Y] << " " << to[Z] << endl;
    }

    //
    // Collect all the faces that are going to be changed
    //
    contractionRegion(v1, rest, changed);

    reshapeVertex(v1, to[X], to[Y], to[Z]);

    for(i=0; i<rest.length(); i++)
	rest(i)->remapTo(v1);

    removeDegeneracy(changed);
}

void Model::maybeFixFace(Face *F)
{
    //
    // Invalid faces do not need to be fixed.
#ifdef SAFETY
    assert( F->isValid() );
#endif

    Vertex *v0=F->vertex(0); Vertex *v1=F->vertex(1); Vertex *v2=F->vertex(2);
    Edge *e0 = F->edge(0); Edge *e1 = F->edge(1); Edge *e2 = F->edge(2);

    bool a=(v0 == v1),  b=(v0 == v2),  c=(v1 == v2);

    if( a && c )
    {
	// This triangle has been reduced to a point
	killEdge(e0);
	killEdge(e1);
	killEdge(e2);

	killFace(F);
    }
    //
    // In the following 3 cases, the triangle has become an edge
    else if( a )
    {
	killEdge(e0);
	e1->remapTo(e2->sym());
	killFace(F);
    }
    else if( b )
    {
	killEdge(e2);
	e0->remapTo(e1->sym());
	killFace(F);
    }
    else if( c )
    {
	killEdge(e1);
	e0->remapTo(e2->sym());
	killFace(F);
    }
    else
    {
	// This triangle remains non-degenerate
    }
}

void Model::removeDegeneracy(face_buffer& changed)
{
    for(int i=0; i<changed.length(); i++)
	maybeFixFace(changed(i));
}

void Model::contractionRegion(Vertex *v1,
			      const vert_buffer& vertices,
			      face_buffer& changed)
{
    changed.reset();

    //
    // First, reset the marks on all reachable vertices
    int i;

    untagFaceLoop(v1);
    for(i=0; i<vertices.length(); i++)
	untagFaceLoop(vertices(i));

    //
    // Now, pick out all the unique reachable faces
    collectFaceLoop(v1, changed);
    for(i=0; i<vertices.length(); i++)
	collectFaceLoop(vertices(i), changed);
}


void Model::contractionRegion(Vertex *v1, Vertex *v2, face_buffer& changed)
{
    changed.reset();

    //
    // Clear marks on all reachable faces
    untagFaceLoop(v1);
    untagFaceLoop(v2);

    //
    // Collect all the unique reachable faces
    collectFaceLoop(v1, changed);
    collectFaceLoop(v2, changed);
}

void Model::contract(Vertex *v1, Vertex *v2, const Vec3& to,
		     face_buffer& changed)
{
#ifdef SAFETY
    assert( v1 != v2);
#endif

    if( logfile && selected_output&OUTPUT_CONTRACTIONS )
	*logfile << "v% (" << v1->uniqID+1 << " " << v2->uniqID+1 << ") "
		 << to[X] << " "
		 << to[Y] << " "
		 << to[Z] << endl;

    //
    // Collect all the faces that are going to be changed
    //
    contractionRegion(v1, v2, changed);

    reshapeVertex(v1, to[X], to[Y], to[Z]);

    //
    // Map v2 ---> v1.  This accomplishes the topological change that we want.
    v2->remapTo(v1);

    removeDegeneracy(changed);
}



ostream& operator<<(ostream& out, Model& M)
{
    int i;
    int uniqVerts = 0;

    out << "#$SMF 1.0" << endl;

//    out_annotate(out, M);

    for(i=0; i<M.vertCount(); i++)
    {
	if( M.vertex(i)->isValid() )
	{
	    M.vertex(i)->tempID = ++uniqVerts;

//	    out_annotate(out, M.vertex(i));
	    const Vertex& v = *M.vertex(i);
	    out << "v " << v[X] << " " << v[Y] << " " << v[Z] << endl;

#ifdef SUPPORT_VCOLOR
	    if( v.props )
		out << ":vc "
		    << v.props->color[X] << " "
		    << v.props->color[Y] << " "
		    << v.props->color[Z] << endl;
#endif
	}
	else
	    M.vertex(i)->tempID = -1;

    }

    for(i=0; i<M.faceCount(); i++)
    {
	if( M.face(i)->isValid() )
	{
	    Face *f = M.face(i);
//	    out_annotate(out, f);
	    out << "f ";
	    Vertex *v0 = (Vertex *)f->vertex(0);
	    Vertex *v1 = (Vertex *)f->vertex(1);
	    Vertex *v2 = (Vertex *)f->vertex(2);
#ifdef SAFETY
	    assert( v0->tempID >= 0 );
	    assert( v1->tempID >= 0 );
	    assert( v2->tempID >= 0 );
#endif
	    out<<v0->tempID << " " <<v1->tempID<< " " <<v2->tempID<< endl;
#ifdef SUPPORT_FCOLOR
	    if( f->props )
		out << ":fc "
		    << f->props->color[X] << " "
		    << f->props->color[Y] << " "
		    << f->props->color[Z] << endl;
#endif
	}
    }

    return out;
}
