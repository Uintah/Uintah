
/*
 *  VertexPrim.h: Base class for primitives that use the Vertex class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/VertexPrim.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>
#include <Malloc/Allocator.h>

GeomVertexPrim::GeomVertexPrim()
{
}

GeomVertexPrim::GeomVertexPrim(const GeomVertexPrim& copy)
: GeomObj(copy), verts(copy.verts.size())
{
    for(int i=0;i<verts.size();i++)
	verts[i]=copy.verts[i]->clone();
}

GeomVertexPrim::~GeomVertexPrim()
{
    for(int i=0;i<verts.size();i++)
	delete verts[i];
}

void GeomVertexPrim::get_bounds(BBox& bb)
{
    for(int i=0;i<verts.size();i++)
	bb.extend(verts[i]->p);
}

void GeomVertexPrim::get_bounds(BSphere& bs)
{
    for(int i=0;i<verts.size();i++)
	bs.extend(verts[i]->p);
}

void GeomVertexPrim::add(const Point& p)
{
    verts.add(scinew GeomVertex(p));
}

void GeomVertexPrim::add(const Point& p, const Vector& normal)
{
    verts.add(scinew GeomNVertex(p, normal));
}

void GeomVertexPrim::add(const Point& p, const MaterialHandle& matl)
{
    verts.add(scinew GeomMVertex(p, matl));
}

void GeomVertexPrim::add(const Point& p, const Vector& normal,
			 const MaterialHandle& matl)
{
    verts.add(scinew GeomNMVertex(p, normal, matl));
}

void GeomVertexPrim::add(GeomVertex* vtx)
{
    verts.add(vtx);
}

GeomVertex::GeomVertex(const Point& p)
: p(p)
{
}

GeomVertex::GeomVertex(const GeomVertex& copy)
: p(copy.p)
{
}

GeomVertex::~GeomVertex()
{
}

GeomVertex* GeomVertex::clone()
{
    return scinew GeomVertex(*this);
}

GeomNVertex::GeomNVertex(const Point& p, const Vector& normal)
: GeomVertex(p), normal(normal)
{
}

GeomNVertex::GeomNVertex(const GeomNVertex& copy)
: GeomVertex(copy), normal(copy.normal)
{
}

GeomVertex* GeomNVertex::clone()
{
    return scinew GeomNVertex(*this);
}

GeomNVertex::~GeomNVertex()
{
}

GeomNMVertex::GeomNMVertex(const Point& p, const Vector& normal,
			   const MaterialHandle& matl)
: GeomNVertex(p, normal), matl(matl)
{
}

GeomNMVertex::GeomNMVertex(const GeomNMVertex& copy)
: GeomNVertex(copy), matl(copy.matl)
{
}

GeomVertex* GeomNMVertex::clone()
{
    return scinew GeomNMVertex(*this);
}

GeomNMVertex::~GeomNMVertex()
{
}

GeomMVertex::GeomMVertex(const Point& p, const MaterialHandle& matl)
: GeomVertex(p), matl(matl)
{
}

GeomMVertex::GeomMVertex(const GeomMVertex& copy)
: GeomVertex(copy), matl(matl)
{
}

GeomVertex* GeomMVertex::clone()
{
    return scinew GeomMVertex(*this);
}

GeomMVertex::~GeomMVertex()
{
}

#ifdef __GNUG__

#include <Classlib/Array1.cc>

template class Array1<GeomVertex*>;

#endif
