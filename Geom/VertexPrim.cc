
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
#include <Classlib/String.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>
#include <Malloc/Allocator.h>

static Persistent* make_GeomVertex()
{
    return scinew GeomVertex(Point(0,0,0));
}

PersistentTypeID GeomVertex::type_id("GeomVertex", "Persistent", make_GeomVertex);

static Persistent* make_GeomNVertex()
{
    return scinew GeomNVertex(Point(0,0,0),Vector(0,0,1));
}

PersistentTypeID GeomNVertex::type_id("GeomNVertex", "GeomVertex", make_GeomNVertex);

static Persistent* make_GeomNMVertex()
{
    return scinew GeomNMVertex(Point(0,0,0), Vector(0,0,1), MaterialHandle(0));
}

PersistentTypeID GeomNMVertex::type_id("GeomNMVertex", "GeomNVertex", make_GeomNMVertex);

static Persistent* make_GeomMVertex()
{
    return scinew GeomMVertex(Point(0,0,0), MaterialHandle(0));
}

PersistentTypeID GeomMVertex::type_id("GeomMVertex", "GeomVertex", make_GeomMVertex);

static Persistent* make_GeomCVertex()
{
    return scinew GeomCVertex(Point(0,0,0), Color(0,0,0));
}

PersistentTypeID GeomCVertex::type_id("GeomCVertex", "GeomVertex", make_GeomCVertex);

PersistentTypeID GeomVertexPrim::type_id("GeomVertexPrim", "GeomObj", 0);

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
    verts.add(new GeomVertex(p));
}

void GeomVertexPrim::add(const Point& p, const Vector& normal)
{
    verts.add(new GeomNVertex(p, normal));
}

void GeomVertexPrim::add(const Point& p, const MaterialHandle& matl)
{
    verts.add(new GeomMVertex(p, matl));
}

void GeomVertexPrim::add(const Point& p, const Color& clr)
{
    verts.add(new GeomCVertex(p, clr));
}

void GeomVertexPrim::add(const Point& p, const Vector& normal,
			 const MaterialHandle& matl)
{
    verts.add(new GeomNMVertex(p, normal, matl));
}

void GeomVertexPrim::add(GeomVertex* vtx)
{
    verts.add(vtx);
}

#define GEOMVERTEXPRIM_VERSION 1

void GeomVertexPrim::io(Piostream& stream)
{
    stream.begin_class("GeomVertexPrim", GEOMVERTEXPRIM_VERSION);
    GeomObj::io(stream);
    Pio(stream, verts);
    stream.end_class();
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

#define GEOMVERTEX_VERSION 1

void GeomVertex::io(Piostream& stream)
{
    stream.begin_class("GeomVertex", GEOMVERTEX_VERSION);
    Pio(stream, p);
    stream.end_class();
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

#define GEOMNVERTEX_VERSION 1

void GeomNVertex::io(Piostream& stream)
{
    stream.begin_class("GeomNVertex", GEOMNVERTEX_VERSION);
    GeomVertex::io(stream);
    Pio(stream, normal);
    stream.end_class();
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

#define GEOMNMVERTEX_VERSION 1

void GeomNMVertex::io(Piostream& stream)
{
    stream.begin_class("GeomNMVertex", GEOMNMVERTEX_VERSION);
    GeomNVertex::io(stream);
    Pio(stream, matl);
    stream.end_class();
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

#define GEOMMVERTEX_VERSION 1

void GeomMVertex::io(Piostream& stream)
{
    stream.begin_class("GeomMVertex", GEOMMVERTEX_VERSION);
    GeomVertex::io(stream);
    Pio(stream, matl);
    stream.end_class();
}

#define GEOMCVERTEX_VERSION 1

void GeomCVertex::io(Piostream& stream)
{
    stream.begin_class("GeomCVertex", GEOMMVERTEX_VERSION);
    GeomVertex::io(stream);
    Pio(stream, color);
    stream.end_class();
}

GeomCVertex::GeomCVertex(const Point& p, const Color& clr)
: GeomVertex(p), color(clr)
{
}

GeomCVertex::GeomCVertex(const GeomCVertex& copy)
: GeomVertex(copy), color(copy.color)
{
}

GeomVertex* GeomCVertex::clone()
{
    return scinew GeomCVertex(*this);
}

GeomCVertex::~GeomCVertex()
{
}

void Pio(Piostream& stream, GeomVertex*& obj)
{
    Persistent* tmp=obj;
    stream.io(tmp, GeomVertex::type_id);
    if(stream.reading())
	obj=(GeomVertex*)tmp;
}

#ifdef __GNUG__

#include <Classlib/Array1.cc>

template class Array1<GeomVertex*>;
template void Pio(Piostream&, Array1<GeomVertex*>&);

#endif

#ifdef __sgi
#if _MIPS_SZPTR == 64
#include <Classlib/Array1.cc>

static void _dummy1_(Piostream& p1, Array1<GeomVertex*>& p2)
{
    Pio(p1, p2);
}

static void _dummy2_(Piostream& p1, Array1<MaterialHandle>& p2)
{
    Pio(p1, p2);
}

#endif
#endif
