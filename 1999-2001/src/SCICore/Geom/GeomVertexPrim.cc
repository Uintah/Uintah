//static char *id="@(#) $Id$";

/*
 *  GeomVertexPrim.cc: Base class for primitives that use the Vertex class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifdef _WIN32
#pragma warning(disable:4291) // quiet the visual C++ compiler
#endif

#include <SCICore/Geom/GeomVertexPrim.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Containers/TrivialAllocator.h>

namespace SCICore {
namespace GeomSpace {

using SCICore::Containers::TrivialAllocator;

static TrivialAllocator GeomVertex_alloc(sizeof(GeomVertex));
static TrivialAllocator GeomNVertex_alloc(sizeof(GeomNVertex));
static TrivialAllocator GeomMVertex_alloc(sizeof(GeomMVertex));
static TrivialAllocator GeomNMVertex_alloc(sizeof(GeomNMVertex));
static TrivialAllocator GeomCVertex_alloc(sizeof(GeomCVertex));

void* GeomVertex::operator new(size_t)
{
  return GeomVertex_alloc.alloc();
}

void GeomVertex::operator delete(void* rp, size_t)
{
  GeomVertex_alloc.free(rp);
}

void* GeomMVertex::operator new(size_t)
{
  return GeomMVertex_alloc.alloc();
}

void GeomMVertex::operator delete(void* rp, size_t)
{
  GeomMVertex_alloc.free(rp);
}

void* GeomNVertex::operator new(size_t)
{
  return GeomNVertex_alloc.alloc();
}

void GeomNVertex::operator delete(void* rp, size_t)
{
  GeomNVertex_alloc.free(rp);
}

void* GeomNMVertex::operator new(size_t)
{
  return GeomNMVertex_alloc.alloc();
}

void GeomNMVertex::operator delete(void* rp, size_t)
{
  GeomNMVertex_alloc.free(rp);
}

void* GeomCVertex::operator new(size_t)
{
  return GeomCVertex_alloc.alloc();
}

void GeomCVertex::operator delete(void* rp, size_t)
{
  GeomCVertex_alloc.free(rp);
}




static Persistent* make_GeomVertex()
{
    return new GeomVertex(Point(0,0,0));
}

PersistentTypeID GeomVertex::type_id("GeomVertex", "Persistent", make_GeomVertex);

static Persistent* make_GeomNVertex()
{
    return new GeomNVertex(Point(0,0,0),Vector(0,0,1));
}

PersistentTypeID GeomNVertex::type_id("GeomNVertex", "GeomVertex", make_GeomNVertex);

static Persistent* make_GeomNMVertex()
{
    return new GeomNMVertex(Point(0,0,0), Vector(0,0,1), MaterialHandle(0));
}

PersistentTypeID GeomNMVertex::type_id("GeomNMVertex", "GeomNVertex", make_GeomNMVertex);

static Persistent* make_GeomMVertex()
{
    return new GeomMVertex(Point(0,0,0), MaterialHandle(0));
}

PersistentTypeID GeomMVertex::type_id("GeomMVertex", "GeomVertex", make_GeomMVertex);

static Persistent* make_GeomCVertex()
{
    return new GeomCVertex(Point(0,0,0), Color(0,0,0));
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

void GeomVertexPrim::add(double t, GeomVertex* vtx)
{
    times.add(t);
    verts.add(vtx);
}

#define GEOMVERTEXPRIM_VERSION 2

void GeomVertexPrim::io(Piostream& stream)
{
    using SCICore::Containers::Pio;

    int version=stream.begin_class("GeomVertexPrim", GEOMVERTEXPRIM_VERSION);
    GeomObj::io(stream);
    if(version >= 2)
      Pio(stream, times);
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
    return new GeomVertex(*this);
}

#define GEOMVERTEX_VERSION 1

void GeomVertex::io(Piostream& stream)
{
    using SCICore::Geometry::Pio;

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
    return new GeomNVertex(*this);
}

GeomNVertex::~GeomNVertex()
{
}

#define GEOMNVERTEX_VERSION 1

void GeomNVertex::io(Piostream& stream)
{
    using SCICore::Geometry::Pio;

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
    return new GeomNMVertex(*this);
}

GeomNMVertex::~GeomNMVertex()
{
}

#define GEOMNMVERTEX_VERSION 1

void GeomNMVertex::io(Piostream& stream)
{
    using SCICore::Containers::Pio;

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
    return new GeomMVertex(*this);
}

GeomMVertex::~GeomMVertex()
{
}

#define GEOMMVERTEX_VERSION 1

void GeomMVertex::io(Piostream& stream)
{
    using SCICore::Containers::Pio;

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
    return new GeomCVertex(*this);
}

GeomCVertex::~GeomCVertex()
{
}

void Pio(Piostream& stream, GeomVertex*& obj)
{
    Persistent* tmp=obj;
    stream.io(tmp, GeomSpace::GeomVertex::type_id);
    if(stream.reading())
	obj=(GeomSpace::GeomVertex*)tmp;
}

}
}



//
// $Log$
// Revision 1.9  1999/11/02 06:06:14  moulding
// added a #ifdef for win32 to quiet the C++ compiler.  This change
// relates to bug # 61 in csafe's bugzilla.
//
// Revision 1.8  1999/09/16 17:43:58  kuzimmer
// corrected new and delete functions for GeomMVertex
//
// Revision 1.7  1999/09/16 17:08:56  kuzimmer
// TrivialAllocator GeomMVertex_alloc(sizeof(GeomMVertex));   was missing,  added again, will prevent core dumps.
//
// Revision 1.6  1999/09/08 02:26:51  sparker
// Various #include cleanups
//
// Revision 1.5  1999/08/23 07:06:33  sparker
// Fix IRIX build
//
// Revision 1.4  1999/08/17 23:50:30  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.3  1999/08/17 06:39:17  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:47  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:55  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:20  dav
// Import sources
//
//
