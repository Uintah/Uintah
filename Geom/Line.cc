
/*
 *  Line.cc:  Line object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Line.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Classlib/TrivialAllocator.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>
#include <Malloc/Allocator.h>

Persistent* make_GeomLine()
{
    return new GeomLine(Point(0,0,0), Point(1,1,1));
}

PersistentTypeID GeomLine::type_id("GeomLine", "GeomObj", make_GeomLine);

static TrivialAllocator Line_alloc(sizeof(GeomLine));

void* GeomLine::operator new(size_t)
{
    return Line_alloc.alloc();
}

void GeomLine::operator delete(void* rp, size_t)
{	
    Line_alloc.free(rp);
}

GeomLine::GeomLine(const Point& p1, const Point& p2)
: GeomObj(), p1(p1), p2(p2)
{
}

GeomLine::GeomLine(const GeomLine& copy)
: GeomObj(), p1(copy.p1), p2(copy.p2)
{
}

GeomLine::~GeomLine()
{
}

GeomObj* GeomLine::clone()
{    return new GeomLine(*this);
}

void GeomLine::get_bounds(BBox& bb)
{
    bb.extend(p1);
    bb.extend(p2);
}

void GeomLine::get_bounds(BSphere& bs)
{
    Point cen(Interpolate(p1, p2, 0.5));
    double rad=(p2-p1).length()/2.;
    bs.extend(cen, rad);
}

void GeomLine::make_prims(Array1<GeomObj*>&,
			  Array1<GeomObj*>& dontfree)
{
    GeomLine* line=this;
    dontfree.add(line);
}

void GeomLine::preprocess()
{
    NOT_FINISHED("GeomLine::preprocess");
}

void GeomLine::intersect(const Ray&, Material*,
			 Hit&)
{
    NOT_FINISHED("GeomLine::intersect");
}

#define GEOMLINE_VERSION 1

void GeomLine::io(Piostream& stream)
{
    stream.begin_class("GeomLine", GEOMLINE_VERSION);
    GeomObj::io(stream);
    Pio(stream, p1);
    Pio(stream, p2);
    stream.end_class();
}

bool GeomLine::saveobj(ostream&, const clString& format, GeomSave*)
{
    NOT_FINISHED("GeomLine::saveobj");
    return false;
}

Persistent* make_GeomLines()
{
    return new GeomLines();
}

PersistentTypeID GeomLines::type_id("GeomLines", "GeomObj", make_GeomLines);

GeomLines::GeomLines()
{
}

GeomLines::GeomLines(const GeomLines& copy)
: pts(copy.pts)
{
}

GeomLines::~GeomLines()
{
}

GeomObj* GeomLines::clone()
{
  return new GeomLines(*this);
}

void GeomLines::make_prims(Array1<GeomObj*>&,
			    Array1<GeomObj*>&)
{
    // Nothing to do...
}

void GeomLines::get_bounds(BBox& bb)
{
  for(int i=0;i<pts.size();i++)
    bb.extend(pts[i]);
}

void GeomLines::get_bounds(BSphere& bs)
{
  for(int i=0;i<pts.size();i++)
    bs.extend(pts[i]);
}

void GeomLines::preprocess()
{
    NOT_FINISHED("GeomLines::preprocess");
}

void GeomLines::intersect(const Ray&, Material*,
			 Hit&)
{
    NOT_FINISHED("GeomLines::intersect");
}

#define GEOMLINES_VERSION 1

void GeomLines::io(Piostream& stream)
{
    stream.begin_class("GeomLines", GEOMLINES_VERSION);
    GeomObj::io(stream);
    Pio(stream, pts);
    stream.end_class();
}

bool GeomLines::saveobj(ostream&, const clString& format, GeomSave*)
{
    NOT_FINISHED("GeomLines::saveobj");
    return false;
}

void GeomLines::add(const Point& p1, const Point& p2)
{
  pts.add(p1);
  pts.add(p2);
}
