
/*
 *  Tetra.cc:  A tetrahedra object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Tetra.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Geom/Line.h>
#include <Geometry/BBox.h>
#include <Malloc/Allocator.h>

Persistent* make_GeomTetra()
{
    return new GeomTetra(Point(0,0,0), Point(0,0,1), Point(0,1,0), Point(1,0,0));
}

PersistentTypeID GeomTetra::type_id("GeomTetra", "GeomObj", make_GeomTetra);

GeomTetra::GeomTetra(const Point& p1, const Point& p2,
		     const Point& p3, const Point& p4)
: GeomObj(), p1(p1), p2(p2), p3(p3), p4(p4)
{
}

GeomTetra::GeomTetra(const GeomTetra& copy)
: GeomObj(copy), p1(copy.p1), p2(copy.p2), p3(copy.p3), p4(copy.p4)
{
}

GeomTetra::~GeomTetra()
{
}

GeomObj* GeomTetra::clone()
{
    return scinew GeomTetra(*this);
}

void GeomTetra::get_bounds(BBox& bb)
{
    bb.extend(p1);
    bb.extend(p2);
    bb.extend(p3);
    bb.extend(p4);
}

void GeomTetra::get_bounds(BSphere&)
{
    NOT_FINISHED("GeomTetra::get_bounds");
}

void GeomTetra::make_prims(Array1<GeomObj*>& free,
			   Array1<GeomObj*>&)
{
    GeomLine* l1=new GeomLine(p1, p2);
//    l1->set_matl(matl);
    free.add(l1);
    GeomLine* l2=new GeomLine(p2, p3);
//    l2->set_matl(matl);
    free.add(l2);
    GeomLine* l3=new GeomLine(p3, p4);
//    l3->set_matl(matl);
    free.add(l3);
    GeomLine* l4=new GeomLine(p1, p4);
//    l4->set_matl(matl);
    free.add(l4);
}

void GeomTetra::preprocess()
{
    NOT_FINISHED("GeomTetra::preprocess");
}

void GeomTetra::intersect(const Ray&, Material*,
			  Hit&)
{
    NOT_FINISHED("GeomTetra::intersect");
}

#define GEOMTETRA_VERSION 1

void GeomTetra::io(Piostream& stream)
{
    stream.begin_class("GeomTetra", GEOMTETRA_VERSION);
    GeomObj::io(stream);
    Pio(stream, p1);
    Pio(stream, p2);
    Pio(stream, p3);
    Pio(stream, p4);
    stream.end_class();
}
