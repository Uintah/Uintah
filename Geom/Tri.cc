
/*
 *  Tri.cc: Triangles...
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Tri.h>
#include <Classlib/NotFinished.h>
#include <Geometry/BBox.h>
#include <Math/MinMax.h>

GeomTri::GeomTri(const Point& p1, const Point& p2, const Point& p3)
: GeomObj(1), p1(p1), p2(p2), p3(p3), n(Cross(p3-p1, p2-p1))
{
    if(n.length2() > 0)
	n.normalize();
}

GeomTri::GeomTri(const GeomTri &copy)
: GeomObj(1), p1(copy.p1), p2(copy.p2), p3(copy.p3), n(copy.n)
{
}

GeomTri::~GeomTri()
{
}

GeomObj* GeomTri::clone()
{
    return new GeomTri(*this);
}

void GeomTri::get_bounds(BBox& bb)
{
    bb.extend(p1);
    bb.extend(p2);
    bb.extend(p3);
}

void GeomTri::make_prims(Array1<GeomObj*>&,
			  Array1<GeomObj*>& dontfree)
{
    GeomTri* tri=this;
    dontfree.add(tri);
}

void GeomTri::intersect(const Ray&, Material*, Hit&)
{
    NOT_FINISHED("GeomTri::intersect");
}
