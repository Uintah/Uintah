
/*
 *  VCTri.cc: Vertex Colored Triangles...
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/VCTri.h>
#include <Classlib/NotFinished.h>
#include <Geometry/BBox.h>
#include <Math/MinMax.h>

GeomVCTri::GeomVCTri(const Point& p1, const Point& p2, const Point& p3,
		     const MaterialHandle& m1, const MaterialHandle &m2,
		     const MaterialHandle& m3)
: GeomObj(1), p1(p1), p2(p2), p3(p3), m1(m1), m2(m2), m3(m3), 
  n(Cross(p3-p1, p2-p1))
{
    if(n.length2() > 0)
	n.normalize();
}

GeomVCTri::GeomVCTri(const GeomVCTri &copy)
: GeomObj(1), p1(copy.p1), p2(copy.p2), p3(copy.p3), m1(copy.m1), m2(copy.m2),
  m3(copy.m3), n(copy.n)
{
}

GeomVCTri::~GeomVCTri()
{
}

GeomObj* GeomVCTri::clone()
{
    return new GeomVCTri(*this);
}

void GeomVCTri::get_bounds(BBox& bb)
{
    bb.extend(p1);
    bb.extend(p2);
    bb.extend(p3);
}

void GeomVCTri::get_bounds(BSphere&)
{
    NOT_FINISHED("GeomVCTri::get_bounds");
}

void GeomVCTri::make_prims(Array1<GeomObj*>&,
			  Array1<GeomObj*>& dontfree)
{
    GeomVCTri* VCTri=this;
    dontfree.add(VCTri);
}

void GeomVCTri::preprocess()
{
    NOT_FINISHED("GeomVCTri::preprocess");
}

void GeomVCTri::intersect(const Ray&, Material*, Hit&)
{
    NOT_FINISHED("GeomVCTri::intersect");
}
