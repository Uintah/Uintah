
/*
 *  TriStrip.cc: Triangle Strip object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/TriStrip.h>
#include <Classlib/NotFinished.h>
#include <Geom/Tri.h>
#include <Geometry/BBox.h>

GeomTriStrip::GeomTriStrip()
: GeomObj(1)
{
}

GeomTriStrip::GeomTriStrip(const GeomTriStrip& copy)
: GeomObj(1), pts(copy.pts), norms(copy.norms)
{
}

GeomTriStrip::~GeomTriStrip() {
}

void GeomTriStrip::make_prims(Array1<GeomObj*>& free,
			      Array1<GeomObj*>&)
{
    if(pts.size() < 3)
	return;
    int n=pts.size()-2;
    for(int i=0;i<n;i++){
	GeomTri* tri=new GeomTri(pts[i], pts[i+1], pts[i+2]);
//	tri->set_matl(matl);
	free.add(tri);
    }
}

GeomObj* GeomTriStrip::clone()
{
    return new GeomTriStrip(*this);
}

void GeomTriStrip::get_bounds(BBox& bb)
{
    for(int i=0;i<pts.size();i++)
	bb.extend(pts[i]);
}

void GeomTriStrip::get_bounds(BSphere&)
{
    NOT_FINISHED("GeomTriStrip::get_bounds");
}

void GeomTriStrip::add(const Point& pt, const Vector& norm)
{
    pts.add(pt);
    norms.add(norm);
}

void GeomTriStrip::preprocess()
{
    NOT_FINISHED("GeomTriStrip::preprocess");
}

void GeomTriStrip::intersect(const Ray& ray, Material* matl, Hit& hit)
{
    int n=pts.size()-2;
    for(int i=0;i<n;i++){
	GeomTri tri(pts[i], pts[i+1], pts[i+2]);
	tri.intersect(ray, matl, hit);
    }
}
