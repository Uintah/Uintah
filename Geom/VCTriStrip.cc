
/*
 *  VCTriStrip.cc: Vertex Colored Triangle Strip object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/VCTriStrip.h>
#include <Classlib/NotFinished.h>
#include <Geom/VCTri.h>
#include <Geometry/BBox.h>

GeomVCTriStrip::GeomVCTriStrip()
: GeomObj(1)
{
}

GeomVCTriStrip::GeomVCTriStrip(const GeomVCTriStrip& copy)
: GeomObj(1), pts(copy.pts), norms(copy.norms), mmatl(copy.mmatl)
{
}

GeomVCTriStrip::~GeomVCTriStrip() {
}

void GeomVCTriStrip::make_prims(Array1<GeomObj*>& free,
			      Array1<GeomObj*>&)
{
    if(pts.size() < 3)
	return;
    int n=pts.size()-2;
    for(int i=0;i<n;i++){
	GeomVCTri* VCTri=new GeomVCTri(pts[i], pts[i+1], pts[i+2],
				       mmatl[i], mmatl[i+1], mmatl[i+2]);
	free.add(VCTri);
    }
}

GeomObj* GeomVCTriStrip::clone()
{
    return new GeomVCTriStrip(*this);
}

void GeomVCTriStrip::get_bounds(BBox& bb)
{
    for(int i=0;i<pts.size();i++)
	bb.extend(pts[i]);
}

void GeomVCTriStrip::get_bounds(BSphere&)
{
    NOT_FINISHED("GeomVCTriStrip::get_bounds");
}

void GeomVCTriStrip::add(const Point& pt, const Vector& norm,
			 const MaterialHandle& mm)
{
    pts.add(pt);
    norms.add(norm);
    mmatl.add(mm);
}

void GeomVCTriStrip::preprocess()
{
    NOT_FINISHED("GeomVCTriStrip::preprocess");
}

void GeomVCTriStrip::intersect(const Ray&, Material*,
			 Hit&)
{
    NOT_FINISHED("GeomVCTriStrip::intersect");
}
