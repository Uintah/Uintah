
/*
 *  Polyline.cc: Polyline object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Polyline.h>
#include <Classlib/NotFinished.h>
#include <Geometry/BBox.h>
#include <Geom/Line.h>

GeomPolyline::GeomPolyline()
: GeomObj(0)
{
}

GeomPolyline::GeomPolyline(const GeomPolyline& copy)
: GeomObj(0), pts(copy.pts)
{
}

GeomPolyline::~GeomPolyline() {
}

GeomObj* GeomPolyline::clone()
{
    return new GeomPolyline(*this);
}

void GeomPolyline::get_bounds(BBox& bb)
{
    for(int i=0;i<pts.size();i++)
	bb.extend(pts[i]);
}

void GeomPolyline::make_prims(Array1<GeomObj*>& free,
			      Array1<GeomObj*>&)
{
    if(pts.size() < 2)
	return;
    int n=pts.size()-1;
    for(int i=0;i<n;i++){
	GeomLine* line=new GeomLine(pts[i], pts[i+1]);
//	line->set_matl(matl);
	free.add(line);
    }
}


void GeomPolyline::intersect(const Ray&, Material*, Hit&)
{
    NOT_FINISHED("GeomPolyline::intersect");
}
