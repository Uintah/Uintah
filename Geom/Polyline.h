
/*
 *  Polyline.h: Polyline object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Polyline_h
#define SCI_Geom_Polyline_h 1

#include <Geom/Geom.h>
#include <Classlib/Array1.h>
#include <Geometry/Point.h>

class GeomPolyline : public GeomObj {
public:
    Array1<Point> pts;

    GeomPolyline();
    GeomPolyline(const GeomPolyline&);
    virtual ~GeomPolyline();

    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);

    virtual void objdraw(DrawInfoOpenGL*);
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
};

#endif /* SCI_Geom_Polyline_h */
