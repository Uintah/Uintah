
/*
 *  Line.h:  Line object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Line_h
#define SCI_Geom_Line_h 1

#include <Geom/Geom.h>
#include <Geometry/Point.h>

class GeomLine : public GeomObj {
public:
    Point p1, p2;

    GeomLine(const Point& p1, const Point& p2);
    GeomLine(const GeomLine&);
    virtual ~GeomLine();

    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);

    virtual void objdraw(DrawInfoOpenGL*);
    virtual void objdraw(DrawInfoX11*);
    virtual double depth(DrawInfoX11*);
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
};

#endif /* SCI_Geom_Line_h */
