
/*
 *  Disc.h:  Disc object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Disc_h
#define SCI_Geom_Disc_h 1

#include <Geom/Geom.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>

class GeomDisc : public GeomObj {
    Vector v1;
    Vector v2;
    Vector zrotaxis;
    double zrotangle;
public:
    Point cen;
    Vector normal;
    double rad;
    int nu;
    int nv;

    void adjust();
    void move(const Point&, const Vector&, double, int nu=20, int nv=2);

    GeomDisc(int nu=20, int nv=2);
    GeomDisc(const Point&, const Vector&, double, int nu=20, int nv=2);
    GeomDisc(const GeomDisc&);
    virtual ~GeomDisc();

    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void objdraw(DrawInfoOpenGL*);
#endif
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void intersect(const Ray& ray, const MaterialHandle& matl,
			   Hit& hit);
};

#endif /* SCI_Geom_Disc_h */
