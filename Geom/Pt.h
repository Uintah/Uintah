
/*
 * Sphere.h: Sphere objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Feb 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_Point_h
#define SCI_Geom_Point_h 1

#include <Geom/Geom.h>
#include <Geometry/Point.h>

class GeomPts : public GeomObj {
public:
    Array1<Point> pts;
    GeomPts(const GeomPts&);
    GeomPts(int size);
    virtual ~GeomPts();
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
    virtual void get_bounds(BSphere&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void preprocess();
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);
    virtual Vector normal(const Point& p, const Hit&);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif /* SCI_Geom_Point_h */
