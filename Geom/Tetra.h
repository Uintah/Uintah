
/*
 *  Tetra.h:  A tetrahedra object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Tetra_h
#define SCI_Geom_Tetra_h 1

#include <Geom/Geom.h>
#include <Geometry/Point.h>
#include <Classlib/Boolean.h>

class GeomTetra : public GeomObj {
public:
    Point p1;
    Point p2;
    Point p3;
    Point p4;

    GeomTetra(const Point&, const Point&, const Point&, const Point&);
    GeomTetra(const GeomTetra&);
    virtual ~GeomTetra();

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

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

#endif /* SCI_Geom_Tetra_h */
