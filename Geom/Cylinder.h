
/*
 *  Cylinder.h: Cylinder Object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Cylinder_h
#define SCI_Geom_Cylinder_h 1

#include <Geom/Geom.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>

class GeomCylinder : public GeomObj {
protected:
    Vector v1;
    Vector v2;

    double height;
    Vector zrotaxis;
    double zrotangle;
public:
    Point bottom;
    Point top;
    Vector axis;
    double rad;
    int nu;
    int nv;
    void adjust();
    void move(const Point&, const Point&, double, int nu=20, int nv=10);

    GeomCylinder(int nu=20, int nv=10);
    GeomCylinder(const Point&, const Point&, double, int nu=20, int nv=10);
    GeomCylinder(const GeomCylinder&);
    virtual ~GeomCylinder();

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
};

class GeomCappedCylinder : public GeomCylinder {
    int nvdisc;
public:
    GeomCappedCylinder(int nu=20, int nv=10, int nvdisc=4);
    GeomCappedCylinder(const Point&, const Point&, double, int nu=20, int nv=10, int nvdisc=4);
    GeomCappedCylinder(const GeomCappedCylinder&);
    virtual ~GeomCappedCylinder();

    virtual GeomObj* clone();
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
};
    
#endif /* SCI_Geom_Cylinder_h */
