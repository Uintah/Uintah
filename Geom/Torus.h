
/*
 * Torus.h: Torus objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   January 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_Torus_h
#define SCI_Geom_Torus_h 1

#include <Geom/Geom.h>
#include <Geometry/Point.h>

class GeomTorus : public GeomObj {
public:
    Point cen;
    Vector axis;
    double rad1;
    double rad2;
    int nu;
    int nv;

    Vector zrotaxis;
    double zrotangle;

    virtual void adjust();
    void move(const Point&, const Vector&, double, double,
	      int nu=50, int nv=8);

    GeomTorus(int nu=50, int nv=8);
    GeomTorus(const Point&, const Vector&, double, double,
	      int nu=50, int nv=8);
    GeomTorus(const GeomTorus&);
    virtual ~GeomTorus();

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
};

class GeomTorusArc : public GeomTorus {
public:
    Vector zero;
    double start_angle;
    double arc_angle;
    Vector xrotaxis;
    double xrotangle;

    virtual void adjust();
    void move(const Point&, const Vector&, double, double,
	      const Vector& zero, double start_angle, double arc_angle,
	      int nu=50, int nv=8);
    GeomTorusArc(int nu=50, int nv=8);
    GeomTorusArc(const Point&, const Vector&, double, double, 
		 const Vector& zero, double start_angle, double arc_angle,
		 int nu=50, int nv=8);
    GeomTorusArc(const GeomTorusArc&);
    virtual ~GeomTorusArc();

    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
    virtual void get_bounds(BSphere&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
};


#endif /* SCI_Geom_Torus_h */
