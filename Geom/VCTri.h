
/*
 *  VCTri.h: Triangles...
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Novemeber 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_VCTri_h
#define SCI_Geom_VCTri_h 1

#include <Geom/Geom.h>
#include <Geom/Material.h>
#include <Geometry/Point.h>

class GeomVCTri : public GeomObj {
public:
    Point p1;
    Point p2;
    Point p3;
    MaterialHandle m1;
    MaterialHandle m2;
    MaterialHandle m3;
    Vector n;

    GeomVCTri(const Point&, const Point&, const Point&, const MaterialHandle&,
	      const MaterialHandle&, const MaterialHandle&);
    GeomVCTri(const GeomVCTri&);
    virtual ~GeomVCTri();

    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
    virtual void get_bounds(BSphere&);

#ifdef SCI_OPENGL
    virtual void objdraw(DrawInfoOpenGL*, Material*);
#endif
    virtual void objdraw(DrawInfoX11*, Material*);
    virtual double depth(DrawInfoX11*);
    virtual void get_hit(Vector&, Point&);
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void preprocess();
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);
};

#endif /* SCI_Geom_VCTri_h */
