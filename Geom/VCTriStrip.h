
/*
 *  VCTriStrip.h: VCTriangle Strip object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_VCTriStrip_h
#define SCI_Geom_VCTriStrip_h 1

#include <Geom/Geom.h>
#include <Geom/Material.h>
#include <Classlib/Array1.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>

class GeomVCTriStrip : public GeomObj {
    Array1<Point> pts;
    Array1<Vector> norms;
    Array1<MaterialHandle> mmatl;
public:
    void add(const Point&, const Vector&, const MaterialHandle&);
    GeomVCTriStrip();
    GeomVCTriStrip(const GeomVCTriStrip&);
    virtual ~GeomVCTriStrip();

    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void objdraw(DrawInfoOpenGL*, Material*);
#endif
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);
};

#endif /* SCI_Geom_VCTriStrip_h */
