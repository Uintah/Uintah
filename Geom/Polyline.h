
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

#include <Geom/VertexPrim.h>

class GeomPolyline : public GeomVertexPrim {
public:
    GeomPolyline();
    GeomPolyline(const GeomPolyline&);
    virtual ~GeomPolyline();

    virtual GeomObj* clone();

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void preprocess();
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);
};

#endif /* SCI_Geom_Polyline_h */
