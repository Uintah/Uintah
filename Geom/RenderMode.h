
/*
 * RenderMode.h:  Object to switch rendering mode
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_RenderMode_h
#define SCI_Geom_RenderMode_h 1

#include <Geom/Geom.h>
#include <Geometry/Point.h>

class GeomRenderMode : public GeomObj {
public:
    enum DrawType {
        WireFrame,
        Flat,
        Gouraud,
        Phong,
    };
private:
    DrawType drawtype;
    GeomObj* child;
public:
    GeomRenderMode(DrawType, GeomObj* child);
    GeomRenderMode(const GeomRenderMode&);
    virtual ~GeomRenderMode();

    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void objdraw(DrawInfoOpenGL*);
#endif
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
};

#endif /* SCI_Geom_RenderMode_h */
