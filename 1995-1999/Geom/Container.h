
/*
 *  Container.h: Base class for container objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   December 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Container_h
#define SCI_Geom_Container_h 1

#include <Geom/Geom.h>

class GeomContainer : public GeomObj {
protected:
    GeomObj* child;
public:
    GeomContainer(GeomObj*);
    GeomContainer(const GeomContainer&);
    virtual ~GeomContainer();
    virtual void get_bounds(BBox&);
    virtual void get_bounds(BSphere&);

    // For OpenGL
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    // For all Painter's algorithm based renderers
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);

    // For Raytracing
    virtual void preprocess();
    virtual void intersect(const Ray& ray, Material* matl,
			   Hit& hit);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};    

#endif
