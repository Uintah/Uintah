
/*
 *  Transform.h:  Transform Properities for Geometry
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1996
 *
 *  Copyright (C) 1996 SCI Group
 */



/****************   WARNING!!!!!!   *******************/
/*****  I didn't get a chance to verify that **********/
/*****  this actually works.  the rendering  **********/
/*****  still needs to be checked.  i didn't **********/
/*****  end up using this code, so it was    **********/
/*****  never checked.  sorry...             **********/
/****************   WARNING!!!!!!  ********************/



#ifndef SCI_Geom_Transform_h
#define SCI_Geom_Transform_h 1

#include <Classlib/Persistent.h>
#include <Classlib/LockingHandle.h>
#include <Datatypes/DenseMatrix.h>
#include <Geom/Container.h>
#include <Geometry/Transform.h>
#include <Multitask/ITC.h>

class GeomTransform : public GeomContainer {
    Transform trans;
public:
    GeomTransform(GeomObj*, const Transform);
    GeomTransform(const GeomTransform&);
    void setTransform(const Transform);
    Transform getTransform();
    virtual ~GeomTransform();
    virtual GeomObj* clone();

    // For OpenGL
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    // For all Painter's algorithm based renderers
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);

    // For Raytracing
    virtual void intersect(const Ray& ray, Material* matl,
			   Hit& hit);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

#endif
