
/*
 *  Geom.h: Displayable Geometry
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Geom_h
#define SCI_Geom_Geom_h 1

#include <Classlib/Array1.h>
#include <Classlib/Handle.h>
#include <Classlib/Persistent.h>
#include <config.h>

class BBox;
class BSphere;
struct DrawInfoOpenGL;
class DrawInfoX11;
class Hit;
class Material;
class Vector;
class Point;
class Ray;
class GeomSave;
class ostream;

class GeomObj : public Persistent {
protected:
    GeomObj* parent;
public:
    GeomObj();
    GeomObj(const GeomObj&);
    virtual ~GeomObj();
    virtual GeomObj* clone() = 0;
    void set_parent(GeomObj*);

    virtual void reset_bbox();
    virtual void get_bounds(BBox&) = 0;
    virtual void get_bounds(BSphere&) = 0;

    // For OpenGL
#ifdef SCI_OPENGL
    void pre_draw(DrawInfoOpenGL*, Material*, int lit);
    virtual void draw(DrawInfoOpenGL*, Material*, double time)=0;
#endif

    // For X11
    void draw(DrawInfoX11*, Material*);
    virtual void objdraw(DrawInfoX11*, Material*);
    virtual double depth(DrawInfoX11*);
    virtual void get_hit(Vector&, Point&);

    // For all Painter's algorithm based renderers
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree) = 0;

    // For Raytracing
    virtual void preprocess()=0;
    virtual void intersect(const Ray& ray, Material* matl,
			   Hit& hit)=0;
    virtual Vector normal(const Point& p, const Hit&);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;

    virtual int saveobj(ostream&, const clString& format, GeomSave*)=0;
};

void Pio(Piostream&, GeomObj*&);

#endif
