
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
#include <variant.h>
#include <Geom/Material.h>

class BBox;
class DrawInfoOpenGL;
class DrawInfoX11;
class GeomPick;
class Hit;
class Vector;
class Point;
class Ray;

class GeomObj {
protected:
    MaterialHandle matl;
    GeomPick* pick;
    int lit;
public:
    GeomObj(int lit);
    GeomObj(const GeomObj&);
    virtual ~GeomObj();
    virtual GeomObj* clone() = 0;


    virtual void reset_bbox();
    virtual void get_bounds(BBox&) = 0;
    void set_matl(const MaterialHandle&);
    void set_pick(GeomPick*);
    GeomPick* get_pick();


    // For OpenGL
#ifdef SCI_OPENGL
    void draw(DrawInfoOpenGL*);
    virtual void objdraw(DrawInfoOpenGL*)=0;
#endif

    // For X11
    void draw(DrawInfoX11*);
    virtual void objdraw(DrawInfoX11*);
    virtual double depth(DrawInfoX11*);
    virtual void get_hit(Vector&, Point&);

    // For all Painter's algorithm based renderers
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree) = 0;

    // For Raytracing
    virtual void intersect(const Ray& ray, const MaterialHandle& matl,
			   Hit& hit)=0;
    virtual Vector normal(const Point& p);
};

#endif
