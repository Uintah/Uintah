
/*
 *  Switch.h:  Turn Geometry on and off
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   January 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_Switch_h
#define SCI_Geom_Switch_h 1

#include <Geom/Container.h>

class GeomSwitch : public GeomContainer {
    int state;
    GeomSwitch(const GeomSwitch&);
public:
    GeomSwitch(GeomObj*, int state=1);
    virtual ~GeomSwitch();
    virtual GeomObj* clone();
    void set_state(int st);
    int get_state();
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

class GeomTimeSwitch : public GeomContainer {
    double tbeg;
    double tend;
    GeomTimeSwitch(const GeomTimeSwitch&);
public:
    GeomTimeSwitch(GeomObj*, double tbeg, double tend);
    virtual ~GeomTimeSwitch();
    virtual GeomObj* clone();

    // For OpenGL
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif /* SCI_Geom_Switch_h */
