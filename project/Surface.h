
/*
 *  Surface.h: The Surface Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Surface_h
#define SCI_project_Surface_h 1

#include <Datatype.h>
#include <Classlib/Array1.h>
#include <Classlib/LockingHandle.h>
#include <Geom.h>
#include <Geometry/Point.h>

class Surface;
typedef LockingHandle<Surface> SurfaceHandle;

class Surface : public Datatype {
public:
    Surface();
    Surface(const Surface& copy);
    virtual ~Surface();
    virtual int inside(const Point& p)=0;
    virtual ObjGroup* getGeomFromSurface()=0;
    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID typeid;
};

struct TSElement {
    int i1; 
    int i2; 
    int i3;
    inline TSElement(int i1, int i2, int i3):i1(i1), i2(i2), i3(i3){}
};

void Pio(Piostream&, TSElement*&);

class TriSurface : public Surface {
    Array1<Point> points;
    Array1<TSElement*> elements;
public:
    TriSurface();
    TriSurface(const TriSurface& copy);
    virtual ~TriSurface();
    virtual int inside(const Point& p);
    virtual ObjGroup* getGeomFromSurface();
    void add_point(const Point& p);
    void add_triangle(int i1, int i2, int i3);
    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID typeid;
};
#endif /* SCI_project_Surface_h */
