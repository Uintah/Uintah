
/*
 *  BasicSurfaces.h: Cylinders and stuff
 *
 *  Written by:
 *   Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Datatypes_BasicSurfaces_h
#define SCI_Datatypes_BasicSurfaces_h 1

#include <Datatypes/Surface.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>

class CylinderSurface : public Surface {
    Point p1;
    Point p2;
    double radius;
    int nu;
    int nv;
    int ndiscu;

    Vector u;
    Vector v;

    Vector axis;
    double rad2;
    double height;
public:
    CylinderSurface(const Point& p1, const Point& p2, double radius,
		    int nu, int nv, int ndiscu);
    CylinderSurface(const CylinderSurface&);
    virtual ~CylinderSurface();
    virtual Surface* clone();
    virtual int inside(const Point& p);
    virtual void get_surfpoints(Array1<Point>&);
    virtual void construct_grid(int, int, int, const Point &, double);
    virtual void construct_grid();

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class PointSurface : public Surface {
    Point pos;
public:
    PointSurface(const Point& pos);
    PointSurface(const PointSurface&);
    virtual ~PointSurface();
    virtual Surface* clone();
    virtual int inside(const Point& p);
    virtual void get_surfpoints(Array1<Point>&);
    virtual void construct_grid(int, int, int, const Point &, double);
    virtual void construct_grid();

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif /* SCI_Datatypes_BasicSurfaces_h */
