
/*
 *  Geom.h: Displayable Geometry
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Geom_h
#define SCI_project_Geom_h 1

#include <Classlib/Array1.h>
#include <Geometry/Point.h>

class MaterialProp;

class GeomObj {
    MaterialProp* matl;
public:
    GeomObj();
    virtual ~GeomObj();
    virtual void draw() = 0;
};

class ObjGroup : public GeomObj {
    Array1<GeomObj*> objs;
public:
    ObjGroup();
    virtual ~ObjGroup();

    void add(GeomObj*);
    int size();
    virtual void draw();
};

class Triangle : public GeomObj {
public:
    Point p1;
    Point p2;
    Point p3;

    Triangle(const Point&, const Point&, const Point&);
    virtual ~Triangle();
    virtual void draw();
};

class Tetra : public GeomObj {
public:
    Point p1;
    Point p2;
    Point p3;
    Point p4;

    Tetra(const Point&, const Point&, const Point&, const Point&);
    virtual ~Tetra();
    virtual void draw();
};

class GeomPt : public GeomObj {
public:
    Point p1;

    GeomPt(const Point&);
    virtual ~GeomPt();
    virtual void draw();
};
#endif
