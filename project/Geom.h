
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
};

class ObjGroup : public GeomObj {
    Array1<GeomObj*> objs;
public:
    ObjGroup();
    virtual ~ObjGroup();

    void add(GeomObj*);
    int size();
};

class Triangle : public GeomObj {
public:
    Point p1;
    Point p2;
    Point p3;

    Triangle(const Point&, const Point&, const Point&);
    virtual ~Triangle();
};

#endif
