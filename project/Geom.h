
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
#include <Geometry/BBox.h>
#include <Color.h>

class BBox;

class MaterialProp {
public:
    Color ambient;
    Color diffuse;
    Color specular;
    double shininess;
    Color emission;
    MaterialProp(const Color&, const Color&, const Color&, double);
    void set();
};

class GeomObj {
    MaterialProp* matl;
public:
    GeomObj();
    virtual ~GeomObj();
    virtual void draw() = 0;
    virtual BBox bbox() = 0;
    void set_matl(MaterialProp*);
};

inline int Hash(const GeomObj*& k, int hash_size)
{
   return ((int)k^(3*hash_size+1))%hash_size;
}

class ObjGroup : public GeomObj {
    Array1<GeomObj*> objs;
    BBox bb;
public:
    ObjGroup();
    virtual ~ObjGroup();

    void add(GeomObj*);
    int size();
    virtual void draw();
    BBox bbox();
};

class Triangle : public GeomObj {
    BBox bb;
public:
    Point p1;
    Point p2;
    Point p3;

    Triangle(const Point&, const Point&, const Point&);
    virtual ~Triangle();
    virtual void draw();
    BBox bbox();
};

class Tetra : public GeomObj {
    BBox bb;
public:
    Point p1;
    Point p2;
    Point p3;
    Point p4;

    Tetra(const Point&, const Point&, const Point&, const Point&);
    virtual ~Tetra();
    virtual void draw();
    BBox bbox();
};

class GeomSphere : public GeomObj {
    BBox bb;
public:
    Point cen;
    double rad;
    int nu;
    int nv;

    GeomSphere(const Point&, double, int nu=20, int nv=10);
    virtual ~GeomSphere();
    virtual void draw();
    BBox bbox();
};

class GeomPt : public GeomObj {
    BBox bb;
public:
    Point p1;

    GeomPt(const Point&);
    virtual ~GeomPt();
    virtual void draw();
    BBox bbox();
};

#endif
