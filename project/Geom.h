
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
#include <Classlib/Stack.h>
#include <Geometry/Point.h>
#include <Geometry/BBox.h>
#include <Color.h>

class BBox;
class DrawInfo;

class MaterialProp {
public:
    Color ambient;
    Color diffuse;
    Color specular;
    double shininess;
    Color emission;
    MaterialProp(const Color&, const Color&, const Color&, double);
    void set(DrawInfo*);
};

struct DrawInfo {
    int polycount;
    MaterialProp* current_matl;
    Stack<MaterialProp*> stack;
    enum DrawType {
	WireFrame,
	Flat,
	Gouraud,
	Phong,
    };
    int pick_mode;
    int edit_mode;
    DrawType drawtype;
    DrawInfo();

    void push_matl(MaterialProp*);
    void pop_matl();
};

class GeomPick {
    Array1<Vector> directions;
    MaterialProp* hightlight;
public:
    GeomPick(const Vector&);
    GeomPick(const Vector&, const Vector&);
    GeomPick(const Vector&, const Vector&, const Vector&);
    GeomPick(const Array1<Vector>&);
    ~GeomPick();
    GeomPick(const GeomPick&);
    int nprincipal();
    Vector principal(int i);
    void set_highlight(MaterialProp* matl);
};

class GeomObj {
protected:
    MaterialProp* matl;
public:
    GeomPick* pick;
    GeomObj();
    GeomObj(const GeomObj&);
    virtual ~GeomObj();
    virtual void draw(DrawInfo*) = 0;
    virtual void get_bounds(BBox&) = 0;
    virtual GeomObj* clone() = 0;
    void set_matl(MaterialProp*);
    void set_pick(GeomPick*);
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
    ObjGroup(const ObjGroup&);
    virtual ~ObjGroup();

    void add(GeomObj*);
    int size();
    virtual void draw(DrawInfo*);
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
};

class Triangle : public GeomObj {
public:
    Point p1;
    Point p2;
    Point p3;
    Vector n;

    Triangle(const Point&, const Point&, const Point&);
    Triangle(const Triangle&);
    virtual ~Triangle();
    virtual void draw(DrawInfo*);
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
};

class Tetra : public GeomObj {
public:
    Point p1;
    Point p2;
    Point p3;
    Point p4;

    Tetra(const Point&, const Point&, const Point&, const Point&);
    Tetra(const Tetra&);
    virtual ~Tetra();
    virtual void draw(DrawInfo*);
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
};

class GeomSphere : public GeomObj {
public:
    Point cen;
    double rad;
    int nu;
    int nv;

    GeomSphere(const Point&, double, int nu=20, int nv=10);
    GeomSphere(const GeomSphere&);
    virtual ~GeomSphere();
    virtual void draw(DrawInfo*);
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
};

class GeomCylinder : public GeomObj {
    Vector v1;
    Vector v2;
public:
    Point bottom;
    Point top;
    Vector axis;
    double rad;
    int nu;
    int nv;

    GeomCylinder(const Point&, const Point&, double, int nu=20, int nv=4);
    GeomCylinder(const GeomCylinder&);
    virtual ~GeomCylinder();
    virtual void draw(DrawInfo*);
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
};

class GeomCone : public GeomObj {
    Vector v1;
    Vector v2;
    double tilt;
public:
    Point bottom;
    Point top;
    Vector axis;
    double bot_rad;
    double top_rad;
    int nu;
    int nv;

    GeomCone(const Point&, const Point&, double, double, int nu=20, int nv=4);
    GeomCone(const GeomCone&);
    virtual ~GeomCone();
    virtual void draw(DrawInfo*);
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
};

class GeomDisc : public GeomObj {
    Vector v1;
    Vector v2;
public:
    Point cen;
    Vector normal;
    double rad;
    int nu;
    int nv;

    GeomDisc(const Point&, const Vector&, double, int nu=20, int nv=2);
    GeomDisc(const GeomDisc&);
    virtual ~GeomDisc();
    virtual void draw(DrawInfo*);
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
};

class GeomPt : public GeomObj {
public:
    Point p1;

    GeomPt(const Point&);
    GeomPt(const GeomPt&);
    virtual ~GeomPt();
    virtual void draw(DrawInfo*);
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
};

#endif
