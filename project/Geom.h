
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
#include <MessageBase.h>
#include <Multitask/ITC.h>

class BBox;
class DrawInfo;
class Module;

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
    int lighting;
    int currently_lit;
    DrawType drawtype;
    DrawInfo();

    void push_matl(MaterialProp*);
    void pop_matl();
};

class GeomPick {
    Array1<Vector> directions;
    MaterialProp* hightlight;
    Mailbox<MessageBase*>* mailbox;
    void* cbdata;
    Module* module;
    GeomPick(const GeomPick&);
public:
    GeomPick(Module*);
    GeomPick(Module* module, const Vector&);
    GeomPick(Module* module, const Vector&, const Vector&);
    GeomPick(Module* module, const Vector&, const Vector&, const Vector&);
    GeomPick(Module* module, const Array1<Vector>&);
    ~GeomPick();
    int nprincipal();
    Vector principal(int i);
    void set_principal(const Vector&);
    void set_principal(const Vector&, const Vector&);
    void set_principal(const Vector&, const Vector&, const Vector&);
    void set_highlight(MaterialProp* matl);
    void set_reply(Mailbox<MessageBase*>*);
    void set_cbdata(void*);

    void pick();
    void moved(int axis, double distance,
	       const Vector& delta);
    void release();
};

class GeomPickMessage : public MessageBase {
public:
    Module* module;
    int axis;
    double distance;
    Vector delta;
    void* cbdata;
    GeomPickMessage(Module*, void*);
    GeomPickMessage(Module*, void*, int);
    GeomPickMessage(Module*, int, double, const Vector&, void*);
    virtual ~GeomPickMessage();
};

class GeomObj {
protected:
    MaterialProp* matl;
    int lit;
public:
    GeomPick* pick;
    GeomObj(int lit);
    GeomObj(const GeomObj&);
    void draw(DrawInfo*);
    virtual ~GeomObj();
    virtual void objdraw(DrawInfo*) = 0;
    virtual void get_bounds(BBox&) = 0;
    virtual GeomObj* clone() = 0;
    void set_matl(MaterialProp*);
    void set_pick(GeomPick*);
};

typedef GeomObj* GeomObjPointer;

inline int Hash(const GeomObjPointer& k, int hash_size)
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
    virtual void objdraw(DrawInfo*);
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
};

class ObjTransform : public GeomObj {
    GeomObj* obj;
    double trans[16];
    Point center;
public:
    ObjTransform(const ObjTransform&);
    ObjTransform(GeomObj *g);
    virtual ~ObjTransform();
    void rotate(double angle, Vector axis);
    void scale(double scl);
    void translate(Vector mtn);
    virtual void objdraw(DrawInfo*);
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
    virtual void objdraw(DrawInfo*);
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
    virtual void objdraw(DrawInfo*);
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
};

class GeomSphere : public GeomObj {
public:
    Point cen;
    double rad;
    int nu;
    int nv;

    GeomSphere();
    GeomSphere(const Point&, double, int nu=20, int nv=10);
    GeomSphere(const GeomSphere&);
    virtual ~GeomSphere();
    virtual void objdraw(DrawInfo*);
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
    void adjust();
    void move(const Point&, double, int nu=20, int nv=10);
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

    GeomCylinder();
    GeomCylinder(const Point&, const Point&, double, int nu=20, int nv=4);
    GeomCylinder(const GeomCylinder&);
    virtual ~GeomCylinder();
    virtual void objdraw(DrawInfo*);
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
    void adjust();
    void move(const Point&, const Point&, double, int nu=20, int nv=4);
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

    GeomCone();
    GeomCone(const Point&, const Point&, double, double, int nu=20, int nv=4);
    GeomCone(const GeomCone&);
    virtual ~GeomCone();
    virtual void objdraw(DrawInfo*);
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
    void adjust();
    void move(const Point&, const Point&, double, double, int nu=20, int nv=4);
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

    GeomDisc();
    GeomDisc(const Point&, const Vector&, double, int nu=20, int nv=2);
    GeomDisc(const GeomDisc&);
    virtual ~GeomDisc();
    virtual void objdraw(DrawInfo*);
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
    void adjust();
    void move(const Point&, const Vector&, double, int nu=20, int nv=2);
};

class GeomPt : public GeomObj {
public:
    Point p1;

    GeomPt(const Point&);
    GeomPt(const GeomPt&);
    virtual ~GeomPt();
    virtual void objdraw(DrawInfo*);
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
};

class GeomLine : public GeomObj {
public:
    Point p1, p2;

    GeomLine(const Point& p1, const Point& p2);
    GeomLine(const GeomLine&);
    virtual ~GeomLine();
    virtual void objdraw(DrawInfo*);
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
};

class GeomPolyLine : public GeomObj {
public:
    Array1<Point> pts;

    GeomPolyLine();
    GeomPolyLine(const GeomPolyLine&);
    virtual ~GeomPolyLine();
    virtual void objdraw(DrawInfo*);
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
};

class GeomTriStrip : public GeomObj {
    Array1<Point> pts;
    Array1<Vector> norms;
public:
    void add(const Point&, const Vector&);
    GeomTriStrip();
    GeomTriStrip(const GeomTriStrip&);
    virtual ~GeomTriStrip();
    virtual void objdraw(DrawInfo*);
    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);
};

#endif
