/*
 *  Roe.h: The Geometry Viewer Window
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_Roe_h
#define SCI_project_module_Roe_h

#include <Classlib/Array1.h>
#include <Geometry/BBox.h>
#include <GL/glx.h>
#include <TCL.h>

class BBox;
class DBContext;
class DrawInfo;
class GeomPick;
class Salmon;
class Vector;
class GeomObj;
class Renderer;

class GeomItem {
public:
    GeomObj* geom;
    clString name;
    int vis;
    int active;
    GeomItem();
    ~GeomItem();
};

class Roe : public TCL {
    Salmon* manager;
    HashTable<clString, Renderer*> renderers;
    Renderer* current_renderer;

    BBox bb;

    DBContext* dbcontext_st;
    
    void make_current();

    Array1<Roe *> kids;
    Roe *parent;
    int firstGen;	// am I first generation? (important for deleting)
    int roeNum;
    makeIndepRoe();
    GLXContext cx;
    int doneInit;
    int last_x;
    int last_y;
    double total_x;
    double total_y;
    double total_z;
    int haveInheritMat;
    double mtnScl;
    double inheritMat[16];

    GeomObj *geomSelected;
    GeomPick* gpick;

    int salmon_count;
    int need_redraw;
    void get_bounds(BBox&);
    Roe(const Roe&);
public:
    clString id;
    Array1<GeomItem *> geomItemA;
    DrawInfo* drawinfo;
    Roe(Salmon *s, const clString& id);
    ~Roe();
    void RoeInit(Salmon *s);
    void itemAdded(GeomObj*, const clString&);
    void itemDeleted(GeomObj*);
    void rotate(double angle, Vector v, Point p);
    void rotate_obj(double angle, const Vector& v, const Point& p);
    void translate(Vector v);
    void scale(Vector v, Point p);
    void addChild(Roe *r);
    void deleteChild(Roe *r);
    void SetParent(Roe *r);
    void SetTop();
    void redrawAll();
    void redraw_if_needed(int always);
    void printLevel(int level, int&flag);

    void mouse_translate(int, int, int, int, int);
    void mouse_scale(int, int, int, int, int);
    void mouse_rotate(int, int, int, int, int);
    void mouse_pick(int, int, int, int, int);

    void tcl_command(TCLArgs&, void*);
    void redraw();
};

#endif
