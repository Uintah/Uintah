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
#include <Classlib/HashTable.h>
#include <Geom/Color.h>
#include <Geom/View.h>
#include <Geometry/BBox.h>
#include <TCL/TCL.h>
#include <TCL/TCLvar.h>

class DBContext;
class GeomObj;
class GeomPick;
class Light;
class Salmon;
class Vector;
class Renderer;

class TCLView : public TCLvar {
    TCLPoint eyep;
    TCLPoint lookat;
    TCLVector up;
    TCLdouble fov;
public:
    TCLView(const clString& name, const clString& id, TCL* tcl);
    ~TCLView();
    TCLView(const TCLView&);

    View get();
    void set(const View&);
};

class Roe : public TCL {
    Salmon* manager;
    HashTable<clString, Renderer*> renderers;
    Renderer* current_renderer;

    BBox bb;

    int need_redraw;
public:
    clString id;

    Roe(Salmon *s, const clString& id);
    Roe(const Roe&);
    ~Roe();

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

    void mouse_translate(int, int, int, int, int);
    void mouse_scale(int, int, int, int, int);
    void mouse_rotate(int, int, int, int, int);
    void mouse_pick(int, int, int, int, int);

    void tcl_command(TCLArgs&, void*);
    void redraw();
    void get_bounds(BBox&);

    // Which of the objects do we draw?
    HashTable<clString, int> visible;

    // Which of the lights are on?
    HashTable<clString, int> light_on;

    // The Camera
    TCLView view;
};

#endif
