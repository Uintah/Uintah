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
#include <Comm/MessageBase.h>
#include <Geom/Color.h>
#include <Geom/View.h>
#include <Geometry/BBox.h>
#include <TCL/TCL.h>
#include <TCL/TCLvar.h>

class DBContext;
class GeomObj;
class GeomPick;
class Light;
class Renderer;
class Salmon;
class SceneItem;
class Vector;

struct ObjTag {
    TCLvarint* visible;
    int tagid;
};

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

class Roe;
typedef void (Roe::*MouseHandler)(int, int x, int y);

class Roe : public TCL {
    Salmon* manager;
    HashTable<clString, Renderer*> renderers;
    Renderer* current_renderer;

    void do_mouse(MouseHandler, TCLArgs&);

    BBox bb;

    int need_redraw;
    int last_x, last_y;
    double total_x, total_y, total_z;
    Point rot_point;
    int rot_point_valid;
    View rot_view;
    double total_scale;
    GeomPick* pick_pick;
    GeomObj* pick_obj;

    void update_mode_string(const char*);
    char* modebuf;
    char* modecommand;

    int maxtag;

    void animate_to_view(const View& v, double time);
public:
    Renderer* get_renderer(const clString&);
    clString id;

    Roe(Salmon *s, const clString& id);
    Roe(const Roe&);
    ~Roe();

    void itemAdded(SceneItem*);
    void itemDeleted(SceneItem*);
    void rotate(double angle, Vector v, Point p);
    void rotate_obj(double angle, const Vector& v, const Point& p);
    void translate(Vector v);
    void scale(Vector v, Point p);
    void addChild(Roe *r);
    void deleteChild(Roe *r);
    void SetParent(Roe *r);
    void SetTop();
    void redrawAll();
    void redraw_if_needed();

    void mouse_translate(int, int, int);
    void mouse_scale(int, int, int);
    void mouse_rotate(int, int, int);
    void mouse_pick(int, int, int);

    void tcl_command(TCLArgs&, void*);
    void redraw();
    void get_bounds(BBox&);

    // Which of the objects do we draw?
    HashTable<clString, ObjTag*> visible;

    // Which of the lights are on?
    HashTable<clString, int> light_on;

    // The Camera
    TCLView view;
    View homeview;

    // Background Color
    TCLColor bgcolor;

    // Shading parameters, etc.
    TCLstring shading;
};

class RoeMouseMessage : public MessageBase {
public:
    clString rid;
    MouseHandler handler;
    int action;
    int x, y;
    RoeMouseMessage(const clString& rid, MouseHandler handler,
		    int action, int x, int y);
    virtual ~RoeMouseMessage();
};

#endif
