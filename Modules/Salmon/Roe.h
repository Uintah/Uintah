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
#include <Devices/Tracker.h>
#include <Geom/Color.h>
#include <Geom/TCLGeom.h>
#include <Geom/View.h>
#include <Geometry/BBox.h>
#include <Geometry/Transform.h>
#include <TCL/TCL.h>
#include <TCL/TCLvar.h>
#include <Modules/Salmon/BallAux.h>

class DBContext;
class GeomObj;
class GeomPick;
class GeomSphere;
class Light;
class Renderer;
class Salmon;
//class SceneItem;

class GeomSalmonItem;
class Vector;
struct DrawInfoOpenGL;
class BallData;
struct GeometryData;
template<class T> class AsyncReply;


struct ObjTag {
    TCLvarint* visible;
    int tagid;
};

class TCLView : public TCLvar {
    TCLPoint eyep;
    TCLPoint lookat;
    TCLVector up;
    TCLdouble fov;
    TCLVector eyep_offset;
public:
    TCLView(const clString& name, const clString& id, TCL* tcl);
    ~TCLView();
    TCLView(const TCLView&);

    View get();
    void set(const View&);
    virtual void emit(ostream& out);
};

class Roe;
typedef void (Roe::*MouseHandler)(int, int x, int y, int state, int btn, int time);
typedef void (Renderer::*RoeVisPMF)(Salmon*, Roe*, GeomObj*);

class Roe : public TCL {
protected:
    friend class Salmon;
    Salmon* manager;
    HashTable<clString, Renderer*> renderers;

    void do_mouse(MouseHandler, TCLArgs&);

    BBox bb;

    int last_x, last_y;
    double total_x, total_y, total_z;
    Point rot_point;
    int rot_point_valid;
    GeomPick* pick_pick;
    GeomObj* pick_obj;

    void update_mode_string(const char*);
    char* modebuf;
    char* modecommand;

    int maxtag;

    Tracker* tracker;
    TCLint tracker_state;
    void head_moved(const TrackerPosition&);
    void flyingmouse_moved(const TrackerPosition&);
    TrackerPosition old_mouse_pos;
    TrackerPosition old_head_pos;
    Point orig_eye;
    Vector frame_up;
    Vector frame_right;
    Vector frame_front;
    int have_trackerdata;
    Point mousep;
    GeomSphere* mouse_obj;
    Array1<GeomObj*> roe_objs;

    void animate_to_view(const View& v, double time);
    void redraw();
    void redraw(double tbeg, double tend, int nframes, double framerate);

    int last_time;
public:
    int inertia_mode;
    BallData *ball;  // this is the ball for arc ball stuff

    double angular_v; // angular velocity for inertia
    View rot_view;    // pre-rotation view
    Transform prev_trans;
    double eye_dist;
    double total_scale;
    int prev_time[3]; // history for quaternions and time
    HVect prev_quat[3];


    Renderer* current_renderer;
    Renderer* get_renderer(const clString&);
    clString id;
    int need_redraw;

    Roe(Salmon *s, const clString& id);
    Roe(const Roe&);
    ~Roe();

    void itemAdded(GeomSalmonItem*);
    void itemDeleted(GeomSalmonItem*);
    void rotate(double angle, Vector v, Point p);
    void rotate_obj(double angle, const Vector& v, const Point& p);
    void translate(Vector v);
    void scale(Vector v, Point p);
    void addChild(Roe *r);
    void deleteChild(Roe *r);
    void SetParent(Roe *r);
    void SetTop();
    void redraw_if_needed();
    void force_redraw();

    void mouse_translate(int, int, int, int, int, int);
    void mouse_scale(int, int, int, int, int, int);
    void mouse_rotate(int, int, int, int, int, int);
    void mouse_pick(int, int, int, int, int, int);

    void tcl_command(TCLArgs&, void*);
    void get_bounds(BBox&);

    void autoview(const BBox&);

    // sets up the state (OGL) for a tool/roe
    void setState(DrawInfoOpenGL*,clString);
    // sets up DI for this drawinfo
    void setDI(DrawInfoOpenGL*,clString);
    // sets up OGL clipping planes...
    void setClip(DrawInfoOpenGL*); 

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

    // Stereo
    TCLint do_stereo;

    TCLint drawimg;

    TCLstring saveprefix;

    // Object processing utility routines
    void do_for_visible(Renderer*, RoeVisPMF);

    void set_current_time(double time);

    void dump_objects(const clString&, const clString& format);

    void getData(int datamask, AsyncReply<GeometryData*>* result);
};

class RoeMouseMessage : public MessageBase {
public:
    clString rid;
    MouseHandler handler;
    int action;
    int x, y;
    int state;
    int btn;
    int time;
    
    
    RoeMouseMessage(const clString& rid, MouseHandler handler,
		    int action, int x, int y, int state, int btn,
		    int time);
    virtual ~RoeMouseMessage();
};

#endif
