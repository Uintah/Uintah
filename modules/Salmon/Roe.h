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
#include <GL/glx.h>
#include <Geometry/BBox.h>

class BBox;
class CallbackData;
class DBContext;
class DialogShellC;
class DrawInfo;
class DrawingAreaC;
class FormC;
class FrameC;
class GLwMDrawC;
class LabelC;
class MainWindowC;
class MenuBarC;
class MenuC;
class Salmon;
class Vector;
class RowColumnC;
class ScrolledWindowC;
class ToggleButtonC;
class PushButtonC;
class SeparatorC;
class GeomObj;

class GeomItem {
public:
    GeomObj* geom;
    ToggleButtonC* btn;
    char name[30];
    int vis;
    GeomItem();
    ~GeomItem();
};

class Roe {
    // User Interface stuff...
    DialogShellC* dialog;
    RowColumnC* wholeWin;
    RowColumnC* left;
    RowColumnC* right;
    GLwMDrawC* graphics;
    RowColumnC* controls;
    RowColumnC* objBox;
    LabelC* objLabel;
    SeparatorC *objSep;
    ScrolledWindowC* objScroll;
    RowColumnC* objRC;
    RowColumnC* shadeBox;
    LabelC* shadeLabel;
    SeparatorC *shadeSep;
    RowColumnC* shadeRC;
    ToggleButtonC* wire;
    ToggleButtonC* flat;
    ToggleButtonC* phong;
    ToggleButtonC* gouraud;
    RowColumnC* lightBox;
    LabelC* lightLabel;
    ScrolledWindowC* lightScroll;
    SeparatorC *lightSep;
    RowColumnC* lightRC;
    ToggleButtonC* ambient;
    ToggleButtonC* point1;
    RowColumnC* opt_proj;
    RowColumnC* proj_fogRC;
    RowColumnC* projRC;
    RowColumnC* options;
    RowColumnC* viewRC;
    ToggleButtonC* orthoB;
    ToggleButtonC* perspB;
    ToggleButtonC* fogB;
    PushButtonC* autoView;
    PushButtonC* setHome;
    PushButtonC* goHome;
    DrawingAreaC* buttons;
    RowColumnC* spawnRC;
    PushButtonC* spawnCh;
    PushButtonC* spawnInd;
    FormC* form;
    FrameC* gr_frame;
    MainWindowC* mainw;
    MenuBarC* menubar;
    MenuC* menu_dials;
    PushButtonC* st_button;
    Salmon* manager;
    BBox bb;

    DBContext* dbcontext_st;
    
    void eventCB(CallbackData*, void*);
    void itemCB(CallbackData*, void*);
    void wireCB(CallbackData*, void*);
    void flatCB(CallbackData*, void*);
    void gouraudCB(CallbackData*, void*);
    void phongCB(CallbackData*, void*);
    void ambientCB(CallbackData*, void*);
    void point1CB(CallbackData*, void*);
    void orthoCB(CallbackData*, void*);
    void perspCB(CallbackData*, void*);
    void fogCB(CallbackData*, void*);
    void goHomeCB(CallbackData*, void*);
    void autoViewCB(CallbackData*, void*);
    void setHomeCB(CallbackData*, void*);
    void spawnChCB(CallbackData*, void*);
    void destroyWidgetCB(CallbackData*, void*); 
    void redrawCB(CallbackData*, void*);
    void initCB(CallbackData*, void*);
    void attach_dials(CallbackData*, void*);
    void DBscale(DBContext*, int, double, double, void*);
    void DBtranslate(DBContext*, int, double, double, void*);
    void DBrotate(DBContext*, int, double, double, void*);
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
    int haveInheritMat;
    double mtnScl;
    double inheritMat[16];

    int modifier_mask;
    int last_btn;

    void update_modifier_widget();
    int old_fh;
    GC gc;
    int buttons_exposed;
    void redraw_buttons(CallbackData*, void*);
    XQColor* mod_colors[8];
    Font modefont;
    clString mode_string;
    void update_mode_string(const clString&);

    int salmon_count;
    int need_redraw;
    void get_bounds(BBox&);
public:
    Array1<GeomItem *> geomItemA;
    DrawInfo* drawinfo;
    Roe(Salmon *s);
    Roe(Salmon *s, double *m, double scl);
    Roe(const Roe&);
    ~Roe();
    void RoeInit(Salmon *s);
    void itemAdded(GeomObj*, char*);
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
    void redraw_if_needed();
    void printLevel(int level, int&flag);

    void mouse_translate(int, int, int, int, int);
    void mouse_scale(int, int, int, int, int);
    void mouse_rotate(int, int, int, int, int);
    void mouse_pick(int, int, int, int, int);
};



#endif
