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
class CallbackData;
class DialogShellC;
class DrawingAreaC;
class FormC;
class FrameC;
class GLwMDrawC;
class Salmon;
class Vector;
class RowColumnC;
class LabelC;
class ScrolledWindowC;
class ToggleButtonC;
class PushButtonC;
class SeparatorC;
class GeomObj;

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
    ToggleButtonC* item1;
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
    RowColumnC* options;
    RowColumnC* viewRC;
    PushButtonC* autoView;
    PushButtonC* setHome;
    PushButtonC* goHome;
    DrawingAreaC* buttons;
    RowColumnC* spawnRC;
    PushButtonC* spawnCh;
    PushButtonC* spawnInd;
    FormC* form;
    FrameC* gr_frame;
    Salmon* manager;

    void btn1upCB(CallbackData*, void*);
    void btn1downCB(CallbackData*, void*);
    void btn1motionCB(CallbackData*, void*);
    void btn2upCB(CallbackData*, void*);
    void btn2downCB(CallbackData*, void*);
    void btn2motionCB(CallbackData*, void*);
    void item1CB(CallbackData*, void*);
    void wireCB(CallbackData*, void*);
    void flatCB(CallbackData*, void*);
    void gouraudCB(CallbackData*, void*);
    void phongCB(CallbackData*, void*);
    void ambientCB(CallbackData*, void*);
    void point1CB(CallbackData*, void*);
    void goHomeCB(CallbackData*, void*);
    void autoViewCB(CallbackData*, void*);
    void setHomeCB(CallbackData*, void*);
    void spawnChCB(CallbackData*, void*);
    void destroyWidgetCB(CallbackData*, void*); 
    void redrawCB(CallbackData*, void*);
    void initCB(CallbackData*, void*);
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
public:
    Roe(Salmon *s);
    Roe(const Roe&);
    ~Roe();
    void rotate(double angle, Vector v);
    void translate(Vector v);
    void scale(Vector v);
    void addChild(Roe *r);
    void deleteChild(Roe *r);
    void SetParent(Roe *r);
    void SetTop();
    void redrawAll();
    void printLevel(int level, int&flag);
};

#endif
