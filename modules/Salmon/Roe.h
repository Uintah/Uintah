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
class CallbackData;
class DialogShellC;
class DrawingAreaC;
class FormC;
class FrameC;
class GLwMDrawC;
class Salmon;
class Vector;

class Roe {
    // User Interface stuff...
    DialogShellC* dialog;
    FormC* form;
    FrameC* gr_frame;
    GLwMDrawC* graphics;
    Salmon* manager;
    Array1<Roe *> kids;
    Roe *parent;
    int firstGen;	// am I first generation? (important for deleting)
    makeIndepRoe();
public:
    Roe(Salmon *s);
    Roe(const Roe&);
    ~Roe();
    void rotate(double angle, Vector v);
    void translate(Vector v);
    void scale(Vector v);
    void redraw();
    void addChild(Roe *r);
    void deleteChild(Roe *r);
    void SetParent(Roe *r);
    void SetParent(Salmon *s);
};

#endif
