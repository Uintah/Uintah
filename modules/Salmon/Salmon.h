/*
 *  Salmon.h: The Geometry Viewer!
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_Salmon_h
#define SCI_project_module_Salmon_h

#include <Module.h>
#include <Classlib/Array1.h>
class CallbackData;
class DialogShellC;
class DrawingAreaC;
class FormC;
class FrameC;
class GeometryIPort;
class GLwMDrawC;
class XQColor;
class Roe;
class GeomObj;

class Salmon : public Module {
    Array1<GeometryIPort*> iports;
    Array1<Roe*> topRoe;
    virtual void do_execute();
    virtual void create_interface();
    virtual int should_execute();
    virtual void reconfigure_iports();
    virtual void reconfigure_oports();
    DrawingAreaC* drawing_a;
    void redraw_widget(CallbackData*, void*);
    XQColor* bgcolor;

    // User Interface stuff...
    DialogShellC* dialog;
    FormC* form;
    FrameC* gr_frame;
    GLwMDrawC* graphics;

    //gotta store the geometry!

public:
    Salmon();
    Salmon(const Salmon&, int deep);
    virtual ~Salmon();
    virtual Module* clone(int deep);
    void addObj(int serial, GeomObj *obj);
    void delObj(int serial);
    void addTopRoe(Roe *r);
    void makeTopRoe();
    void delTopRoe(Roe *r);
};

#endif
