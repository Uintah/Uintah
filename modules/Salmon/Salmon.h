
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
#include <Geometry/Point.h>
class CallbackData;
class DialogShellC;
class DrawingAreaC;
class FormC;
class FrameC;
class GeometryIPort;
class GLwMDrawC;
class XQColor;

class Salmon : public Module {
    Array1<GeometryIPort*> iports;
    virtual void do_execute();
    virtual void create_widget();
    virtual int should_execute();
    DrawingAreaC* drawing_a;
    void redraw_widget(CallbackData*, void*);
    XQColor* bgcolor;

    // User Interface stuff...
    DialogShellC* dialog;
    FormC* form;
    FrameC* gr_frame;
    GLwMDrawC* graphics;
public:
    Salmon();
    Salmon(const Salmon&, int deep);
    virtual ~Salmon();
    virtual Module* clone(int deep);
};

#endif
