
/*
 *  FrameTest.h: Visualization module
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_FrameTest_h
#define SCI_project_module_FrameTest_h

#include <Dataflow/Module.h>
#include <Widgets/FrameWidget.h>
class GeometryOPort;
class MUI_slider_real;


class FrameTest : public Module {
    GeometryOPort* ogeom;
    int abort_flag;

private:
   int init;
    int widget_id;
    double widget_scale;

    FrameWidget* widget;

    virtual void geom_moved(int, double, const Vector&, void*);
public:
    FrameTest(const clString& id);
    FrameTest(const FrameTest&, int deep);
    virtual ~FrameTest();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual void mui_callback(void*, int);
};

#endif
