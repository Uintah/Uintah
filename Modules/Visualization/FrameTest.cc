/*
 *  FrameTest.cc:  
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/GeometryPort.h>
#include <Geometry/Point.h>
#include <Geom/Geom.h>
#include <Geom/Group.h>
#include <TCL/TCLvar.h>

#include <Widgets/ArrowWidget.h>
#include <Widgets/FrameWidget.h>
#include <Widgets/GuageWidget.h>

#include <iostream.h>

const Index NumWidgetTypes = 3;
enum WidgetTypes {Arrow, Frame, Guage};

class FrameTest : public Module {
    GeometryOPort* ogeom;

private:
    int init;
    int widget_id;
    TCLdouble widget_scale;
    TCLint widget_type;

    BaseWidget* widgets[NumWidgetTypes];

    virtual void geom_moved(int, double, const Vector&, void*);
public:
    FrameTest(const clString& id);
    FrameTest(const FrameTest&, int deep);
    virtual ~FrameTest();
    virtual Module* clone(int deep);
    virtual void execute();
};

static Module* make_FrameTest(const clString& id)
{
    return new FrameTest(id);
}

static RegisterModule db1("Fields", "FrameTest", make_FrameTest);
static RegisterModule db2("Visualization", "FrameTest", make_FrameTest);

static clString widget_name("FrameTest Widget");

FrameTest::FrameTest(const clString& id)
: Module("FrameTest", id, Source), widget_scale("widget_scale", id, this),
  widget_type("widget_type", id, this)
{
    // Create the output port
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);

    widgets[Arrow]=new ArrowWidget(this, .1);
    widgets[Frame]=new FrameWidget(this, .1);
    widgets[Frame]=new GuageWidget(this, .1);
    widget_scale.set(.1);
    init = 1;

    widget_type.set(1);
}

FrameTest::FrameTest(const FrameTest& copy, int deep)
: Module(copy, deep),
  widget_scale("widget_scale", id, this),
  widget_type("widget_type", id, this)
{
    NOT_FINISHED("FrameTest::FrameTest");
}

FrameTest::~FrameTest()
{
}

Module* FrameTest::clone(int deep)
{
    return new FrameTest(*this, deep);
}

void FrameTest::execute()
{
    if (init == 1) {
        init = 0;
        widget_id=ogeom->addObj(widgets[0]->GetWidget(), widget_name);
        widget_id=ogeom->addObj(widgets[1]->GetWidget(), widget_name+"2");
        widget_id=ogeom->addObj(widgets[1]->GetWidget(), widget_name+"3");
    }

    widgets[widget_type.get()]->SetScale(widget_scale.get());
    widgets[widget_type.get()]->execute();
    ogeom->flushViews();
}

void FrameTest::geom_moved(int axis, double dist, const Vector& delta,
			   void* cbdata)
{
    cerr << "Moved called..." << endl;

    widgets[widget_type.get()]->geom_moved(axis, dist, delta, cbdata);
    
    if(!abort_flag){
	abort_flag=1;
	want_to_execute();
    }
}

