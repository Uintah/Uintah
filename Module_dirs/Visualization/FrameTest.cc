/*
 *  FrameTest.cc:  
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/GeometryPort.h>
#include <Geometry/Point.h>
#include <TCL/TCLvar.h>

#include <Widgets/PointWidget.h>
#include <Widgets/ArrowWidget.h>
#include <Widgets/GuageWidget.h>
#include <Widgets/FrameWidget.h>
#include <Widgets/ScaledFrameWidget.h>
#include <Widgets/SquareWidget.h>
#include <Widgets/ScaledSquareWidget.h>
#include <Widgets/BoxWidget.h>
#include <Widgets/ScaledBoxWidget.h>
#include <Widgets/CubeWidget.h>
#include <Widgets/ScaledCubeWidget.h>

#include <iostream.h>

const Index NumWidgetTypes = 11;
enum WidgetTypes {Point, Arrow, Guage, Frame, SFrame, Square, SSquare, Box, SBox, Cube, SCube};

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

    widgets[Point]=new PointWidget(this, .1);
    widgets[Arrow]=new ArrowWidget(this, .1);
    widgets[Guage]=new GuageWidget(this, .1);
    widgets[Frame]=new FrameWidget(this, .1);
    widgets[SFrame]=new ScaledFrameWidget(this, .1);
    widgets[Square]=new SquareWidget(this, .1);
    widgets[SSquare]=new ScaledSquareWidget(this, .1);
    widgets[Box]=new BoxWidget(this, .1);
    widgets[SBox]=new ScaledBoxWidget(this, .1);
    widgets[Cube]=new CubeWidget(this, .1);
    widgets[SCube]=new ScaledCubeWidget(this, .1);

    widget_scale.set(.1);
    init = 1;

    widget_type.set(Frame);
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
        widget_id=ogeom->addObj(widgets[Point]->GetWidget(), widget_name+"_0");
        widget_id=ogeom->addObj(widgets[Arrow]->GetWidget(), widget_name+"_1");
        widget_id=ogeom->addObj(widgets[Guage]->GetWidget(), widget_name+"_2");
        widget_id=ogeom->addObj(widgets[Frame]->GetWidget(), widget_name+"_3");
	widget_id=ogeom->addObj(widgets[SFrame]->GetWidget(), widget_name+"_4");
        widget_id=ogeom->addObj(widgets[Square]->GetWidget(), widget_name+"_5");
	widget_id=ogeom->addObj(widgets[SSquare]->GetWidget(), widget_name+"_6");
        widget_id=ogeom->addObj(widgets[Box]->GetWidget(), widget_name+"_7");
	widget_id=ogeom->addObj(widgets[SBox]->GetWidget(), widget_name+"_8");
        widget_id=ogeom->addObj(widgets[Cube]->GetWidget(), widget_name+"_9");
	widget_id=ogeom->addObj(widgets[SCube]->GetWidget(), widget_name+"_10");
    }

    for(int i=0;i<NumWidgetTypes;i++)
       widgets[i]->SetState(0);
    
    widgets[widget_type.get()]->SetState(1);
    widgets[widget_type.get()]->SetScale(widget_scale.get());
    widgets[widget_type.get()]->execute();
    ogeom->flushViews();
}

void FrameTest::geom_moved(int axis, double dist, const Vector& delta,
			   void* cbdata)
{
    cerr << "Moved called..." << endl;

    widgets[widget_type.get()]->geom_moved(axis, dist, delta, cbdata);
    cout << "Gauge ratio " << ((GuageWidget*)widgets[Guage])->GetRatio() << endl;
    
    if(!abort_flag){
	abort_flag=1;
	want_to_execute();
    }
}

