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

#include <Modules/Visualization/FrameTest.h>

#include <Classlib/NotFinished.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/GeometryPort.h>
#include <Geometry/Point.h>
#include <Geom/Geom.h>
#include <Geom/Group.h>

#include <iostream.h>
#include <fstream.h>

static Module* make_FrameTest(const clString& id)
{
    return new FrameTest(id);
}

static RegisterModule db1("Fields", "FrameTest", make_FrameTest);
static RegisterModule db2("Visualization", "FrameTest", make_FrameTest);

static clString widget_name("FrameTest Widget");

FrameTest::FrameTest(const clString& id)
: Module("FrameTest", id, Source)
{
    // Create the output port
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);

    widget=new FrameWidget(this);
    init = 1;

#ifdef OLDUI
    MUI_slider_real* slider=new MUI_slider_real("Adaption rate", &widget_scale,
						MUI_widget::Immediate, 0);
    slider->set_minmax(0, 10);
    add_ui(slider);
#endif
}

FrameTest::FrameTest(const FrameTest& copy, int deep)
: Module(copy, deep)
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
        widget_id=ogeom->addObj(widget->GetWidget(), widget_name);
    }
    abort_flag=0;
    widget->SetScale(100.0);
    widget->execute();
}

void FrameTest::mui_callback(void*, int)
{
    if(!abort_flag){
	abort_flag=1;
	want_to_execute();
    }
}

void FrameTest::geom_moved(int axis, double dist, const Vector& delta,
			   void* cbdata)
{
    cerr << "Moved called..." << endl;
    
    widget->geom_moved(axis, dist, delta, cbdata);
    
    if(!abort_flag){
	abort_flag=1;
	want_to_execute();
    }
}

