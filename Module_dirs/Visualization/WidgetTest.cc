/*
 *  WidgetTest.cc:  
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
#include <Widgets/CrosshairWidget.h>
#include <Widgets/GuageWidget.h>
#include <Widgets/RingWidget.h>
#include <Widgets/FixedFrameWidget.h>
#include <Widgets/FrameWidget.h>
#include <Widgets/ScaledFrameWidget.h>
#include <Widgets/SquareWidget.h>
#include <Widgets/ScaledSquareWidget.h>
#include <Widgets/BoxWidget.h>
#include <Widgets/ScaledBoxWidget.h>
#include <Widgets/CubeWidget.h>
#include <Widgets/ScaledCubeWidget.h>

#include <iostream.h>

const Index NumWidgetTypes = 14;
enum WidgetTypes { WT_Point, WT_Arrow, WT_Cross, WT_Guage, WT_Ring,
		   WT_FFrame, WT_Frame, WT_SFrame, WT_Square, WT_SSquare,
		   WT_Box, WT_SBox, WT_Cube, WT_SCube };

class WidgetTest : public Module {
   GeometryOPort* ogeom;
   CrowdMonitor widget_lock;

private:
   int init;
   int widget_id;
   TCLdouble widget_scale;
   TCLint widget_type;

   BaseWidget* widgets[NumWidgetTypes];

   virtual void geom_moved(int, double, const Vector&, void*);
public:
   WidgetTest(const clString& id);
   WidgetTest(const WidgetTest&, int deep);
   virtual ~WidgetTest();
   virtual Module* clone(int deep);
   virtual void execute();
};

static Module* make_WidgetTest(const clString& id)
{
   return new WidgetTest(id);
}

static RegisterModule db1("Fields", "WidgetTest", make_WidgetTest);
static RegisterModule db2("Visualization", "WidgetTest", make_WidgetTest);

static clString module_name("WidgetTest");

WidgetTest::WidgetTest(const clString& id)
: Module("WidgetTest", id, Source), widget_scale("widget_scale", id, this),
  widget_type("widget_type", id, this)
{
   // Create the output port
   ogeom = new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
   add_oport(ogeom);

   widgets[WT_Point] = new PointWidget(this, &widget_lock, .1);
   widgets[WT_Arrow] = new ArrowWidget(this, &widget_lock, .1);
   widgets[WT_Cross] = new CrosshairWidget(this, &widget_lock, .1);
   widgets[WT_Guage] = new GuageWidget(this, &widget_lock, .1);
   widgets[WT_Ring] = new RingWidget(this, &widget_lock, .1);
   widgets[WT_FFrame] = new FixedFrameWidget(this, &widget_lock, .1);
   widgets[WT_Frame] = new FrameWidget(this, &widget_lock, .1);
   widgets[WT_SFrame] = new ScaledFrameWidget(this, &widget_lock, .1);
   widgets[WT_Square] = new SquareWidget(this, &widget_lock, .1);
   widgets[WT_SSquare] = new ScaledSquareWidget(this, &widget_lock, .1);
   widgets[WT_Box] = new BoxWidget(this, &widget_lock, .1);
   widgets[WT_SBox] = new ScaledBoxWidget(this, &widget_lock, .1);
   widgets[WT_Cube] = new CubeWidget(this, &widget_lock, .1);
   widgets[WT_SCube] = new ScaledCubeWidget(this, &widget_lock, .1);

   widget_scale.set(.1);
   init = 1;

   widget_type.set(WT_Frame);
}

WidgetTest::WidgetTest(const WidgetTest& copy, int deep)
: Module(copy, deep),
  widget_scale("widget_scale", id, this),
  widget_type("widget_type", id, this)
{
   NOT_FINISHED("WidgetTest::WidgetTest");
}

WidgetTest::~WidgetTest()
{
}

Module* WidgetTest::clone(int deep)
{
   return new WidgetTest(*this, deep);
}

void WidgetTest::execute()
{
   if (init == 1) {
      init = 0;
      GeomGroup* w = new GeomGroup;
      for(int i = 0; i < NumWidgetTypes; i++)
	 w->add(widgets[i]->GetWidget());
      widget_id = ogeom->addObj(w, module_name, &widget_lock);
   }

   widget_lock.write_lock();
   for(int i = 0; i < NumWidgetTypes; i++)
      widgets[i]->SetState(0);
   
   widgets[widget_type.get()]->SetState(1);
   widgets[widget_type.get()]->SetScale(widget_scale.get());
   widget_lock.write_unlock();
   widgets[widget_type.get()]->execute();
   ogeom->flushViews();
}

void WidgetTest::geom_moved(int axis, double dist, const Vector& delta,
			    void* cbdata)
{
   widgets[widget_type.get()]->geom_moved(axis, dist, delta, cbdata);
   cout << "Gauge ratio " << ((GuageWidget*)widgets[WT_Guage])->GetRatio() << endl;
   cout << "Ring angle " << ((RingWidget*)widgets[WT_Ring])->GetRatio() << endl;
   
   if(!abort_flag) {
      abort_flag = 1;
      want_to_execute();
   }
}

