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
#include <Widgets/CriticalPointWidget.h>
#include <Widgets/CrosshairWidget.h>
#include <Widgets/GaugeWidget.h>
#include <Widgets/RingWidget.h>
#include <Widgets/FrameWidget.h>
#include <Widgets/ScaledFrameWidget.h>
#include <Widgets/BoxWidget.h>
#include <Widgets/ScaledBoxWidget.h>
#include <Widgets/ViewWidget.h>
#include <Widgets/LightWidget.h>
#include <Widgets/PathWidget.h>

#include <iostream.h>

enum WidgetTypes { WT_Point, WT_Arrow, WT_Crit, WT_Cross, WT_Gauge, WT_Ring,
		   WT_Frame, WT_SFrame, WT_Box, WT_SBox, WT_View, WT_Light,
		   WT_Path, NumWidgetTypes };

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
   virtual void geom_release(void*);
public:
   WidgetTest(const clString& id);
   WidgetTest(const WidgetTest&, int deep);
   virtual ~WidgetTest();
   virtual Module* clone(int deep);
   virtual void execute();

   virtual void tcl_command(TCLArgs&, void*);
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
  widget_type("widget_type", id, this), init(1)
{
   // Create the output port
   ogeom = new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
   add_oport(ogeom);

   float INIT(0.01);

   widgets[WT_Point] = new PointWidget(this, &widget_lock, INIT);
   widgets[WT_Arrow] = new ArrowWidget(this, &widget_lock, INIT);
   widgets[WT_Crit] = new CriticalPointWidget(this, &widget_lock, INIT);
   widgets[WT_Cross] = new CrosshairWidget(this, &widget_lock, INIT);
   widgets[WT_Gauge] = new GaugeWidget(this, &widget_lock, INIT);
   widgets[WT_Ring] = new RingWidget(this, &widget_lock, INIT);
   widgets[WT_Frame] = new FrameWidget(this, &widget_lock, INIT);
   widgets[WT_SFrame] = new ScaledFrameWidget(this, &widget_lock, INIT);
   widgets[WT_Box] = new BoxWidget(this, &widget_lock, INIT);
   widgets[WT_SBox] = new ScaledBoxWidget(this, &widget_lock, INIT);
   widgets[WT_View] = new ViewWidget(this, &widget_lock, INIT);
   widgets[WT_Light] = new LightWidget(this, &widget_lock, INIT);
   widgets[WT_Path] = new PathWidget(this, &widget_lock, INIT);
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
   widgets[widget_type.get()]->execute();
   ogeom->flushViews();
}

void WidgetTest::geom_moved(int, double, const Vector&, void*)
{
    // Update Arrow/Critical point widget
    if ((widgets[WT_Arrow]->ReferencePoint()-Point(0,0,0)).length2() >= 1e-6)
	((ArrowWidget*)widgets[WT_Arrow])->SetDirection((Point(0,0,0)-widgets[WT_Arrow]->ReferencePoint()).normal());
    if ((widgets[WT_Crit]->ReferencePoint()-Point(0,0,0)).length2() >= 1e-6)
	((CriticalPointWidget*)widgets[WT_Crit])->SetDirection((Point(0,0,0)-widgets[WT_Crit]->ReferencePoint()).normal());
    widgets[widget_type.get()]->execute();

    widget_lock.read_lock();
    cout << "Gauge ratio " << ((GaugeWidget*)widgets[WT_Gauge])->GetRatio() << endl;
    cout << "Ring angle " << ((RingWidget*)widgets[WT_Ring])->GetRatio() << endl;
    cout << "FOV " << ((ViewWidget*)widgets[WT_View])->GetFOV() << endl;
    widget_lock.read_unlock();
}

void WidgetTest::geom_release(void*)
{
    if(!abort_flag){
	abort_flag=1;
	want_to_execute();
    }
}


void WidgetTest::tcl_command(TCLArgs& args, void* userdata)
{
   if(args.count() < 2){
      args.error("WidgetTest needs a minor command");
      return;
   }
   if (args[1] == "nextmode") {
      widgets[widget_type.get()]->NextMode();
      if(!abort_flag){
	 abort_flag=1;
	 want_to_execute();
      }
   } else if(args[1] == "select"){
       // Select the appropriate widget
       reset_vars();
       widget_lock.write_lock();
       for(int i = 0; i < NumWidgetTypes; i++)
	   widgets[i]->SetState(0);
       widgets[widget_type.get()]->SetState(1);
   
       widgets[widget_type.get()]->SetScale(widget_scale.get());
       widgets[widget_type.get()]->execute();
       widget_lock.write_unlock();
   } else if(args[1] == "scale"){
       widget_lock.write_lock();
       widgets[widget_type.get()]->SetScale(widget_scale.get());
       widgets[widget_type.get()]->execute();
       widget_lock.write_unlock();
       ogeom->flushViews();
    } else {
      Module::tcl_command(args, userdata);
   }
}
