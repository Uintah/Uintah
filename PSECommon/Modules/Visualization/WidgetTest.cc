//static char *id="@(#) $Id$";

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

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Thread/CrowdMonitor.h>

#include <PSECore/Widgets/PointWidget.h>
#include <PSECore/Widgets/ArrowWidget.h>
#include <PSECore/Widgets/CriticalPointWidget.h>
#include <PSECore/Widgets/CrosshairWidget.h>
#include <PSECore/Widgets/GaugeWidget.h>
#include <PSECore/Widgets/RingWidget.h>
#include <PSECore/Widgets/FrameWidget.h>
#include <PSECore/Widgets/ScaledFrameWidget.h>
#include <PSECore/Widgets/BoxWidget.h>
#include <PSECore/Widgets/ScaledBoxWidget.h>
#include <PSECore/Widgets/ViewWidget.h>
#include <PSECore/Widgets/LightWidget.h>
#include <PSECore/Widgets/PathWidget.h>

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace PSECore::Widgets;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

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

   virtual void widget_moved(int);

public:
   WidgetTest(const clString& id);
   virtual ~WidgetTest();
   virtual void execute();

   virtual void tcl_command(TCLArgs&, void*);
};

Module* make_WidgetTest(const clString& id) {
  return new WidgetTest(id);
}

static clString module_name("WidgetTest");

WidgetTest::WidgetTest(const clString& id)
: Module("WidgetTest", id, Source), widget_lock("WidgetTest widget lock"),
    widget_scale("widget_scale", id, this),
  widget_type("widget_type", id, this),
  init(1)
{
   // Create the output port
   ogeom = scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
   add_oport(ogeom);

   float INIT(0.01);

   widgets[WT_Point] = scinew PointWidget(this, &widget_lock, INIT);
   widgets[WT_Arrow] = scinew ArrowWidget(this, &widget_lock, INIT);
   widgets[WT_Crit] = scinew CriticalPointWidget(this, &widget_lock, INIT);
   widgets[WT_Cross] = scinew CrosshairWidget(this, &widget_lock, INIT);
   widgets[WT_Gauge] = scinew GaugeWidget(this, &widget_lock, INIT);
   widgets[WT_Ring] = scinew RingWidget(this, &widget_lock, INIT);
   widgets[WT_Frame] = scinew FrameWidget(this, &widget_lock, INIT);
   widgets[WT_SFrame] = scinew ScaledFrameWidget(this, &widget_lock, INIT);
   widgets[WT_Box] = scinew BoxWidget(this, &widget_lock, INIT);
   widgets[WT_SBox] = scinew ScaledBoxWidget(this, &widget_lock, INIT);
   widgets[WT_View] = scinew ViewWidget(this, &widget_lock, INIT);
   widgets[WT_Light] = scinew LightWidget(this, &widget_lock, INIT);
   widgets[WT_Path] = scinew PathWidget(this, &widget_lock, INIT);
}

WidgetTest::~WidgetTest()
{
}

void WidgetTest::execute()
{
   if (init == 1) {
      init = 0;
      GeomGroup* w = scinew GeomGroup;
      int i;
      for(i = 0; i < NumWidgetTypes; i++)
	 w->add(widgets[i]->GetWidget());
      widget_id = ogeom->addObj(w, module_name, &widget_lock);
      for(i = 0; i < NumWidgetTypes; i++)
	 widgets[i]->Connect(ogeom);
   }
   cerr << "There are currently " << ogeom->getNRoe() << " active roe\n";
   GeometryData* data=ogeom->getData(0, GEOM_ALLDATA);
   cerr << "data=" << data << endl;
}

void WidgetTest::widget_moved(int last)
{
    if(last)
	cerr << "Last callback...\n";
   cerr << "WidgetTest: begin widget_moved" << endl;
   // Update Arrow/Critical point widget
   if ((widgets[WT_Arrow]->ReferencePoint()-Point(0,0,0)).length2() >= 1e-6)
      ((ArrowWidget*)widgets[WT_Arrow])->SetDirection((Point(0,0,0)-widgets[WT_Arrow]->ReferencePoint()).normal());
   if ((widgets[WT_Crit]->ReferencePoint()-Point(0,0,0)).length2() >= 1e-6)
      ((CriticalPointWidget*)widgets[WT_Crit])->SetDirection((Point(0,0,0)-widgets[WT_Crit]->ReferencePoint()).normal());
   
   widget_lock.readLock();
   cout << "Gauge ratio " << ((GaugeWidget*)widgets[WT_Gauge])->GetRatio() << endl;
   cout << "Ring angle " << ((RingWidget*)widgets[WT_Ring])->GetRatio() << endl;
   cout << "FOV " << ((ViewWidget*)widgets[WT_View])->GetFOV() << endl;
   widget_lock.readUnlock();

   // If your module needs to execute when the widget moves, add these lines:
   //if(last && !abort_flag){
   //    abort_flag=1;
   //    want_to_execute();
   //}
   cerr << "WidgetTest: end widget_moved" << endl;
}


void WidgetTest::tcl_command(TCLArgs& args, void* userdata)
{
#if 0
   if(args.count() < 2){
      args.error("WidgetTest needs a minor command");
      return;
   }
#endif
   if (args[1] == "nextmode") {
      widgets[widget_type.get()]->NextMode();
   } else if(args[1] == "select"){
       // Select the appropriate widget
       reset_vars();
       widget_lock.writeLock();
       for(int i = 0; i < NumWidgetTypes; i++)
	   widgets[i]->SetState(0);
       widgets[widget_type.get()]->SetState(1);
       widget_lock.writeUnlock();
       widgets[widget_type.get()]->SetScale(widget_scale.get());
   } else if(args[1] == "scale"){
      reset_vars();
      widgets[widget_type.get()]->SetScale(widget_scale.get());
   } else if(args[1] == "ui"){
      widgets[widget_type.get()]->ui();
   } else {
      Module::tcl_command(args, userdata);
   }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.7  1999/10/07 02:07:09  sparker
// use standard iostreams and complex type
//
// Revision 1.6  1999/08/29 00:46:49  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.5  1999/08/25 03:48:12  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:18:00  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:20:13  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:55  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:18  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:34  dav
// Import sources
//
//
