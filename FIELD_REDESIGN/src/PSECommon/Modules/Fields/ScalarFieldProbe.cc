/*----------------------------------------------------------------------
CLASS
    ScalarFieldProbe

    Interactive tester/probe for scalar fields

OVERVIEW TEXT

KEYWORDS


AUTHOR
    Alexei Samsonov
    Department of Computer Science
    University of Utah
    June 2000

    Copyright (C) 2000 SCI Group

LOG
    Created June 2000
----------------------------------------------------------------------*/



#include <SCICore/Thread/CrowdMonitor.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Widgets/CrosshairWidget.h>
#include <SCICore/Geom/Switch.h>
#include <SCICore/Datatypes/ScalarFieldUG.h>
#include <math.h>

#include <PSECommon/share/share.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Widgets;    
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::Containers;
using SCICore::Geometry::Point;

const clString EMPTY_MSG="Not Available";

class PSECommonSHARE ScalarFieldProbe : public Module {
  
  ScalarFieldIPort* iPort;
  ScalarFieldOPort* oPort;
  GeometryOPort*    ogeom;

  // probe part
  CrosshairWidget* widget;
  clString         execMsg;
  double           widget_scale;
  TCLdouble        w_x, w_y, w_z;
  TCLstring        field_value;
  GeomID           widget_id;
 
  
  // statistical part
  TCLstring minValue;
  TCLstring maxValue;
  TCLstring bnd1;
  TCLstring bnd2;
  TCLstring gridType;
 
  ScalarFieldHandle hSF;
  clString  exec_msg;
  TCLstring num_elements;

public:
  ScalarFieldProbe(const clString& id);
  
  virtual ~ScalarFieldProbe();
  CrowdMonitor widget_lock;
  virtual void execute();
  
  virtual void widget_moved(int lock);
  virtual void tcl_command(TCLArgs&, void*);
};


extern "C" PSECommonSHARE Module* make_ScalarFieldProbe(const clString& id) {
  return new ScalarFieldProbe(id);
}

ScalarFieldProbe::ScalarFieldProbe(const clString& id)
  : Module("ScalarFieldProbe", id, Source),
  widget_lock("ScalarFieldProbe widget lock"),
  w_x("w_x", id, this),
  w_y("w_y", id, this),
  w_z("w_z", id, this),
  field_value("field_value", id, this),
  widget_id(0),
  minValue ("minValue", id, this),
  maxValue ("maxValue", id, this),
  bnd1 ("bnd1", id, this),
  bnd2 ("bnd2", id, this),
  gridType ("gridType", id, this),
  num_elements("num_elements", id, this)
{
   iPort=scinew ScalarFieldIPort(this, "Input Data", ScalarFieldIPort::Atomic);
   add_iport(iPort);
   
   oPort=scinew ScalarFieldOPort(this, "Output Data", ScalarFieldIPort::Atomic);
   add_oport(oPort);
   
   ogeom=scinew GeometryOPort(this, "Widget Output", GeometryIPort::Atomic);
   add_oport(ogeom);

   minValue.set(EMPTY_MSG);
   maxValue.set(EMPTY_MSG);
   gridType.set(EMPTY_MSG);
   field_value.set(EMPTY_MSG);
   bnd1.set(EMPTY_MSG);
   bnd2.set(EMPTY_MSG);
   num_elements.set(EMPTY_MSG);

   w_x.set(0);
   w_y.set(0);
   w_z.set(0);

   widget_scale=0.1;
   widget=scinew CrosshairWidget(this, &widget_lock, widget_scale);
   widget->SetScale(widget_scale);
   widget->SetPosition(Point(0, 0, 0));
}

ScalarFieldProbe::~ScalarFieldProbe()
{
   ogeom->delObj(widget_id);
   ogeom->flushViews();
}

void ScalarFieldProbe::execute()
{
  if (!widget_id){ 
      widget_id=ogeom->addObj(widget->GetWidget(), clString("Crosshair"), &widget_lock);
      ogeom->flushViews();
  }

  if (!iPort->get(hSF))
    {
       minValue.set(EMPTY_MSG);
       maxValue.set(EMPTY_MSG);
       bnd1.set(EMPTY_MSG);
       bnd2.set(EMPTY_MSG);
       gridType.set(EMPTY_MSG);
       field_value.set(EMPTY_MSG);
       num_elements.set(EMPTY_MSG);

       TCL::execute(id+" update_field ");
       TCL::execute(id+" update_widg ");
       return;
    }
   
   Point pos=widget->GetPosition();
   w_x.set(pos.x());
   w_y.set(pos.y());
   w_z.set(pos.z());
    
   double fv=0;
   if (hSF->interpolate(pos, fv))
     field_value.set(to_string(fv));
   else
     field_value.set(EMPTY_MSG);
     TCL::execute(id+" update_widg ");
  
   if (exec_msg!="widget_moved"){
    
    clString typeName;
    int nelems=0;
  
    if (hSF->getRG()) typeName="Regular";
    else if ( ScalarFieldUG* hUGSF=hSF->getUG()){ 
      typeName="Unstructured";
      nelems=hUGSF->mesh->elems.size();
    }
    else typeName="Unknown";
  
    gridType.set(typeName);    
    
    if (nelems>0)
      num_elements.set(to_string(nelems));
    else
      num_elements.set(EMPTY_MSG);

    double min=0, max=0;
    hSF->get_minmax(min, max);
  
    minValue.set(to_string(min));
    maxValue.set(to_string(max));
    
    Point bound1, bound2;
    hSF->get_bounds(bound1, bound2);
    bnd1.set("X1="+to_string(float(bound1.x()))+", Y1="+to_string(float(bound1.y()))+",  Z1="+to_string(float(bound1.z())));
    bnd2.set("X2="+to_string(float(bound2.x()))+", Y2="+to_string(float(bound2.y()))+",  Z2="+to_string(float(bound2.z())));
    
    // finding appropriate scale for widget
    double dx=bound2.x()-bound1.x();
    double dy=bound2.y()-bound1.y();
    double dz=bound2.z()-bound1.z();

    widget_scale=sqrt (dx*dx+dy*dy+dz*dz)/80; // ?????
    widget->SetScale(widget_scale);
    widget->SetPosition(Point(bound1.x(), bound1.y(), bound1.z()));
    ogeom->flushViews();
    
    TCL::execute(id+" update_field ");
    TCL::execute(id+" update_widg ");
  }
  exec_msg="";
  oPort->send(hSF);
}

void ScalarFieldProbe::tcl_command(TCLArgs& args, void* userdata)
{
  if (args[1] == "WidgetOn"){
     widget->SetState(1);
     widget->SetScale(widget_scale);
  }
  else if (args[1] == "WidgetOff"){
     widget->SetState(0);
  }
  else {
     Module::tcl_command(args, userdata);
  }
  ogeom->flushViews();
}

void ScalarFieldProbe::widget_moved(int lock)
{    
    Point pos=widget->GetPosition();
    w_x.set(pos.x());
    w_y.set(pos.y());
    w_z.set(pos.z());
   
    TCL::execute(id+" update_widg ");

    if(lock && !abort_flag){
	abort_flag=1;
	exec_msg="widget_moved";
	want_to_execute();
    }
}

} // End namespace Modules
} // End namespace PSECommon










