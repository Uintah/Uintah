
/*
 *  Streamline.cc:  Generate streamlines from a field...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Streamline/Streamline.h>
#include <ScalarFieldRG.h>
#include <ScalarFieldUG.h>
#include <Geom.h>
#include <GeometryPort.h>
#include <ModuleList.h>
#include <MUI.h>
#include <NotFinished.h>
#include <Geometry/Point.h>
#include <iostream.h>
#include <fstream.h>

static Module* make_Streamline()
{
    return new Streamline;
}

static RegisterModule db1("Fields", "Streamline", make_Streamline);
static RegisterModule db2("Visualization", "Streamline", make_Streamline);

static clString widget_name("Streamline Widget");
static clString surface_name("Streamline");

Streamline::Streamline()
: UserModule("Streamline", Filter)
{
    // Create the input ports
    infield=new VectorFieldIPort(this, "Vector Field", ScalarFieldIPort::Atomic);
    add_iport(infield);
    //incolormap=new ColormapIPort(this, "Colormap");
    //add_iport(incolormap);
    incolorfield=new ScalarFieldIPort(this, "Color Field", ScalarFieldIPort::Atomic);
    add_iport(incolorfield);

    // Create the output port
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);

    widgettype=WPoint;
    MUI_choice* choice=new MUI_choice("Source: ", &widgettype,
				      MUI_widget::Immediate);
    choice->add_choice("Point");
    choice->add_choice("Line");
    choice->add_choice("Square");
    add_ui(choice);

    markertype=MLine;
    choice=new MUI_choice("Marker: ", &markertype, MUI_widget::Immediate);
    choice->add_choice("Line");
    choice->add_choice("Ribbon");
    add_ui(choice);

    lineradius=0.1;
    MUI_slider_real* slider=new MUI_slider_real("Line radius", &lineradius,
						MUI_widget::Immediate, 0);
    slider->set_minmax(0, 1);
    add_ui(slider);

    alg=AEuler;
    choice=new MUI_choice("Algorithm: ", &alg, MUI_widget::Immediate);
    choice->add_choice("Euler");
    choice->add_choice("4th Order Runge-Kutta");
    choice->add_choice("Stream Function");
    add_ui(choice);

    stepsize=0.1;
    slider=new MUI_slider_real("Step size", &stepsize,
			       MUI_widget::Immediate, 0);
    slider->set_minmax(0, 10);
    add_ui(slider);

    maxsteps=100;
    MUI_slider_int* islider=new MUI_slider_int("Max steps", &maxsteps,
					       MUI_widget::Immediate, 0);
    islider->set_minmax(0, 1000);
    add_ui(islider);

    adaptive_steps=1;
    add_ui(new MUI_onoff_switch("Adaptive stepsize", &adaptive_steps,
				MUI_widget::Immediate));

    maxfactor=10;
    slider=new MUI_slider_real("Max adaption factor", &maxfactor,
			       MUI_widget::Immediate, 0);
    slider->set_minmax(1, 100);
    add_ui(slider);

    arate=.5;
    slider=new MUI_slider_real("Adaption rate", &arate,
			       MUI_widget::Immediate, 0);
    slider->set_minmax(0, 1);
    add_ui(slider);

    widgettype_changed=1;
    need_p1=1;

    widget_point_matl=new MaterialProp(Color(0,0,0), Color(.54, .60, 1),
				       Color(.5,.5,.5), 20);
    widget_edge_matl=new MaterialProp(Color(0,0,0), Color(.54, .60, .66),
				      Color(.5,.5,.5), 20);
    widget_slider_matl=new MaterialProp(Color(0,0,0), Color(.83, .60, .66),
					Color(.5,.5,.5), 20);
    widget_highlight_matl=new MaterialProp(Color(0,0,0), Color(.7,.7,.7),
					   Color(0,0,.6), 20);
    widget_id=0;
    streamline_id=0;
}

Streamline::Streamline(const Streamline& copy, int deep)
: UserModule(copy, deep)
{
    NOT_FINISHED("Streamline::Streamline");
}

Streamline::~Streamline()
{
}

Module* Streamline::clone(int deep)
{
    return new Streamline(*this, deep);
}

void Streamline::execute()
{
    abort_flag=0;
    if(streamline_id)
	ogeom->delObj(streamline_id);
    VectorFieldHandle field;
    if(!infield->get(field))
	return;
    if(need_p1){
	Point min, max;
	field->get_bounds(min, max);
	p1=Interpolate(min, max, 0.5);
	need_p1=0;
    }
    widget_scale=0.05*field->longest_dimension();
    if(widgettype_changed){
	widgettype_changed=0;
	if(widget_id)
	    ogeom->delObj(widget_id);
	widget=new ObjGroup;
	switch(widgettype){
	case WPoint:
	    {
		widget_p1=new GeomSphere(p1, 1*widget_scale);
		widget_p1->set_matl(widget_point_matl);
		GeomPick* pick=new GeomPick(this, Vector(1,0,0),
					    Vector(0,1,0),
					    Vector(0,0,1));
		pick->set_highlight(widget_highlight_matl);
		widget->pick=pick;
		widget->add(widget_p1);
	    }
	    break;
	case WLine:
	    NOT_FINISHED("Line widget");
	    break;
	case WSquare:
	    NOT_FINISHED("Square widget");
	    break;
	}
	widget_id=ogeom->addObj(widget, widget_name);
    }
    // Update the widget...
    switch(widgettype){
    case WPoint:
	widget_p1->cen=p1;
	widget_p1->rad=1*widget_scale;
	widget_p1->adjust();
	break;
    case WLine:
	NOT_FINISHED("Line widget");
	break;
    case WSquare:
	NOT_FINISHED("Square widget");
	break;
    }
    ObjGroup* group=new ObjGroup;

    // Temporary algorithm...
    Point p(p1);
    GeomPolyLine* line=new GeomPolyLine;
    for(int i=0;i<maxsteps;i++){
	line->pts.add(p);
	Vector v;
	if(!field->interpolate(p, v))
	    break;
	p+=(v*stepsize);
    }
    if(line->pts.size() < 2)
	delete line;
    else
	group->add(line);

    if(group->size() == 0){
	delete group;
	streamline_id=0;
    } else {
	streamline_id=ogeom->addObj(group, surface_name);
    }
}

void Streamline::mui_callback(void*, int)
{
    if(!abort_flag){
	abort_flag=1;
	want_to_execute();
    }
    widgettype_changed=1;
}

void Streamline::geom_moved(int, double, const Vector& delta, void*)
{
    switch(widgettype){
    case WPoint:
	p1+=delta;
	break;
    case WLine:
	NOT_FINISHED("Line widget");
	break;
    case WSquare:
	NOT_FINISHED("Square widget");
	break;
    }
    if(!abort_flag){
	abort_flag=1;
	want_to_execute();
    }
}
