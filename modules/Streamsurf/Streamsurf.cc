/*
 *  Streamsurf.cc:  Generate Streamsurfs from a field...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Streamsurf/Streamsurf.h>
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

static Module* make_Streamsurf()
{
    return new Streamsurf;
}

static RegisterModule db1("Fields", "Streamsurf", make_Streamsurf);
static RegisterModule db2("Visualization", "Streamsurf", make_Streamsurf);

static clString widget_name("Streamsurf Widget");
static clString streamsurf_name("Streamsurf");

struct SSurf {
    Point p;
    int outside;
    SSurf(const Point&);
    int advance(const VectorFieldHandle&, int alg, double);
};

Streamsurf::Streamsurf()
: UserModule("Streamsurf", Filter)
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

    alg=AEuler;
    MUI_choice* choice=new MUI_choice("Algorithm: ", &alg, MUI_widget::Immediate);
    choice->add_choice("Euler");
    choice->add_choice("4th Order Runge-Kutta");
    choice->add_choice("Stream Function");
    add_ui(choice);

    stepsize=0.1;
    MUI_slider_real* slider=new MUI_slider_real("Step size", &stepsize,
						MUI_widget::Immediate, 0);
    slider->set_minmax(0, 10);
    add_ui(slider);

    maxsteps=100;
    MUI_slider_int* islider=new MUI_slider_int("Max steps", &maxsteps,
					       MUI_widget::Immediate, 0);
    islider->set_minmax(0, 1000);
    add_ui(islider);

    maxangle=5;
    slider=new MUI_slider_real("Maximum surface angle", &maxangle,
			       MUI_widget::Immediate, 0);
    slider->set_minmax(0, 90);
    add_ui(slider);
    need_p1=1;

    widget_point_matl=new MaterialProp(Color(0,0,0), Color(.54, .60, 1),
				       Color(.5,.5,.5), 20);
    widget_edge_matl=new MaterialProp(Color(0,0,0), Color(.54, .60, .66),
				      Color(.5,.5,.5), 20);
    widget_slider_matl=new MaterialProp(Color(0,0,0), Color(.83, .60, .66),
					Color(.5,.5,.5), 20);
    widget_highlight_matl=new MaterialProp(Color(0,0,0), Color(.7,.7,.7),
					   Color(0,0,.6), 20);
    need_widget=1;
    widget_id=0;
    streamsurf_id=0;
}

Streamsurf::Streamsurf(const Streamsurf& copy, int deep)
: UserModule(copy, deep)
{
    NOT_FINISHED("Streamsurf::Streamsurf");
}

Streamsurf::~Streamsurf()
{
}

Module* Streamsurf::clone(int deep)
{
    return new Streamsurf(*this, deep);
}

void Streamsurf::execute()
{
    abort_flag=0;
    if(streamsurf_id)
	ogeom->delObj(streamsurf_id);
    VectorFieldHandle field;
    if(!infield->get(field))
	return;
    if(need_p1){
	Point min, max;
	field->get_bounds(min, max);
	p1=Interpolate(min, max, 0.5);
	double s=field->longest_dimension();
	p2=p1+Vector(1,0,0)*(s/10);
	Vector sp(p2-p1);
	slider1_dist=sp.length()/10;
	need_p1=0;
    }
    widget_scale=0.01*field->longest_dimension();
    if(need_widget){
	need_widget=0;
	cerr << "Rebuilding widget" << endl;
	if(widget_id)
	    ogeom->delObj(widget_id);
	widget=new ObjGroup;
	widget_p1=new GeomSphere;
	widget_p1->pick=new GeomPick(this);
	widget_p1->pick->set_highlight(widget_highlight_matl);
	widget_p1->pick->set_cbdata((void*)1);
	widget_p1->set_matl(widget_point_matl);
	widget_p2=new GeomSphere;
	widget_p2->pick=new GeomPick(this);
	widget_p2->pick->set_highlight(widget_highlight_matl);
	widget_p2->pick->set_cbdata((void*)2);
	widget_p2->set_matl(widget_point_matl);
	widget_edge1=new GeomCylinder;
	widget_edge1->pick=new GeomPick(this);
	widget_edge1->pick->set_highlight(widget_highlight_matl);
	widget_edge1->pick->set_cbdata((void*)3);
	widget_edge1->set_matl(widget_edge_matl);
	widget_slider1=new ObjGroup;
	widget_slider1body=new GeomCylinder;
	widget_slider1cap1=new GeomDisc;
	widget_slider1cap2=new GeomDisc;
	widget_slider1->add(widget_slider1body);
	widget_slider1->add(widget_slider1cap1);
	widget_slider1->add(widget_slider1cap2);
	widget_slider1->pick=new GeomPick(this);
	widget_slider1->pick->set_highlight(widget_highlight_matl);
	widget_slider1->pick->set_cbdata((void*)4);
	widget_slider1->set_matl(widget_slider_matl);
	widget->add(widget_p1);
	widget->add(widget_p2);
	widget->add(widget_edge1);
	widget->add(widget_slider1);
	widget->pick=new GeomPick(this);
	widget_id=ogeom->addObj(widget, widget_name);
    }
    // Update the widget...
    widget_p1->move(p1, 1*widget_scale);
    widget_p2->move(p2, 1*widget_scale);
    widget_edge1->move(p1, p2, 0.5*widget_scale);
    Vector spvec(p2-p1);
    spvec.normalize();
    Point sp(p1+spvec*slider1_dist);
    Point sp2(sp+spvec*(widget_scale*0.5));
    Point sp1(sp-spvec*(widget_scale*0.5));
    widget_slider1body->move(sp1, sp2, 1*widget_scale);
    widget_slider1cap1->move(sp2, spvec, 1*widget_scale);
    widget_slider1cap2->move(sp1, -spvec, 1*widget_scale);
    Vector v1,v2;
    spvec.find_orthogonal(v1, v2);
    widget_p1->pick->set_principal(spvec, v1, v2);
    widget_p2->pick->set_principal(spvec, v1, v2);
    widget_edge1->pick->set_principal(spvec, v1, v2);
    widget_slider1->pick->set_principal(spvec);


    ObjGroup* group=new ObjGroup;

    Array1<SSurf*> ssurfs;
    Vector line(p2-p1);
    double l=line.length();
    for(double x=0;x<=l;x+=slider1_dist){
	ssurfs.add(new SSurf(p1+line*(x/l)));
    }

    int n=0;
    int groupid=0;
    Array1<GeomTriStrip*> strips(ssurfs.size()-1);
    Vector vv1(ssurfs[1]->p-ssurfs[0]->p);
    Vector vv2;
    field->interpolate(ssurfs[0]->p, vv2);
    Vector norm(Cross(vv1, vv2));
    norm.normalize();
    for(int i=0;i<strips.size();i++){
	strips[i]=new GeomTriStrip;
	strips[i]->add(ssurfs[i]->p, norm);
	if(i<strips.size()-1){
	    Vector v1(ssurfs[i+2]->p-ssurfs[i]->p);
	    Vector v2;
	    field->interpolate(ssurfs[i+1]->p, v2);
	    norm=Cross(v1, v2);
	    norm.normalize();
	} else {
	    Vector v1(ssurfs[i+1]->p-ssurfs[i]->p);
	    Vector v2;
	    field->interpolate(ssurfs[i+1]->p, v2);
	    norm=Cross(v1, v2);
	    norm.normalize();
	}
	strips[i]->add(ssurfs[i+1]->p, norm);
	group->add(strips[i]);
    }
    for(i=0;i<maxsteps;i++){
	double ss=stepsize;
	for(int i=0;i<ssurfs.size();i++)
	    n+=ssurfs[i]->advance(field, alg, ss);
	Vector v1(ssurfs[1]->p-ssurfs[0]->p);
	Vector v2;
	field->interpolate(ssurfs[0]->p, v2);
	Vector norm(Cross(v1, v2));
	norm.normalize();
	for(i=0;i<strips.size();i++){
	    SSurf* s1=ssurfs[i];
	    SSurf* s2=ssurfs[i+1];
	    if(!s1->outside && !s2->outside){
		if(i>0 && ssurfs[i-1]->outside){
		    // Recompute normal...
		    Vector v1(ssurfs[i+1]->p-ssurfs[i]->p);
		    Vector v2;
		    field->interpolate(ssurfs[i]->p, v2);
		    norm=Cross(v1, v2);
		    norm.normalize();
		}
		strips[i]->add(s1->p, norm);
		if(i<strips.size()-1 && !ssurfs[i+2]->outside){
		    Vector v1(ssurfs[i+2]->p-ssurfs[i]->p);
		    Vector v2;
		    field->interpolate(ssurfs[i+1]->p, v2);
		    norm=Cross(v1, v2);
		    norm.normalize();
		} else {
		    Vector v1(ssurfs[i+1]->p-ssurfs[i]->p);
		    Vector v2;
		    field->interpolate(ssurfs[i+1]->p, v2);
		    norm=Cross(v1, v2);
		    norm.normalize();
		}
		strips[i]->add(s2->p, norm);
	    }
	}
    }
    if(groupid)
	ogeom->delObj(groupid);
    if(group->size() == 0){
	delete group;
	streamsurf_id=0;
    } else {
	streamsurf_id=ogeom->addObj(group, streamsurf_name);
    }
}

void Streamsurf::mui_callback(void*, int)
{
    if(!abort_flag){
	abort_flag=1;
	want_to_execute();
    }
}

void Streamsurf::geom_moved(int axis, double dist, const Vector& delta,
			    void* cbdata)
{
    switch((int)cbdata){
    case 1:
	p1+=delta;
	break;
    case 2:
	p2+=delta;
	break;
    case 3:
	p1+=delta;
	p2+=delta;
	break;
    case 4:
	{
	    if(axis==0){
		slider1_dist+=dist;
	    } else {
		slider1_dist-=dist;
	    }
	    Vector pv(p2-p1);
	    double l=pv.length();
	    if(slider1_dist < 0.01*l)
		slider1_dist=0.01*l;
	    else if(slider1_dist > l)
		slider1_dist=l;
	    break;
	}
	break;
    }
    if(!abort_flag){
	abort_flag=1;
	want_to_execute();
    }
}

SSurf::SSurf(const Point& p)
: p(p)
{
    outside=0;
}

int SSurf::advance(const VectorFieldHandle& field, int alg,
		   double stepsize)
{
    if(outside)
	return 0;
    switch(alg){
    case Streamsurf::AEuler:
	{
	    Vector v;
	    if(!field->interpolate(p, v)){
		outside=1;
		return 0;
	    }
	    p+=(v*stepsize);
	}
	break;
    case Streamsurf::ARK4:
	{
	    NOT_FINISHED("SSurf::advance for RK4");
	}
	break;
    case Streamsurf::AStreamFunction:
	{
	    NOT_FINISHED("SSurf::advance for Stream Function");
	}
	break;
    }
    return 1;
}
