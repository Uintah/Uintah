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

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldUG.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/VectorField.h>
#include <Datatypes/VectorFieldPort.h>
#include <Geom/Cylinder.h>
#include <Geom/Disc.h>
#include <Geom/Geom.h>
#include <Geom/Group.h>
#include <Geom/Pick.h>
#include <Geom/Polyline.h>
#include <Geom/Sphere.h>
#include <Geometry/Point.h>
#include <TCL/TCLvar.h>

#include <iostream.h>

class Streamline : public Module {
    VectorFieldIPort* infield;
//    ColormapPort* incolormap;
    ScalarFieldIPort* incolorfield;
    GeometryOPort* ogeom;

    TCLstring widgettype;
    clString oldwidgettype;
    TCLstring markertype;
    TCLdouble lineradius;
    TCLstring algorithm;

    TCLvardouble stepsize;
    TCLvarint maxsteps;

    int need_p1;
    Point p1;
    Point p2;
    Point p3;
    Point p4;
    double slider1_dist;
    double slider2_dist;

    GeomGroup* widget;
    GeomSphere* widget_p1;
    GeomSphere* widget_p2;
    GeomSphere* widget_p3;
    GeomSphere* widget_p4;
    GeomCylinder* widget_edge1;
    GeomCylinder* widget_edge2;
    GeomCylinder* widget_edge3;
    GeomCylinder* widget_edge4;
    GeomGroup* widget_slider1;
    GeomGroup* widget_slider2;
    GeomCylinder* widget_slider1body;
    GeomCylinder* widget_slider2body;
    GeomDisc* widget_slider1cap1;
    GeomDisc* widget_slider1cap2;
    GeomDisc* widget_slider2cap1;
    GeomDisc* widget_slider2cap2;
    GeomPick* widget_p1_pick;
    GeomPick* widget_p2_pick;
    GeomPick* widget_p3_pick;
    GeomPick* widget_p4_pick;
    GeomPick* widget_edge1_pick;
    GeomPick* widget_edge2_pick;
    GeomPick* widget_edge3_pick;
    GeomPick* widget_edge4_pick;
    GeomPick* widget_slider1_pick;
    GeomPick* widget_slider2_pick;
    int widget_id;
    double widget_scale;

    int streamline_id;

    MaterialHandle widget_point_matl;
    MaterialHandle widget_edge_matl;
    MaterialHandle widget_slider_matl;
    MaterialHandle widget_highlight_matl;
    MaterialHandle matl;

    virtual void geom_moved(int, double, const Vector&, void*);
public:
    Streamline(const clString& id);
    Streamline(const Streamline&, int deep);
    virtual ~Streamline();
    virtual Module* clone(int deep);
    virtual void execute();
};

static Module* make_Streamline(const clString& id)
{
    return new Streamline(id);
}

static RegisterModule db1("Fields", "Streamline", make_Streamline);
static RegisterModule db2("Visualization", "Streamline", make_Streamline);

static clString widget_name("Streamline Widget");
static clString streamline_name("Streamline");

class SLine {
    Point p;
    GeomPolyline* line;
    int outside;
public:
    SLine(GeomGroup*, const Point&);
    int advance(const VectorFieldHandle&, int rk4, double);
};

Streamline::Streamline(const clString& id)
: Module(streamline_name, id, Filter),
  widgettype("source", id, this),
  markertype("marker", id, this), lineradius("lineradius", id, this),
  algorithm("algorithm", id, this), stepsize("stepsize", id, this),
  maxsteps("maxsteps", id, this)
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

    need_p1=1;

    widget_point_matl=new Material(Color(0,0,0), Color(.54, .60, 1),
				   Color(.5,.5,.5), 20);
    widget_edge_matl=new Material(Color(0,0,0), Color(.54, .60, .66),
				  Color(.5,.5,.5), 20);
    widget_slider_matl=new Material(Color(0,0,0), Color(.83, .60, .66),
				    Color(.5,.5,.5), 20);
    widget_highlight_matl=new Material(Color(0,0,0), Color(.7,.7,.7),
				       Color(0,0,.6), 20);
    matl=new Material(Color(0,0,0), Color(0,0,.6),
			  Color(0,0,0.5), 20);
    widget_id=0;
    streamline_id=0;
}

Streamline::Streamline(const Streamline& copy, int deep)
: Module(copy, deep),
  widgettype("source", id, this),
  markertype("marker", id, this), lineradius("lineradius", id, this),
  algorithm("algorithm", id, this), stepsize("stepsize", id, this),
  maxsteps("maxsteps", id, this)
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
    if(streamline_id)
	ogeom->delObj(streamline_id);
    VectorFieldHandle field;
    if(!infield->get(field))
	return;
    if(need_p1){
	Point min, max;
	field->get_bounds(min, max);
	p1=Interpolate(min, max, 0.5);
	double s=field->longest_dimension();
	p2=p1+Vector(1,0,0)*(s/10);
	p3=p1+Vector(0,1,0)*(s/10);
	p4=p1+Vector(1,1,0)*(s/10);
	Vector sp(p2-p1);
	slider1_dist=slider2_dist=sp.length()/10;
	need_p1=0;
    }
    widget_scale=0.01*field->longest_dimension();
    if(widgettype.get() != oldwidgettype){
	oldwidgettype=widgettype.get();
	if(widget_id)
	    ogeom->delObj(widget_id);
	widget=new GeomGroup;
	if(widgettype.get() == "Point"){
	    widget_p1=new GeomSphere(p1, 1*widget_scale);
	    widget_p1->set_matl(widget_point_matl);
	    GeomPick* pick=new GeomPick(this, Vector(1,0,0),
					Vector(0,1,0),
					Vector(0,0,1));
	    pick->set_highlight(widget_highlight_matl);
	    widget->set_pick(pick);
	    widget->add(widget_p1);
	} else if(widgettype.get() == "Line"){
	    widget_p1=new GeomSphere;
	    GeomPick* p=new GeomPick(this);
	    p->set_highlight(widget_highlight_matl);
	    p->set_cbdata((void*)1);
	    widget_p1->set_pick(p);
	    widget_p1->set_matl(widget_point_matl);
	    widget_p2=new GeomSphere;
	    p=new GeomPick(this);
	    p->set_highlight(widget_highlight_matl);
	    p->set_cbdata((void*)2);
	    widget_p2->set_pick(p);
	    widget_p2->set_matl(widget_point_matl);
	    widget_edge1=new GeomCylinder;
	    p=new GeomPick(this);
	    p->set_highlight(widget_highlight_matl);
	    p->set_cbdata((void*)3);
	    widget_edge1->set_pick(p);
	    widget_edge1->set_matl(widget_edge_matl);
	    widget_slider1=new GeomGroup;
	    widget_slider1body=new GeomCylinder;
	    widget_slider1cap1=new GeomDisc;
	    widget_slider1cap2=new GeomDisc;
	    widget_slider1->add(widget_slider1body);
	    widget_slider1->add(widget_slider1cap1);
	    widget_slider1->add(widget_slider1cap2);
	    p=new GeomPick(this);
	    p->set_highlight(widget_highlight_matl);
	    p->set_cbdata((void*)4);
	    widget_slider1->set_pick(p);
	    widget_slider1->set_matl(widget_slider_matl);
	    widget->add(widget_p1);
	    widget->add(widget_p2);
	    widget->add(widget_edge1);
	    widget->add(widget_slider1);
	    widget->set_pick(new GeomPick(this));
	} else if(widgettype.get() == "Square"){
	    NOT_FINISHED("Square widget");
	} else {
	    error("Unknown widget type: "+widgettype.get());
	}
	widget_id=ogeom->addObj(widget, widget_name);
    }
    // Upedate the widget...
    if(widgettype.get() == "Point"){
	widget_p1->move(p1, 1*widget_scale);
    } else if(widgettype.get() == "Line"){
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
	widget_p1->get_pick()->set_principal(spvec, v1, v2);
	widget_p2->get_pick()->set_principal(spvec, v1, v2);
	widget_edge1->get_pick()->set_principal(spvec, v1, v2);
	widget_slider1->get_pick()->set_principal(spvec);
    } else if(widgettype.get() == "Square"){
	NOT_FINISHED("Square widget");
    }
    GeomGroup* group=new GeomGroup;
    group->set_matl(matl);

    // Temporary algorithm...
    Array1<SLine*> slines;
    if(widgettype.get() == "Point"){
	slines.add(new SLine(group, p1));
    } else if(widgettype.get() == "Line"){
	Vector line(p2-p1);
	double l=line.length();
	for(double x=0;x<=l;x+=slider1_dist){
	    slines.add(new SLine(group, p1+line*(x/l)));
	}
    } else if(widgettype.get() == "Square"){
	Vector line1(p2-p1);
	Vector line2(p3-p1);
	double l1=line1.length();
	double l2=line2.length();
	for(double x=0;x<=l1;x+=slider1_dist){
	    Point p(p1+line1*(x/l1));
	    for(double y=0;y<=l2;y+=slider2_dist){
		slines.add(new SLine(group, p+line2*(y/l2)));
	    }
	}
    }

    int n=0;
    int groupid=0;
    int alg=(algorithm.get()=="RK4");
    for(int i=0;i<maxsteps.get();i++){
	int oldn=n;
	double ss=stepsize.get();
	for(int i=0;i<slines.size();i++)
	    n+=slines[i]->advance(field, alg, ss);
	if(abort_flag || n==oldn)
	    break;
	if(n>500){
	    if(!ogeom->busy()){
		n=0;
		if(groupid)
		    ogeom->delObj(groupid);
		groupid=ogeom->addObj(group->clone(), streamline_name);
		ogeom->flushViews();
	    }
	}
    }
    if(groupid)
	ogeom->delObj(groupid);
    if(group->size() == 0){
	delete group;
	streamline_id=0;
    } else {
	streamline_id=ogeom->addObj(group, streamline_name);
    }
}

void Streamline::geom_moved(int axis, double dist, const Vector& delta,
			    void* cbdata)
{
    if(widgettype.get() == "Point"){
	p1+=delta;
    } else if(widgettype.get() == "Line"){
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
	}
    } else if(widgettype.get() == "Square"){
	NOT_FINISHED("Square widget");
    } else {
	error("Unknown widgettype in Streamline");
    }
    if(!abort_flag){
	abort_flag=1;
	want_to_execute();
    }
}

SLine::SLine(GeomGroup* group, const Point& p)
: p(p), line(new GeomPolyline)
{
    group->add(line);
    outside=0;
    line->pts.add(p);
}


int SLine::advance(const VectorFieldHandle& field, int rk4,
		   double stepsize)
{
    if(outside)
	return 0;
    if(rk4){
	    NOT_FINISHED("SLine::advance for RK4");
    } else {
	Vector v;
	if(!field->interpolate(p, v)){
	    outside=1;
	    return 0;
	}
	p+=(v*stepsize);
    }
    line->pts.add(p);
    return 1;
}
