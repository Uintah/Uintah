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
#include <NotFinished.h>
#include <Geometry/Point.h>
#include <iostream.h>
#include <fstream.h>

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
    GeomPolyLine* line;
    int outside;
public:
    SLine(ObjGroup*, const Point&);
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
    abort_flag=0;
    if(streamline_id)
	ogeom->delObj(streamline_id);
    VectorFieldHandle field;
    if(!infield->get(field))
	return;
    if(need_p1){
	Point min, max;
	field->get_bounds(min, max);
	cerr << "min=" << min.string() << endl;
	cerr << "max=" << max.string() << endl;
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
    cerr << "longest_dimension=" << field->longest_dimension() << endl;
    if(widgettype.get() != oldwidgettype){
	oldwidgettype=widgettype.get();
	cerr << "Rebuilding widget" << endl;
	if(widget_id)
	    ogeom->delObj(widget_id);
	widget=new ObjGroup;
	if(widgettype.get() == "Point"){
	    widget_p1=new GeomSphere(p1, 1*widget_scale);
	    widget_p1->set_matl(widget_point_matl);
	    GeomPick* pick=new GeomPick(this, Vector(1,0,0),
					Vector(0,1,0),
					Vector(0,0,1));
	    pick->set_highlight(widget_highlight_matl);
	    widget->pick=pick;
	    widget->add(widget_p1);
	} else if(widgettype.get() == "Line"){
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
	widget_p1->pick->set_principal(spvec, v1, v2);
	widget_p2->pick->set_principal(spvec, v1, v2);
	widget_edge1->pick->set_principal(spvec, v1, v2);
	widget_slider1->pick->set_principal(spvec);
    } else if(widgettype.get() == "Square"){
	NOT_FINISHED("Square widget");
    }
    ObjGroup* group=new ObjGroup;
    group->set_matl(new MaterialProp(Color(0,0,0), Color(0,0,.6),
				     Color(0,0,0.5), 20));

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
    cerr << "maxsteps=" << maxsteps.get() << endl;
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
    cerr << "Moved called..." << endl;
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
    }
    if(!abort_flag){
	abort_flag=1;
	want_to_execute();
    }
}

SLine::SLine(ObjGroup* group, const Point& p)
: p(p), line(new GeomPolyLine)
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
