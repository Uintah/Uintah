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
#include <Modules/Visualization/Streamsurf.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldUG.h>
#include <Geom/Geom.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Math/Trig.h>
#include <iostream.h>
#include <fstream.h>

static Module* make_Streamsurf(const clString& id)
{
    return new Streamsurf(id);
}

static RegisterModule db1("Fields", "Streamsurf", make_Streamsurf);
static RegisterModule db2("Visualization", "Streamsurf", make_Streamsurf);

static clString widget_name("Streamsurf Widget");
static clString streamsurf_name("Streamsurf");

struct SSLine {
    Point p;
    Point op;
    int outside;
    SSLine(const Point&);
    int advance(const VectorFieldHandle&, int alg, double);
    int have_normal;
    Vector normal;
};

struct SSurf {
    SSLine* l1;
    SSLine* l2;
    SSurf* left;
    SSurf* right;
    GeomTriStrip* tri;
    int split;
    int outside;
    void advance(const VectorFieldHandle&);
    void newtri(const VectorFieldHandle&, ObjGroup*);
    SSurf(const VectorFieldHandle&, ObjGroup*, SSLine*, SSLine*);
};

Streamsurf::Streamsurf(const clString& id)
: Module("Streamsurf", id, Filter)
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
#ifdef OLDUI
    MUI_choice* choice=new MUI_choice("Algorithm: ", &alg, MUI_widget::Immediate);
    choice->add_choice("Euler");
    choice->add_choice("4th Order Runge-Kutta");
    choice->add_choice("Stream Function");
    add_ui(choice);

    maxsteps=100;
    MUI_slider_int* islider=new MUI_slider_int("Max steps", &maxsteps,
					       MUI_widget::Immediate, 0);
    islider->set_minmax(0, 1000);
    add_ui(islider);

    maxangle=5;
    MUI_slider_real* slider=new MUI_slider_real("Maximum surface angle",
						&maxangle,
						MUI_widget::Immediate, 0);
    slider->set_minmax(0, 90);
    add_ui(slider);
    need_p1=1;
#endif

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
: Module(copy, deep)
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
    int groupid=0;

    Array1<SSLine*> slines;
    Point op1(p1);
    Point op2(p2);
    double d2=4*slider1_dist*slider1_dist;
    double ca=Cos(maxangle);
    double ss=slider1_dist/2;
    Vector line(op2-op1);
    double l=line.length();
    for(double x=0;x<=l;x+=slider1_dist)
	slines.add(new SSLine(op1+line*(x/l)));
    Array1<SSurf*> ssurfs(slines.size()-1);
    for(int i=0;i<ssurfs.size();i++)
	ssurfs[i]=new SSurf(field, group, slines[i], slines[i+1]);
    for(i=0;i<ssurfs.size();i++){
	if(i>0)
	    ssurfs[i]->left=ssurfs[i-1];
	if(i<ssurfs.size()-1)
	    ssurfs[i]->right=ssurfs[i+1];
    }
    int n=0;
    for(int iter=0;iter<maxsteps;iter++){
	update_progress(iter, maxsteps);
	if(abort_flag)
	    break;
	if(n > 500 && !ogeom->busy()){
	    n=0;
	    if(groupid)
		ogeom->delObj(groupid);
	    groupid=ogeom->addObj(group->clone(), streamsurf_name);
	    ogeom->flushViews();
	}
	int oldn=n;
	for(int i=0;i<slines.size();i++)
	    n+=slines[i]->advance(field, alg, ss);
	if(n==oldn)
	   break;
	for(i=0;i<ssurfs.size();i++)
	    ssurfs[i]->advance(field);
	for(i=0;i<ssurfs.size();i++)
	    ssurfs[i]->split=0;
	for(i=0;i<ssurfs.size();i++){
	    SSurf* surf=ssurfs[i];
	    if(!surf->split && !surf->outside){
		// Check lengths...
		Vector v1(surf->l2->p-surf->l1->p);
		if(v1.length2() > d2){
		    surf->split=1;
		}
		if(!surf->split && surf->right){
		    Vector v2(surf->right->l2->p-surf->l2->p);
		    v1.normalize();
		    v2.normalize();
		    double a=Dot(v1, v2);
		    if(a<ca){
			surf->split=1;
			surf->right->split=1;
		    }
		}
	    }
	}
	// Now do the splitting...
    	int size=ssurfs.size();
	for(i=0;i<size;i++){
	    SSurf* surf=ssurfs[i];
	    if(surf->split){
		Point newp(Interpolate(surf->l1->op, surf->l2->op, 0.5));
		SSLine* newline=new SSLine(newp);
		// advance it...
		for(int ii=0;ii<=iter;ii++)
		    newline->advance(field, alg, ss);
		// Save the advanced point for this - use the interpolated
		// one - to avoid holes
		Point save(newline->p);
		newline->p=Interpolate(surf->l1->p, surf->l2->p, 0.5);

		SSurf* newsurf=new SSurf(field, group, newline, surf->l2);
		newsurf->right=surf->right;
		newsurf->left=surf;
		surf->right=newsurf;
		surf->l2=newline;
		surf->newtri(field, group);
		newline->p=save;
		ssurfs.add(newsurf);
		slines.add(newline);
	    }
	}
    }
    if(groupid)
	ogeom->delObj(groupid);
    for(i=0;i<ssurfs.size();i++)
	delete ssurfs[i];
    for(i=0;i<slines.size();i++)
	delete slines[i];
    if(group->size() == 0){
	delete group;
	streamsurf_id=0;
    } else {
	streamsurf_id=ogeom->addObj(group, streamsurf_name);
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

SSLine::SSLine(const Point& p)
: p(p), op(p)
{
    outside=0;
    have_normal=0;
}

int SSLine::advance(const VectorFieldHandle& field, int alg,
		   double stepsize)
{
    if(outside)
	return 0;
    have_normal=0;
    switch(alg){
    case Streamsurf::AEuler:
	{
	    Vector v;
	    if(!field->interpolate(p, v)){
		outside=1;
		return 0;
	    }
	    if(v.length2() > 0){
		v.normalize();
		p+=(v*stepsize);
	    } else {
		// Stagnation...
		outside=1;
	    }
	}
	break;
    case Streamsurf::ARK4:
	{
	    NOT_FINISHED("SSLine::advance for RK4");
	}
	break;
    case Streamsurf::AStreamFunction:
	{
	    NOT_FINISHED("SSLine::advance for Stream Function");
	}
	break;
    }
    return 1;
}

void SSurf::advance(const VectorFieldHandle& field)
{
    if(l1->outside || l2->outside){
	outside=1;
	return;
    }
    if(!l1->have_normal){
	Vector v1;
	if(left && !left->l1->outside){
	    v1=Vector(l2->p-left->l1->p);
	} else {
	    v1=Vector(l2->p-l1->p);
	}
	Vector v2;
	if(!field->interpolate(l1->p, v2)){
	    outside=1;
	    l1->outside=1;
	    if(left)left->outside=1;
	    return;
	}
	l1->normal=Cross(v2, v1);
	l1->have_normal=1;
    }
    if(!l2->have_normal){
	Vector v1;
	if(right && !right->l2->outside){
	    v1=Vector(right->l2->p-l1->p);
	} else {
	    v1=Vector(l2->p-l1->p);
	}
	Vector v2;
	if(!field->interpolate(l2->p, v2)){
	    outside=1;
	    l2->outside=1;
	    if(right)right->outside=1;
	    return;
	}
	l2->normal=Cross(v2, v1);
	l2->have_normal=1;
    }
    tri->add(l1->p, l1->normal);
    tri->add(l2->p, l2->normal);
}

SSurf::SSurf(const VectorFieldHandle& field,
	     ObjGroup* group, SSLine* l1, SSLine* l2)
: l1(l1), l2(l2), tri(new GeomTriStrip)
{
    outside=0;
    group->add(tri);
    left=right=0;
    advance(field);
}

void SSurf::newtri(const VectorFieldHandle& field, ObjGroup* group)
{
    tri=new GeomTriStrip;
    group->add(tri);
    advance(field);
}
