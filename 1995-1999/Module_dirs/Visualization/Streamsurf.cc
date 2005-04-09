
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

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldUG.h>
#include <Datatypes/VectorField.h>
#include <Datatypes/VectorFieldPort.h>
#include <Geom/Cylinder.h>
#include <Geom/Disc.h>
#include <Geom/Geom.h>
#include <Geom/Group.h>
#include <Geom/Material.h>
#include <Geom/Pick.h>
#include <Geom/Sphere.h>
#include <Geom/TriStrip.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Math/Trig.h>
#include <TCL/TCLvar.h>
#include <iostream.h>

class Streamsurf : public Module {
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
    TCLvardouble maxangle;

    TCLdouble range_min;
    TCLdouble range_max;

    int need_p1;
    int need_widget;
    Point p1;
    Point p2;
    double slider1_dist;

    GeomGroup* widget;
    GeomSphere* widget_p1;
    GeomSphere* widget_p2;
    GeomCylinder* widget_edge1;
    GeomGroup* widget_slider1;
    GeomCylinder* widget_slider1body;
    GeomDisc* widget_slider1cap1;
    GeomDisc* widget_slider1cap2;
    GeomPick* pick_p1;
    GeomPick* pick_p2;
    GeomPick* pick_edge1;
    GeomPick* pick_slider1;
    int widget_id;
    double widget_scale;

    int streamsurf_id;

    MaterialHandle widget_point_matl;
    MaterialHandle widget_edge_matl;
    MaterialHandle widget_slider_matl;
    MaterialHandle widget_highlight_matl;
    MaterialHandle matl;

    virtual void geom_moved(int, double, const Vector&, void*);
    virtual void geom_release(void*);
public:
    Streamsurf(const clString& id);
    Streamsurf(const Streamsurf&, int deep);
    virtual ~Streamsurf();
    virtual Module* clone(int deep);
    virtual void execute();
};

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
    void newtri(const VectorFieldHandle&, GeomGroup*);
    SSurf(const VectorFieldHandle&, GeomGroup*, SSLine*, SSLine*);
};

Streamsurf::Streamsurf(const clString& id)
: Module("Streamsurf", id, Filter),
  widgettype("source", id, this),
  markertype("markertype", id, this), lineradius("lineradius", id, this),
  algorithm("algorithm", id, this), stepsize("stepsize", id, this),
  maxsteps("maxsteps", id, this), range_min("range_min", id, this),
  range_max("range_max", id, this), maxangle("maxangle", id, this)
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


    widget_point_matl=new Material(Color(0,0,0), Color(.54, .60, 1),
				   Color(.5,.5,.5), 20);
    widget_edge_matl=new Material(Color(0,0,0), Color(.54, .60, .66),
				  Color(.5,.5,.5), 20);
    widget_slider_matl=new Material(Color(0,0,0), Color(.83, .60, .66),
				    Color(.5,.5,.5), 20);
    widget_highlight_matl=new Material(Color(0,0,0), Color(.7,.7,.7),
				       Color(0,0,.6), 20);
    need_widget=1;
    widget_id=0;
    streamsurf_id=0;
}

Streamsurf::Streamsurf(const Streamsurf& copy, int deep)
: Module(copy, deep),
  widgettype("source", id, this),
  markertype("markertype", id, this), lineradius("lineradius", id, this),
  algorithm("algorithm", id, this), stepsize("stepsize", id, this),
  maxsteps("maxsteps", id, this), range_min("range_min", id, this),
  range_max("range_max", id, this), maxangle("maxangle", id, this)
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
    widget_scale=0.05*field->longest_dimension();
    if(widgettype.get() != oldwidgettype){
	oldwidgettype=widgettype.get();
	if(widget_id)
	    ogeom->delObj(widget_id);
	widget=new GeomGroup;
	if(widgettype.get() == "Point"){
	    widget_p1=new GeomSphere(p1, 1*widget_scale);
	    GeomMaterial* matl=new GeomMaterial(widget_p1, widget_point_matl);
	    GeomPick* pick=new GeomPick(matl, this, Vector(1,0,0),
					Vector(0,1,0),
					Vector(0,0,1));
	    pick->set_highlight(widget_highlight_matl);
	    widget->add(pick);
	} else if(widgettype.get() == "Line"){
	    GeomGroup* pts=new GeomGroup;
	    widget_p1=new GeomSphere(p1, 1*widget_scale);
	    pick_p1=new GeomPick(widget_p1, this);
	    pick_p1->set_highlight(widget_highlight_matl);
	    pick_p1->set_cbdata((void*)1);
	    pts->add(pick_p1);
	    widget_p2=new GeomSphere(p2, 1*widget_scale);
	    pick_p2=new GeomPick(widget_p2, this);
	    pick_p2->set_highlight(widget_highlight_matl);
	    pick_p2->set_cbdata((void*)2);
	    pts->add(pick_p2);
	    GeomMaterial* m1=new GeomMaterial(pts, widget_point_matl);
	    widget_edge1=new GeomCylinder(p1, p2, 0.5*widget_scale);
	    pick_edge1=new GeomPick(widget_edge1, this);
	    pick_edge1->set_highlight(widget_highlight_matl);
	    pick_edge1->set_cbdata((void*)3);
	    GeomMaterial* m2=new GeomMaterial(pick_edge1, widget_edge_matl);
	    widget_slider1=new GeomGroup;
	    Vector spvec(p2-p1);
	    spvec.normalize();
	    Point sp(p1+spvec*slider1_dist);
	    Point sp2(sp+spvec*(widget_scale*0.5));
	    Point sp1(sp-spvec*(widget_scale*0.5));
	    widget_slider1body=new GeomCylinder(sp1, sp2, 1*widget_scale);
	    widget_slider1cap1=new GeomDisc(sp2, spvec, 1*widget_scale);
	    widget_slider1cap2=new GeomDisc(sp1, -spvec, 1*widget_scale);
	    widget_slider1->add(widget_slider1body);
	    widget_slider1->add(widget_slider1cap1);
	    widget_slider1->add(widget_slider1cap2);
	    pick_slider1=new GeomPick(widget_slider1, this);
	    pick_slider1->set_highlight(widget_highlight_matl);
	    pick_slider1->set_cbdata((void*)4);
	    GeomMaterial* m3=new GeomMaterial(pick_slider1, widget_slider_matl);
	    widget->add(m1);
	    widget->add(m2);
	    widget->add(m3);
	    Vector v1,v2;
	    spvec.find_orthogonal(v1, v2);
	    pick_p1->set_principal(spvec, v1, v2);
	    pick_p2->set_principal(spvec, v1, v2);
	    pick_edge1->set_principal(spvec, v1, v2);
	    pick_slider1->set_principal(spvec);
	} else if(widgettype.get() == "Square"){
	    NOT_FINISHED("Square widget");
	} else {
	    error("Unknown widget type: "+widgettype.get());
	}
	widget_id=ogeom->addObj(widget, widget_name);
    }
    GeomGroup* group=new GeomGroup;
    GeomMaterial* matlobj=new GeomMaterial(group, matl);
    int groupid=0;

    Array1<SSLine*> slines;
    Point op1(p1);
    Point op2(p2);
    double d2=4*slider1_dist*slider1_dist;
    double ca=Cos(maxangle.get());
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
    int alg=(algorithm.get()=="RK4");
    for(int iter=0;iter<maxsteps.get();iter++){
	update_progress(iter, maxsteps.get());
	if(abort_flag)
	    break;
	if(n > 500 && !ogeom->busy()){
	    n=0;
	    if(groupid)
		ogeom->delObj(groupid);
	    groupid=ogeom->addObj(matlobj->clone(), streamsurf_name);
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
	streamsurf_id=ogeom->addObj(matlobj, streamsurf_name);
    }
}

void Streamsurf::geom_moved(int axis, double dist, const Vector& delta,
			    void* cbdata)
{
    if(widgettype.get() == "Point"){
	p1+=delta;
	widget_p1->move(p1, 1*widget_scale);
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
	// Reconfigure...
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
	pick_p1->set_principal(spvec, v1, v2);
	pick_p2->set_principal(spvec, v1, v2);
	pick_edge1->set_principal(spvec, v1, v2);
	pick_slider1->set_principal(spvec);
	widget->reset_bbox();
    } else if(widgettype.get() == "Square"){
	NOT_FINISHED("Square widget");
    } else {
	error("Unknown widgettype in Streamline");
    }
}

void Streamsurf::geom_release(void*)
{
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

int SSLine::advance(const VectorFieldHandle& field, int rk4,
		   double stepsize)
{
    if(outside)
	return 0;
    have_normal=0;
    if(rk4){
	NOT_FINISHED("SSLine::advance for RK4");
    } else {
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
	     GeomGroup* group, SSLine* l1, SSLine* l2)
: l1(l1), l2(l2), tri(new GeomTriStrip)
{
    outside=0;
    group->add(tri);
    left=right=0;
    advance(field);
}

void SSurf::newtri(const VectorFieldHandle& field, GeomGroup* group)
{
    tri=new GeomTriStrip;
    group->add(tri);
    advance(field);
}
