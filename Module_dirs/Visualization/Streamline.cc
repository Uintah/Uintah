/*
  TODO:
1) Line Widget
2) Square Widget
3) Ring Widget
4) Tubes DONE
5) Ribbons
6) Surfaces
7) colormap and sfield
8) RK4
9) PC
10) Stream function
11) Time animation DONE
12) Position animation
*/


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
#include <Math/Trig.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/ColormapPort.h>
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
#include <Geom/Material.h>
#include <Geom/TriStrip.h>
#include <Geom/Group.h>
#include <Geom/Pick.h>
#include <Geom/Polyline.h>
#include <Geom/Sphere.h>
#include <Geom/Switch.h>
#include <Geom/Tube.h>
#include <Geometry/Point.h>
#include <Widgets/RingWidget.h>
#include <Widgets/PointWidget.h>
#include <Widgets/ScaledFrameWidget.h>

#include <iostream.h>

class Streamline;

struct SLTracer {
    int inside;
    double s,t;
    Point p;
    Vector grad;

    SLTracer(const Point&, double s, double t,
	     const VectorFieldHandle& vfield);
    virtual ~SLTracer();
    virtual int advance(const VectorFieldHandle&, double stepsize)=0;
};

struct SLEulerTracer : public SLTracer {
    SLEulerTracer(const Point&, double s, double t,
		  const VectorFieldHandle& vfield);
    virtual ~SLEulerTracer();
    virtual int advance(const VectorFieldHandle&, double stepsize);
};    

struct SLRK4Tracer : public SLTracer {
    SLRK4Tracer(const Point&, double s, double t,
		const VectorFieldHandle& vfield);
    virtual ~SLRK4Tracer();
    virtual int advance(const VectorFieldHandle&, double stepsize);
};    

struct SLSource {
    Streamline* sl;
    BaseWidget* widget;
    clString name;
    int selected;
    SLSource(Streamline* sl, const clString& name);
    virtual ~SLSource();
    virtual void find(const Point&, const Vector& axis, double scale)=0;
    virtual Point trace_start(double s, double t)=0;
    virtual void get_n(int& s, int& t)=0;
    void select();
    void deselect();
};

struct SLPointSource : public SLSource {
    PointWidget* pw;
public:
    SLPointSource(Streamline* sl);
    virtual ~SLPointSource();
    virtual void find(const Point&, const Vector& axis, double scale);
    virtual Point trace_start(double s, double t);
    virtual void get_n(int& s, int& t);
};

class Streamline : public Module {
    VectorFieldIPort* infield;
    ColormapIPort* incolormap;
    ScalarFieldIPort* incolorfield;
    GeometryOPort* ogeom;

    Array1<SLSource*> sources;
    int first_execute;
    GeomGroup* widget_group;
    int widget_geomid;
    int geomid;

    // Manipulate groups for animation
    GeomGroup* get_group(double t);
    double anim_begin;
    double anim_end;
    int anim_steps;
    Array1<GeomGroup*> anim_groups;
    void make_anim_groups(const clString& animation, GeomGroup* group,
			  double total_time);

    virtual void geom_moved(int, double, const Vector&, void*);
    virtual void geom_release(void*);

    void make_tracers(SLSource*, Array1<SLTracer*>&,
		      const VectorFieldHandle& vfield);
    void do_streamline(SLSource* source, const VectorFieldHandle&,
		       double stepsize, int maxsteps,
		       const ScalarFieldHandle& sfield,
		       const ColormapHandle& cmap);
    void do_streamtube(SLSource* source, const VectorFieldHandle&,
		       double stepsize, int maxsteps,
		       const ScalarFieldHandle& sfield,
		       const ColormapHandle& cmap,
		       double tubesize);

    GeomVertex* get_vertex(const Point& p,
			   const ScalarFieldHandle& sfield,
			   const ColormapHandle& cmap);
public:
    CrowdMonitor widget_lock;

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
static clString module_name("Streamline");

Streamline::Streamline(const clString& id)
: Module(module_name, id, Filter), first_execute(1)
{
    // Create the input ports
    infield=new VectorFieldIPort(this, "Vector Field",
				 ScalarFieldIPort::Atomic);
    add_iport(infield);
    incolorfield=new ScalarFieldIPort(this, "Color Field",
				      ScalarFieldIPort::Atomic);
    add_iport(incolorfield);
    incolormap=new ColormapIPort(this, "Colormap", ColormapIPort::Atomic);
    add_iport(incolormap);

    // Create the output port
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
    geomid=0;

    // Set up one of each of the possible sources
    sources.add(new SLPointSource(this));
}

Streamline::Streamline(const Streamline& copy, int deep)
: Module(copy, deep)
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
    // Get the data from the ports...
    VectorFieldHandle vfield;
    if(!infield->get(vfield))
	return;
    ScalarFieldHandle sfield;
    incolorfield->get(sfield);
    ColormapHandle cmap;
    int have_cmap=incolormap->get(cmap);
    if(!have_cmap)
	sfield=0;

    // Setup the widgets - first time only
    if(first_execute){
	first_execute=0;
	widget_group=new GeomGroup;
	for(int i=0;i<sources.size();i++)
	    widget_group->add(sources[i]->widget->GetWidget());
	widget_geomid=ogeom->addObj(widget_group, widget_name, &widget_lock);
    }

    // Find the current source
    SLSource* source=0;
    clString sname;
    if(!get_tcl_stringvar(id, "source", sname)){
	error("Error reading source variable");
	return;
    }
    for(int i=0;i<sources.size();i++)
	if(sources[i]->name == sname)
	    source=sources[i];
    if(!source){
	error("Illegal name of source: "+sname);
	return;
    }

    // See if we need to find the field
    int need_find;
    
    if(!get_tcl_boolvar(id, "need_find", need_find)){
	error("Error reading need_find variable");
	return;
    }
    if(need_find){
	// Find the field
	Point min, max;
	vfield->get_bounds(min, max);
	Point cen(AffineCombination(min, 0.5, max, 0.5));
	Vector axis;
	if(!vfield->interpolate(cen, axis)){
	    // No field???
	    error("Can't find center of field");
	    return;
	}
	axis.normalize();
	double scale=vfield->longest_dimension()/10;
	set_tclvar(id, "need_find", "false");
	source->find(cen, axis, scale);
    }

    // Update the 3D Widget...
    widget_lock.write_lock();
    if(!source->selected){
	for(int i=0;i<sources.size();i++)
	    if(sources[i]->selected)
		sources[i]->deselect();
	source->select();
    }
    widget_lock.write_unlock();

    // Calculate Streamlines
    double stepsize;
    if(!get_tcl_doublevar(id, "stepsize", stepsize)){
	error("Error reading stepsize variable");
	return;
    }
    int maxsteps;
    if(!get_tcl_intvar(id, "maxsteps", maxsteps)){
	error("Error reading maxsteps variable");
	return;
    }

    // Set up animations
    clString animation;
    if(!get_tcl_stringvar(id, "animation", animation)){
	error("Error reading animation variable");
	return;
    }
    GeomGroup* group=new GeomGroup;
    make_anim_groups(animation, group, maxsteps*stepsize);

    clString markertype;
    if(!get_tcl_stringvar(id, "markertype", markertype)){
	error("Error reading markertype variable");
	return;
    }
    if(markertype == "Line"){
	do_streamline(source, vfield, stepsize, maxsteps,
		      sfield, cmap);
    } else if(markertype == "Tube"){
	double tubesize;
	if(!get_tcl_doublevar(id, "tubesize", tubesize)){
	    error("Error reading tubesize variable");
	    return;
	}
	do_streamtube(source, vfield, stepsize, maxsteps,
		      sfield, cmap, tubesize);
    } else if(markertype == "Ribbon"){
//	do_streamribbon();
    } else if(markertype == "Surface"){
//	do_streamsurface();
    } else {
	error("Unknown marketype");
	return;
    }
    // Remove the old and add the new
    if(geomid)
	ogeom->delObj(geomid);
    geomid=ogeom->addObj(group, module_name);

    // Flush it all out..
    ogeom->flushViews();
}

void Streamline::make_tracers(SLSource* source, Array1<SLTracer*>& tracers,
			      const VectorFieldHandle& vfield)
{
    int ns, nt;
    source->get_n(ns, nt);
    clString alg;
    if(!get_tcl_stringvar(id, "algorithm", alg)){
	error("Error reading algorithm variable");
	return;
    }
    enum ALGS {
	Euler, RK4,
    } alg_enum;
    if(alg == "Euler"){
	alg_enum=Euler;
    } else if(alg == "RK4"){
	alg_enum=RK4;
    } else {
	error("Unknown algorithm");
	return;
    }
    for(int s=0;s<ns;s++){
	for(int t=0;t<nt;t++){
	    Point start(source->trace_start(s, t));
	    switch(alg_enum){
	    case Euler:
		tracers.add(new SLEulerTracer(start, s, t, vfield));
		break;
	    case RK4:
		tracers.add(new SLRK4Tracer(start, s, t, vfield));
		break;
	    }
	}
    }
}

//
// Considering the current animation mode and parameters, return
// the appropriate object group for time t
//
GeomGroup* Streamline::get_group(double t)
{
    int n=int(anim_groups.size()*(t-anim_begin)/(anim_end-anim_begin));
    if(n<0)
	n=0;
    else if(n>=anim_groups.size())
	n=anim_groups.size()-1;
    return anim_groups[n];
}

void Streamline::make_anim_groups(const clString& animation, GeomGroup* top,
				  double total_time)
{
    if(animation == "Time"){
	if(!get_tcl_intvar(id, "anim_steps", anim_steps)){
	    error("Error reading anim_steps variable");	
	    return;
	}
	anim_begin=0;
	anim_end=total_time;
    } else {
	if(animation != "None"){
	    error("Unknown value in animation variable: "+animation);
	}
	anim_begin=0;
	anim_end=1;
	anim_steps=1;
    }
    anim_groups.remove_all();
    cerr << "anim_steps=" << anim_steps << endl;
    if(anim_steps == 1){
	anim_groups.add(top);
    } else {
	for(int i=0;i<anim_steps;i++){
	    double tbeg=0;
	    double tend=0;
	    if(animation == "Time"){
		tbeg=double(i)/double(anim_steps)*total_time;
		tend=1.e100;
	    }
	    GeomGroup* timegroup=new GeomGroup;
	    anim_groups.add(timegroup);
	    GeomTimeSwitch* timeswitch=new GeomTimeSwitch(timegroup, tbeg, tend);
	    top->add(timeswitch);
	}
    }
}

GeomVertex* Streamline::get_vertex(const Point& p,
				   const ScalarFieldHandle& sfield,
				   const ColormapHandle& cmap)
{
    if(sfield.get_rep()){
	double sval;
	if(sfield->interpolate(p, sval)){
	    MaterialHandle matl(cmap->lookup(sval));
	    return new GeomMVertex(p, matl);
	}
    }
    return new GeomVertex(p);
}

void Streamline::do_streamline(SLSource* source,
			       const VectorFieldHandle& field,
			       double stepsize, int maxsteps,
			       const ScalarFieldHandle& sfield,
			       const ColormapHandle& cmap)
{
    Array1<SLTracer*> tracers;
    make_tracers(source, tracers, field);
    Array1<GeomPolyline*> lines(tracers.size());
    double t=0;
    GeomGroup* group=0;
    int step=0;
    int ninside=1;
    while(step< maxsteps && ninside){
	step++;
	// If the group is discontinued, we have to start new polylines
	GeomGroup* newgroup=get_group(t);
	if(newgroup != group){
	    group=newgroup;
	    for(int i=0;i<lines.size();i++){
		if(tracers[i]->inside)
		    lines[i]=new GeomPolyline;
		else
		    lines[i]=0;
		group->add(lines[i]);
		GeomVertex* vtx=get_vertex(tracers[i]->p, sfield, cmap);
		lines[i]->add(vtx);
	    }
	}
	t+=stepsize;

	// Advance the tracers
	ninside=0;
	for(int i=0;i<tracers.size();i++){
	    int inside=tracers[i]->advance(field, stepsize);
	    if(inside){
		GeomVertex* vtx=get_vertex(tracers[i]->p, sfield, cmap);
		lines[i]->add(vtx);
	    }
	    ninside+=inside;
	}
    }
}

void Streamline::do_streamtube(SLSource* source,
			       const VectorFieldHandle& field,
			       double stepsize, int maxsteps,
			       const ScalarFieldHandle& sfield,
			       const ColormapHandle& cmap,
			       double tubesize)
{
    Array1<SLTracer*> tracers;
    make_tracers(source, tracers, field);
    Array1<GeomTube*> tubes(tracers.size());
    double t=0;
    GeomGroup* group=0;
    int step=0;
    int ninside=1;
    while(step< maxsteps && ninside){
	step++;
	// If the group is discontinued, we have to start new tubes
	GeomGroup* newgroup=get_group(t);
	if(newgroup != group){
	    group=newgroup;
	    for(int i=0;i<tubes.size();i++){
		if(tracers[i]->inside){
		    tubes[i]=new GeomTube;
		    group->add(tubes[i]);
		    Vector grad=tracers[i]->grad;
		    field->interpolate(tracers[i]->p, grad);
		    GeomVertex* vtx=get_vertex(tracers[i]->p, sfield, cmap);
		    tubes[i]->add(vtx, tubesize, grad);
		} else {
		    tubes[i]=0;
		}
	    }
	}
	t+=stepsize;

	// Advance the tracers
	ninside=0;
	for(int i=0;i<tracers.size();i++){
	    int inside=tracers[i]->advance(field, stepsize);
	    if(inside){
		Vector grad=tracers[i]->grad;
		field->interpolate(tracers[i]->p, grad);
		GeomVertex* vtx=get_vertex(tracers[i]->p, sfield, cmap);
		tubes[i]->add(vtx, tubesize, grad);
	    }
	    ninside+=inside;
	}
    }
}

void Streamline::geom_moved(int axis, double dist, const Vector& delta,
			    void* cbdata)
{
    for(int i=0;i<sources.size();i++){
	if(sources[i]->selected){
	    sources[i]->widget->geom_moved(axis, dist, delta, cbdata);
	    sources[i]->widget->execute();
	    ogeom->flushViews();
	    break;
	}
    }
}

void Streamline::geom_release(void*)
{
    if(!abort_flag){
	abort_flag=1;
	want_to_execute();
    }
}

SLSource::SLSource(Streamline* sl, const clString& name)
: sl(sl), name(name), selected(0), widget(0)
{
}

SLSource::~SLSource()
{
    if(widget)
	delete widget;
}

void SLSource::select()
{
    if(widget)
	widget->SetState(1);
    selected=1;
}

void SLSource::deselect()
{
    if(widget)
	widget->SetState(0);
    selected=0;
}

SLPointSource::SLPointSource(Streamline* sl)
: SLSource(sl, "Point")
{
    widget=pw=new PointWidget(sl, &sl->widget_lock, 1);
}

SLPointSource::~SLPointSource()
{
}

void SLPointSource::find(const Point& start, const Vector&, double scale)
{
    pw->SetPosition(start);
    pw->SetScale(scale/5);
}

Point SLPointSource::trace_start(double s, double t)
{
    s=t=0;
    return pw->GetVar(PointW_Point);
}

void SLPointSource::get_n(int& ns, int& nt)
{
    ns=1;
    nt=1;
}

SLTracer::SLTracer(const Point& p, double s, double t,
		   const VectorFieldHandle& vfield)
: p(p), s(s), t(t), inside(1)
{
    // Interpolate initial gradient
    if(!vfield->interpolate(p, grad))
	inside=0;
}

SLTracer::~SLTracer()
{
}

SLEulerTracer::SLEulerTracer(const Point& p, double s, double t,
			     const VectorFieldHandle& vfield)
: SLTracer(p, s, t, vfield)
{
}

SLEulerTracer::~SLEulerTracer()
{
}

int SLEulerTracer::advance(const VectorFieldHandle& vfield, double stepsize)
{
    if(!vfield->interpolate(p, grad)){
	inside=0;
    } else {
	p+=(grad*stepsize);
    }
    return inside;
}

SLRK4Tracer::SLRK4Tracer(const Point& p, double s, double t,
			 const VectorFieldHandle& vfield)
: SLTracer(p, s, t, vfield)
{
}

SLRK4Tracer::~SLRK4Tracer()
{
}

int SLRK4Tracer::advance(const VectorFieldHandle&, double)
{
    NOT_FINISHED("SLRK4Tracer::advance");
    return 0;
}
