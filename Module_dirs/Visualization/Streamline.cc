/*
  TODO:
1) Delete tracers!
2) Square Widget
6) Surfaces IMPLEMENTED, not tested
8) RK4
9) PC
10) Stream function
13) Multiple sources  **
Keep them registered...
14) Redo user interface **
hook up user interface buttons
15) direction switch - downstream, upstream, both
16) update_progress
17) watch abort_flag
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

#include <Classlib/HashTable.h>
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
#include <Widgets/GaugeWidget.h>
#include <Widgets/PointWidget.h>
#include <Widgets/RingWidget.h>
#include <Widgets/ScaledFrameWidget.h>

#include <iostream.h>

class Streamline;

struct SLTracer {
    Point p;
    double s,t;
    int inside;
    Vector grad;

    SLTracer(const Point&, double s, double t,
	     const VectorFieldHandle& vfield);
    virtual ~SLTracer();
    virtual int advance(const VectorFieldHandle&, double stepsize, int skip)=0;
};

struct SLEulerTracer : public SLTracer {
    SLEulerTracer(const Point&, double s, double t,
		  const VectorFieldHandle& vfield);
    virtual ~SLEulerTracer();
    virtual int advance(const VectorFieldHandle&, double stepsize, int skip);
};    

struct SLRK4Tracer : public SLTracer {
    SLRK4Tracer(const Point&, double s, double t,
		const VectorFieldHandle& vfield);
    virtual ~SLRK4Tracer();
    virtual int advance(const VectorFieldHandle&, double stepsize, int skip);
};    

struct SLSource {
    Streamline* sl;
    clString name;
    BaseWidget* widget;
    int selected;
    SLSource(Streamline* sl, const clString& name);
    virtual ~SLSource();
    virtual void find(const Point&, const Vector& axis, double scale)=0;
    virtual Point trace_start(double s, double t)=0;
    virtual void get_n(int& s, int& t)=0;
    virtual Vector ribbon_direction(double s, double t, const Point&,
				    const VectorFieldHandle& vfield)=0;
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
    virtual Vector ribbon_direction(double s, double t, const Point&,
				    const VectorFieldHandle& vfield);
};

struct SLLineSource : public SLSource {
    GaugeWidget* gw;
public:
    SLLineSource(Streamline* sl);
    virtual ~SLLineSource();
    virtual void find(const Point&, const Vector& axis, double scale);
    virtual Point trace_start(double s, double t);
    virtual void get_n(int& s, int& t);
    virtual Vector ribbon_direction(double s, double t, const Point&,
				    const VectorFieldHandle& vfield);
};

struct SLRingSource : public SLSource {
    RingWidget* rw;
public:
    SLRingSource(Streamline* sl);
    virtual ~SLRingSource();
    virtual void find(const Point&, const Vector& axis, double scale);
    virtual Point trace_start(double s, double t);
    virtual void get_n(int& s, int& t);
    virtual Vector ribbon_direction(double s, double t, const Point&,
				    const VectorFieldHandle& vfield);
};

struct SLSourceInfo {
    int sid;
    GeomGroup* widget_group;
    int widget_geomid;
    int geomid;

    Array1<SLSource*> sources;
    SLSource* source;

    int need_find;
    // Manipulate groups for animation
    GeomGroup* get_group(double t);
    double anim_begin;
    double anim_end;
    int anim_steps;
    Array1<GeomGroup*> anim_groups;
    void make_anim_groups(const clString& animation, GeomGroup* group,
			  double total_time, int timesteps);

    SLSourceInfo(int sid, Streamline* module, GeometryOPort* ogeom);
    void pick_source(const clString& source, const VectorFieldHandle& vfield,
		     Streamline* module);
};

class Streamline : public Module {
    VectorFieldIPort* infield;
    ColormapIPort* incolormap;
    ScalarFieldIPort* incolorfield;
    GeometryOPort* ogeom;

    int first_execute;

    HashTable<int, SLSourceInfo*> source_info;

    virtual void geom_moved(int, double, const Vector&, void*);
    virtual void geom_release(void*);

    enum ALGS {
	Euler, RK4,
    };

    void make_tracers(SLSource*, Array1<SLTracer*>&,
		      const VectorFieldHandle& vfield,
		      ALGS alg_enum, double width=0);
    SLTracer* make_tracer(SLSource* source, double s, double t,
			  const VectorFieldHandle& vfield,
			  ALGS alg_enum);
    SLTracer* make_tracer(const Point& p,
			  double s, double t,
			  const VectorFieldHandle& vfield,
			  ALGS alg_enum);
    void do_streamline(SLSourceInfo* si, const VectorFieldHandle&,
		       double stepsize, int maxsteps, int skip,
		       const ScalarFieldHandle& sfield,
		       const ColormapHandle& cmap, ALGS alg_enum);
    void do_streamtube(SLSourceInfo* si, const VectorFieldHandle&,
		       double stepsize, int maxsteps, int skip,
		       const ScalarFieldHandle& sfield,
		       const ColormapHandle& cmap, ALGS alg_enum,
		       double tubesize);
    void do_streamribbon(SLSourceInfo* si, const VectorFieldHandle&,
			 double stepsize, int maxsteps, int skip,
			 const ScalarFieldHandle& sfield,
			 const ColormapHandle& cmap, ALGS alg_enum,
			 double ribbonsize);
    void do_streamsurface(SLSourceInfo* si, const VectorFieldHandle&,
			  double stepsize, int maxsteps, int skip,
			  const ScalarFieldHandle& sfield,
			  const ColormapHandle& cmap, ALGS alg_enum,
			  double maxbend);
    inline SLTracer* left_tracer(const Array1<SLTracer*> tracers, int i){
	return tracers[i*2];
    }
    inline SLTracer* right_tracer(const Array1<SLTracer*> tracers, int i){
	return tracers[i*2+1];
    }
	    
    GeomVertex* get_vertex(const Point& p,
			   const ScalarFieldHandle& sfield,
			   const ColormapHandle& cmap);
    GeomVertex* get_vertex(const Point& p,
			   const ScalarFieldHandle& sfield,
			   const ColormapHandle& cmap,
			   const Vector& normal);
public:
    CrowdMonitor widget_lock;

    Streamline(const clString& id);
    Streamline(const Streamline&, int deep);
    virtual ~Streamline();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual void tcl_command(TCLArgs& args, void* userdata);
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

    HashTableIter<int, SLSourceInfo*> iter(&source_info);
    for(iter.first();iter.ok();++iter){
	// Find the current source...
	SLSourceInfo* si=iter.get_data();
	clString sidstr(id+"-"+to_string(si->sid));
	cerr << "Processing source: " << si->sid << endl;

	si->source=0;
	clString sname;
	if(!get_tcl_stringvar(sidstr, "source", sname)){
	    error("Error reading source variable");
	    return;
	}

	si->pick_source(sname, vfield, this);

	// Calculate Streamlines
	double stepsize;
	if(!get_tcl_doublevar(sidstr, "stepsize", stepsize)){
	    error("Error reading stepsize variable");
	    return;
	}
	int maxsteps;
	if(!get_tcl_intvar(sidstr, "maxsteps", maxsteps)){
	    error("Error reading maxsteps variable");
	    return;
	}
	int skip;
	if(!get_tcl_intvar(sidstr, "skip", skip)){
	    error("Error reading skip variable");
	    return;
	}
	maxsteps/=skip;

	// Set up animations
	clString animation;
	if(!get_tcl_stringvar(sidstr, "animation", animation)){
	    error("Error reading animation variable");
	    return;
	}
	int anim_timesteps;
	if(!get_tcl_intvar(sidstr, "anim_timesteps", anim_timesteps)){
	    error("Error reading anim_timesteps variable");
	    return;
	}
	GeomGroup* group=new GeomGroup;
	si->make_anim_groups(animation, group, maxsteps*stepsize,
			     anim_timesteps);

	clString markertype;
	if(!get_tcl_stringvar(sidstr, "markertype", markertype)){
	    error("Error reading markertype variable");
	    return;
	}

	// Figure out which algorithm to use
	clString alg;
	if(!get_tcl_stringvar(sidstr, "algorithm", alg)){
	    error("Error reading algorithm variable");
	    return;
	}
	ALGS alg_enum;
	if(alg == "Euler"){
	    alg_enum=Euler;
	} else if(alg == "RK4"){
	    alg_enum=RK4;
	} else {
	    error("Unknown algorithm");
	    return;
	}

	// Do it...
	if(markertype == "Line"){
	    do_streamline(si, vfield, stepsize, maxsteps, skip,
			  sfield, cmap, alg_enum);
	} else if(markertype == "Tube"){
	    double tubesize;
	    if(!get_tcl_doublevar(sidstr, "tubesize", tubesize)){
		error("Error reading tubesize variable");
		return;
	    }
	    do_streamtube(si, vfield, stepsize, maxsteps, skip,
			  sfield, cmap, alg_enum, tubesize);
	} else if(markertype == "Ribbon"){
	    double ribbonsize;
	    if(!get_tcl_doublevar(sidstr, "ribbonsize", ribbonsize)){
		error("Error reading ribbonsize variable");
		return;
	    }
	    do_streamribbon(si, vfield, stepsize, maxsteps, skip,
			    sfield, cmap, alg_enum, ribbonsize);
	} else if(markertype == "Surface"){
	    double maxbend;
	    if(!get_tcl_doublevar(sidstr, "maxbend", maxbend)){
		error("Error reading ribbonsize variable");
		return;
	    }
	    do_streamsurface(si, vfield, stepsize, maxsteps, skip,
			     sfield, cmap, alg_enum, maxbend);
	} else {
	    error("Unknown marketype");
	    return;
	}
	// Remove the old and add the new
	if(si->geomid)
	    ogeom->delObj(si->geomid);
	si->geomid=ogeom->addObj(group, module_name+to_string(si->sid));

    }
    // Flush it all out..
    ogeom->flushViews();
}

SLTracer* Streamline::make_tracer(SLSource* source, double s, double t,
				  const VectorFieldHandle& vfield,
				  ALGS alg_enum)
{
    Point start(source->trace_start(s, t));
    return make_tracer(start, s, t, vfield, alg_enum);
}

SLTracer* Streamline::make_tracer(const Point& start,
				  double s, double t,
				  const VectorFieldHandle& vfield,
				  ALGS alg_enum)
{
    switch(alg_enum){
    case Euler:
	return new SLEulerTracer(start, s, t, vfield);
    case RK4:
	return new SLRK4Tracer(start, s, t, vfield);
    }
    return 0;
}

void Streamline::make_tracers(SLSource* source, Array1<SLTracer*>& tracers,
			      const VectorFieldHandle& vfield,
			      ALGS alg_enum,
			      double ribbonsize)
{
    int ns, nt;
    source->get_n(ns, nt);
    for(int s=0;s<ns;s++){
	for(int t=0;t<nt;t++){
	    Point start(source->trace_start(s, t));
	    if(ribbonsize == 0){
		tracers.add(make_tracer(start, s, t, vfield, alg_enum));
	    } else {
		Vector v(source->ribbon_direction(s, t, start, vfield)
			 *(ribbonsize/2));
		tracers.add(make_tracer(start-v, s, t, vfield, alg_enum));
		tracers.add(make_tracer(start+v, s, t, vfield, alg_enum));
	    }
	}
    }
}

//
// Considering the current animation mode and parameters, return
// the appropriate object group for time t
//
GeomGroup* SLSourceInfo::get_group(double t)
{
    int n=int(anim_groups.size()*(t-anim_begin)/(anim_end-anim_begin));
    if(n<0)
	n=0;
    else if(n>=anim_groups.size())
	n=anim_groups.size()-1;
    return anim_groups[n];
}

void SLSourceInfo::make_anim_groups(const clString& animation, GeomGroup* top,
				    double total_time, int timesteps)
{
    if(animation == "Time"){
	anim_steps=timesteps;
	anim_begin=0;
	anim_end=total_time;
    } else {
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

GeomVertex* Streamline::get_vertex(const Point& p,
				   const ScalarFieldHandle& sfield,
				   const ColormapHandle& cmap,
				   const Vector& normal)
{
    if(sfield.get_rep()){
	double sval;
	if(sfield->interpolate(p, sval)){
	    MaterialHandle matl(cmap->lookup(sval));
	    return new GeomNMVertex(p, normal, matl);
	}
    }
    return new GeomNVertex(p, normal);
}

void Streamline::do_streamline(SLSourceInfo* si,
			       const VectorFieldHandle& field,
			       double stepsize, int maxsteps, int skip,
			       const ScalarFieldHandle& sfield,
			       const ColormapHandle& cmap,
			       ALGS alg_enum)
{
    SLSource* source=si->source;
    Array1<SLTracer*> tracers;
    make_tracers(source, tracers, field, alg_enum);
    Array1<GeomPolyline*> lines(tracers.size());
    double t=0;
    GeomGroup* group=0;
    int step=0;
    int ninside=1;
    while(step< maxsteps && ninside){
	step++;
	// If the group is discontinued, we have to start new polylines
	GeomGroup* newgroup=si->get_group(t);
	if(newgroup != group){
	    group=newgroup;
	    for(int i=0;i<lines.size();i++){
		if(tracers[i]->inside){
		    lines[i]=new GeomPolyline;
		    group->add(lines[i]);
		    GeomVertex* vtx=get_vertex(tracers[i]->p, sfield, cmap);
		    lines[i]->add(vtx);
		} else {
		    lines[i]=0;
		}
	    }
	}
	t+=stepsize;

	// Advance the tracers
	ninside=0;
	for(int i=0;i<tracers.size();i++){
	    int inside=tracers[i]->advance(field, stepsize, skip);
	    if(inside){
		GeomVertex* vtx=get_vertex(tracers[i]->p, sfield, cmap);
		lines[i]->add(vtx);
	    }
	    ninside+=inside;
	}
    }
}

void Streamline::do_streamtube(SLSourceInfo* si,
			       const VectorFieldHandle& field,
			       double stepsize, int maxsteps, int skip,
			       const ScalarFieldHandle& sfield,
			       const ColormapHandle& cmap,
			       ALGS alg_enum,
			       double tubesize)
{
    Array1<SLTracer*> tracers;
    SLSource* source=si->source;
    make_tracers(source, tracers, field, alg_enum);
    Array1<GeomTube*> tubes(tracers.size());
    double t=0;
    GeomGroup* group=0;
    int step=0;
    int ninside=1;
    while(step< maxsteps && ninside){
	step++;
	// If the group is discontinued, we have to start new tubes
	GeomGroup* newgroup=si->get_group(t);
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
	    int inside=tracers[i]->advance(field, stepsize, skip);
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

void Streamline::do_streamribbon(SLSourceInfo* si,
				 const VectorFieldHandle& field,
				 double stepsize, int maxsteps, int skip,
				 const ScalarFieldHandle& sfield,
				 const ColormapHandle& cmap,
				 ALGS alg_enum,
				 double ribbonsize)

{
    Array1<SLTracer*> tracers;
    SLSource* source=si->source;
    make_tracers(source, tracers, field, alg_enum, ribbonsize);
    Array1<GeomTriStrip*> ribbons(tracers.size()/2);
    double t=0;
    GeomGroup* group=0;
    int step=0;
    int ninside=1;
    while(step< maxsteps && ninside){
	step++;
	// If the group is discontinued, we have to start new ribbons
	GeomGroup* newgroup=si->get_group(t);
	if(newgroup != group){
	    group=newgroup;
	    for(int i=0;i<ribbons.size();i++){
		if(left_tracer(tracers, i)->inside
		   && right_tracer(tracers, i)->inside){
		    ribbons[i]=new GeomTriStrip;
		    group->add(ribbons[i]);
		    // Compute vector between points
		    Point p1(left_tracer(tracers, i)->p);
		    Point p2(right_tracer(tracers, i)->p);
		    Vector vec(p2-p1);
		    // Left tracer
		    Vector grad=left_tracer(tracers, i)->grad;
		    field->interpolate(left_tracer(tracers, i)->p, grad);
		    Vector n1(Cross(vec, grad));
		    GeomVertex* vtx=get_vertex(left_tracer(tracers, i)->p,
					       sfield, cmap, n1);
		    ribbons[i]->add(vtx);

		    // Right tracer
		    grad=right_tracer(tracers, i)->grad;
		    field->interpolate(right_tracer(tracers, i)->p, grad);
		    Vector n2(Cross(vec, grad));
		    vtx=get_vertex(right_tracer(tracers, i)->p,
				   sfield, cmap, n2);
		    ribbons[i]->add(vtx);
		} else {
		    ribbons[i]=0;
		}
	    }
	}
	t+=stepsize;

	// Advance the tracers
	ninside=0;
	for(int i=0;i<ribbons.size();i++){
	    int linside=left_tracer(tracers, i)->advance(field, stepsize, skip);
	    int rinside=right_tracer(tracers, i)->advance(field, stepsize, skip);
	    if(linside && rinside){
		// Normalize the distance between them...
		Point p1(left_tracer(tracers, i)->p);
		Point p2(right_tracer(tracers, i)->p);
		Vector vec(p2-p1);
		double l=vec.normalize();
		double d1=(l-ribbonsize)/2;
		double d2=(l+ribbonsize)/2;
		left_tracer(tracers, i)->p=p1+vec*d1;
		right_tracer(tracers, i)->p=p1+vec*d2;

		// Get left vertex
		Vector grad=left_tracer(tracers, i)->grad;
		field->interpolate(left_tracer(tracers, i)->p, grad);
		Vector n1(Cross(vec, grad));
		GeomVertex* lvtx=get_vertex(left_tracer(tracers, i)->p,
					   sfield, cmap, n1);

		// Get right vertex
		grad=right_tracer(tracers, i)->grad;
		field->interpolate(right_tracer(tracers, i)->p, grad);
		Vector n2(Cross(vec, grad));
		GeomVertex* rvtx=get_vertex(right_tracer(tracers, i)->p,
			       sfield, cmap, n2);

		ribbons[i]->add(lvtx);
		ribbons[i]->add(rvtx);
	    } else {
		// Make sure that we stop doing work for both tracers...
		left_tracer(tracers, i)->inside=0;
		right_tracer(tracers, i)->inside=0;
	    }
	    ninside+=(linside && rinside);
	}
    }
}

void Streamline::do_streamsurface(SLSourceInfo* si,
				  const VectorFieldHandle& field,
				  double stepsize, int maxsteps, int skip,
				  const ScalarFieldHandle& sfield,
				  const ColormapHandle& cmap,
				  ALGS alg_enum,
				  double maxbend)
{
    Array1<SLTracer*> tracers;
    SLSource* source=si->source;
    make_tracers(source, tracers, field, alg_enum);
    if(tracers.size() <= 1){
	error("Can't make a surface out of this!!!");
	return;
    }
    Array1<GeomTriStrip*> surfs(tracers.size()-1);
    double t=0;
    GeomGroup* group=0;
    int step=0;
    int ninside=1;
    int splitlast=0;
    double maxcosangle=Cos(maxbend);
    while(step< maxsteps && ninside){
	step++;
	// If the group is discontinued, we have to start new surfaces
	GeomGroup* newgroup=si->get_group(t);
	if(newgroup != group || splitlast){
	    group=newgroup;
	    for(int i=0;i<surfs.size();i++){
		if( (newgroup != group || !surfs[i]) &&
		   tracers[i]->inside && tracers[i+1]->inside){
		    surfs[i]=new GeomTriStrip;
		    group->add(surfs[i]);
		    // Compute vector between points
		    Point p1(tracers[i]->p);
		    Point p2(tracers[i+1]->p);
		    Vector vec(p2-p1);

		    // Left tracer
		    Vector grad=tracers[i]->grad;
		    field->interpolate(tracers[i]->p, grad);
		    Vector n1(Cross(grad, vec));
		    GeomVertex* vtx=get_vertex(tracers[i]->p,
					       sfield, cmap, n1);
		    surfs[i]->add(vtx);

		    // Right tracer
		    grad=tracers[i+1]->grad;
		    field->interpolate(tracers[i+1]->p, grad);
		    Vector n2(Cross(grad, -vec));
		    vtx=get_vertex(tracers[i]->p,
				   sfield, cmap, n2);
		    surfs[i]->add(vtx);
		} else {
		    surfs[i]=0;
		}
	    }
	}
	t+=stepsize;

	// Advance the tracers
	ninside=0;
	for(int i=0;i<tracers.size();i++)
	    ninside+=tracers[i]->advance(field, stepsize, skip);

	// Draw new points...
	for(i=0;i<surfs.size();i++){
	    if(tracers[i]->inside && tracers[i+1]->inside){
		Point p1(left_tracer(tracers, i)->p);
		Point p2(right_tracer(tracers, i)->p);
		Vector vec(p2-p1);
		// Get left vertex
		SLTracer* left=tracers[i];
		Vector grad=left->grad;
		field->interpolate(left->p, grad);
		Vector n1(Cross(grad, vec));
		GeomVertex* lvtx=get_vertex(left->p,
					    sfield, cmap, n1);

		// Get right vertex
		SLTracer* right=tracers[i+1];
		grad=right->grad;
		field->interpolate(right->p, grad);
		Vector n2(Cross(grad, -vec));
		GeomVertex* rvtx=get_vertex(right->p,
					    sfield, cmap, n2);

		surfs[i]->add(lvtx);
		surfs[i]->add(rvtx);
	    }
	}

	// See if any of the surfaces need to be split
	Array1<int> split(surfs.size());
	split[0]=0;
	for(i=1;i<surfs.size();i++){
	    split[i]=0;
	    Vector v1(tracers[i-1]->p - tracers[i]->p);
	    Vector v2(tracers[i+1]->p - tracers[i]->p);
	    v1.normalize();
	    v2.normalize();
	    double cosangle=Dot(v1, v2);
	    if(cosangle < maxcosangle){
		// Split them both
		split[i-1]=split[i]=1;
	    }
	}

	int nsplit=0;
	for(i=0;i<surfs.size();i++)
	    nsplit+=split[i];
	if(nsplit > 0){
	    // Insert the new tracers into the array...
	    Array1<SLTracer*> new_tracers(tracers.size()+nsplit);
	    int newp=0;
	    for(int i=0;i<tracers.size();i++){
		// Insert the old tracer
		new_tracers[newp++]=tracers[i];

		// Insert new ones...
		if(i < surfs.size() && split[i]){
		    double s=(tracers[i]->s+tracers[i+1]->s);
		    double t=(tracers[i]->t+tracers[i+1]->t);
		    SLTracer* tracer=make_tracer(source, s, t,
						 field, alg_enum);
		    new_tracers[newp++]=tracer;

		    // Advect it to the current point
		    for(int i=0;i<=step;i++)
			tracer->advance(field, stepsize, skip);
		}
	    }

	    // Copy the new ones over
	    tracers=new_tracers;

	    // Rebuild the TriStrips
	    Array1<GeomTriStrip*> new_surfs(surfs.size()+nsplit);
	    newp=0;
	    for(i=0;i<surfs.size();i++){
		if(split[i]){
		    // Make new tristrips next time around
		    new_surfs[newp++]=0;
		    new_surfs[newp++]=0;
		} else {
		    // Continue to use the old one
		    new_surfs[newp++]=surfs[i];
		}
	    }
	    // Copy the new ones over
	    surfs=new_surfs;
	    splitlast=1;
	} else {
	    splitlast=0;
	}
    }
}

void Streamline::geom_moved(int, double, const Vector&,
			    void*)
{
}

void Streamline::geom_release(void*)
{
    if(!abort_flag){
	abort_flag=1;
	want_to_execute();
    }
}

SLSource::SLSource(Streamline* sl, const clString& name)
: sl(sl), name(name), widget(0), selected(0)
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
    pw->SetScale(scale/50);
}

Point SLPointSource::trace_start(double s, double t)
{
    s=t=0;
    return pw->GetPosition();
}

void SLPointSource::get_n(int& ns, int& nt)
{
    ns=1;
    nt=1;
}

Vector SLPointSource::ribbon_direction(double, double,
				       const Point& p,
				       const VectorFieldHandle& vfield)
{
    Vector grad(0,0,1); // Default, in case we are outside of the field
    vfield->interpolate(p, grad);
    Vector v1, v2;
    grad.find_orthogonal(v1, v2);
    return v1;
}
    

SLLineSource::SLLineSource(Streamline* sl)
: SLSource(sl, "Line")
{
    widget=gw=new GaugeWidget(sl, &sl->widget_lock, 1);
}

SLLineSource::~SLLineSource()
{
}

void SLLineSource::find(const Point& start, const Vector& downstream,
			double scale)
{
    Vector v1, v2;
    downstream.find_orthogonal(v1, v2);
    double dist=scale/3.;
    Point p1(start-v1*dist);
    Point p2(start+v1*dist);
    gw->SetEndpoints(p1, p2);
    gw->SetScale(scale/50);
    gw->execute();
}

Point SLLineSource::trace_start(double s, double)
{
    Point p1, p2;
    gw->GetEndpoints(p1, p2);
    Vector axis(p2-p1);
    double ratio=gw->GetRatio();
    return p1+axis*s*ratio;
}

void SLLineSource::get_n(int& ns, int& nt)
{
    double ratio=gw->GetRatio();
    if(ratio < 1.e-3)
	ns=1;
    else
	ns=Floor(1./ratio);
    nt=1;
}

Vector SLLineSource::ribbon_direction(double, double,
				      const Point&,
				      const VectorFieldHandle&)
{
    Point p1, p2;
    gw->GetEndpoints(p1, p2);
    Vector axis(p2-p1);
    axis.normalize();
    return axis;
}

SLRingSource::SLRingSource(Streamline* sl)
: SLSource(sl, "Ring")
{
    widget=rw=new RingWidget(sl, &sl->widget_lock, 1);
}

SLRingSource::~SLRingSource()
{
}

void SLRingSource::find(const Point& start, const Vector& downstream,
			double scale)
{
    rw->SetPosition(start, downstream, scale/10);
    rw->SetScale(scale/50);
    rw->execute();
}

Point SLRingSource::trace_start(double s, double)
{
    double ratio=rw->GetRatio();
    double angle=s*ratio*2*Pi;
    Point cen;
    Vector normal;
    double rad;
    rw->GetPosition(cen, normal, rad);
    Vector v1, v2;
    rw->GetPlane(v1, v2);
    return cen+v1*(rad*Cos(angle))+v2*(rad*Sin(angle));
}

void SLRingSource::get_n(int& ns, int& nt)
{
    double ratio=rw->GetRatio();
    if(ratio < 1.e-3)
	ns=1;
    else
	ns=Floor(1./ratio);
    nt=1;
}

Vector SLRingSource::ribbon_direction(double s, double,
				      const Point&,
				      const VectorFieldHandle&)
{
    double ratio=rw->GetRatio();
    double angle=s*ratio*2*Pi;
    Vector v1, v2;
    rw->GetPlane(v1, v2);
    return v1*Sin(angle)+v2*Cos(angle);
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

int SLEulerTracer::advance(const VectorFieldHandle& vfield, double stepsize, int skip)
{
    for(int i=0;i<skip;i++){
	if(!vfield->interpolate(p, grad)){
	    inside=0;
	    return 0;
	} else {
	    p+=(grad*stepsize);
	}
    }
    return 1;
}

SLRK4Tracer::SLRK4Tracer(const Point& p, double s, double t,
			 const VectorFieldHandle& vfield)
: SLTracer(p, s, t, vfield)
{
}

SLRK4Tracer::~SLRK4Tracer()
{
}

int SLRK4Tracer::advance(const VectorFieldHandle&, double, int)
{
    NOT_FINISHED("SLRK4Tracer::advance");
    return 0;
}

void Streamline::tcl_command(TCLArgs& args, void* userdata)
{
    if(args.count() < 2){
	args.error("Streamline needs a minor command");
	return;
    }
    if(args[1] == "newsource"){
	if(args.count() != 3){
	    args.error("newsource must have a SID");
	    return;
	}
	int sid;
	if(!args[2].get_int(sid)){
	    args.error("newsource has a bad SID");
	    return;
	}
	source_info.insert(sid, new SLSourceInfo(sid, this, ogeom));
    } else if(args[1] == "need_find"){
	if(args.count() != 3){
	    args.error("need_find must have a SID");
	    return;
	}
	int sid;
	if(!args[2].get_int(sid)){
	    args.error("need_find has a bad SID");
	    return;
	}
	SLSourceInfo* si;
	if(!source_info.lookup(sid, si)){
	    args.error("bad SID for need_find");
	    return;
	}
	si->need_find=1;
    } else {
	Module::tcl_command(args, userdata);
    }
}

SLSourceInfo::SLSourceInfo(int sid, Streamline* module, GeometryOPort* ogeom)
: sid(sid), widget_group(0), widget_geomid(0), geomid(0),
  source(0), need_find(1)
{
    sources.add(new SLPointSource(module));
    sources.add(new SLLineSource(module));
    sources.add(new SLRingSource(module));
    // Make the group;
    widget_group=new GeomGroup;
    for(int i=0;i<sources.size();i++)
	widget_group->add(sources[i]->widget->GetWidget());
    widget_geomid=ogeom->addObj(widget_group,
				widget_name+to_string(sid),
				&module->widget_lock);
}

void SLSourceInfo::pick_source(const clString& sname,
			       const VectorFieldHandle& vfield,
			       Streamline* module)
{
    SLSource* newsource=0;
    for(int i=0;i<sources.size();i++)
	if(sources[i]->name == sname)
	    newsource=sources[i];
    if(!newsource){
	module->error("No such source!");
	return;
    }
    if(newsource != source){
	for(i=0;i<sources.size();i++)
	    sources[i]->widget->SetState(sources[i] == newsource);
	if(source){
	    Point olds(source->widget->ReferencePoint());
	    Point news(source->widget->ReferencePoint());
	    Vector moved(news-olds);
	    newsource->widget->MoveDelta(moved);
	}
	source=newsource;
    }

    // See if we need to find the field
    if(need_find){
	// Find the field
	Point min, max;
	vfield->get_bounds(min, max);
	Point cen(AffineCombination(min, 0.5, max, 0.5));
	Vector axis;
	if(!vfield->interpolate(cen, axis)){
	    // No field???
	    module->error("Can't find center of field");
	    return;
	}
	axis.normalize();
	double scale=vfield->longest_dimension();
	need_find=0;
	cerr << "scale=" << scale << endl;
	for(int i=0;i<sources.size();i++)
	    sources[i]->find(cen, axis, scale);
    }
}
