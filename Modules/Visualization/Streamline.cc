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
#include <Datatypes/ColorMapPort.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldUG.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/VectorField.h>
#include <Datatypes/VectorFieldUG.h>
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
#include <Malloc/Allocator.h>
#include <Math/MinMax.h>
#include <Multitask/Task.h>
#include <Widgets/GaugeWidget.h>
#include <Widgets/PointWidget.h>
#include <Widgets/RingWidget.h>
#include <Widgets/ScaledFrameWidget.h>

#include <iostream.h>
#include <values.h>
#include <strstream.h>

class Streamline;

struct SLTracer {
    Point p;
    double s,t;
    int inside;
    int ix;
    Vector grad;
    double sign;
    int startstep;
    double starttime;

    SLTracer(const Point&, double s, double t,
	     const VectorFieldHandle& vfield, double sign);
    virtual ~SLTracer();
    virtual int advance(const VectorFieldHandle&, double stepsize, int skip)=0;
};

struct SLEulerTracer : public SLTracer {
    int stagnate;
    SLEulerTracer(const Point&, double s, double t,
		  const VectorFieldHandle& vfield, double sign);
    virtual ~SLEulerTracer();
    virtual int advance(const VectorFieldHandle&, double stepsize, int skip);
};    

#if 0
struct SLExactTracer : public SLTracer {
    SLExactTracer(const Point&, double s, double t,
		  const VectorFieldHandle& vfield, double sign);
    virtual ~SLExactTracer();
    virtual int advance(const VectorFieldHandle&, double stepsize, int skip);
};   
#endif 

struct SLRK4Tracer : public SLTracer {
    int stagnate;
    SLRK4Tracer(const Point&, double s, double t,
		const VectorFieldHandle& vfield, double sign);
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

    virtual void update_position(const clString& base, const clString& varname)=0;
    virtual void reposition(const clString& pos)=0;
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
    virtual void update_position(const clString& base, const clString& varname);
    virtual void reposition(const clString& pos);
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
    virtual void update_position(const clString& base, const clString& varname);
    virtual void reposition(const clString& pos);
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
    virtual void update_position(const clString& base, const clString& varname);
    virtual void reposition(const clString& pos);
};

struct SLSquareSource : public SLSource {
    ScaledFrameWidget* fw;
public:
    SLSquareSource(Streamline* sl);
    virtual ~SLSquareSource();
    virtual void find(const Point&, const Vector& axis, double scale);
    virtual Point trace_start(double s, double t);
    virtual void get_n(int& s, int& t);
    virtual Vector ribbon_direction(double s, double t, const Point& p,
				    const VectorFieldHandle& vfield);
    virtual void update_position(const clString& base, const clString& varname);
    virtual void reposition(const clString& pos);
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
    ColorMapIPort* inColorMap;
    ScalarFieldIPort* incolorfield;
    GeometryOPort* ogeom;

    int first_execute;

    HashTable<int, SLSourceInfo*> source_info;

    virtual void geom_moved(GeomPick*, int, double, const Vector&, void*);
    virtual void geom_release(GeomPick*, void*);

    enum ALGS {
	Exact, Euler, RK4,
    };
    ALGS alg_enum;

    void make_tracers(SLSource*, Array1<SLTracer*>&,
		      const VectorFieldHandle& vfield,
		      double width=0);
    SLTracer* make_tracer(SLSource* source, double s, double t,
			  const VectorFieldHandle& vfield,
			  double sign);
    SLTracer* make_tracer(const Point& p,
			  double s, double t,
			  const VectorFieldHandle& vfield,
			  double sign);
    Array1<SLTracer*> tracers;
    int np;
    double stepsize;
    int maxsteps;
    int skip;
    Array1<GeomPolyline*> lines;
    Array1<GeomTube*> tubes;
    double tubesize;
    Mutex grouplock;
    VectorFieldHandle field;
    ScalarFieldHandle sfield;
    ColorMapHandle cmap;
    SLSourceInfo* si;

    int upstream, downstream;
  int drawmode;
  double drawdist;

public:
    void do_streamline(SLSourceInfo* si);
    void parallel_streamline(int proc);
    void do_streamtube(SLSourceInfo* si);
    void parallel_streamtube(int proc);
    void do_streamribbon(SLSourceInfo* si, double ribbonsize);
    void parallel_streamribbon(int proc);
    void do_streamsurface(SLSourceInfo* si,
			  double maxbend);
    inline SLTracer* left_tracer(const Array1<SLTracer*> tracers, int i){
	return tracers[i*2];
    }
    inline SLTracer* right_tracer(const Array1<SLTracer*> tracers, int i){
	return tracers[i*2+1];
    }
	    
    GeomVertex* get_vertex(double t, double maxt, const Point& p,
			   const ScalarFieldHandle& sfield,
			   const ColorMapHandle& cmap);
    GeomVertex* get_vertex(double t, double maxt, const Point& p,
			   const ScalarFieldHandle& sfield,
			   const ColorMapHandle& cmap,
			   const Vector& normal);
    GeomVertex* get_vertex(double t, double maxt, const Point& p,
			   const ScalarFieldHandle& sfield,
			   const ColorMapHandle& cmap,
			   const Vector& normal,
			   int& ix);
public:
    CrowdMonitor widget_lock;

    Streamline(const clString& id);
    Streamline(const Streamline&, int deep);
    virtual ~Streamline();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual void tcl_command(TCLArgs& args, void* userdata);
};

extern "C" {
Module* make_Streamline(const clString& id)
{
    return scinew Streamline(id);
}
};

static clString widget_name("Streamline Widget");
static clString module_name("Streamline");

Streamline::Streamline(const clString& id)
: Module(module_name, id, Filter), first_execute(1)
{
    // Create the input ports
    infield=scinew VectorFieldIPort(this, "Vector Field",
				 ScalarFieldIPort::Atomic);
    add_iport(infield);
    incolorfield=scinew ScalarFieldIPort(this, "Color Field",
				      ScalarFieldIPort::Atomic);
    add_iport(incolorfield);
    inColorMap=scinew ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_iport(inColorMap);

    // Create the output port
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
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
    return scinew Streamline(*this, deep);
}

void Streamline::execute()
{
    // Get the data from the ports...
    if(!infield->get(field))
	return;
    incolorfield->get(sfield);
    int have_cmap=inColorMap->get(cmap);
    if(!have_cmap)
	sfield=0;

    HashTableIter<int, SLSourceInfo*> iter(&source_info);
    for(iter.first();iter.ok();++iter){
	// Find the current source...
        si=iter.get_data();
	clString sidstr(id+"-"+to_string(si->sid));
	cerr << "Processing source: " << si->sid << endl;

	si->source=0;
	clString sname;
	if(!get_tcl_stringvar(sidstr, "source", sname)){
	    error("Error reading source variable");
	    return;
	}

	si->pick_source(sname, field, this);
	si->source->update_position(sidstr, "position");

	// Calculate Streamlines
	if(!get_tcl_doublevar(sidstr, "stepsize", stepsize)){
	    error("Error reading stepsize variable");
	    return;
	}
	if(!get_tcl_intvar(sidstr, "maxsteps", maxsteps)){
	    error("Error reading maxsteps variable");
	    return;
	}
	if(!get_tcl_intvar(sidstr, "skip", skip)){
	    error("Error reading skip variable");
	    return;
	}
	clString dir;
	if(!get_tcl_stringvar(sidstr, "direction", dir)){
	    error("Error reading direction variable");
	    return;
	}
	if(dir == "Upstream"){
	  upstream=1;
	  downstream=0;
	} else if(dir == "Downstream"){
	  downstream=1;
	  upstream=0;
	} else if(dir == "Both"){
	  upstream=downstream=1;
	} else {
	  error("Bad value in direction variable");
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
	GeomGroup* group=scinew GeomGroup;
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
	if(alg == "Exact"){
	    alg_enum=Exact;
	} else if(alg == "Euler"){
	    alg_enum=Euler;
	} else if(alg == "RK4"){
	    alg_enum=RK4;
	} else {
	    error("Unknown algorithm");
	    return;
	}

	// Do it...
	if(markertype == "Line"){
	  if(!get_tcl_intvar(sidstr, "drawmode", drawmode)){
	    error("Error reading drawmode variable");
	    return;
	  }
	  if(!get_tcl_doublevar(sidstr, "drawdist", drawdist)){
	    error("Error reading drawdist variable");
	    return;
	  }
	  do_streamline(si);
	} else if(markertype == "Tube"){
	    if(!get_tcl_doublevar(sidstr, "tubesize", tubesize)){
		error("Error reading tubesize variable");
		return;
	    }
	    do_streamtube(si);
	} else if(markertype == "Ribbon"){
	    double ribbonsize;
	    if(!get_tcl_doublevar(sidstr, "ribbonsize", ribbonsize)){
		error("Error reading ribbonsize variable");
		return;
	    }
	    do_streamribbon(si, ribbonsize);
	} else if(markertype == "Surface"){
	    double maxbend;
	    if(!get_tcl_doublevar(sidstr, "maxbend", maxbend)){
		error("Error reading ribbonsize variable");
		return;
	    }
	    do_streamsurface(si, maxbend);
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
    field=0;
    sfield=0;
    cmap=0;
}

SLTracer* Streamline::make_tracer(SLSource* source, double s, double t,
				  const VectorFieldHandle& vfield,
				  double sign)
{
    Point start(source->trace_start(s, t));
    return make_tracer(start, s, t, vfield, sign);
}

SLTracer* Streamline::make_tracer(const Point& start,
				  double s, double t,
				  const VectorFieldHandle& vfield,
				  double sign)
{
    switch(alg_enum){
#if 0
    case Exact:
	return scinew SLExactTracer(start, s, t, vfield, sign);
#endif
    case Euler:
	return scinew SLEulerTracer(start, s, t, vfield, sign);
    case RK4:
	return scinew SLRK4Tracer(start, s, t, vfield, sign);
    }
    return 0;
}

void Streamline::make_tracers(SLSource* source, Array1<SLTracer*>& tracers,
			      const VectorFieldHandle& vfield,
			      double ribbonsize)
{
    int ns, nt;
    source->get_n(ns, nt);
    for(int s=0;s<ns;s++){
	for(int t=0;t<nt;t++){
	    Point start(source->trace_start(s, t));
	    if(ribbonsize == 0){
	      if(downstream)
		tracers.add(make_tracer(start, s, t, vfield, 1));
	      if(upstream)
		tracers.add(make_tracer(start, s, t, vfield, -1));
	    } else {
		Vector v(source->ribbon_direction(s, t, start, vfield)
			 *(ribbonsize/2));
		if(downstream){
		  tracers.add(make_tracer(start-v, s, t, vfield, 1));
		  tracers.add(make_tracer(start+v, s, t, vfield, 1));
		}
		if(upstream){
		  tracers.add(make_tracer(start-v, s, t, vfield, -1));
		  tracers.add(make_tracer(start+v, s, t, vfield, -1));
		}
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
				    double total_time, int)
{
    anim_begin=0;
    anim_end=1;
    anim_steps=1;
    anim_groups.remove_all();
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
	    GeomGroup* timegroup=scinew GeomGroup;
	    anim_groups.add(timegroup);
	    double ntbeg=(tbeg-anim_begin)/(anim_end-anim_begin);
	    double ntend=(tend-anim_begin)/(anim_end-anim_begin);
	    GeomTimeSwitch* timeswitch=scinew GeomTimeSwitch(timegroup, ntbeg, ntend);
	    top->add(timeswitch);
	}
    }
}

GeomVertex* Streamline::get_vertex(double t, double maxt, const Point& p,
				   const ScalarFieldHandle& sfield,
				   const ColorMapHandle& cmap)
{
    if(sfield.get_rep()){
	double sval;
	if(sfield->interpolate(p, sval)){
	    MaterialHandle matl(cmap->lookup(sval));
	    return new GeomCVertex(p, matl->diffuse);
	}
    } else if(cmap.get_rep()){
      MaterialHandle matl(cmap->lookup2(p.z()/-5200*50)); //cmap->lookup2(t/maxt));
      return new GeomCVertex(p, matl->diffuse);
    }
    return new GeomVertex(p);
}

GeomVertex* Streamline::get_vertex(double t, double maxt, const Point& p,
				   const ScalarFieldHandle& sfield,
				   const ColorMapHandle& cmap,
				   const Vector& normal)
{
    Vector n(normal);
    double l=n.length2();
    if(l < 1.e-6){
	l*=100000;
	n*=100000;
    }
    if(l > 1.e-6){
	n.normalize();
    }
    if(sfield.get_rep()){
	double sval;
	if(sfield->interpolate(p, sval)){
	    MaterialHandle matl(cmap->lookup(sval));
	    return new GeomNMVertex(p, n, matl);
	}
    } else if(cmap.get_rep()){
      MaterialHandle matl(cmap->lookup2(t/maxt));
      return new GeomNMVertex(p, n, matl);
    }
    return new GeomNVertex(p, n);
}

GeomVertex* Streamline::get_vertex(double t, double maxt, const Point& p,
				   const ScalarFieldHandle& sfield,
				   const ColorMapHandle& cmap,
				   const Vector& normal,
				   int& ix)
{
    Vector n(normal);
    if(n.length2() > 1.e-6)
	n.normalize();
    if(sfield.get_rep()){
	double sval;
	if(sfield->interpolate(p, sval, ix)){
	    MaterialHandle matl(cmap->lookup(sval));
	    return new GeomNMVertex(p, n, matl);
	}
    } else if(cmap.get_rep()){
      MaterialHandle matl(cmap->lookup2(t/maxt));
      return new GeomNMVertex(p, n, matl);
    }
    return new GeomNVertex(p, n);
}

void Streamline::parallel_streamline(int proc)
{
  GeomGroup* group=0;
  int st=proc*tracers.size()/np;
  int et=(proc+1)*tracers.size()/np;
  double maxt=maxsteps*stepsize;
#if 0
  double t=0;
  int step=0;
  int ninside=1;
  while(step< maxsteps && ninside){
    step++;
    // If the group is discontinued, we have to start new polylines
    GeomGroup* newgroup=si->get_group(t);
    if(newgroup != group){
      group=newgroup;
      for(int i=st;i<et;i++){
	if(tracers[i]->inside){
	  lines[i]=scinew GeomPolyline;
	  grouplock.lock();
	  group->add(lines[i]);
	  grouplock.unlock();
	  GeomVertex* vtx=get_vertex(t, maxt, tracers[i]->p, sfield, cmap);
	  lines[i]->add(t/maxt, vtx);
	} else {
	    lines[i]=0;
	}
      }
    }
    t+=stepsize;
      
    // Advance the tracers
    ninside=0;
    for(int i=st;i<et;i++){
      int inside=tracers[i]->advance(field, stepsize, skip);
      if(inside){
	GeomVertex* vtx=get_vertex(t, maxt, tracers[i]->p, sfield, cmap);
	lines[i]->add(t/maxt, vtx);
      }
      ninside+=inside;
    }
  }
#else
  group=new GeomGroup;
  double ss=stepsize/maxt;
  for(int i=st;i<et;i++){
    int step=tracers[i]->startstep;
    double t=tracers[i]->starttime;
    int inside=1;
    GeomPolylineTC* line=new GeomPolylineTC(drawmode, drawdist);
    while(step++< maxsteps && inside){
      inside=tracers[i]->advance(field, stepsize, skip);
      if(inside){
	line->add(t, tracers[i]->p,
		  cmap->lookup2(tracers[i]->p.z()*50/-5200)->diffuse);
      } else {
	if(tracers[i]->p.x() >= 1280){
	  SLTracer* ot=tracers[i];
	  Point p(ot->p);
	  cerr << "Went out: x=" << p.x() << endl;
	  if(p.x() >= 1280){
	    cerr << "Went out on right\n";
	    p.x(p.x()-1280);
	  } else if(p.x()<= 0){
	    cerr << "Went out on left\n";
	    p.x(p.x()+1280);
	  }
	  SLTracer* nt=make_tracer(p, ot->s, ot->t, field, ot->sign);
	  tracers[i]=nt;
	  delete ot;
	  nt->startstep=step;
	  nt->starttime=t;
	  i--;
	}
      }
      t+=ss;
    }
    group->add(line);
  }
  grouplock.lock();
  GeomGroup* maingroup=si->get_group(0);
  maingroup->add(group);
  grouplock.unlock();
#endif
  for(i=st;i<et;i++){
    delete tracers[i];
  }
}

static void do_parallel_streamline(void* obj, int proc)
{
  Streamline* module=(Streamline*)obj;
  module->parallel_streamline(proc);
}

void Streamline::do_streamline(SLSourceInfo* si)
{
    tracers.remove_all();
    make_tracers(si->source, tracers, field, alg_enum);
    lines.resize(tracers.size());
    np=Min(tracers.size(), Task::nprocessors());
    Task::multiprocess(np, do_parallel_streamline, this);
}

void Streamline::parallel_streamtube(int proc)
{
    int st=proc*tracers.size()/np;
    int et=(proc+1)*tracers.size()/np;
    double t=0;
    GeomGroup* group=0;
    int ninside=1;
    double maxt=maxsteps*stepsize;
    int step=0;
    while(step< maxsteps && ninside){
	step++;
	// If the group is discontinued, we have to start new tubes
	GeomGroup* newgroup=si->get_group(t);
	if(newgroup != group){
	    group=newgroup;
	    for(int i=st;i<et;i++){
		if(tracers[i]->inside){
		    tubes[i]=scinew GeomTube;
		    grouplock.lock();
		    group->add(tubes[i]);
		    grouplock.unlock();
		    Vector grad=tracers[i]->grad;
		    GeomVertex* vtx=get_vertex(t, maxt, tracers[i]->p, sfield, cmap);
		    if(grad.length2() < 1.e-6)
			grad=Vector(0,0,1);
		    tubes[i]->add(vtx, tubesize, grad);
		} else {
		    tubes[i]=0;
		}
	    }
	}
	t+=stepsize;

	// Advance the tracers
	ninside=0;
	for(int i=st;i<et;i++){
	    int inside=tracers[i]->advance(field, stepsize, skip);
	    if(inside){
		Vector grad=tracers[i]->grad;
		GeomVertex* vtx=get_vertex(t, maxt, tracers[i]->p, sfield, cmap);
		if(grad .length2() < 1.e-6)
		    grad=Vector(0,0,1);
		tubes[i]->add(vtx, tubesize, grad);
	    }
	    ninside+=inside;
	}
    }
    for(int i=st;i<et;i++){
	delete tracers[i];
    }
}

static void do_parallel_streamtube(void* obj, int proc)
{
  Streamline* module=(Streamline*)obj;
  module->parallel_streamtube(proc);
}


void Streamline::do_streamtube(SLSourceInfo* si)
{
    SLSource* source=si->source;
    tracers.remove_all();
    make_tracers(source, tracers, field, alg_enum);
    tubes.resize(tracers.size());
    np=Min(tracers.size(), Task::nprocessors());
    Task::multiprocess(np, do_parallel_streamtube, this);
}

void Streamline::do_streamribbon(SLSourceInfo* si,
				 double ribbonsize)

{
    tracers.remove_all();
    SLSource* source=si->source;
    make_tracers(source, tracers, field, ribbonsize);
    Array1<GeomTriStrip*> ribbons(tracers.size()/2);
    double t=0;
    GeomGroup* group=0;
    int step=0;
    int ninside=1;
    double maxt=maxsteps*stepsize;
    while(step< maxsteps && ninside){
	step++;
	// If the group is discontinued, we have to start new ribbons
	GeomGroup* newgroup=si->get_group(t);
	if(newgroup != group){
	    group=newgroup;
	    for(int i=0;i<ribbons.size();i++){
		if(left_tracer(tracers, i)->inside
		   && right_tracer(tracers, i)->inside){
		    ribbons[i]=scinew GeomTriStrip;
		    group->add(ribbons[i]);
		    // Compute vector between points
		    Point p1(left_tracer(tracers, i)->p);
		    Point p2(right_tracer(tracers, i)->p);
		    Vector vec(p2-p1);
		    // Left tracer
		    Vector grad=left_tracer(tracers, i)->grad;
		    //field->interpolate(left_tracer(tracers, i)->p, grad);
		    Vector n1(Cross(vec, grad));
		    GeomVertex* vtx=get_vertex(t, maxt, left_tracer(tracers, i)->p,
					       sfield, cmap, n1);
		    ribbons[i]->add(vtx);

		    // Right tracer
		    grad=right_tracer(tracers, i)->grad;
		    //field->interpolate(right_tracer(tracers, i)->p, grad);
		    Vector n2(Cross(vec, grad));
		    vtx=get_vertex(t, maxt, right_tracer(tracers, i)->p,
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
		Vector g1(left_tracer(tracers, i)->grad);
		Vector g2(right_tracer(tracers, i)->grad);
		if(g1.length2() < 1.e-6 || g2.length2() < 1.e-6){
		    left_tracer(tracers, i)->inside=0;
		    right_tracer(tracers, i)->inside=0;
		} else {
		    Point p1(left_tracer(tracers, i)->p);
		    Point p2(right_tracer(tracers, i)->p);
		    g1.normalize();
		    g2.normalize();
		    p2-=g2*Dot(g1, (p2-p1));
		    Vector vec(p2-p1);
		    if(vec.length2() < 1.e-6){
			left_tracer(tracers, i)->inside=0;
			right_tracer(tracers, i)->inside=0;
		    } else {
			double l=vec.normalize();
			double d1=(l-ribbonsize)/2;
			double d2=(l+ribbonsize)/2;
			left_tracer(tracers, i)->p=p1+vec*d1;
			right_tracer(tracers, i)->p=p1+vec*d2;

			// Get left vertex
			Vector grad=left_tracer(tracers, i)->grad;
			//field->interpolate(left_tracer(tracers, i)->p, grad);
			Vector n1(Cross(vec, grad));
			GeomVertex* lvtx=get_vertex(t, maxt, left_tracer(tracers, i)->p,
						    sfield, cmap, n1);

			// Get right vertex
			grad=right_tracer(tracers, i)->grad;
			//field->interpolate(right_tracer(tracers, i)->p, grad);
			Vector n2(Cross(vec, grad));
			GeomVertex* rvtx=get_vertex(t, maxt, right_tracer(tracers, i)->p,
						    sfield, cmap, n2);

			ribbons[i]->add(lvtx);
			ribbons[i]->add(rvtx);
		    }
		}
	    } else {
		// Make sure that we stop doing work for both tracers...
		left_tracer(tracers, i)->inside=0;
		right_tracer(tracers, i)->inside=0;
	    }
	    ninside+=(linside && rinside);
	}
    }
    for(int i=0;i<tracers.size();i++){
	delete tracers[i];
    }
}

void Streamline::do_streamsurface(SLSourceInfo* si,
				  double maxbend)
{
    tracers.remove_all();
    SLSource* source=si->source;
    make_tracers(source, tracers, field, alg_enum);
    if(tracers.size() <= 1){
	error("Can't make a surface out of this!!!");
	return;
    }
    Array1<GeomTriStrip*> surfs(tracers.size()-1);
    for(int i=0;i<surfs.size();i++)
	surfs[0]=0;
    double t=0;
    GeomGroup* group=0;
    int step=0;
    int ninside=1;
    int splitlast=0;
    double maxcosangle=Cos(maxbend);
    double maxt=maxsteps*stepsize;
    while(step< maxsteps && ninside){
	step++;
	// If the group is discontinued, we have to start new surfaces
	GeomGroup* newgroup=si->get_group(t);
	if(newgroup != group || splitlast){
	    for(int i=0;i<surfs.size();i++){
		if( (newgroup != group || !surfs[i]) &&
		   tracers[i]->inside && tracers[i+1]->inside){
		    surfs[i]=scinew GeomTriStrip;
		    newgroup->add(surfs[i]);
		    // Compute vector between points
		    Point p1(tracers[i]->p);
		    Point p2(tracers[i+1]->p);
		    Vector vec(p2-p1);

		    // Left tracer
		    Vector grad=tracers[i]->grad;
		    //field->interpolate(tracers[i]->p, grad);
		    Vector n1(Cross(vec, grad));
		    grad=tracers[i+1]->grad;
		    //field->interpolate(tracers[i+1]->p, grad);
		    Vector n2(Cross(vec, grad));
		    if(n1.length2() < 1.e-8){
			if(n2.length2() < 1.e-8){
			    surfs[i]=0;
			    cerr << "Degenerate normals. stopping surface...\n";
			} else {
			    n1=n2;
			}
		    } else {
			if(n2.length2() < 1.e-8){
			    n2=n1;
			}
		    }
		    if(surfs[i]){
			GeomVertex* vtx=get_vertex(t, maxt, tracers[i]->p,
						   sfield, cmap, n1);
			surfs[i]->add(vtx);

			// Right tracer
			vtx=get_vertex(t, maxt, tracers[i+1]->p,
				       sfield, cmap, n2);
			surfs[i]->add(vtx);
		    }
		} else {
		    surfs[i]=0;
		}
	    }
	    group=newgroup;
	}
	t+=stepsize;

	// Advance the tracers
	ninside=0;
	int i;
	for(i=0;i<tracers.size();i++)
	    ninside+=tracers[i]->advance(field, stepsize, skip);

	// Draw new points...
	for(i=0;i<surfs.size();i++){
	    if(tracers[i]->inside && tracers[i+1]->inside){
		Point p1(tracers[i]->p);
		Point p2(tracers[i+1]->p);
		Vector vec(p2-p1);
		// Get left vertex
		SLTracer* left=tracers[i];
		Vector grad=left->grad;
		//field->interpolate(left->p, grad);
		Vector n1(Cross(vec, grad));
		SLTracer* right=tracers[i+1];
		grad=right->grad;
		Vector n2(Cross(vec, grad));
		if(n1.length2() < 1.e-8){
		    if(n2.length2() < 1.e-8){
			surfs[i]=0;
			cerr << "2. Degenerate normals, stopping surface...\n";
		    } else {
			n1=n2;
		    }
		} else {
		    if(n2.length2() < 1.e-8){
			n2=n1;
		    }
		}
		if(surfs[i]){
		    GeomVertex* lvtx=get_vertex(t, maxt, left->p,
						sfield, cmap, n1);

		    // Get right vertex
		    //field->interpolate(right->p, grad);
		    GeomVertex* rvtx=get_vertex(t, maxt, right->p,
						sfield, cmap, n2);

		    surfs[i]->add(lvtx);
		    surfs[i]->add(rvtx);
		}
	    }
	}

	// See if any of the surfaces need to be split
#if 0
	Array1<int> split(surfs.size());
	split[0]=0;
	for(i=1;i<surfs.size();i++){
	    split[i]=0;
	    Vector v1(tracers[i-1]->p - tracers[i]->p);
	    Vector v2(tracers[i+1]->p - tracers[i]->p);
	    if(v1.length2() > 1.e-8 && v2.length2() > 1.e-8){
		v1.normalize();
		v2.normalize();
		double cosangle=Dot(v1, v2);
		if(cosangle < maxcosangle){
		    // Split them both
		    split[i-1]=split[i]=1;
		}
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
#endif
	    splitlast=0;
#if 0
	}
#endif
    }
}

void Streamline::geom_moved(GeomPick*, int, double, const Vector&,
			    void*)
{
}

void Streamline::geom_release(GeomPick*, void*)
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
    widget=pw=scinew PointWidget(sl, &sl->widget_lock, 1);
}

SLPointSource::~SLPointSource()
{
}

void SLPointSource::find(const Point& start, const Vector&, double scale)
{
    pw->SetPosition(start);
    pw->SetScale(scale/50);
}

Point SLPointSource::trace_start(double /*s*/, double /*t*/)
{
    return pw->GetPosition();
}

void SLPointSource::get_n(int& ns, int& nt)
{
    ns=1;
    nt=1;
}

void SLPointSource::update_position(const clString& base, const clString& varname)
{
  char buf[1000];
  ostrstream out(buf, 1000);
  out << pw->GetPosition();
  sl->set_tclvar(base, varname, buf);
}

void SLPointSource::reposition(const clString& pos)
{
  istrstream buf(pos());
  Point p;
  buf >> p;
  if(buf){
    pw->SetPosition(p);
  } else {
    sl->error("Error parsing position string");
  }
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
    widget=gw=scinew GaugeWidget(sl, &sl->widget_lock, 1);
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
    gw->SetScale(scale/50);
    gw->SetRatio(.1);
    gw->SetEndpoints(p1, p2);
}

Point SLLineSource::trace_start(double s, double)
{
    Point p1, p2;
    gw->GetEndpoints(p1, p2);
    Vector axis(p2-p1);
    double ratio=gw->GetRatio();
    return p1+axis*s*ratio;
}

void SLLineSource::reposition(const clString& pos)
{
  istrstream buf(pos());
  Point p1, p2;
  double ratio;
  buf >> p1 >> p2 >> ratio;
  if(buf){
    gw->SetRatio(ratio);
    gw->SetEndpoints(p1, p2);
  } else {
    sl->error("Error parsing position string");
  }
}

void SLLineSource::update_position(const clString& base, const clString& varname)
{
  char buf[1000];
  ostrstream out(buf, 1000);
  Point p1, p2;
  gw->GetEndpoints(p1, p2);
  double ratio=gw->GetRatio();
  out << p1 << " " << p2 << " " << ratio << '\0';
  sl->set_tclvar(base, varname, buf);
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
    widget=rw=scinew RingWidget(sl, &sl->widget_lock, 1);
    rw->SetRatio(0.05);
}

SLRingSource::~SLRingSource()
{
}

void SLRingSource::find(const Point& start, const Vector& downstream,
			double scale)
{
    rw->SetPosition(start, downstream, scale/10);
    rw->SetRadius(scale/4);
    rw->SetScale(scale/50);
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

void SLRingSource::reposition(const clString& pos)
{
  istrstream buf(pos());
  Point cen;
  Vector normal;
  double rad;
  double ratio;
  buf >> cen >> normal >> rad >> ratio;
  if(buf){
    rw->SetPosition(cen, normal, rad);
    rw->SetRadius(rad);
    rw->SetRatio(ratio);
  } else {
    sl->error("Error parsing position string");
  }
}

void SLRingSource::update_position(const clString& base, const clString& varname)
{
  char buf[1000];
  ostrstream out(buf, 1000);
  Point cen;
  Vector normal;
  double rad;
  rw->GetPosition(cen, normal, rad);
  double ratio=rw->GetRatio();
  out << cen << " " << normal << " " << rad << " " << ratio << '\0';
  sl->set_tclvar(base, varname, buf);
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

SLSquareSource::SLSquareSource(Streamline* sl)
: SLSource(sl, "Square")
{
    widget=fw=scinew ScaledFrameWidget(sl, &sl->widget_lock, 1);
    fw->SetRatioR(0.1);
    fw->SetRatioD(0.1);
}

SLSquareSource::~SLSquareSource()
{
}

void SLSquareSource::find(const Point& start, const Vector& downstream,
			double scale)
{
    Vector v1, v2;
    downstream.find_orthogonal(v1, v2);
    fw->SetPosition(start, start+v1*(scale*0.3), start+v2*(scale*0.3));
    fw->SetScale(scale/50);
}

Point SLSquareSource::trace_start(double s, double t)
{
    Point 	corner, center, R, D;
    fw->GetPosition( center, R, D);
    Vector v1 = R - center,
           v2 = D - center;
         
    // calculate the corner and the
    // u and v vectors of the cutting plane
    corner = (center - v1) - v2;
    int ns, nt;
    get_n(ns, nt);
    Vector u = v1 * (2.0/ns),
           v = v2 * (2.0/nt);

    Point p(corner+u*s+v*t);
    return p;
}

void SLSquareSource::reposition(const clString& pos)
{
  istrstream buf(pos());
  Point center, R, D;
  double ratioR, ratioD;
  buf >> center >> R >> D >> ratioR >> ratioD;
  if(buf){
    fw->SetPosition(center, R, D);
    fw->SetRatioR(ratioR);
    fw->SetRatioD(ratioD);
  } else {
    sl->error("Error parsing position string");
  }
}

void SLSquareSource::update_position(const clString& base, const clString& varname)
{
  char buf[1000];
  ostrstream out(buf, 1000);
  Point center, R, D;
  fw->GetPosition(center, R, D);
  double ratioR=fw->GetRatioR();
  double ratioD=fw->GetRatioD();
  out << center << " " << R << " " << D << " " << ratioR << " " << ratioD << '\0';
  sl->set_tclvar(base, varname, buf);
}

void SLSquareSource::get_n(int& ns, int& nt)
{
    ns=fw->GetRatioR()*500;
    nt=fw->GetRatioD()*500;
}

Vector SLSquareSource::ribbon_direction(double, double,
				      const Point&,
				      const VectorFieldHandle&)
{
    Point 	corner, center, R, D;
    fw->GetPosition( center, R, D);
    Vector v1 = R - center,
           v2 = D - center;
    return v1;
}

SLTracer::SLTracer(const Point& p, double s, double t,
		   const VectorFieldHandle& vfield, double sign)
: p(p), s(s), t(t), inside(1), sign(sign), startstep(0), starttime(0)
{
    // Interpolate initial gradient
    if(!vfield->interpolate(p, grad))
	inside=0;
}

SLTracer::~SLTracer()
{
}

#if 0

SLExactTracer::SLExactTracer(const Point& p, double s, double t,
			     const VectorFieldHandle& vfield, double sign)
: SLTracer(p, s, t, vfield, sign)
{
    ix=0;
}

SLExactTracer::~SLExactTracer()
{
}

int SLExactTracer::advance(const VectorFieldHandle& vfield, double stepsize, int skip)
{
    double s=stepsize*skip;
    s=100000000000000000000.;
    double total=0;
    VectorFieldUG* ug=vfield->getUG();
    if(!ug){
	inside=0;
	return 0;
    }
    if(ug->typ != VectorFieldUG::ElementValues){
	inside=0;
	return 0;
    }
    Mesh* mesh=ug->mesh.get_rep();
    if(ix==-1){
	if(!mesh->locate(p, ix)){
	    inside=0;
	    return 0;
	}
    }
    int done=0;
    while(!done){
	Element* e=mesh->elems[ix];
	grad=ug->data[ix];

    again:
	int f=-1;
	double min=MAXDOUBLE;
	for(int j=0;j<4;j++){
	    double denom=Dot(e->g[j],grad);
	    if(denom < 1.e-10 && denom > -1.e-10){
		//cerr << "Denom is zero (g=" << e->g[j] << ", grad=" << grad << endl;
	    } else {
		double tf=-(e->a[j]+Dot(e->g[j], p))/denom;
		if(tf>1.e-6 && tf<min){
		    f=j;
		    min=tf;
		}
	    }
	}
	if(f==-1){
	    if(grad.length2() < 1.e-10){
		cerr << "RETURNING because of stagnation!\n";
		inside=0;
		return 0;
	    }
#if 0
	    cerr << "NO MIN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
	    for(int i=0;i<4;i++){
		double tf=-(e->a[i]+Dot(e->g[i], p))/Dot(e->g[i], grad);
		cerr << "tf[" << i << "]=" << tf << endl;
	    }
#endif
	    p-=grad*10;
	    if(!mesh->locate(p, ix)){
		inside=0;
		return 0;
	    }
	    goto again;
	    inside=0;
	    return 0;
	}
	if(total+min > s){
	    min=s-total;
	    done=1;
	} else {
	    // Go to our neighbor next time...
	    ix=e->face(f);
	    if(ix==-1){
		inside=0;
		return done;
	    }
	}
	min*=1.1;
	p+=grad*min;
	
	if(!mesh->locate(p, ix)){
	    inside=0;
	    return 1;
	}
	total+=min;
	return 1;
    }
    return 1;
}
#endif

SLEulerTracer::SLEulerTracer(const Point& p, double s, double t,
			     const VectorFieldHandle& vfield,
			     double sign)
: SLTracer(p, s, t, vfield, sign)
{
    stagnate=0;
    ix=0;
}

SLEulerTracer::~SLEulerTracer()
{
}

int SLEulerTracer::advance(const VectorFieldHandle& vfield, double stepsize, int skip)
{
    if(stagnate){
	inside=0;
	return 0;
    }
    for(int i=0;i<skip;i++){
	if(!vfield->interpolate(p, grad, ix)){
	    inside=0;
	    return 0;
	} else {
	  if(grad.length2() < 1.e-6){
	    stagnate=1;
	  }
#if 0
	    if(grad.length2() < 1.e-6){
		grad*=100000;
		if(grad.length2() < 1.e-6){
		    stagnate=1;
		} else {
		    grad.normalize();
		}
	    } else {
		grad.normalize();
	    }
#endif
	    grad*=sign;
	    p+=(grad*stepsize);
	}
    }
    return 1;
}

SLRK4Tracer::SLRK4Tracer(const Point& p, double s, double t,
			 const VectorFieldHandle& vfield, double sign)
: SLTracer(p, s, t, vfield, sign)
{
    stagnate=0;
    ix=0;
}

SLRK4Tracer::~SLRK4Tracer()
{
}

int SLRK4Tracer::advance(const VectorFieldHandle& vfield, double stepsize, int skip)
{
    Vector F1, F2, F3, F4;

    if(stagnate){
	inside=0;
	return 0;
    }
    for(int i=0;i<skip;i++){
	if(!vfield->interpolate(p, grad, ix)){
	    inside=0;
	    return 0;
	} else {
	    if(grad.length2() < 1.e-6){
		grad*=100000;
		if(grad.length2() < 1.e-6){
		    stagnate=1;
		} else {
		    grad.normalize();
		}
	    } else {
		grad.normalize();
	    }
	    grad*=sign;
	    F1 = grad*stepsize;
	    if(!vfield->interpolate(p+F1*0.5, grad, ix)){
		inside=0;
		return 0;
	    } else {
		if(grad.length2() < 1.e-6){
		    grad*=100000;
		    if(grad.length2() < 1.e-6){
			stagnate=1;
		    } else {
			grad.normalize();
		    }
		} else {
		    grad.normalize();
		}
		grad*=sign;
		F2 = grad*stepsize;
		if(!vfield->interpolate(p+F2*0.5, grad, ix)){
		    inside=0;
		    return 0;
		} else {
		    if(grad.length2() < 1.e-6){
			grad*=100000;
			if(grad.length2() < 1.e-6){
			    stagnate=1;
			} else {
			    grad.normalize();
			}
		    } else {
			grad.normalize();
		    }
		    grad*=sign;
		    F3 = grad*stepsize;
		    if(!vfield->interpolate(p+F3, grad, ix)){
			inside=0;
			return 0;
		    } else {
			if(grad.length2() < 1.e-6){
			    grad*=100000;
			    if(grad.length2() < 1.e-6){
				stagnate=1;
			    } else {
				grad.normalize();
			    }
			} else {
			    grad.normalize();
			}
			grad*=sign;
			F4 = grad * stepsize;
		    }
		}
	    }
	}
    
	p += (F1 + F2 * 2.0 + F3 * 2.0 + F4) / 6.0;
	if(!vfield->interpolate(p, grad, ix)){
	    inside=0;
	    return 0;
	}
	if(grad.length2() < 1.e-6){
	    grad*=100000;
	    if(grad.length2() < 1.e-6){
		stagnate=1;
	    } else {
		grad.normalize();
	    }
	} else {
	    grad.normalize();
	}
	grad*=sign;
    }
    return 1;
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
    } else if(args[1] == "reposition"){
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
	clString pos;
	clString sidstr(id+"-"+to_string(si->sid));
	if(!get_tcl_stringvar(sidstr, "position", pos)){
	  args.error("Error reading position");
	  return;
	}
	si->source->reposition(pos);
    } else {
	Module::tcl_command(args, userdata);
    }
}

SLSourceInfo::SLSourceInfo(int sid, Streamline* module, GeometryOPort* ogeom)
: sid(sid), widget_group(0), widget_geomid(0), geomid(0),
  source(0), need_find(1)
{
    sources.add(scinew SLPointSource(module));
    sources.add(scinew SLLineSource(module));
    sources.add(scinew SLRingSource(module));
    sources.add(scinew SLSquareSource(module));
    // Make the group;
    widget_group=scinew GeomGroup;
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
	for(int i=0;i<sources.size();i++)
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
	if(axis.length2() != 0)
	    axis.normalize();
	else
	    axis=Vector(0,0,1);
	double scale=vfield->longest_dimension();
	need_find=0;
	for(int i=0;i<sources.size();i++)
	    sources[i]->find(cen, axis, scale);
    }
}

#ifdef __GNUG__

#include <Classlib/Array1.cc>

template class Array1<SLSource*>;
template class Array1<SLTracer*>;
template class Array1<GeomPolyline*>;
template class Array1<GeomTube*>;
template class Array1<GeomTriStrip*>;

#include <Classlib/HashTable.cc>
template class HashTable<int, SLSourceInfo*>;
template class HashTableIter<int, SLSourceInfo*>;
template class HashKey<int, SLSourceInfo*>;

#endif
