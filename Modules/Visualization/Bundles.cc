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
 *  Bundles.cc:  Generate Bundless from a field...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Tester/RigorousTest.h>
#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Math/Trig.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColorMapPort.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/TensorField.h>
#include <Datatypes/TensorFieldPort.h>
#include <Geom/Geom.h>
#include <Geom/Material.h>
#include <Geom/Group.h>
#include <Geom/Pick.h>
#include <Geom/Switch.h>
#include <Geom/Line.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>
#include <Math/MinMax.h>
#include <Multitask/Task.h>
#include <Widgets/RingWidget.h>

#include <iostream.h>
#include <values.h>
#include <strstream.h>

class Bundles;

struct BundleTracer {
    Point p;
    double s,t;
    int inside;
    int ix;
    Vector grad;
    double sign;
    int startstep;
    double starttime;

    BundleTracer(const Point&, double s, double t,
	     const TensorFieldHandle& vfield, double sign);
    virtual ~BundleTracer();
    virtual int advance(const TensorFieldHandle&, double stepsize, int skip)=0;
};

struct BundleEulerTracer : public BundleTracer {
    int stagnate;
    BundleEulerTracer(const Point&, double s, double t,
		  const TensorFieldHandle& vfield, double sign);
    virtual ~BundleEulerTracer();
    virtual int advance(const TensorFieldHandle&, double stepsize, int skip);
};    

struct BundleRK4Tracer : public BundleTracer {
    int stagnate;
    BundleRK4Tracer(const Point&, double s, double t,
		const TensorFieldHandle& vfield, double sign);
    virtual ~BundleRK4Tracer();
    virtual int advance(const TensorFieldHandle&, double stepsize, int skip);
};    

struct BundleSource {
    Bundles* Bundle;
    clString name;
    BaseWidget* widget;
    int selected;
    BundleSource(Bundles* Bundle, const clString& name);
    virtual ~BundleSource();
    virtual void find(const Point&, const Vector& axis, double scale)=0;
    virtual Point trace_start(double s, double t)=0;
    virtual void get_n(int& s, int& t)=0;
    virtual Vector ribbon_direction(double s, double t, const Point&,
				    const TensorFieldHandle& vfield)=0;
    void select();
    void deselect();

    virtual void update_position(const clString& base, const clString& varname)=0;
    virtual void reposition(const clString& pos)=0;
};

struct BundleRingSource : public BundleSource {
    RingWidget* rw;
public:
    BundleRingSource(Bundles* Bundle);
    virtual ~BundleRingSource();
    virtual void find(const Point&, const Vector& axis, double scale);
    virtual Point trace_start(double s, double t);
    virtual void get_n(int& s, int& t);
    virtual Vector ribbon_direction(double s, double t, const Point&,
				    const TensorFieldHandle& vfield);
    virtual void update_position(const clString& base, const clString& varname);
    virtual void reposition(const clString& pos);
};

struct BundleSourceInfo {
    int sid;
    GeomGroup* widget_group;
    int widget_geomid;
    int geomid;

    Array1<BundleSource*> sources;
    BundleSource* source;

    int need_find;
    BundleSourceInfo(int sid, Bundles* module, GeometryOPort* ogeom);
    void pick_source(const clString& source, const TensorFieldHandle& field,
		     Bundles* module);
};

class Bundles : public Module {
    TensorFieldIPort* infield;
    ColorMapIPort* inColorMap;
    GeometryOPort* ogeom;

    int first_execute;

    BundleSourceInfo* source_info;

    virtual void geom_moved(GeomPick*, int, double, const Vector&, void*);
    virtual void geom_release(GeomPick*, void*);

    enum ALGS {
	Euler, RK4
    };
    ALGS alg_enum;

    void make_tracers(BundleSource*, Array1<BundleTracer*>&,
		      const TensorFieldHandle& vfield,
		      double width=0);
    BundleTracer* make_tracer(BundleSource* source, double s, double t,
			  const TensorFieldHandle& vfield,
			  double sign);
    BundleTracer* make_tracer(const Point& p,
			  double s, double t,
			  const TensorFieldHandle& vfield,
			  double sign);
    Array1<BundleTracer*> tracers;
    int np;
    double stepsize;
    int maxsteps;
    int skip;
    TexGeomLines      *line_batch; // all of the stream lines are here...
    double tubesize;
    Mutex grouplock;
    TensorFieldHandle field;
    ColorMapHandle cmap;
    BundleSourceInfo* si;

    int upstream, downstream;
    int drawmode;
    double drawdist;

public:
    void do_bundles(BundleSourceInfo* si);
    void parallel_bundles(int proc);
    inline BundleTracer* left_tracer(const Array1<BundleTracer*> tracers, int i){
	return tracers[i*2];
    }
    inline BundleTracer* right_tracer(const Array1<BundleTracer*> tracers, int i){
	return tracers[i*2+1];
    }
	    
public:
    CrowdMonitor widget_lock;

    Bundles(const clString& id);
    Bundles(const Bundles&, int deep);
    virtual ~Bundles();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual void tcl_command(TCLArgs& args, void* userdata);
};

extern "C" {
Module* make_Bundles(const clString& id)
{
    return scinew Bundles(id);
}
}

static clString widget_name("Bundles Widget");
static clString module_name("Bundles");

Bundles::Bundles(const clString& id)
: Module(module_name, id, Filter), first_execute(1),line_batch(0)
{
    // Create the input ports
    infield=scinew TensorFieldIPort(this, "Tensor Field",
				 TensorFieldIPort::Atomic);
    add_iport(infield);

    inColorMap=scinew ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_iport(inColorMap);

    // Create the output port
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
}

Bundles::Bundles(const Bundles& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("Bundles::Bundles");
}

Bundles::~Bundles()
{
}

Module* Bundles::clone(int deep)
{
    return scinew Bundles(*this, deep);
}

void Bundles::execute()
{
    // Get the data from the ports...
    if(!infield->get(field))
	return;
    int have_cmap=inColorMap->get(cmap);

	GeomGroup *group;

	// Find the current source...
        si=source_info;
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

	// Calculate Bundless
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
	  if(!get_tcl_intvar(sidstr, "drawmode", drawmode)){
	    error("Error reading drawmode variable");
	    return;
	  }
	  if(!get_tcl_doublevar(sidstr, "drawdist", drawdist)){
	    error("Error reading drawdist variable");
	    return;
	  }
	  do_bundles(si);
	}
	// Remove the old and add the new
	if(si->geomid)
	    ogeom->delObj(si->geomid);
	si->geomid=ogeom->addObj(group, module_name+to_string(si->sid));

    // Flush it all out..
    ogeom->flushViews();
    field=0;
    cmap=0;
}

BundleTracer* Bundles::make_tracer(BundleSource* source, double s, double t,
				  const TensorFieldHandle& vfield,
				  double sign)
{
    Point start(source->trace_start(s, t));
    return make_tracer(start, s, t, vfield, sign);
}

BundleTracer* Bundles::make_tracer(const Point& start,
				  double s, double t,
				  const TensorFieldHandle& vfield,
				  double sign)
{
    switch(alg_enum){
    case Euler:
	return scinew BundleEulerTracer(start, s, t, vfield, sign);
    case RK4:
	return scinew BundleRK4Tracer(start, s, t, vfield, sign);
    }
    return 0;
}

void Bundles::make_tracers(BundleSource* source, Array1<BundleTracer*>& tracers,
			      const TensorFieldHandle& vfield,
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


void Bundles::geom_moved(GeomPick*, int, double, const Vector&,
			    void*)
{
}

void Bundles::geom_release(GeomPick*, void*)
{
    if(!abort_flag){
	abort_flag=1;
	want_to_execute();
    }
}

void Bundles::parallel_bundles(int proc)
{
  GeomGroup* group=0;
  int st=proc*tracers.size()/np;
  int et=(proc+1)*tracers.size()/np;
  double maxt=maxsteps*stepsize;
  group=new GeomGroup;
  double ss=stepsize/maxt;
  for(int i=st;i<et;i++){
    int step=tracers[i]->startstep;
    double t=tracers[i]->starttime;
    int inside=1;

//    GeomPolylineTC* line=new GeomPolylineTC(drawmode, drawdist);


    if(inside){


//	line->add(t, tracers[i]->p,
//		  Color(.5,0,0));/*cmap->lookup2(tracers[i]->p.z()*50/-5200)->diffuse);*/

    }
    while(step++< maxsteps && inside){
      inside=tracers[i]->advance(field, stepsize, skip);
      if(inside){


//	line->add(t, tracers[i]->p,
//		  Color(.5,0,0));/*cmap->lookup2(tracers[i]->p.z()*50/-5200)->diffuse);*/


      } else {
	if(tracers[i]->p.x() >= 1280){
	  BundleTracer* ot=tracers[i];
	  Point p(ot->p);
	  cerr << "Went out: x=" << p.x() << endl;
	  if(p.x() >= 1280){
	    cerr << "Went out on right\n";
	    p.x(p.x()-1280);
	  } else if(p.x()<= 0){
	    cerr << "Went out on left\n";
	    p.x(p.x()+1280);
	  }
	  BundleTracer* nt=make_tracer(p, ot->s, ot->t, field, ot->sign);
	  tracers[i]=nt;
	  delete ot;
	  nt->startstep=step;
	  nt->starttime=t;
	  i--;
	}
      }
      t+=ss;
    }


//    group->add(line);


  }
  grouplock.lock();


//  GeomGroup* maingroup=si->get_group(0);
//  maingroup->add(group);


  grouplock.unlock();
  for(i=st;i<et;i++){
    delete tracers[i];
  }
}

static void do_parallel_bundles(void* obj, int proc)
{
  Bundles* module=(Bundles*)obj;
  module->parallel_bundles(proc);
}

void Bundles::do_bundles(BundleSourceInfo* si)
{
    tracers.remove_all();
    make_tracers(si->source, tracers, field);

//    lines.resize(tracers.size());

    np=Min(tracers.size(), Task::nprocessors());
    Task::multiprocess(np, do_parallel_bundles, this);
}

BundleSource::BundleSource(Bundles* Bundle, const clString& name)
: Bundle(Bundle), name(name), widget(0), selected(0)
{
}

BundleSource::~BundleSource()
{
    if(widget)
	delete widget;
}

void BundleSource::select()
{
    if(widget)
	widget->SetState(1);
    selected=1;
}

void BundleSource::deselect()
{
    if(widget)
	widget->SetState(0);
    selected=0;
}

BundleRingSource::BundleRingSource(Bundles* Bundle)
: BundleSource(Bundle, "Ring")
{
    widget=rw=scinew RingWidget(Bundle, &Bundle->widget_lock, 1);
    rw->SetRatio(0.05);
}

BundleRingSource::~BundleRingSource()
{
}

void BundleRingSource::find(const Point& start, const Vector& downstream,
			double scale)
{
    rw->SetPosition(start, downstream, scale/10);
    rw->SetRadius(scale/4);
    rw->SetScale(scale/50);
}

Point BundleRingSource::trace_start(double s, double)
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

void BundleRingSource::reposition(const clString& pos)
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
    Bundle->error("Error parsing position string");
  }
}

void BundleRingSource::update_position(const clString& base, const clString& varname)
{
  char buf[1000];
  ostrstream out(buf, 1000);
  Point cen;
  Vector normal;
  double rad;
  rw->GetPosition(cen, normal, rad);
  double ratio=rw->GetRatio();
  out << cen << " " << normal << " " << rad << " " << ratio << '\0';
  Bundle->set_tclvar(base, varname, buf);
}


void BundleRingSource::get_n(int& ns, int& nt)
{
    double ratio=rw->GetRatio();
    if(ratio < 1.e-3)
	ns=1;
    else
	ns=Floor(1./ratio);
    nt=1;
}

Vector BundleRingSource::ribbon_direction(double s, double,
				      const Point&,
				      const TensorFieldHandle&)
{
    double ratio=rw->GetRatio();
    double angle=s*ratio*2*Pi;
    Vector v1, v2;
    rw->GetPlane(v1, v2);
    return v1*Sin(angle)+v2*Cos(angle);
}

BundleTracer::BundleTracer(const Point& p, double s, double t,
		   const TensorFieldHandle& field, double sign)
: p(p), s(s), t(t), inside(1), sign(sign), startstep(0), starttime(0)
{
    // Interpolate initial gradient


//    if(!field->interpolate(p, grad))


	inside=0;
}

BundleTracer::~BundleTracer()
{
}


BundleEulerTracer::BundleEulerTracer(const Point& p, double s, double t,
			     const TensorFieldHandle& field,
			     double sign)
: BundleTracer(p, s, t, field, sign)
{
    stagnate=0;
    ix=0;
}

BundleEulerTracer::~BundleEulerTracer()
{
}

int BundleEulerTracer::advance(const TensorFieldHandle& field, double stepsize, int skip)
{
    if(stagnate){
	inside=0;
	return 0;
    }
    for(int i=0;i<skip;i++){


//	if(!field->interpolate(p, grad, ix)){


	if(0){
	    inside=0;
	    return 0;
	} else {
	  if(grad.length2() < 1.e-6){
	    stagnate=1;
	  }
	    grad*=sign;
	    p+=(grad*stepsize);
	}
    }
    return 1;
}

BundleRK4Tracer::BundleRK4Tracer(const Point& p, double s, double t,
			 const TensorFieldHandle& field, double sign)
: BundleTracer(p, s, t, field, sign)
{
    stagnate=0;
    ix=0;
}

BundleRK4Tracer::~BundleRK4Tracer()
{
}

int BundleRK4Tracer::advance(const TensorFieldHandle& field, double stepsize, int skip)
{
    Vector F1, F2, F3, F4;

    if(stagnate){
	inside=0;
	return 0;
    }
    for(int i=0;i<skip;i++){


//	if(!field->interpolate(p, grad, ix)){


	if (0){
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


//	    if(!field->interpolate(p+F1*0.5, grad, ix)){

	    if (0) {
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


//		if(!field->interpolate(p+F2*0.5, grad, ix)){
		if(0) {

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


//		    if(!field->interpolate(p+F3, grad, ix)){

		    if(0){
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


//	if(!field->interpolate(p, grad, ix)){

	
	if(0){
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

void Bundles::tcl_command(TCLArgs& args, void* userdata)
{
    if(args.count() < 2){
	args.error("Bundles needs a minor command");
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
	source_info=new BundleSourceInfo(sid, this, ogeom);
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
	source_info->need_find=1;
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
	BundleSourceInfo* si;
	clString pos;
	clString sidstr(id+"-"+to_string(si->sid));
	if(!get_tcl_stringvar(sidstr, "position", pos)){
	  args.error("Error reading position");
	  return;
	}
	source_info->source->reposition(pos);
    } else {
	Module::tcl_command(args, userdata);
    }
}

BundleSourceInfo::BundleSourceInfo(int sid, Bundles* module, GeometryOPort* ogeom)
: sid(sid), widget_group(0), widget_geomid(0), geomid(0),
  source(0), need_find(1)
{
    sources.add(scinew BundleRingSource(module));
    // Make the group;
    widget_group=scinew GeomGroup;
    for(int i=0;i<sources.size();i++)
	widget_group->add(sources[i]->widget->GetWidget());
    widget_geomid=ogeom->addObj(widget_group,
				widget_name+to_string(sid),
				&module->widget_lock);
}

void BundleSourceInfo::pick_source(const clString& sname,
			       const TensorFieldHandle& field,
			       Bundles* module)
{
    BundleSource* newsource=0;
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


//	field->get_bounds(min, max);


	Point cen(AffineCombination(min, 0.5, max, 0.5));
	Vector axis;


//	if(!field->interpolate(cen, axis)){


	if(0){
	    // No field???
	    module->error("Can't find center of field");
	    return;
	}


	if(axis.length2() != 0)
	    axis.normalize();
	else
	    axis=Vector(0,0,1);
	double scale;


//	scale=field->longest_dimension();


	need_find=0;
	for(int i=0;i<sources.size();i++)
	    sources[i]->find(cen, axis, scale);
    }
}

#ifdef __GNUG__
/*
 * These template instantiations can't go in templates.cc, because
 * the classes are defined in this file.
 */
#include <Classlib/Array1.cc>
template class Array1<VertBatch>;
#endif
