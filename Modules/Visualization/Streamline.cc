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
#include <Geom/VCTriStrip.h>
#include <Geometry/Point.h>
#include <Widgets/RingWidget.h>
#include <Widgets/PointWidget.h>
#include <Widgets/ScaledFrameWidget.h>

#include <iostream.h>

class Streamline;

class SLTracer {
    Point p;
    int outside;
    double s,t;
public:
    SLTracer(const Point&, double s, double t);
    int advance(const VectorFieldHandle&, double stepsize);
};

struct SLSource {
    Streamline* sl;
    BaseWidget* widget;
    clString name;
    int selected;
    int geomid;
    SLSource(Streamline* sl, BaseWidget*, const clString& name);
    virtual ~SLSource();
    virtual void find(const Point&, const Vector& axis)=0;
    virtual SLTracer* make_tracer(double s, double t)=0;
    virtual void get_n(int& s, int& t)=0;
    void select();
    void deselect();
};

struct SLPointSource : public SLSource {
    PointWidget* pw;
public:
    SLPointSource(Streamline* sl);
    virtual ~SLPointSource();
    virtual void find(const Point&, const Vector& axis);
    virtual SLTracer* make_tracer(double s, double t);
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

    virtual void geom_moved(int, double, const Vector&, void*);
    virtual void geom_release(void*);
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
    new GeomPolyline;
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
    int have_sfield=incolorfield->get(sfield);
    ColormapHandle cmap;
    int have_cmap=incolormap->get(cmap);
    if(have_sfield && !have_cmap)
	have_sfield=0;


    // The material to be used on the last step when we exit the field
    MaterialHandle outmatl(new Material(Color(0,0,0), Color(1,1,1),
					Color(1,1,1), 10.0));

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
    clString sname(get_tcl_stringvar(id, "source"));
    for(int i=0;i<sources.size();i++)
	if(sources[i]->name == sname)
	    source=sources[i];
    if(!source){
	error("Illegal name of source: "+sname);
	return;
    }

    // See if we need to find the field
    if(get_tcl_boolvar(id, "need_find")){
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
	axis*=vfield->longest_dimension()/10;
	set_tclvar(id, "need_find", "false");
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

    // Flush it all out..
    ogeom->flushViews();
}

void Streamline::geom_moved(int axis, double dist, const Vector& delta,
			    void* cbdata)
{
    for(int i=0;i<sources.size();i++){
	if(sources[i]->selected){
	    cerr << "axis=" << axis << "dist=" << dist << "delta=" << delta << endl;
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

SLSource::SLSource(Streamline* sl, BaseWidget* widget, const clString& name)
: sl(sl), widget(widget), name(name), geomid(0), selected(0)
{
}

SLSource::~SLSource()
{
    delete widget;
}

void SLSource::select()
{
    widget->SetState(1);
    selected=1;
}

void SLSource::deselect()
{
    widget->SetState(0);
    selected=0;
}

SLPointSource::SLPointSource(Streamline* sl)
: SLSource(sl, new PointWidget(sl, &sl->widget_lock, 1), "Point")
{
}

SLPointSource::~SLPointSource()
{
}

void SLPointSource::find(const Point& start, const Vector& downstream)
{
    NOT_FINISHED("SLPointSource::find");
}

SLTracer* SLPointSource::make_tracer(double s, double t)
{
    s=t=0;
    return new SLTracer(widget->GetVar(PointW_Point), s, t);
}

void SLPointSource::get_n(int& ns, int& nt)
{
    ns=1;
    nt=1;
}

SLTracer::SLTracer(const Point& p, double s, double t)
: p(p), s(s), t(t), outside(0)
{
}
