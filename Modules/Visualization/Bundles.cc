/*
 *  Bundles.cc:  Generate bundles from a field...
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Tester/RigorousTest.h>
#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Math/Trig.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColorMapPort.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/TensorField.h>
#include <Datatypes/TensorFieldPort.h>
#include <Datatypes/VectorField.h>
#include <Datatypes/VectorFieldPort.h>
#include <Geom/Geom.h>
#include <Geom/Material.h>
#include <Geom/Group.h>
#include <Geom/Pick.h>
#include <Geom/Switch.h>
#include <Geom/Line.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>
#include <Math/MusilRNG.h>
#include <Math/Expon.h>
#include <Multitask/Task.h>
#include <Widgets/RingWidget.h>

#include <iostream.h>
#include <values.h>
#include <strstream.h>

MusilRNG mr;
TensorFieldBase *tfield;
VectorField *vfield;
ScalarField *sfield;
ColorMap* cmap;

Vector matvecmult3by3(double A[][3], const Vector& v) {
    double x[3];
    x[0]=v.x(); x[1]=v.y(); x[2]=v.z();
    double b[3];
    b[0]=b[1]=b[2]=0;
    for (int i=0; i<3; i++)
	for (int j=0; j<3; j++)
	    b[i]+=A[i][j]*x[j];
    return Vector(b[0], b[1], b[2]);
}

Vector problisticAdvect(double t[][3], const Vector &inDir) {
    // make a vector v, which will be a random vector *in* a sphere
    double x(mr()-.5);
    double y(mr()-.5);
    double z(mr()-.5);
    Vector v(x,y,z);
    v.normalize();
    double r(mr());
    r=Cbrt(r);
    v*=r;

    // now, push v through the tensor and make its dot-product with inDir pos
    Vector newDir=matvecmult3by3(t, v);
    if (Dot(newDir,inDir) < 0) newDir*=-1;
    return newDir;
}

struct Fiber {
    Point pos;
    double stepsize;
    int nsteps;
    Vector dir;
    int inside;
    int stagnate;
    int iDir;

    Fiber(const Point &p, int ns, double ss, const Vector &d, int idir)
	: pos(p), nsteps(ns), stepsize(ss), dir(d),
          inside(1), stagnate(0), iDir(idir) {};
    ~Fiber() {};
    int advanceV();
    int advanceT();
    void advect(TexGeomLines *);
};

void Fiber::advect(TexGeomLines *lines) {
    double lifetime=mr();
    lifetime=1-lifetime*lifetime;
    Array1<double> time;
    Array1<Point> posn;
    Array1<Color> clrs;

    posn.add(pos);
    if (sfield && cmap) {
	Color cv(0,0,0);
	double sval;
	if (sfield->interpolate(pos,sval)) {
	    MaterialHandle matl(cmap->lookup(sval));
	    cv = matl->diffuse;
	}
	clrs.add(cv);
    }
    for (int i=0; i<nsteps*lifetime && inside; i++) {
	if ((tfield && advanceT()) || (!tfield && advanceV())) {
	    posn.add(pos);
	    if (sfield && cmap) {
		Color cv(0,0,0);
		double sval;
		if (sfield->interpolate(pos,sval)) {
		    MaterialHandle matl(cmap->lookup(sval));
		    cv = matl->diffuse;
		}
		clrs.add(cv);
	    }
	}
    }
    if (posn.size() != clrs.size()) cerr << "ERROR -- time.size()="<<time.size()<<" posn.size()="<<posn.size()<<" clrs.size()="<<clrs.size()<<"\n";
    if (posn.size()>1) lines->batch_add(time, posn, clrs);
}

int Fiber::advanceV() {
    Vector F1, F2, F3, F4, grad;
    Point p(pos);
    int ix;

    if(stagnate){
	inside=0;
	return 0;
    }
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
    F1 = grad*(stepsize*iDir);

    if(!vfield->interpolate(p+F1*0.5, grad, ix)){
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
    F2 = grad*(stepsize*iDir);

    if(!vfield->interpolate(p+F2*0.5, grad, ix)){
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
    F3 = grad*(stepsize*iDir);

    if(!vfield->interpolate(p+F3, grad, ix)){
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
    F4 = grad*(stepsize*iDir);
    
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
    pos=p;
    dir=grad;
    return 1;
}

int Fiber::advanceT() {
    Vector F1, F2, F3, F4, grad;
    int ix;
    Point p(pos);
    double tens[3][3];

    if(stagnate){
	inside=0;
	return 0;
    }
    if(!tfield->interpolate(p, tens, ix)){
	inside=0;
	return 0;
    }	
    grad=problisticAdvect(tens, dir);
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
    F1 = grad*stepsize;

    if(!tfield->interpolate(p+F1*0.5, tens, ix)){
	inside=0;
	return 0;
    }
    grad=problisticAdvect(tens, dir);
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
    F2 = grad*stepsize;

    if(!tfield->interpolate(p+F2*0.5, tens, ix)){
	inside=0;
	return 0;
    }
    grad=problisticAdvect(tens, dir);
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
    F3 = grad*stepsize;

    if(!tfield->interpolate(p+F3, tens, ix)){
	inside=0;
	return 0;
    }
    grad=problisticAdvect(tens, dir);
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
    F4 = grad * stepsize;
    
    p += (F1 + F2 * 2.0 + F3 * 2.0 + F4) / 6.0;
    if(!tfield->interpolate(p, tens, ix)){
	inside=0;
	return 0;
    }
    grad=problisticAdvect(tens, dir);
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
    pos=p;
    dir=grad;
    return 1;
}

struct FiberBundle {
    RingWidget* rw;
    Array1<Fiber*> fibers;
    Point mid;
    Vector dir;
    int step;
    double rad;

    FiberBundle(const Array1<Point> &pts, int nsteps, const Vector &v, 
		int nfibers, double stepsize, double r, int idir);
    ~FiberBundle() {};
    GeomObj *advect(int niters);
};

FiberBundle::FiberBundle(const Array1<Point> &pts, int nsteps, const Vector &v,
			 int nfibers, double stepsize, double r, int idir) {
    cerr << "Creating bundle in direction "<<v<<"...\n";
    rad=r;
    step=0;
    fibers.resize(nfibers);
    for (int i=0; i<nfibers; i++) {
      // find the position of each fiber, nad the step when it will be created
	fibers[i]=new Fiber(pts[i], nsteps, stepsize, v, idir);
    }
}

GeomObj *FiberBundle::advect(int niters) {
    cerr << "Advecting fibers through field...\n";
    TexGeomLines *lines=new TexGeomLines;
    lines->alpha=1.;

    for (int i=0; i<niters; i++) {
	Vector newCtr(0,0,0);
	Vector newDir(0,0,0);
	for (int j=0; j<fibers.size(); j++) {
	    fibers[j]->advect(lines);
	    newCtr+=fibers[j]->pos.vector();
	    newDir+=fibers[j]->dir;
	}
	newCtr=newCtr/fibers.size();
	newDir=newDir/fibers.size();
	for (j=0; j<fibers.size(); j++) {
	    Vector disp(mr()-.5, mr()-.5, mr()-.5);
	    fibers[j]->pos=newCtr.point()+disp*(rad*2);
	    fibers[j]->dir=newDir;
	    fibers[j]->inside=1;
	    fibers[j]->stagnate=0;
	}
    }
    return lines;
}

class Bundles : public Module {
    TensorFieldIPort* itfport;
    VectorFieldIPort* ivfport;
    ScalarFieldIPort* isfport;
    ColorMapIPort* icmport;
    GeometryOPort* ogport;

    FiberBundle* fb_fwd;
    FiberBundle* fb_bwd;
    int first_execute;
    virtual void geom_moved(GeomPick*, int, double, const Vector&, void*);
    virtual void geom_release(GeomPick*, void*);

    TCLint nsteps;
    TCLint nfibers;
    TCLdouble stepsize;
    TCLint whichdir;
    TCLint niters;

    TensorFieldHandle tfh;
    VectorFieldHandle vfh;
    ColorMapHandle cmh;
    ScalarFieldHandle sfh;

    clString msg;

    RingWidget *rw;
    CrowdMonitor widget_lock;
    int gid;
public:
    Bundles(const clString& id);
    Bundles(const Bundles&, int deep);
    virtual ~Bundles();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_Bundles(const clString& id)
{
    return scinew Bundles(id);
}
}

static clString module_name("Bundles");

Bundles::Bundles(const clString& id)
: Module(module_name, id, Filter), first_execute(1), gid(0),
  nsteps("nsteps", id, this), nfibers("nfibers", id, this),
  stepsize("stepsize", id, this), whichdir("whichdir", id, this),
  niters("niters", id, this)
{
    // Create the input ports
    itfport=scinew TensorFieldIPort(this, "Tensor Field",
				    TensorFieldIPort::Atomic);
    add_iport(itfport);
    ivfport=scinew VectorFieldIPort(this, "Vector Field",
				    VectorFieldIPort::Atomic);
    add_iport(ivfport);

    isfport=scinew ScalarFieldIPort(this, "Scalar Field",
				    ScalarFieldIPort::Atomic);
    add_iport(isfport);

    icmport=scinew ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_iport(icmport);

    // Create the output port
    ogport=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogport);
}

Bundles::Bundles(const Bundles& copy, int deep)
: Module(copy, deep), first_execute(1),
  nsteps("nsteps", id, this), nfibers("nfibers", id, this),
  stepsize("stepsize", id, this), whichdir("whichdir", id, this),
  niters("niters", id, this)
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
    int havetensors;
    if(!itfport->get(tfh)) {
	havetensors=0;
	tfield=tfh.get_rep();
    } else havetensors=1;

    if (!havetensors)
	if (!ivfport->get(vfh)) return;
	else vfield=vfh.get_rep();

    if (icmport->get(cmh)) cmap=cmh.get_rep();
    else cmap=0;

    if (isfport->get(sfh)) sfield=sfh.get_rep();
    else sfield=0;

    if (first_execute) {
	Point fmin, fmax, fctr;
	Vector fdiag;
	if (havetensors) {
	    TensorField<double> *tfd;
	    tfd=dynamic_cast<TensorField<double> *>(tfh.get_rep());
	    if (tfd) tfd->get_bounds(fmin,fmax);
	} else {
	    vfh->get_bounds(fmin, fmax);
	}
	fdiag=(fmax-fmin);
	fctr=fmin+fdiag*.33;
	double fscale=fdiag.length();
	rw=scinew RingWidget(this, &widget_lock, fscale/1000.);
	rw->SetRatio(0.5);
	Point ppp;
	Vector vvv;
	double sss;
	rw->GetPosition(ppp, vvv, sss);
	rw->SetPosition(fctr, vvv, fscale/200.);

	GeomObj *w=rw->GetWidget();
	ogport->addObj(w, clString("Ring Widget"), &widget_lock);
	rw->Connect(ogport);
	fb_fwd=fb_bwd=0;
	first_execute=0;
    }
    Array1<Point> pts(nfibers.get());
    Point ctr;
    Vector nrml;
    double rad;
    rw->GetPosition(ctr, nrml, rad);
    // build the array of points that'll be the start positions
    for (int i=0; i<pts.size(); i++) {
	Vector disp(mr()-.5, mr()-.5, mr()-.5);
	pts[i]=ctr+disp*(rad*2);
    }
    if (fb_fwd) {delete fb_fwd; fb_fwd=0;}
    if (fb_bwd) {delete fb_bwd; fb_bwd=0;}
    int whichDir=whichdir.get();
    if (whichDir==0 || whichDir==2)
	fb_fwd=new FiberBundle(pts, nsteps.get(), nrml, nfibers.get(), 
			       stepsize.get(), rad, 1);
    if (whichDir==1 || whichDir==2)
	fb_bwd=new FiberBundle(pts, nsteps.get(), -nrml, nfibers.get(),
			       stepsize.get(), rad, -1);	

    GeomGroup *g=new GeomGroup;
    if (fb_fwd) g->add(fb_fwd->advect(niters.get()));
    if (fb_bwd) g->add(fb_bwd->advect(niters.get()));

    if (gid) ogport->delObj(gid);
    if (g)
	gid = ogport->addObj(g, "Bundles");
}

void Bundles::geom_moved(GeomPick*, int, double, const Vector&,
			    void*)
{
}

void Bundles::geom_release(GeomPick*, void*)
{
    if(!abort_flag){
	abort_flag=1;
	msg="ringmoved";
	want_to_execute();
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
