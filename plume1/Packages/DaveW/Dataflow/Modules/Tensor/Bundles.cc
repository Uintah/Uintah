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

#include <Packages/DaveW/Core/Datatypes/General/TensorFieldPort.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/MeshPort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Dataflow/Ports/VectorFieldPort.h>
#include <Dataflow/Widgets/GaugeWidget.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomPick.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Expon.h>
#include <Core/Math/MusilRNG.h>
#include <Core/Math/Trig.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
using std::cerr;
#include <values.h>

namespace DaveW {
using namespace DaveW;
using namespace SCIRun;

MusilRNG *mr=0;
TensorFieldBase *tfield;
VectorField *vfield;
ScalarField *sfield;
ScalarField *anisofield;
ColorMap* cmap;
double pnctr=0;
double dmrcl=0;
double diffScale=1;
int bunds;
Color myclrs[7];
int myseed;

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

Vector problisticAdvect(double t[][3], const Vector &inDir, double aniso) {
    // make a vector v, which will be a random vector *in* a sphere
    double x((*mr)()-.5);
    double y((*mr)()-.5);
    double z((*mr)()-.5);
    Vector v(x,y,z);
    v.normalize();
    double r((*mr)());
    r=Cbrt(r);
    v*=r;

    // now, push v through the tensor and make its dot-product with inDir pos
    Vector newDir=matvecmult3by3(t, v);
    if (newDir.length()<0.000000001) newDir.x(1);
    newDir.normalize();

    if (Dot(newDir,inDir) < 0) newDir*=-1;

    double mm=pnctr*(1-aniso*aniso);
    
//    if (aniso<.1) cerr << "  v="<<v<<" newDir="<<newDir<<" inDir="<<inDir<<" mm="<<mm;

    newDir=Interpolate(newDir, inDir, mm);
//    if (aniso<.1) cerr << " finDir="<<newDir;
    return newDir;
}

Vector advectionDiffusion(double t[][3], const Vector &inDir) {
    Vector v(inDir);
    v.normalize();

    // now, push v through the tensor and make its dot-product with inDir pos
    Vector newDir=matvecmult3by3(t, v);
    if (newDir.length()<0.000000001) newDir.x(1);
    newDir.normalize();

    if (Dot(newDir,inDir) < 0) newDir*=-1;
    return newDir;
}

struct Fiber {
    Point pos;
    double stepsize;
    Vector dir;
    int inside;
    int stagnate;
    int iDir;
    int lifetime;
    Array1<double> time;
    Array1<Point> posn;
    Array1<Color> clrs;
    int step;
    
    Fiber(const Point &p, double ss, const Vector &d, int idir)
	: pos(p), stepsize(ss), dir(d), step(0),
          inside(1), stagnate(0), iDir(idir) {};
    ~Fiber() {};
    int advanceV();
    int advanceT();
    int advanceT_AD();
    void advectFirst();
    void advect();
};

void Fiber::advectFirst() {
    time.resize(0);
    posn.resize(0);
    clrs.resize(0);

    if (sfield && cmap) {
	Color cv(0,0,0);
	double sval;
	if (sfield->interpolate(pos,sval)) {
	    MaterialHandle matl(cmap->lookup(sval));
	    cv = matl->diffuse;
	    posn.add(pos);
	    clrs.add(cv);
	} else {
	    inside=0;
	}
    } else {
	posn.add(pos);
	clrs.add(myclrs[myseed%7]);
    }
    step=1;
}

void Fiber::advect() {
    step++;
    int ok=1;

    if (tfield) ok=advanceT_AD();
    else if (vfield) ok=advanceV();

    if (ok) {
	Point last=posn[posn.size()-1];
	if ((last-pos).length2() > 0.000001) {
	    posn.add(pos);
	    if (sfield && cmap) {
		Color cv(0,0,0);
		double sval;
		if (sfield->interpolate(pos,sval)) {
		    MaterialHandle matl(cmap->lookup(sval));
		    cv = matl->diffuse;
		}
		clrs.add(cv);
	    } else {
		clrs.add(myclrs[myseed%7]);
	    }
	}
    }
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
    double aniso=0;

    if(stagnate){
	inside=0;
	return 0;
    }
    if(!tfield->interpolate(p, tens, ix)){
	inside=0;
	return 0;
    }	
    if(anisofield) anisofield->interpolate(p, aniso, ix);
    grad=problisticAdvect(tens, dir, aniso);
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
    if(anisofield) anisofield->interpolate(p, aniso, ix);
    grad=problisticAdvect(tens, dir, aniso);
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
    if(anisofield) anisofield->interpolate(p, aniso, ix);
    grad=problisticAdvect(tens, dir, aniso);
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
    if(anisofield) anisofield->interpolate(p, aniso, ix);
    grad=problisticAdvect(tens, dir, aniso);
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
    if(anisofield) anisofield->interpolate(p, aniso, ix);
    grad=problisticAdvect(tens, dir, aniso);
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

void printMat3x3(double t[3][3]) {
    for (int i=0; i<3; i++) {
	cerr <<"     ";
	for (int j=0; j<3; j++)
	    cerr << t[i][j]<<" ";
	cerr <<"\n";
    }
}

int Fiber::advanceT_AD() {
    Vector grad;
    int ix;
    double tens[3][3];

//    cerr << "At point "<<p<<"  dir="<<dir<<"  tensor=\n";

    if(stagnate){
	inside=0;
	return 0;
    }
    if(!tfield->interpolate(pos, tens, ix)){
	inside=0;
	return 0;
    }	
//    printMat3x3(tens);

    grad=advectionDiffusion(tens, dir);
    grad*=1./diffScale;
    
    Vector fpc;
    if (tfield->m_vectorsGood) {
	tfield->m_e_vectors[0].interpolate(pos,fpc,ix);
	if (Dot(fpc,dir)<0) fpc=-fpc;
    } else {
	fpc=grad;
    }

    double cl;
    if (!anisofield) {
	cerr << "BUILDING ANISOTROPY VOLUME FOR ADVECT_AD\n";
	Point min, max;
	tfield->get_bounds(min, max);
	ScalarFieldRGdouble *afield = scinew ScalarFieldRGdouble();
	afield->set_bounds(min, max);
	afield->resize(tfield->m_height, tfield->m_width, tfield->m_slices);
	for (int z = 0; z < tfield->m_slices; z++)
	    for (int y = 0; y < tfield->m_width; y++) 
		for (int x = 0; x < tfield->m_height; x++) {
		    double e1 = tfield->m_e_values[0].grid(x,y,z);
		    double e2 = tfield->m_e_values[1].grid(x,y,z);
		    double e3 = tfield->m_e_values[2].grid(x,y,z);
		    float ins = tfield->m_inside(x,y,z);
		    double average = (e1 + e2 + e3)/3.0;
		    if (average < 0.00000001)
			{
//			    printf("hack: %f\n", average);
			    average = 0.00000001;
			}
		    afield->grid(x,y,z) = (e1 - e3)/(3.0 * average)*ins;
		}
	anisofield=afield;
    }
    
    anisofield->interpolate(pos,cl,ix);
    int ii, jj, kk;
    tfield->m_e_values[0].locate(pos, ii, jj, kk);
    float ins=0;
    if (ii>=0 && ii<tfield->m_height && jj>=0 && jj<tfield->m_width && 
	kk>=0 && kk<tfield->m_slices)
	ins=tfield->m_inside(ii,jj,kk);

    if (ins<.5) { stagnate=1; return 0;} 

    double K=(1-dmrcl)*(cl-1)+1;
    double alpha=(1-dmrcl)*(1-cl);
    
    Vector dir=fpc*K + (dir*(1-pnctr)+grad*pnctr)*alpha;
    if (dir.length2() < 1.e-10) {stagnate=1; return 0;}

    double dirl=1./dir.length();

    pos=pos+dir*(dirl*stepsize);

    return 1;
}

struct FiberBundle {
    GaugeWidget* gw;
    Array1<Fiber*> fibers;
    Point mid;
    Vector dir;
    int step;
    double radx;
    double rady;
    double radz;
    FiberBundle(const Array1<Point> &pts, const Vector &v, 
		int nfibers, double stepsize, double rx, 
		double ry, double rz, int idir);
    ~FiberBundle() {};
    GeomObj *advect(int niters, int nsteps);
};

FiberBundle::FiberBundle(const Array1<Point> &pts, const Vector &v,
			 int nfibers, double stepsize, double rx, 
			 double ry, double rz, int idir) {
//    cerr << "Creating bundle in direction "<<v<<"...\n";
    radx=rx;
    rady=ry;
    radz=rz;
    step=0;
    fibers.resize(nfibers);
    for (int i=0; i<nfibers; i++) {
      // find the position of each fiber, and the step when it will be created
	fibers[i]=new Fiber(pts[i], stepsize, v, idir);
    }
}

GeomObj *FiberBundle::advect(int niters, int nsteps) {
//    cerr << "Advecting fibers through field...\n";
//    cerr << "niters="<<niters<<"  nsteps="<<nsteps<<"\n";
    TexGeomLines *lines=new TexGeomLines;
    lines->alpha=1.;

    Point mid;
    Vector newDir;
    int needAvg;
    int i,j;
    for (i=0; i<niters*nsteps; i++) {
	needAvg=0;
	for (j=0; j<fibers.size(); j++) {
	    if (fibers[j]->step == -1) {  // need to reseed
		double x((*mr)()-.5);
		double y((*mr)()-.5);
		double z((*mr)()-.5);
		Vector v(x,y,z);
		v.normalize();
		double r((*mr)());
		r=Cbrt(r);
		v*=r;
		v.x(v.x()*radx);
		v.y(v.y()*rady);
		v.z(v.z()*radz);
		fibers[j]->pos=mid+v;
		fibers[j]->dir=newDir;
		fibers[j]->inside=1;
		fibers[j]->stagnate=0;
		fibers[j]->step=0;
	    }
	    if (fibers[j]->step == 0) {
		double lt=(*mr)();
		fibers[j]->lifetime=(1-lt*lt)*nsteps;
//		cerr << "fiber["<<j<<"] lifetime="<<fibers[j]->lifetime<<"\n";
		fibers[j]->advectFirst();
	    } else {
		fibers[j]->advect();
	    }
	    if (fibers[j]->step >= fibers[j]->lifetime ||
		!fibers[j]->inside || fibers[j]->stagnate) {
		if (fibers[j]->posn.size()>1) 
		    if (sfield && cmap) {
			lines->batch_add(fibers[j]->time, fibers[j]->posn, 
					 fibers[j]->clrs);
		    } else {
			lines->batch_add(fibers[j]->time, fibers[j]->posn,
					 fibers[j]->clrs);
		    }
		if (!fibers[j]->inside) { 
		    if ((j==0) && bunds) {  // our main bundler died!
			fibers.resize(0);
		    } else {
			fibers.remove(j);
			j--; 
		    }
		} else {
		    needAvg=1;
		    fibers[j]->step=-1;
		}
	    }
	}
	if (needAvg) {
	    if (bunds && fibers.size()) { // follow fiber 0!
		mid=fibers[0]->pos;
		newDir=fibers[0]->dir;
	    } else {
		mid = Point(0,0,0);
		newDir=Vector(0,0,0);
		for (j=0; j<fibers.size(); j++) {
		    mid+=fibers[j]->pos.vector();
		    newDir+=fibers[j]->dir;
		}
		mid.x(mid.x()/fibers.size());
		mid.y(mid.y()/fibers.size());
		mid.z(mid.z()/fibers.size());
		newDir=newDir/fibers.size();
	    }
	}
    }
    return lines;
}

class Bundles : public Module {
    TensorFieldIPort* itfport;
    VectorFieldIPort* ivfport;
    ScalarFieldIPort* isfport;
    ScalarFieldIPort* isf2port;
    ColorMapIPort* icmport;
    GeometryOPort* ogport;

    FiberBundle* fb_fwd;
    FiberBundle* fb_bwd;
    int first_execute;
    virtual void geom_release(GeomPick*, void*);

    GuiInt nsteps;
    GuiInt nfibers;
    GuiDouble stepsize;
    GuiDouble puncture;
    GuiDouble demarcelle;
    GuiInt whichdir;
    GuiInt niters;
    GuiInt uniform;
    GuiInt bundlers;
    GuiDouble bundleradx;
    GuiDouble bundlescy;
    GuiDouble bundlescz;
    GuiDouble startx;
    GuiDouble starty;
    GuiDouble startz;
    GuiDouble endx;
    GuiDouble endy;
    GuiDouble endz;
    GuiString seed;
    TensorFieldHandle tfh;
    VectorFieldHandle vfh;
    ColorMapHandle cmh;
    ScalarFieldHandle sfh;
    ScalarFieldHandle sfh2;

    clString msg;

    GaugeWidget *gw;
    CrowdMonitor widget_lock;
    int gid;
public:
    Bundles(const clString& id);
    virtual ~Bundles();
    virtual void execute();
    void tcl_command( TCLArgs&, void * );
};

extern "C" Module* make_Bundles(const clString& id)
{
    return scinew Bundles(id);
}

static clString module_name("Bundles");

Bundles::Bundles(const clString& id)
: Module(module_name, id, Filter), first_execute(1), gid(0),
  nsteps("nsteps", id, this), nfibers("nfibers", id, this),
  stepsize("stepsize", id, this), whichdir("whichdir", id, this),
  niters("niters", id, this), bundleradx("bundleradx", id, this),
  bundlescy("bundlescy", id, this), bundlescz("bundlescz", id, this),
  puncture("puncture", id, this), demarcelle("demarcelle", id, this),
  startx("startx", id, this), starty("starty", id, this), 
  startz("startz", id, this), endx("endx", id, this),
  endy("endy", id, this), endz("endz", id, this),
  uniform("uniform", id, this), seed("seed", id, this),
  bundlers("bundlers", id, this), widget_lock("Bundles widget_lock")
{
    // Create the input ports
    itfport=scinew TensorFieldIPort(this, "Tensor Field",
				    TensorFieldIPort::Atomic);
    add_iport(itfport);
    ivfport=scinew VectorFieldIPort(this, "Vector Field",
				    VectorFieldIPort::Atomic);
    add_iport(ivfport);

    isfport=scinew ScalarFieldIPort(this, "Color",
				    ScalarFieldIPort::Atomic);
    add_iport(isfport);

    isf2port=scinew ScalarFieldIPort(this, "Anisotropy",
				    ScalarFieldIPort::Atomic);
    add_iport(isf2port);

    icmport=scinew ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_iport(icmport);

    // Create the output port
    ogport=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogport);
    anisofield=0;
    sfield=0;

    myclrs[0]=Color(.7,.1,.1);
    myclrs[1]=Color(.1,.7,.1);
    myclrs[2]=Color(.1,.1,.7);
    myclrs[3]=Color(.7,.7,.1);
    myclrs[4]=Color(.7,.1,.7);
    myclrs[5]=Color(.1,.7,.7);
    myclrs[6]=Color(.6,.6,.6);

}

Bundles::~Bundles()
{
}

void Bundles::execute()
{
    // Get the data from the ports...
    int havetensors;
    if(!itfport->get(tfh)) {
	havetensors=0;
	tfield=0;
    } else {
	havetensors=1;
	tfield=tfh.get_rep();
	double min;
	if (tfield->m_valuesGood) {
	    tfield->m_e_values[0].get_minmax(min, diffScale);
	    diffScale/=4;
	    if (diffScale<1.e-10) diffScale=1;
	}
    }
    if (!havetensors)
	if (!ivfport->get(vfh)) return;
	else vfield=vfh.get_rep();
    if (icmport->get(cmh)) cmap=cmh.get_rep();
    else cmap=0;
    if (isfport->get(sfh)) sfield=sfh.get_rep();
    if (isf2port->get(sfh2)) anisofield=sfh2.get_rep();
    pnctr=puncture.get();
    dmrcl=demarcelle.get();

    if (mr) delete(mr);
    int sd=atoi((seed.get())());
    myseed=sd;
    mr = new MusilRNG(abs(sd));
    cerr << "SEEDING MR WITH "<<abs(sd)<<"\n";
    if (sd>=0) {
	clString seedval(to_string(atoi((seed.get())())+1));
	seed.set(seedval);
    }
    bunds=bundlers.get();
    if (first_execute) {
	Point fmin, fmax, fctr;
	Vector fdiag;
	if (havetensors) {
	    tfh->get_bounds(fmin, fmax);
	} else {
	    vfh->get_bounds(fmin, fmax);
	}
	fdiag=(fmax-fmin);
	fctr=fmin+fdiag*.33;
	double fscale=fdiag.length();
	gw=scinew GaugeWidget(this, &widget_lock, fscale/1000.0, true);
	bundleradx.set(fscale/1000.);
	reset_vars();
	Vector dir(1,0,0);
	if (vfield) vfield->interpolate(fctr, dir);
	else if (tfield && tfield->m_vectorsGood) 
	    if (tfield->m_e_vectors[0].interpolate(fctr, dir)) {
		cerr << "Using eigenvect dir: "<<dir<<"\n";
	    }
	gw->SetEndpoints(fctr, fctr+(dir*3));
	gw->SetRatio(0.1);
	GeomObj *w=gw->GetWidget();
	ogport->addObj(w, clString("Gauge Widget"), &widget_lock);
	gw->Connect(ogport);
	fb_fwd=fb_bwd=0;
	first_execute=0;
    }
    Array1<Point> pts(nfibers.get());
    Point ctr, pt1;
    double radx, rady, radz;
    gw->GetEndpoints(ctr, pt1);
    Vector dir=pt1-ctr;
    dir.normalize();
    radx=bundleradx.get();
    rady=bundlescy.get()*radx;
    radz=bundlescz.get()*radx;

    cerr << "radx="<<radx<<"\n";
    cerr << "rady="<<rady<<"\n";
    cerr << "radz="<<radz<<"\n";

    gw->SetScale(radx);
    // build the array of points that'll be the start positions

#if 0
    if (uniform.get()) {
	if (pts.size()) {
	    pts[0]=ctr;
	    for (int i=1; i<pts.size(); i++) {
		double dd=i/(pts.size()-1.);
		Vector disp((dd-.5)*2*radx,(dd-.5)*2*rady,(dd-.5)*2*radz);
		pts[i]=ctr+disp;
	    }
	}
    } else {
	if (pts.size()) {
	    pts[0]=ctr;
	    for (int i=1; i<pts.size(); i++) {
		Vector disp(((*mr)()-.5)*2*radx, ((*mr)()-.5)*2*rady, ((*mr)()-.5)*2*radz);
		pts[i]=ctr+disp;
	    }
	}
    }
#endif
    Vector lngth=pt1-ctr;
    if (pts.size()) {
	lngth*=1./pts.size();
	Point curr(ctr+lngth/2);
	for (int ii=0; ii<pts.size(); ii++) { pts[ii]=curr; curr+=lngth; }
    }
    if (fb_fwd) {delete fb_fwd; fb_fwd=0;}
    if (fb_bwd) {delete fb_bwd; fb_bwd=0;}
    int whichDir=whichdir.get();
    
    if (whichDir==0 || whichDir==2)
	fb_fwd=new FiberBundle(pts, dir, nfibers.get(), 
			       stepsize.get(), radx, rady, radz, 1);
    if (whichDir==1 || whichDir==2)
	fb_bwd=new FiberBundle(pts, dir*(-1), nfibers.get(),
			       stepsize.get(), radx, rady, radz, -1);	

    GeomGroup *g=new GeomGroup;
    if (fb_fwd) g->add(fb_fwd->advect(niters.get(), nsteps.get()));
    if (fb_bwd) g->add(fb_bwd->advect(niters.get(), nsteps.get()));

    if (gid) ogport->delObj(gid);
    if (g)
	gid = ogport->addObj(g, "Bundles");
}

void Bundles::geom_release(GeomPick*, void*)
{
    if(!abort_flag){
	abort_flag=1;
	msg="ringmoved";
	want_to_execute();
    }
}

void Bundles::tcl_command(TCLArgs& args, void* userdata) {
    if (args[1] == "set_points") {
	if (gw) {
	    Point p1, p2;
	    gw->GetEndpoints(p1,p2);
	    startx.set(p1.x());
	    starty.set(p1.y());
	    startz.set(p1.z());
	    endx.set(p2.x());
	    endy.set(p2.y());
	    endz.set(p2.z());
	} else {
	    cerr << "Error - can't set points, since we don't have a widget yet!\n";
	}
    } else if (args[1] == "get_points") {
	if (gw) {
	    Point p1, p2;
	    p1.x(startx.get());
	    p1.y(starty.get());
	    p1.z(startz.get());
	    p2.x(endx.get());
	    p2.y(endy.get());
	    p2.z(endz.get());
	    gw->SetEndpoints(p1,p2);
	    want_to_execute();
	} else {
	    cerr << "Error - can't set points, since we don't have a widget yet!\n";	    
	}
    } else {
        Module::tcl_command(args, userdata);
    }
}
} // End namespace DaveW


