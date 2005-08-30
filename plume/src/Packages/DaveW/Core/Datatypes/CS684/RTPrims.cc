
/*
 *  RTPrims.cc:  Ray-tracing primitives -- Ray, Sphere, ...
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Packages/DaveW/Core/Datatypes/CS684/RTPrims.h>
#include <Packages/DaveW/Core/Datatypes/CS684/RadPrims.h>
#include <Packages/DaveW/Core/Datatypes/CS684/Spectrum.h>
#include <Core/Containers/String.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/Material.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>

#include <iostream>
using std::cerr;
#include <math.h>
#include <stdio.h>

namespace DaveW {
using namespace SCIRun;


PersistentTypeID BRDF::type_id("BRDF", "Datatype", 0);

BRDF::BRDF(Representation rep)
: rep(rep)
{	
}

BRDF::BRDF(const BRDF& copy)
: rep(copy.rep)
{
}

BRDF::~BRDF() {
}

#define BRDF_VERSION 1
void BRDF::io(Piostream& stream) {
 
  /*int version=*/stream.begin_class("BRDF", BRDF_VERSION);
  int* repp=(int*)&rep;
  SCIRun::Pio(stream, *repp);
  stream.end_class();
}

Lambertian::Lambertian() 
: BRDF(lambertian) {
}

Lambertian::Lambertian(const Lambertian& copy)
: BRDF(copy) {
}

BRDF* Lambertian::clone() {
    return scinew Lambertian(*this);
}

static Persistent* make_Lambertian()
{
    return scinew Lambertian;
}

Lambertian::~Lambertian() {
}

#define LAMBERTIAN_VERSION 1
void Lambertian::io(Piostream& stream) {
    /* int version=*/stream.begin_class("Lambertian", LAMBERTIAN_VERSION);
    BRDF::io(stream);
    stream.end_class();
}

PersistentTypeID Lambertian::type_id("Lambertian", "BRDF", make_Lambertian);

void Lambertian::direction(double x, double y, double &theta, double &phi) {
    theta=asin(sqrt(x));
    phi=2*M_PI*y;
}
    
RTCamera::RTCamera() : 
 init(0) {
}

RTCamera::RTCamera(const RTCamera& c) :
 apperture(c.apperture), zoom(c.zoom), fDist(c.fDist), fLength(c.fLength),
 u(c.u), v(c.v), w(c.w), view(c.view), init(c.init) {
}

RTCamera::RTCamera(const View& v) :
 view(v), init(0) {
}

RTCamera* RTCamera::clone() {
    return scinew RTCamera(*this);
}

//    Vector Up(0,1,0);
//    Vector n(-camera.View);
//    n.normalize();
//    double dott=Dot(Up,n);
//    if (dott>.9) {
//	Up=Vector(1,0,0);
//	dott=Dot(Up,n);
//    }
//    Vector Upp(Up-n*dott);
//    Upp.normalize();
//    Vector v(Upp);
//    Vector u=Cross(n,v);
//    camera.u=-u;
//    camera.v=v;
//    camera.w=n;

void RTCamera::initialize() {
    if (init) return;
    if (fLength == 0) fLength=0.05;	// default 50mm lens
    w=view.lookat()-view.eyep();
    fDist=w.length();
    w.normalize();
    v=view.up();
    u=Cross(w,v);
    zoom=1./tan(view.fov()/2);
    init=1;
}

RTCamera::~RTCamera() {
}

static Persistent* make_RTCamera() {
    return scinew RTCamera;
}

#define RTCamera_VERSION 1
void RTCamera::io(Piostream& stream) {
  using DaveW::Pio;
  using SCIRun::Pio;
  
  /* int version=*/stream.begin_class("RTCamera", RTCamera_VERSION);
  Pio(stream, view);
  Pio(stream, apperture);
  Pio(stream, fLength);
  stream.end_class();
}
    
PersistentTypeID RTCamera::type_id("RTCamera", "Persistent", make_RTCamera);

RTMaterial::RTMaterial() : emitting(0), name(clString("material")) {
    brdf=scinew Lambertian;
}

RTMaterial::RTMaterial(clString nm) : emitting(0), name(nm) {
    brdf=scinew Lambertian;
}

RTMaterial::RTMaterial(const RTMaterial &c)
: diffuse(c.diffuse), emission(c.emission), base(c.base),
  emitting(c.emitting), brdf(c.brdf), spectral(c.spectral)
{
}

RTMaterial::RTMaterial(MaterialHandle &m, const Spectrum &d, const Spectrum &e,
		       BRDFHandle& b, clString nm)
: base(m), diffuse(d), emission(e), brdf(b), emitting(1), name(nm), spectral(1)
{
}

RTMaterial::RTMaterial(MaterialHandle &m, const Spectrum &d, clString nm)
: base(m), diffuse(d), emitting(0), name(nm)
{
    brdf=scinew Lambertian;
}

RTMaterial::RTMaterial(MaterialHandle &m, clString nm)
: base(m), emitting(0), name(nm)
{
    brdf=scinew Lambertian;
}

RTMaterial& RTMaterial::operator=(const RTMaterial& c) {
    base=c.base;
    diffuse=c.diffuse;
    spectral=c.spectral;
    emission=c.emission;
    emitting=c.emitting;
    brdf=c.brdf;
    name=c.name;
    return *this;
}

RTMaterial::~RTMaterial() {
}

RTMaterial* RTMaterial::clone() {
    return scinew RTMaterial(*this);
}

static Persistent* make_RTMaterial() {
    return scinew RTMaterial;
}

PersistentTypeID RTMaterial::type_id("RTMaterial", "Persistent", make_RTMaterial);

#define RTMaterial_VERSION 2
void RTMaterial::io(Piostream& stream) {
    using DaveW::Pio;
    using SCIRun::Pio;

    /*int version=*/stream.begin_class("RTMaterial", RTMaterial_VERSION);
    Pio(stream, name);		      
    Pio(stream, diffuse);
    Pio(stream, emission);
    Pio(stream, base);
    Pio(stream, brdf);
    Pio(stream, spectral);
    Pio(stream, emitting);
    stream.end_class();
}

RTRay::RTRay() : spectrum(0) {
    origin=Point(0,0,0);
    dir=Vector(0,0,1);
    energy=1;
    nu.val=1;
    nu.prev=0;
    pixel=0;
}

RTRay::RTRay(const RTRay& copy)
: origin(copy.origin), dir(copy.dir), energy(copy.energy),
  nu(copy.nu), spectrum(copy.spectrum), pixel(copy.pixel)
{
}

RTRay::RTRay(const Point& origin, const Vector& d, Array1<double> &spectrum, 
	     Pixel *pixel, double energy, double n)
: origin(origin), dir(d), energy(energy), spectrum(spectrum), pixel(pixel)
{
    dir.normalize();
    nu.val=n;
    nu.prev=0;
}

RTRay::RTRay(const Point& origin, const Point& p2, Array1<double>& spectrum,
	     Pixel *pixel, double energy, double n)
: origin(origin), dir(p2-origin), energy(energy), spectrum(spectrum),
  pixel(pixel)
{
    dir.normalize();
    nu.val=n;
    nu.prev=0;
}

RTRay::~RTRay() {
}

RTHit::RTHit() :
 valid(0), epsilon(0.00001) {
}

RTHit::RTHit(const RTHit& copy) :
 valid(copy.valid), t(copy.t), p(copy.p), epsilon(copy.epsilon), 
 side(copy.side), obj(copy.obj) {
}

RTHit::~RTHit() {
}

int RTHit::hit(double t1, const Point& p1, int s, RTObject* ob, int f) {
    if (t1<epsilon || (valid && t<t1)) return 0;
    valid=1;
    t=t1;
    p=p1;
    side=s;
    obj=ob;
    face=f;
    return 1;
}

PersistentTypeID RTObject::type_id("RTObject", "Datatype", 0);

RTObject::RTObject(Representation rep, clString nm)
: rep(rep), visible(1), name(nm)
{	
    MaterialHandle m=scinew Material(Color(0,0,.6));
    matl=scinew RTMaterial(m, "material");
    
}

double RTObject::area() {
    cerr << "Error: area not implemented for this type of object!\n";
    return 0;
}

void RTObject::destroyTempSpectra() {
    if (matl->temp_emission.size()>0)
	matl->temp_emission.resize(0);
    if (matl->temp_diffuse.size()>0)
	matl->temp_diffuse.resize(0);
}

void RTObject::buildTempSpectra(double min, double max, int num) {
    if (num<2) {
	cerr << "can't build a spectra with less than two elements!\n";
	return;
    }
    if (matl->emitting && matl->temp_emission.size()==0) {
	matl->temp_emission.resize(num);
	matl->emission.rediscretize(matl->temp_emission, min, max);
    }
//    if (matl->spectral && matl->temp_diffuse.size()==0) {
    if (matl->temp_diffuse.size()==0) {
	matl->temp_diffuse.resize(num);
	matl->diffuse.rediscretize(matl->temp_diffuse, min, max);
    }
}
	    
RTObject::RTObject(const RTObject& copy)
: rep(copy.rep), matl(copy.matl), visible(copy.visible), mesh(copy.mesh)
{
}

Vector RTObject::BRDF(const Point& p, int side, Vector vec, int face) {

    // build our local coordinate frame (u,v,w)
    Vector w=normal(p, side, vec, face);
    Vector Up(0,1,0);
    double dott=Dot(Up,w);
    if (dott>.9 || dott<-.9) {
	Up=Vector(1,0,0);
	dott=Dot(Up,w);
    }
    Vector v(Up-w*dott);
    v.normalize();
    Vector u=Cross(v,w);

    // swap rows and columns to get inverse transform
    //    which will take a vector in local coords to global coords
    Vector uprime(u.x(), v.x(), w.x());
    Vector vprime(u.y(), v.y(), w.y());
    Vector wprime(u.z(), v.z(), w.z());

    // we want a vector (r) that's angle (theta,phi) w/ probability
    // cos(theta)/(2*Pi)
    double x=mr();
    double y=mr();
//    double theta=(M_PI/2.) * (1-cos(x*(M_PI/2.)));
//    double theta=(M_PI/2.)*x;
//    double theta=(M_PI/2.)*(sin(x*M_PI/2.)*sin(x*M_PI/2.));
    double theta, phi;
    theta=phi=0;
    matl->brdf->direction(x,y,theta,phi);

    Vector rprime(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta));
    Vector r(Dot(rprime,uprime), Dot(rprime,vprime), Dot(rprime,wprime));
    return r;
}

Point RTObject::getSurfacePoint(double x, double y) {
    cerr << "Error:  getSurfacePoint not implemented for this type of object!\n";
    return Point(x,y,0);
}

RTSphere* RTObject::getSphere() {
    if (rep==Sphere)
	return (RTSphere*)this;
    else
	return 0;
}

RTBox* RTObject::getBox() {
    if (rep==Box)
	return (RTBox*)this;
    else
	return 0;
}

RTRect* RTObject::getRect() {
    if (rep==Rect)
	return (RTRect*)this;
    else
	return 0;
}

RTTris* RTObject::getTris() {
    if (rep==Tris)
	return (RTTris*)this;
    else
	return 0;
}

RTTrin* RTObject::getTrin() {
    if (rep==Trin)
	return (RTTrin*)this;
    else
	return 0;
}

RTPlane* RTObject::getPlane() {
    if (rep==Plane)
	return (RTPlane*)this;
    else
	return 0;
}

#define RTObject_VERSION 1
void RTObject::io(Piostream& stream) {
    using DaveW::Pio;
    using SCIRun::Pio;

    /*int version=*/stream.begin_class("RTObject", RTObject_VERSION);
    int* repp=(int*)&rep;
    Pio(stream, name);
    Pio(stream, *repp);
    Pio(stream, visible);
    Pio(stream, matl);
    stream.end_class();
}

RTObject::~RTObject() {
}

static Persistent* make_RTSphere()
{
    return scinew RTSphere;
}

#define RTSphere_VERSION 1
void RTSphere::io(Piostream& stream) {
    using DaveW::Pio;
    using SCIRun::Pio;
    /* int version=*/stream.begin_class("RTSphere", RTSphere_VERSION);
    RTObject::io(stream);
    Pio(stream, center);
    Pio(stream, radius);
    stream.end_class();
}

PersistentTypeID RTSphere::type_id("RTSphere", "RTObject", make_RTSphere);

RTSphere::RTSphere()
: RTObject(Sphere, clString("sphere")) {
    center=Point(0,0,0);
    radius=1;
}

RTSphere::RTSphere(const RTSphere& copy)
: center(copy.center), radius(copy.radius), RTObject(copy)
{
}

RTSphere::RTSphere(const Point& center, double radius, RTMaterialHandle m, clString nm)
: center(center), radius(radius), RTObject(Sphere, nm)
{
    matl=m;
}

RTSphere::~RTSphere() {
}

RTObject* RTSphere::clone()
{
    return scinew RTSphere(*this);
}

int RTSphere::intersect(const RTRay& ray, RTHit &hit) {
    Vector co=center-ray.origin;
    Vector v=ray.dir;
    Point o=ray.origin;

    if (hit.valid && (hit.t < (co.length()-radius))) return 0;	// too far

    double radical=Dot(v,co)*Dot(v,co)-Dot(v,v)*(Dot(co,co)-radius*radius);
    if (radical < 0) return 0;
    double t;
    if (radical < hit.epsilon) {
	t=Dot(v,co)/Dot(v,v);
	return (hit.hit(t,o+v*t,1,this)); // ray hits silouette from a distance
    }
    double rad=sqrt(radical);	
    t=(Dot(v,co)-rad)/Dot(v,v);
    if (t<hit.epsilon) {
	t=(Dot(v,co)+rad)/Dot(v,v);
	return (hit.hit(t,o+v*t,-1,this)); // ray starts in sphere
    }
    return (hit.hit(t,o+v*t,1,this));	  // ray hits front of sphere
}

static Persistent* make_RTPlane()
{
    return scinew RTPlane;
}

PersistentTypeID RTPlane::type_id("RTPlane", "RTObject", make_RTPlane);

RTPlane::RTPlane()
: RTObject(Plane, clString("plane")) {
}

RTPlane::RTPlane(const RTPlane& copy)
: d(copy.d), n(copy.n), RTObject(copy)
{
}

RTPlane::RTPlane(double d, const Vector &n, RTMaterialHandle m, clString nm)
: d(d), n(n), RTObject(Plane, nm)
{
    matl=m;
}

RTPlane::~RTPlane() {
}

RTObject* RTPlane::clone()
{
    return scinew RTPlane(*this);
}

int RTPlane::intersect(const RTRay& ray, RTHit &hit) {
    double denom=Dot(n,ray.dir);
    if (Abs(denom)<hit.epsilon) return 0;
    double t=-(Dot(n,ray.origin)+d)/denom;
    return hit.hit(t,ray.origin+ray.dir*t,1,this);
}

#define RTPlane_VERSION 1
void RTPlane::io(Piostream& stream) {
    using DaveW::Pio;
    using SCIRun::Pio;
    /* int version=*/stream.begin_class("RTPlane", RTPlane_VERSION);
    RTObject::io(stream);
    Pio(stream, d);
    Pio(stream, n);
    stream.end_class();
}

static Persistent* make_RTBox()
{
    return scinew RTBox;
}

PersistentTypeID RTBox::type_id("RTBox", "RTObject", make_RTBox);

RTBox::RTBox()
: RTObject(Box, clString("box")), d(Vector(1,1,1)) {
}

RTBox::RTBox(const RTBox& copy)
: center(copy.center), d(copy.d), RTObject(copy)
{
}

RTBox::RTBox(const Point &center, const Vector &d, RTMaterialHandle m, clString nm)
: center(center), d(d), RTObject(Box, nm)
{
    matl=m;
}

RTBox::~RTBox() {
}

RTObject* RTBox::clone()
{
    return scinew RTBox(*this);
}

Vector RTBox::normal(const Point &p, int side, Vector, int) {
    double dx=p.x()-center.x();
    double dy=p.y()-center.y();
    double dz=p.z()-center.z();
//    cerr << "dx="<<dx<<"  dy="<<dy<<"  dz="<<dz<<"  d="<<d<<"\n";
    if (Abs(dx-d.x())<0.00001) return Vector(1,0,0)*side;
    if (Abs(dx+d.x())<0.00001) return Vector(-1,0,0)*side;
    if (Abs(dy-d.y())<0.00001) return Vector(0,1,0)*side;
    if (Abs(dy+d.y())<0.00001) return Vector(0,-1,0)*side;
    if (Abs(dz-d.z())<0.00001) return Vector(0,0,1)*side;
    if (Abs(dz+d.z())<0.00001) return Vector(0,0,-1)*side;
    cerr << "TOO FAR FROM OBJECT!  NO INTERSECTION!\n";
    return Vector(1,1,1);
}

int RTBox::intersect(const RTRay& ray, RTHit &hit) {
    Point minB, maxB, tminB, tmaxB, coord;
    double bestd=100000;

    Point origin(ray.origin);
    Vector dir(ray.dir);

    minB=center-d;
    maxB=center+d;

    tminB=((minB-origin)/dir).point();
    tmaxB=((maxB-origin)/dir).point();

    int mini=-1;
    int in[3];
    in[0]=in[1]=in[2]=0;


    double iX,iY,iZ;
    if (tminB.x()>0.001) {
	iY=origin.y()+dir.y()*tminB.x();
	iZ=origin.z()+dir.z()*tminB.x();
	if ((iY<minB.y())||(iY>maxB.y())||(iZ<minB.z())||(iZ>maxB.z()))
	    tminB.x(-1);
	else if (tminB.x()<bestd) {
	    bestd=tminB.x();
	    mini=0;
	}
    } else in[0]++;
    if (tmaxB.x()>0.001) {
	in[0]++;
	iY=origin.y()+dir.y()*tmaxB.x();
	iZ=origin.z()+dir.z()*tmaxB.x();
	if ((iY<minB.y())||(iY>maxB.y())||(iZ<minB.z())||(iZ>maxB.z()))
	    tmaxB.x(-1);
	else if (tmaxB.x()<bestd) {
	    bestd=tmaxB.x();
	    mini=3;
	}
    }
    if (tminB.y()>0.001) {
	iX=origin.x()+dir.x()*tminB.y();
	iZ=origin.z()+dir.z()*tminB.y();
	if ((iX<minB.x())||(iX>maxB.x())||(iZ<minB.z())||(iZ>maxB.z()))
	    tminB.y(-1);
	else if (tminB.y()<bestd) {
	    bestd=tminB.y();
	    mini=1;
	}
    } else in[1]++;
    if (tmaxB.y()>0.001) {
	in[1]++;
	iX=origin.x()+dir.x()*tmaxB.y();
	iZ=origin.z()+dir.z()*tmaxB.y();
	if ((iX<minB.x())||(iX>maxB.x())||(iZ<minB.z())||(iZ>maxB.z()))
	    tmaxB.y(-1);
	else if (tmaxB.y()<bestd) {
	    bestd=tmaxB.y();
	    mini=4;
	}
    }
    if (tminB.z()>0.001) {
	iX=origin.x()+dir.x()*tminB.z();
	iY=origin.y()+dir.y()*tminB.z();
	if ((iY<minB.y())||(iY>maxB.y())||(iX<minB.x())||(iX>maxB.x()))
	    tminB.z(-1);
	else if (tminB.z()<bestd) {
	    bestd=tminB.z();
	    mini=2;
	}
    } else in[2]++;
    if (tmaxB.z()>0.001) {
	in[2]++;
	iX=origin.x()+dir.x()*tmaxB.z();
	iY=origin.y()+dir.y()*tmaxB.z();
	if ((iY<minB.y())||(iY>maxB.y())||(iX<minB.x())||(iX>maxB.x()))
	    tmaxB.z(-1);
	else if (tmaxB.z()<bestd) {
	    bestd=tmaxB.z();
	    mini=5;
	}
    }
    if (mini==-1) return 0;

    int side;
    if (in[mini%3]==2) side=-1; else side=1;
    return hit.hit(bestd,ray.origin+ray.dir*bestd,side,this);
}

#define RTBox_VERSION 1
void RTBox::io(Piostream& stream) {
    using DaveW::Pio;
    using SCIRun::Pio;
    /* int version=*/stream.begin_class("RTBox", RTBox_VERSION);
    RTObject::io(stream);
    Pio(stream, center);
    Pio(stream, d);
    stream.end_class();
}

static Persistent* make_RTRect()
{
    return scinew RTRect;
}

#define RTRect_VERSION 1
void RTRect::io(Piostream& stream) {
    using DaveW::Pio;
    using SCIRun::Pio;
    /* int version=*/stream.begin_class("RTRect", RTRect_VERSION);
    RTObject::io(stream);
    Pio(stream, c);
    Pio(stream, v1);
    Pio(stream, v2);
    Pio(stream, surfArea);
    stream.end_class();
}

PersistentTypeID RTRect::type_id("RTRect", "RTObject", make_RTRect);

RTRect::RTRect()
: RTObject(Rect, clString("rect")) {
    c=Point(0,0,0);
    v1=Vector(1,0,0);
    v2=Vector(0,1,0);
    surfArea=Cross(v1,v2).length()*2;
}

RTRect::RTRect(const RTRect& copy)
: c(copy.c), v1(copy.v1), v2(copy.v2), surfArea(copy.surfArea), RTObject(copy)
{
}

RTRect::RTRect(const Point& c, const Vector& v1, const Vector& v2, 
	       RTMaterialHandle m, clString nm)
: c(c), v1(v1), v2(v2), RTObject(Rect, nm)
{
    matl=m;
    surfArea=Cross(v1,v2).length()*2;
}

RTRect::~RTRect() {
}

RTObject* RTRect::clone()
{
    return scinew RTRect(*this);
}

Point RTRect::getSurfacePoint(double x, double y) {
    x=x*1.9-0.95;
    y=y*1.9-0.95;
    return c+((v1+v2)*(x/2.))+((v1-v2)*(y/2.));
}

void orderNormal(int i[], const Vector& v) {
    if (fabs(v.x())>fabs(v.y())) {
        if (fabs(v.y())>fabs(v.z())) {  // x y z
            i[0]=0; i[1]=1; i[2]=2;
        } else if (fabs(v.z())>fabs(v.x())) {   // z x y
            i[0]=2; i[1]=0; i[2]=1;
        } else {                        // x z y
            i[0]=0; i[1]=2; i[2]=1;
        }
    } else {
        if (fabs(v.x())>fabs(v.z())) {  // y x z
            i[0]=1; i[1]=0; i[2]=2;
        } else if (fabs(v.z())>fabs(v.y())) {   // z y x
            i[0]=2; i[1]=1; i[2]=0;
        } else {                        // y z x
            i[0]=1; i[1]=2; i[2]=0;
        }
    }
}       


/* Graphics Gems page 390 & 735 -- Didier Badouel*/
int RTRect::intersect(const RTRay& ray, RTHit &hit) {
    double P[3], t, alpha, beta;
    double u0,u1,u2,w0,w1,w2;
    int i[3];
    double V[3][3];
    int inter;

    Vector n(Cross(v1,v2));
    n.normalize();
    
    double dis=-Dot(n,c);
    t=-(dis+Dot(n,ray.origin))/Dot(n,ray.dir);
    if (t<hit.epsilon) return 0;
    if (hit.valid && t>hit.t) return 0;

    V[0][0]=c.x()+v2.x();
    V[0][1]=c.y()+v2.y();
    V[0][2]=c.z()+v2.z();
    
    V[1][0]=c.x()+v1.x();
    V[1][1]=c.y()+v1.y();
    V[1][2]=c.z()+v1.z();

    V[2][0]=c.x()-v1.x();
    V[2][1]=c.y()-v1.y();
    V[2][2]=c.z()-v1.z();

    orderNormal(i,n);

    P[0]= ray.origin.x()+ray.dir.x()*t;
    P[1]= ray.origin.y()+ray.dir.y()*t;
    P[2]= ray.origin.z()+ray.dir.z()*t;

    u0=P[i[1]]-V[0][i[1]];
    w0=P[i[2]]-V[0][i[2]];
    inter=0;
    u1=V[1][i[1]]-V[0][i[1]];
    w1=V[1][i[2]]-V[0][i[2]];
    u2=V[2][i[1]]-V[0][i[1]];
    w2=V[2][i[2]]-V[0][i[2]];
    if (u1==0) {
        beta=u0/u2;
        if ((beta >= 0.) && (beta <= 1.)) {
            alpha = (w0-beta*w2)/w1;
            if ((alpha>=0.) && (alpha<=1.)) inter=1;
        }       
    } else {
        beta=(w0*u1-u0*w1)/(w2*u1-u2*w1);
        if ((beta >= 0.)&&(beta<=1.)) {
            alpha=(u0-beta*u2)/u1;
            if ((alpha>=0.) && (alpha<=1.)) inter=1;
        }
    }
    if (!inter) return 0;
    return (hit.hit(t,Point(P[0],P[1],P[2]),1,this));
}

static Persistent* make_RTTris()
{
    return scinew RTTris;
}

#define RTTris_VERSION 1
void RTTris::io(Piostream& stream) {
    using DaveW::Pio;
    using SCIRun::Pio;
    /* int version=*/stream.begin_class("RTTris", RTTris_VERSION);
    RTObject::io(stream);    
    Pio(stream, surf);
    Pio(stream, bb);		     
    stream.end_class();
}

PersistentTypeID RTTris::type_id("RTTris", "RTObject", make_RTTris);

RTTris::RTTris()
: RTObject(Tris, clString("tris")) {
}

RTTris::RTTris(const RTTris& copy)
: surf(copy.surf), RTObject(copy)
{
}

RTTris::RTTris(const SurfaceHandle& surf, RTMaterialHandle m, clString nm)
: surf(surf), RTObject(Tris, nm)
{
    matl=m;
}

RTTris::~RTTris() {
}

RTObject* RTTris::clone()
{
    return scinew RTTris(*this);
}

Vector RTTris::normal(const Point&, int side, Vector, int face) {
    TriSurfFieldace* ts=surf->getTriSurfFieldace();
    TSElement* e=ts->elements[face];
    Vector n(Cross(ts->points[e->i2]-ts->points[e->i1],
		   ts->points[e->i3]-ts->points[e->i1]));
    return n*side;
}

int RTTris::intersect(const RTRay& ray, RTHit &hit) {
    TriSurfFieldace* ts=surf->getTriSurfFieldace();
    if (!bb.valid()) {
	for (int i=0; i<ts->points.size(); i++) {
	    bb.extend(ts->points[i]);
	}
    }
    Point p;
    if (!bb.Intersect(ray.origin, ray.dir, p)) return 0;
    int gotone=0;
    for (int i=0; i<ts->elements.size(); i++) {
	gotone |= intersect(ray, hit, i);
    }
    return gotone;
}

int RTTris::intersect(const RTRay& ray, RTHit &hit, int face) {
    double P[3], t, alpha, beta;
    double u0,u1,u2,v0,v1,v2;
    int i[3];
    double V[3][3];
    int inter;

    TriSurfFieldace *ts=surf->getTriSurfFieldace();
    TSElement* e=ts->elements[face];
    Point p1(ts->points[e->i1]);
    Point p2(ts->points[e->i2]);
    Point p3(ts->points[e->i3]);

    Vector n(Cross(p2-p1, p3-p1));
    n.normalize();
    
    double dis=-Dot(n,p1);
    t=-(dis+Dot(n,ray.origin))/Dot(n,ray.dir);
    if (t<hit.epsilon) return 0;
    if (hit.valid && t>hit.t) return 0;

    V[0][0]=p1.x();
    V[0][1]=p1.y();
    V[0][2]=p1.z();
    
    V[1][0]=p2.x();
    V[1][1]=p2.y();
    V[1][2]=p2.z();

    V[2][0]=p3.x();
    V[2][1]=p3.y();
    V[2][2]=p3.z();

    orderNormal(i,n);

    P[0]= ray.origin.x()+ray.dir.x()*t;
    P[1]= ray.origin.y()+ray.dir.y()*t;
    P[2]= ray.origin.z()+ray.dir.z()*t;

    u0=P[i[1]]-V[0][i[1]];
    v0=P[i[2]]-V[0][i[2]];
    inter=0;
    u1=V[1][i[1]]-V[0][i[1]];
    v1=V[1][i[2]]-V[0][i[2]];
    u2=V[2][i[1]]-V[0][i[1]];
    v2=V[2][i[2]]-V[0][i[2]];
    if (u1==0) {
        beta=u0/u2;
        if ((beta >= 0.) && (beta <= 1.)) {
            alpha = (v0-beta*v2)/v1;
            if ((alpha>=0.) && ((alpha+beta)<=1.)) inter=1;
        }       
    } else {
        beta=(v0*u1-u0*v1)/(v2*u1-u2*v1);
        if ((beta >= 0.)&&(beta<=1.)) {
            alpha=(u0-beta*u2)/u1;
            if ((alpha>=0.) && ((alpha+beta)<=1.)) inter=1;
        }
    }
    if (!inter) return 0;
    return (hit.hit(t,Point(P[0],P[1],P[2]),1,this,face));
}

static Persistent* make_RTTrin()
{
    return scinew RTTrin;
}

#define RTTrin_VERSION 1
void RTTrin::io(Piostream& stream) {
    using DaveW::Pio;
    using SCIRun::Pio;
    /* int version=*/stream.begin_class("RTTrin", RTTrin_VERSION);
    RTObject::io(stream);    
    Pio(stream, surf);		     
    Pio(stream, bb);		     
    stream.end_class();
}

PersistentTypeID RTTrin::type_id("RTTrin", "RTObject", make_RTTrin);

RTTrin::RTTrin()
: RTObject(Trin, clString("trin")) {
}

RTTrin::RTTrin(const RTTrin& copy)
: surf(copy.surf), RTObject(copy)
{
}

RTTrin::RTTrin(const SurfaceHandle& surf, const Array2<Vector> &vectors, RTMaterialHandle m, clString nm)
: surf(surf), vectors(vectors), RTObject(Trin, nm)
{
    matl=m;
}

RTTrin::~RTTrin() {
}

RTObject* RTTrin::clone()
{
    return scinew RTTrin(*this);
}

Vector RTTrin::normal(const Point& p, int side, Vector, int face) {
    double P[3], t, alpha, beta;
    double u0,u1,u2,v0,v1,v2;
    int i[3];
    double V[3][3];
    int inter;

    TriSurfFieldace *ts=surf->getTriSurfFieldace();
    TSElement* e=ts->elements[face];
    Point p1(ts->points[e->i1]);
    Point p2(ts->points[e->i2]);
    Point p3(ts->points[e->i3]);

    Vector n(Cross(p2-p1, p3-p1));
    n.normalize();
    
    V[0][0]=p1.x();
    V[0][1]=p1.y();
    V[0][2]=p1.z();
    
    V[1][0]=p2.x();
    V[1][1]=p2.y();
    V[1][2]=p2.z();

    V[2][0]=p3.x();
    V[2][1]=p3.y();
    V[2][2]=p3.z();

    orderNormal(i,n);

    P[0]= p.x();
    P[1]= p.y();
    P[2]= p.z();

    u0=P[i[1]]-V[0][i[1]];
    v0=P[i[2]]-V[0][i[2]];
    u1=V[1][i[1]]-V[0][i[1]];
    v1=V[1][i[2]]-V[0][i[2]];
    u2=V[2][i[1]]-V[0][i[1]];
    v2=V[2][i[2]]-V[0][i[2]];
    if (u1==0) {
        beta=u0/u2;
        if ((beta >= 0.) && (beta <= 1.)) {
            alpha = (v0-beta*v2)/v1;
        }       
    } else {
        beta=(v0*u1-u0*v1)/(v2*u1-u2*v1);
        if ((beta >= 0.)&&(beta<=1.)) {
            alpha=(u0-beta*u2)/u1;
        }
    }
    Vector N(vectors(face,i[0])*(1-(alpha+beta))+vectors(face,i[1])*alpha+vectors(face, i[1])*beta);
    N.normalize();
    return N*side;
}

int RTTrin::intersect(const RTRay& ray, RTHit &hit) {
    TriSurfFieldace* ts=surf->getTriSurfFieldace();
    if (!bb.valid()) {
	for (int i=0; i<ts->points.size(); i++) {
	    bb.extend(ts->points[i]);
	}
    }
    Point p;
    if (!bb.Intersect(ray.origin, ray.dir, p)) return 0;
    int gotone=0;
    for (int i=0; i<ts->elements.size(); i++) {
	gotone |= intersect(ray, hit, i);
    }
    return gotone;
}

int RTTrin::intersect(const RTRay& ray, RTHit &hit, int face) {
    double P[3], t, alpha, beta;
    double u0,u1,u2,v0,v1,v2;
    int i[3];
    double V[3][3];
    int inter;

    TriSurfFieldace *ts=surf->getTriSurfFieldace();
    TSElement* e=ts->elements[face];
    Point p1(ts->points[e->i1]);
    Point p2(ts->points[e->i2]);
    Point p3(ts->points[e->i3]);

    Vector n(Cross(p2-p1, p3-p1));
    n.normalize();
    
    double dis=-Dot(n,p1);
    t=-(dis+Dot(n,ray.origin))/Dot(n,ray.dir);
    if (t<hit.epsilon) return 0;
    if (hit.valid && t>hit.t) return 0;

    V[0][0]=p1.x();
    V[0][1]=p1.y();
    V[0][2]=p1.z();
    
    V[1][0]=p2.x();
    V[1][1]=p2.y();
    V[1][2]=p2.z();

    V[2][0]=p3.x();
    V[2][1]=p3.y();
    V[2][2]=p3.z();

    orderNormal(i,n);

    P[0]= ray.origin.x()+ray.dir.x()*t;
    P[1]= ray.origin.y()+ray.dir.y()*t;
    P[2]= ray.origin.z()+ray.dir.z()*t;

    u0=P[i[1]]-V[0][i[1]];
    v0=P[i[2]]-V[0][i[2]];
    inter=0;
    u1=V[1][i[1]]-V[0][i[1]];
    v1=V[1][i[2]]-V[0][i[2]];
    u2=V[2][i[1]]-V[0][i[1]];
    v2=V[2][i[2]]-V[0][i[2]];
    if (u1==0) {
        beta=u0/u2;
        if ((beta >= 0.) && (beta <= 1.)) {
            alpha = (v0-beta*v2)/v1;
            if ((alpha>=0.) && ((alpha+beta)<=1.)) inter=1;
        }       
    } else {
        beta=(v0*u1-u0*v1)/(v2*u1-u2*v1);
        if ((beta >= 0.)&&(beta<=1.)) {
            alpha=(u0-beta*u2)/u1;
            if ((alpha>=0.) && ((alpha+beta)<=1.)) inter=1;
        }
    }
    if (!inter) return 0;
    return (hit.hit(t,Point(P[0],P[1],P[2]),1,this,face));
}


// Assume Ray I is pointing at surface, Vector N is pointing away
RTRay Reflect(const RTRay& I, const Vector& N) {
    RTRay R(I);
    R.dir=Vector(N*(-2*Dot(N,I.dir))+I.dir);
    return R;
}

int Snell(const RTRay& I, const Vector& N, RTRay& T) {
    double nu=I.nu.val/T.nu.val;
    double Ci=Dot(N,I.dir);
    if (Ci<0) Ci*=-1;
    double radical=1+Square(nu)*(Square(Ci)-1);
    if (radical < 0) return 0;
    T.dir=Vector(N*(nu*Ci-sqrt(radical))+I.dir*nu);
    return 1;
}

double Fres(const RTRay& I, Vector N, double nu_trans) {
    double nu=nu_trans/I.nu.val;
    double R0=Square(nu-1)/Square(nu+1);
    double dot=Dot(-N,I.dir);
    double oneMinusCosTheta=1-dot;
    return R0+Square(Square(oneMinusCosTheta))*oneMinusCosTheta*(1-R0);
}

void Pio(Piostream& stream, RTLight& l)
{
    using DaveW::Pio;
    using SCIRun::Pio;
    stream.begin_cheap_delim();
    Pio(stream, l.pos);
    Pio(stream, l.color);
    Pio(stream, l.visible);
    stream.end_cheap_delim();
}
} // End namespace DaveW


