
/*
 *  RadPrims.cc:  Radiosity primitives -- Mesh, ...
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Packages/DaveW/Core/Datatypes/CS684/DRaytracer.h>
#include <Packages/DaveW/Core/Datatypes/CS684/RTPrims.h>
#include <Packages/DaveW/Core/Datatypes/CS684/RadPrims.h>
#include <Packages/DaveW/Core/Datatypes/CS684/Spectrum.h>
#include <Core/Containers/String.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/Material.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Plane.h>
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
using DaveW::Pio;
using SCIRun::Pio;

RadObj::RadObj()
{	
}

RadObj::RadObj(const RadObj& copy)
: i1(copy.i1), i2(copy.i2), i3(copy.i3), children(copy.children), 
  mesh(copy.mesh), rad(copy.rad), area(copy.area)
{
}

RadObj::RadObj(int i1, int i2, int i3, double area, const Color &rad, 
	       RadMesh* rm)
: i1(i1), i2(i2), i3(i3), area(area), rad(rad), mesh(rm)
{
}

RadObj::~RadObj() {
}

RadObj* RadObj::clone()
{
    return scinew RadObj(*this);
}

Point RadObj::rndPt() {
    Point p1, p2, p3;
    p1=mesh->pts[i1]; p2=mesh->pts[i2]; p3=mesh->pts[i3];
    Vector v1, v2;
    v1=p1-p3;
    v2=p2-p3;
    double alpha=mr();
    double beta=mr();
    if (alpha+beta>1) {alpha=1-alpha; beta=1-beta;}
    return Point(p3+v1*alpha+v2*beta);
}

double RadObj::computeVis(RadObj* so, DRaytracer* rt, int nsamp) {
    double vis=0;
    for (int i=0; i<nsamp; i++) {
	Point ps(so->rndPt());
	Point pr(rndPt());
	Vector v(pr-ps);
	RTObject* rcv=mesh->obj.get_rep();
	//RTObject* src=so->mesh->obj.get_rep();
	
	// send a ray from src and point ps in direction v... 
	// ...if it hits rcv then pr is visible
	
	RTRay r;
	r.origin = ps;
	r.dir = v;
	RTHit hit;
	rt->scene.findIntersect(r, hit, 0);
	if (hit.obj == rcv) vis+=1;
    }
    return vis/nsamp;
}

// see if (and node from) receiver is behind source -- if so, return 0
// otherwise

double RadObj::computeFF(RadObj* so, int nsamp) {
    Point rp1, sp1, rp2, sp2, rp3, sp3;
    rp1 = mesh->pts[i1];
    rp2 = mesh->pts[i2];
    rp3 = mesh->pts[i3];
    sp1 = so->mesh->pts[so->i1];
    sp2 = so->mesh->pts[so->i2];
    sp3 = so->mesh->pts[so->i3];
    Plane rpl(rp1, rp2, rp3);
    Plane spl(sp1, sp2, sp3);
    if ((rpl.eval_point(sp1)<=0) &&
	(rpl.eval_point(sp2)<=0) &&
	(rpl.eval_point(sp3)<=0))
	return 0;

    Vector rv4,k4;
    Point sp4;
    int haveFour=0;

    if ((rpl.eval_point(sp1)<0) ||
	(rpl.eval_point(sp2)<0) ||
	(rpl.eval_point(sp3)<0)) {

//	cerr << "Looking for new points...\n";

	// we need to find new sp points that are the crossing points
	int numGoodPts=0;
	if (rpl.eval_point(sp1)>=0) numGoodPts++;
	if (rpl.eval_point(sp2)>=0) numGoodPts++;
	if (rpl.eval_point(sp3)>=0) numGoodPts++;
	if (numGoodPts == 2) {
	    haveFour=1;
	    Point in1, in2, out, new1, new2;
	    if (rpl.eval_point(sp1)<=0) {in1=sp2; in2=sp3; out=sp1;}
	    else if (rpl.eval_point(sp2)<=0) {in1=sp3; in2=sp1; out=sp2;}
	    else {in1=sp1; in2=sp2; out=sp3;}
	    double A=rpl.normal().x();
	    double B=rpl.normal().y();
	    double C=rpl.normal().z();
	    double D=rpl.eval_point(Point(0,0,0));
	    double x0,x1,y0,y1,z0,z1,t;
	    
	    x0=in2.x(); y0=in2.y(); z0=in2.z();
	    x1=out.x(); y1=out.y(); z1=out.z();
	    t=(A*x1+B*y1+C*z1+D)/(A*(x1-x0)+B*(y1-y0)+C*(z1-z0));
	    new1=AffineCombination(in2,t,out,(1-t));
	    
	    x0=in1.x(); y0=in1.y(); z0=in1.z();
	    x1=out.x(); y1=out.y(); z1=out.z();
	    t=(A*x1+B*y1+C*z1+D)/(A*(x1-x0)+B*(y1-y0)+C*(z1-z0));
	    new2=AffineCombination(in1,t,out,(1-t));
	    
	    sp1=in1; sp2=in2;
	    sp3=new1; sp4=new2;
	} else {
	    Point in, out1, out2, new1, new2;
	    if (rpl.eval_point(sp1)>=0) {in=sp1; out1=sp2; out2=sp3;}
	    else if (rpl.eval_point(sp2)>=0) {in=sp2; out1=sp3; out2=sp1;}
	    else {in=sp3; out1=sp1; out2=sp2;}
	    double A=rpl.normal().x();
	    double B=rpl.normal().y();
	    double C=rpl.normal().z();
	    double D=rpl.eval_point(Point(0,0,0));
	    double x0,x1,y0,y1,z0,z1,t;
	    
	    x0=in.x(); y0=in.y(); z0=in.z();
	    x1=out1.x(); y1=out1.y(); z1=out1.z();
	    t=(A*x1+B*y1+C*z1+D)/(A*(x1-x0)+B*(y1-y0)+C*(z1-z0));
	    new1=AffineCombination(in,t,out1,(1-t));
	    
	    x0=in.x(); y0=in.y(); z0=in.z();
	    x1=out2.x(); y1=out2.y(); z1=out2.z();
	    t=(A*x1+B*y1+C*z1+D)/(A*(x1-x0)+B*(y1-y0)+C*(z1-z0));
	    new2=AffineCombination(in,t,out2,(1-t));
	    
	    sp1=in; sp2=new1; sp3=new2;
	}
    }

    Vector rn(rpl.normal());
    Vector sn(spl.normal());
    double ff=0;
    for (int i=0; i<nsamp; i++) {
	Point rp(rndPt());
	if (spl.eval_point(rp)<=0) continue;
	Vector rv1(sp1-rp);
	Vector rv2(sp2-rp);
	Vector rv3(sp3-rp);
	if (haveFour) {rv4=sp4-rp; rv4.normalize();}
	rv1.normalize(); rv2.normalize(); rv3.normalize();
	Vector k1(Cross(rv1,rv2));
	Vector k2(Cross(rv2,rv3));
	Vector k3(Cross(rv3,rv1));
	if (haveFour) {k3=Cross(rv3,rv4); k4=Cross(rv4,rv1); k4.normalize();}
	k1.normalize();
	k2.normalize();
	k3.normalize();
	if (haveFour) 
	    ff+=-1/(2.*3.141592) * (Dot(rn * acos(Dot(rv1,rv2)), k1) +
				    Dot(rn * acos(Dot(rv2,rv3)), k2) +
				    Dot(rn * acos(Dot(rv3,rv4)), k3) +
				    Dot(rn * acos(Dot(rv4,rv1)), k4));
	else
	    ff += -1/(2.*3.141592) * (Dot(rn * acos(Dot(rv1,rv2)), k1) +
				      Dot(rn * acos(Dot(rv2,rv3)), k2) +
				      Dot(rn * acos(Dot(rv3,rv1)), k3));
    }
//    if (ff<0) {
//	cerr << "Negative FF!\n";
//	ff*=-1;
//    }
    return ff/nsamp;
}
			
#define FF_EPSILON 0.0001
#define VIS_EPSILON 0.01
#define AREA_EPSILON 150
			  
void RadObj::createLink(RadObj* so, DRaytracer* rt, int nvissamp, int nffsamp){
    double FF=computeFF(so, nffsamp);
//    cerr << "From "<<so->mesh->obj->name()<<" to "<<mesh->obj->name<<": ";
//    cerr << "FF="<<FF<<" ";
    if (FF > FF_EPSILON) {
	double vis=computeVis(so, rt, nvissamp);
//	cerr << "vis="<<vis;
	if (vis > VIS_EPSILON) {
//	    cerr << " ADDING LINK!";
	    RadLink* rl=new RadLink(FF, vis, so, area);
	    links.add(rl);
	}
    }
//    cerr << "\n";
}

Color RadObj::radPushPull(const Color& down) {
    Color up;
    Color tmp;
    if (!children.size()) {
	up=mesh->emit_coeff + gathered + down;
    } else {
	up = Color(0,0,0);
	for (int c=0; c<children.size(); c++) {
	    tmp = children[c]->radPushPull(gathered + down);
	    up += tmp/4.;
	}
    }
    rad = up;
    return up;
}

double RadObj::allFF() {
    double levelFF=0;
    for (int l=0; l<links.size(); l++) {
	RadLink *link = links[l];
	levelFF += link->FF*link->vis;
    }
    for (int c=0; c<children.size(); c++) {
	levelFF += children[c]->allFF()/4.;
    }
    return levelFF;
}
    
void RadObj::gatherRad() {
    gathered = Color(0,0,0);
    for (int l=0; l<links.size(); l++) {
	RadLink *link = links[l];
	gathered += link->src->rad*link->FF*link->vis*mesh->rho_coeff;
    }
    for (int c=0; c<children.size(); c++) {
	children[c]->gatherRad();
    }
}

RadObj* RadObj::subdivide(RadLink* rl) {
//    if (children.size()) return this;

    RadObj* src=rl->src;
//    if (src->children.size()) return src;

    RadObj* chosen;
    if (area > src->area) chosen=this;
    else chosen=src;

    if (chosen->children.size()) return chosen;

    // ok, we have to subdivide this chosen patch...

    RadMesh *chm = chosen->mesh;

    int i1=chosen->i1;
    int i2=chosen->i2;
    int i3=chosen->i3;
    Point p1(chm->pts[i1]);
    Point p2(chm->pts[i2]);
    Point p3(chm->pts[i3]);
    Vector v1((p2-p3)/2);
    Vector v2((p3-p1)/2);
    Vector v3((p1-p2)/2);
    Point e1(p2-v1);
    Point e2(p3-v2);
    Point e3(p1-v3);
    int i4=chm->pts.size();
    int i5=i4+1;
    int i6=i5+1;
    chm->pts.add(e3);
    chm->pts.add(e1);
    chm->pts.add(e2);
    chosen->children.add(new RadObj(i1, i4, i6, area/4, chosen->rad, chm));
    chosen->children.add(new RadObj(i4, i2, i5, area/4, chosen->rad, chm));
    chosen->children.add(new RadObj(i5, i3, i6, area/4, chosen->rad, chm));
    chosen->children.add(new RadObj(i4, i5, i6, area/4, chosen->rad, chm));
    return chosen;
}    
    
void RadObj::refineLinks(double radEpsilon, DRaytracer *rt, int nvissamp,
			 int nffsamp) {
    for (int l=0; l<links.size(); l++) {
	 if (links[l]->oracle2(this, radEpsilon)) {
	     RadObj* which = subdivide(links[l]);
	     if (!which) continue;
//	     cerr << "SUBDIVIDING --";
	     if (which == this) {
//		 cerr << children.size()<<" CHILDREN (this)!\n";
		 for (int c=0; c<children.size(); c++) {
		     RadObj *co = children[c];
		     co->createLink(links[l]->src, rt, nvissamp, nffsamp);
		 }
	     } else {
//		 cerr << links[l]->src->children.size()<<" CHILDREN (that)!\n";
		 for (int c=0; c<links[l]->src->children.size(); c++) {
		     RadObj *co = links[l]->src->children[c];
		     this->createLink(co, rt, nvissamp, nffsamp);
		 }
	     }
	     links.remove(l);
	 }
    }
}

void RadObj::refineAllLinks(double radEpsilon, DRaytracer *rt, int nvissamp,
			    int nffsamp) {
    for (int c=0; c<children.size(); c++) 
	children[c]->refineAllLinks(radEpsilon, rt, nvissamp, nffsamp);
    refineLinks(radEpsilon, rt, nffsamp, nvissamp);
}

int RadObj::ancestorOf(RadObj *ro) {
    if (ro == this) return 1;
    for (int i=0; i<children.size(); i++) 
	if (children[i]->ancestorOf(ro)) return 1;
    return 0;
}

#define RadObj_VERSION 1
void RadObj::io(Piostream& stream) {

    /* int version=*/stream.begin_class("RadObj", RadObj_VERSION);
    Pio(stream, i1);
    Pio(stream, i2);
    Pio(stream, i3);
    Pio(stream, area);
    Pio(stream, rad);
//    Pio(stream, mesh);
//    Pio(stream, children);
    stream.end_class();
}

static Persistent* make_RadObj()
{
    return scinew RadObj;
}

PersistentTypeID RadObj::type_id("RadObj", "Datatype", make_RadObj);


RadLink::RadLink()
{	
}

RadLink::RadLink(const RadLink& copy)
: FF(copy.FF), vis(copy.vis), src(copy.src), rcvArea(copy.rcvArea)
{
}

RadLink::RadLink(double FF, double vis, RadObj *src, double rcvArea)
: FF(FF), vis(vis), src(src), rcvArea(rcvArea)
{
}

RadLink::~RadLink() {
}

RadLink* RadLink::clone()
{
    return scinew RadLink(*this);
}

int RadLink::oracle2(RadObj* rcv, double radEpsilon) {
    if (src->area < AREA_EPSILON && rcv->area < AREA_EPSILON) return 0;
    if (src->rad.r()+src->rad.g()+src->rad.b() == 0) return 0;
    if (error() < radEpsilon) return 0; else return 1;
}

#define RadLink_VERSION 1
void RadLink::io(Piostream& stream) {

    /* int version=*/stream.begin_class("RadLink", RadLink_VERSION);
    Pio(stream, FF);
    Pio(stream, vis);
    Pio(stream, rcvArea);
//    Pio(stream, src);
    stream.end_class();
}

static Persistent* make_RadLink()
{
    return scinew RadLink;
}

PersistentTypeID RadLink::type_id("RadLink", "Datatype", make_RadLink);


RadMesh::RadMesh()
{	
}

RadMesh::RadMesh(const RadMesh& copy)
: rho_coeff(copy.rho_coeff), emit_coeff(copy.emit_coeff), nrml(copy.nrml),
  emitting(copy.emitting), pts(copy.pts), ts(copy.ts), patches(copy.patches), 
  obj(copy.obj), dl(copy.dl)
{
}

void bldQuadTree(TriSurfFieldace& ts, Point nw, Point ne, Point se, 
		 Point sw, int n) {
    if (n<1) {
	cerr << "ERROR - can't build a quadTree less than one across!\n";
	return;
    }
    n++;
    ts.points.resize(0);
    ts.elements.resize(0);
    Vector dx(ne-nw);
//    Vector ddx;
//    if (n<64) ddx=dx/128;
//    else ddx=dx/(2*n-1);
    dx=dx/(n-1);

    Vector dy(sw-nw);
//    Vector ddy;
//    if (n<64) ddy=dy/128;
//    else ddy=dy/(2*n-1);
    dy=dy/(n-1);

    Point cx(nw);
    Point cy(nw);
    int npts=0;
//    cerr << "n="<<n<<"\n";
    for (int j=0; j<n; j++, cy=cy+dy) {
	cx=cy;
	for (int i=0; i<n; i++, cx=cx+dx, npts++) {
//	    Point cast(cx);
//	    if (i==0) cast+=ddx; else if (i==n-1) cast-=ddx;
//	    if (j==0) cast+=ddy; else if (j==n-1) cast-=ddy;
//	    inset.add(cast);
	    ts.points.add(cx);
//	    cerr << "Added point "<<mesh.points.size()<<": "<<cx<<"\n";
	    if (j!=0 && i!=0) {
		ts.add_triangle(npts, npts-1, npts-n-1);
		ts.add_triangle(npts, npts-n-1, npts-n);
//		cerr <<"Added elem ("<<npts<<","<<npts-1<<","<<npts-n-1<<")\n";
//		cerr <<"Added elem ("<<npts<<","<<npts-n-1<<","<<npts-n<<")\n";
	    }
	}
    }
}
    
RadMesh::RadMesh(RTObjectHandle& rto, DRaytracer* rt, int dl) : dl(dl) {
    obj=rto;
    emitting = rto->matl->emitting;
    rho_coeff = rt->spectrumToClr(rto->matl->temp_diffuse);
    if (!emitting) emit_coeff = Color(0,0,0);
    else emit_coeff = rt->spectrumToClr(rto->matl->temp_emission);

    RTTris *rtt=rto->getTris();
    RTRect *rtr=rto->getRect();
    if (rtt) {
	//	ts=(*rtt->surf->getTriSurfFieldace());
	ASSERTFAIL("You need to implement operator= for TriSurfFieldace and Surface before this will work");
// HACK -- need to fix this at some points...
	nrml=Vector(1,0,0);
    } else if (rtr) {
	Vector v(Cross(rtr->v1, rtr->v2));
	v.normalize();
	nrml=v;
	Point nw(rtr->c+rtr->v1);
	Point se(rtr->c-rtr->v1);
	Point ne(rtr->c+rtr->v2);
	Point sw(rtr->c-rtr->v2);
//	int n=pow(2, dl);
//	cerr << "n="<<n<<"   dl="<<dl<<"\n";
//	bldQuadTree(ts, nw, ne, se, sw, n, inset);
	bldQuadTree(ts, nw, ne, se, sw, dl);
    }
}

RadMesh::~RadMesh() {
}

RadMesh* RadMesh::clone()
{
    return scinew RadMesh(*this);
}

#define RadMesh_VERSION 1
void RadMesh::io(Piostream& stream) {
    using DaveW::Pio;
    using SCIRun::Pio;

    /* int version=*/stream.begin_class("RadMesh", RadMesh_VERSION);
    Pio(stream, rho_coeff);
    Pio(stream, emit_coeff);
    Pio(stream, nrml);
    Pio(stream, emitting);
    Pio(stream, pts);
//    Pio(stream, inset);
    Pio(stream, ts);
//    Pio(stream, patches);
    Pio(stream, obj);
    Pio(stream, dl);
    stream.end_class();
}

static Persistent* make_RadMesh()
{
    return scinew RadMesh;
}

PersistentTypeID RadMesh::type_id("RadMesh", "Datatype", make_RadMesh);
} // End namespace DaveW

