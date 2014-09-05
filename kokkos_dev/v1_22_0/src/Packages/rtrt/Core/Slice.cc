
#include <Packages/rtrt/Core/CatmullRomSpline.h>
#include <Packages/rtrt/Core/Slice.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/VolumeDpy.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <stdio.h>
#include <fstream>
#include <unistd.h>
#include <iostream>
#include <fcntl.h>
extern Mutex io_lock_;

using namespace std;
using namespace rtrt;

template<class T, class A, class B>
Slice<T,A,B>::Slice(VolumeDpy* dpy, PlaneDpy* pdpy,
		    HVolume<T,A,B>* share)
    : VolumeBase(this, dpy), pdpy(pdpy)
{
    min=share->min;
    datadiag=share->datadiag;
    sdiag=share->sdiag;
    nx=share->nx;
    ny=share->ny;
    nz=share->nz;
    isdiag=Vector(nx-1,ny-1,nz-1)/datadiag;
    blockdata.share(share->blockdata);
    datamin=share->datamin;
    datamax=share->datamax;

    CatmullRomSpline<Color> spline(
	    Color(.4,.4,.4),
	    Color(.4,.4,1),
	    Color(.4,.4,1),
	    Color(.4,.4,1),
	    Color(.4,1,.4),
	    Color(.4,1,.4),
	    Color(.4,1,.4),
	    Color(1,1,.4), 
	    Color(1,1,.4), 
	    Color(1,1,.4), 
	    Color(1,.4,.4),
	    Color(1,.4,.4)
	    );
    int ncolors=1000;
    matls.resize(ncolors);
    float Ka=.2;
    float Kd=.8;
    float Ks=0;
    float refl=0;
    float specpow=0;
    for(int i=0;i<ncolors;i++){
	float frac=float(i)/(ncolors-1);
	Color c(spline(frac));
	matls[i]=new Phong(c*Kd, c*Ks, specpow, refl);
    }
}

template<class T, class A, class B>
Slice<T,A,B>::~Slice()
{
}

template<class T, class A, class B>
void Slice<T,A,B>::io(SCIRun::Piostream &str)
{
  ASSERTFAIL("Pio for Slice<T,A,B> not implemented");
}

template<class T, class A, class B>
void Slice<T,A,B>::preprocess(double, int&, int&)
{
}

template<class T, class A, class B>
void Slice<T,A,B>::compute_bounds(BBox& bbox, double offset)
{
    bbox.extend(min-Vector(offset,offset,offset));
    bbox.extend(min+datadiag+Vector(offset,offset,offset));
}

namespace rtrt {
extern int HitCell(const Ray& r, const Point& pmin, const Point& pmax, 
		   float rho[2][2][2], float iso, double tmin, double tmax, double& t);
extern Vector GradientCell(const Point& pmin, const Point& pmax,
			   const Point& p, float rho[2][2][2]);
} // end namespace rtrt

template<class T, class A, class B>
void Slice<T,A,B>::intersect(Ray& ray, HitInfo& hit,
			    DepthStats*, PerProcessorContext*)
{
    Vector dir(ray.direction());
    Point orig(ray.origin());
    double dt=Dot(dir, n);
    if(dt < 1.e-6 && dt > -1.e-6)
	return;
    double t=(d-Dot(n, orig))/dt;
    Point p(orig+dir*t);
    // Compute the data value at this point...
    Vector pn((p-min)*isdiag);
    // A redundant check, but necessary to avoid problems with
    // conversion to int...

    if(p.x()<0 || p.y()<0 || p.z()<0)
	return;
    if(p.x()>nx || p.y()>ny || p.z()>nz)
	return;
    int ix=(int)pn.x();
    int iy=(int)pn.y();
    int iz=(int)pn.z();
    if(ix>=0 && iy>=0 && iz>= 0 && ix+1<nx && iy+1<ny && iz+1<nz){
	float fx=pn.x()-ix;
	float fy=pn.y()-iy;
	float fz=pn.z()-iz;
	T p000=blockdata(ix,iy,iz);
	T p001=blockdata(ix,iy,iz+1);
	T p010=blockdata(ix,iy+1,iz);
	T p011=blockdata(ix,iy+1,iz+1);
	T p100=blockdata(ix+1,iy,iz);
	T p101=blockdata(ix+1,iy,iz+1);
	T p110=blockdata(ix+1,iy+1,iz);
	T p111=blockdata(ix+1,iy+1,iz+1);
	float p00=p000*(1-fz)+p001*fz;
	float p01=p010*(1-fz)+p011*fz;
	float p10=p100*(1-fz)+p101*fz;
	float p11=p110*(1-fz)+p111*fz;
	float p0=p00*(1-fy)+p01*fy;
	float p1=p10*(1-fy)+p11*fy;
	float p=p0*(1-fx)+p1*fx;
	if(p < dpy->isoval)
	    return;
	if(hit.hit(this, t)){
	    double* data=(double*)hit.scratchpad;
	    *data=p;
	}
    }
}

template<class T, class A, class B>
void Slice<T,A,B>::shade(Color& result, const Ray& ray, const HitInfo& hit,
			 int depth, double atten,
			 const Color& accumcolor, Context* cx)
{
    double* data=(double*)hit.scratchpad;
    int idx=(*data-datamin)/(datamax-datamin)*matls.size();
    if(idx<0)
	idx=0;
    if(idx>=matls.size())
       idx=matls.size()-1;
    matls[idx]->shade(result, ray, hit, depth, atten, accumcolor, cx);
}

template<class T, class A, class B>
void Slice<T,A,B>::animate(double, bool& changed)
{
    if(n != pdpy->n || d != pdpy->d){
	changed=true;
	n=pdpy->n;
	d=pdpy->d;
    }
}

template<class T, class A, class B>
Vector Slice<T,A,B>::normal(const Point&, const HitInfo&)
{
    Vector nn(n);
    nn.normalize();
    return nn;
}

template<class T, class A, class B>
void Slice<T,A,B>::compute_hist(int, int*,
				float, float)
{
}    

template<class T, class A, class B>
void Slice<T,A,B>::get_minmax(float& min, float& max)
{
    min=datamin;
    max=datamax;
}
