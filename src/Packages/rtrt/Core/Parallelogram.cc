
#include "Parallelogram.h"
#include "Ray.h"
#include "Light.h"
#include "HitInfo.h"
#include "BBox.h"
#include "MiscMath.h"
#include "Stats.h"
#include "UV.h"

using namespace rtrt;

Parallelogram::Parallelogram(Material* matl, const Point& anchor, const Vector& u,
	   const Vector& v)
    : Object(matl, this), anchor(anchor), u(u), v(v)
{
    n=u.cross(v);
    n.normalize();
    d=n.dot(anchor);
    //double l1=1./u.length2();
    //double l2=1./v.length2();

    un=u;
    du=un.normalize();
    vn=v;
    dv=vn.normalize();

    //this->u*=l1;
    //this->v*=l2;
    d1=this->u.dot(anchor);
    d2=this->v.dot(anchor);
}

Parallelogram::~Parallelogram()
{
}

void Parallelogram::intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
		     PerProcessorContext*)
{
    st->parallelogram_isect++;
    Vector dir(ray.direction());
    Point orig(ray.origin());
    double dt=dir.dot(n);
    if(dt < 1.e-6 && dt > -1.e-6)
	return;
    double t=(d-n.dot(orig))/dt;
    if(hit.was_hit && t>hit.min_t)
	return;
    Point p(orig+dir*t);
#if 0
    double a1=u.dot(p)-d1;
    if(a1 > 1 || a1 < -1)
	return;
    double a2=v.dot(p)-d2;
    if(a2 > 1 || a2 < -1)
	return;
#else
    Vector vi(p-anchor);
    double a1 = un.dot(vi);
    if (a1 < 0 || a1 > du)
      return;
    double a2 = vn.dot(vi);
    if (a2 < 0 || a2 > dv)
      return;
    
#endif
    st->parallelogram_hit++;
    hit.hit(this, t);
}

Vector Parallelogram::normal(const Point&, const HitInfo&)
{
    return n;
}

void Parallelogram::light_intersect(Light*, const Ray& ray,
			   HitInfo&, double dist, Color& atten,
			   DepthStats* st, PerProcessorContext*)
{
    st->rect_light_isect++;
    Vector dir(ray.direction());
    Point orig(ray.origin());
    double dt=dir.dot(n);
    if(dt < 1.e-6 && dt > -1.e-6)
	return;
    double t=(d-n.dot(orig))/dt;


    if(t<=1.e-6 || t>dist)
	return;

    Point p(orig+dir*t);
    //Vector pv(p-anchor);

    //double delta=light->radius*t/dist;

    Vector vi(p-anchor);
    double a1 = un.dot(vi);
    if (a1 < 0 || a1 > du)
	return; // miss
    double a2 = vn.dot(vi);
    if (a2 < 0 || a2 > dv)
	return; // miss
    atten=Color(0,0,0);
    return;
#if 0
    double a1=un.dot(pv);
    if(a1 > du+delta || a1 < -(du+delta))
	return;
    double a2=vn.dot(pv);
    if(a2 > dv+delta || a2 < -(dv+delta))
	return;
    if(a1<du && a1>-du && a2<dv && a2>-dv){
	atten=Color(0,0,0);
	st->rect_light_hit++;
	return;
    }

    double t1;
    double t2;
    if(Abs(a1)>du){
	t1=(Abs(a1)-du)/delta;
    } else {
	t1=0;
    }
    if(Abs(a2)>dv){
	t2=(Abs(a2)-dv)/delta;
    } else {
	t2=0;
    }

    double g1=3*t1*t1-2*t1*t1*t1;
    double g2=3*t2*t2-2*t2*t2*t2;
    double g=1-(1-g1)*(1-g2);
    st->rect_light_penumbra++;
    atten=g<atten.luminance()?Color(g,g,g):atten;
#endif
}

void Parallelogram::compute_bounds(BBox& bbox, double offset)
{
    bbox.extend(anchor+vn*(dv+offset)+un*(du+offset));
    bbox.extend(anchor+un*(du+offset)-vn*(offset));
    bbox.extend(anchor-un*(offset)+vn*(dv+offset));
    bbox.extend(anchor-un*(offset)-vn*(offset));
}

void Parallelogram::uv(UV& uv, const Point& hitpos, const HitInfo&)
{
    Vector p(hitpos-anchor);
#if 0
    double uu=(un.dot(p)*dv+1)*0.5;
    double vv=(vn.dot(p)*du+1)*0.5;
#else
    double uu=(un.dot(p)/(du));
    double vv=(vn.dot(p)/(dv));
#endif
    uv.set(uu,vv);
}
