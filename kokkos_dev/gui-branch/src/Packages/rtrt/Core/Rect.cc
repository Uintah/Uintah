
#include "Rect.h"
#include "Ray.h"
#include "Light.h"
#include "HitInfo.h"
#include "BBox.h"
#include "MiscMath.h"
#include "Stats.h"
#include "UV.h"

using namespace rtrt;

Rect::Rect(Material* matl, const Point& cen, const Vector& u,
	   const Vector& v)
    : Object(matl, this), cen(cen), u(u), v(v)
{
    n=u.cross(v);
    n.normalize();
    d=n.dot(cen);
    double l1=1./u.length2();
    double l2=1./v.length2();

    un=u;
    du=un.normalize();
    vn=v;
    dv=vn.normalize();

    this->u*=l1;
    this->v*=l2;
    d1=this->u.dot(cen);
    d2=this->v.dot(cen);
}

Rect::~Rect()
{
}

void Rect::intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
		     PerProcessorContext*)
{
    st->rect_isect++;
    Vector dir(ray.direction());
    Point orig(ray.origin());
    double dt=dir.dot(n);
    if(dt < 1.e-6 && dt > -1.e-6)
	return;
    double t=(d-n.dot(orig))/dt;
    if(hit.was_hit && t>hit.min_t)
	return;
    Point p(orig+dir*t);
    double a1=u.dot(p)-d1;
    if(a1 > 1 || a1 < -1)
	return;
    double a2=v.dot(p)-d2;
    if(a2 > 1 || a2 < -1)
	return;
    hit.hit(this, t);
}

Vector Rect::normal(const Point&, const HitInfo&)
{
    return n;
}

void Rect::light_intersect(Light* light, const Ray& ray,
			   HitInfo&, double dist, Color& atten,
			   DepthStats* st, PerProcessorContext*)
{
    st->rect_light_isect++;
    Vector dir(ray.direction());
    Point orig(ray.origin());
    double dt=dir.dot(n);
    if(dt < 1.e-4 && dt > -1.e-4)
	return;
    double t=(d-n.dot(orig))/dt;


    if(t<=1.e-6 || t>dist)
	return;

    Point p(orig+dir*t);
    Vector pv(p-cen);

    double delta=light->radius*t/dist;

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
}

void Rect::compute_bounds(BBox& bbox, double offset)
{
    bbox.extend(cen+vn*(dv+offset)+un*(du+offset));
    bbox.extend(cen-vn*(dv+offset)+un*(du+offset));
    bbox.extend(cen+vn*(dv+offset)-un*(du+offset));
    bbox.extend(cen-vn*(dv+offset)-un*(du+offset));
}

void Rect::uv(UV& uv, const Point& hitpos, const HitInfo&)
{
    Vector p(hitpos-cen);
    double uu=(un.dot(p)*dv+1)*0.5;
    double vv=(vn.dot(p)*du+1)*0.5;
    uv.set(uu,vv);
}
