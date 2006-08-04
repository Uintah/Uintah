
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Core/Math/MiscMath.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/UV.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* rect_maker() {
  return new Rect();
}

// initialize the static member type_id
PersistentTypeID Rect::type_id("Rect", "Object", rect_maker);

Rect::Rect(Material* matl, const Point& cen, const Vector& u,
	   const Vector& v)
    : Object(matl, this), cen(cen), u(u), v(v), tex_scale(Vector(1,1,0))
{
    n=Cross(u, v);
    n.normalize();
    d=Dot(n, cen);
    double l1=1./u.length2();
    double l2=1./v.length2();

    un=u;
    du=un.normalize();
    vn=v;
    dv=vn.normalize();

    this->u*=l1;
    this->v*=l2;
    d1=Dot(this->u, cen);
    d2=Dot(this->v, cen);
}

Rect::~Rect()
{
}

void Rect::intersect(Ray& ray, HitInfo& hit, DepthStats* st,
		     PerProcessorContext*)
{
    st->rect_isect++;
    Vector dir(ray.direction());
    Point orig(ray.origin());
    double dt=Dot(dir, n);
    if(dt < 1.e-6 && dt > -1.e-6)
	return;
    double t=(d-Dot(n, orig))/dt;
    if(hit.was_hit && t>hit.min_t)
	return;
    Point p(orig+dir*t);
    double a1=Dot(u, p)-d1;
    if(a1 > 1 || a1 < -1)
	return;
    double a2=Dot(v, p)-d2;
    if(a2 > 1 || a2 < -1)
	return;
    hit.hit(this, t);
}

void Rect::light_intersect(Ray& ray, HitInfo& hit, Color& /*atten*/,
			   DepthStats* st, PerProcessorContext*)
{
    st->rect_isect++;
    Vector dir(ray.direction());
    Point orig(ray.origin());
    double dt=Dot(dir, n);
    if(dt < 1.e-6 && dt > -1.e-6)
	return;
    double t=(d-Dot(n, orig))/dt;
    if(t>hit.min_t)
	return;
    Point p(orig+dir*t);
    double a1=Dot(u, p)-d1;
    if(a1 > 1 || a1 < -1)
	return;
    double a2=Dot(v, p)-d2;
    if(a2 > 1 || a2 < -1)
	return;
    hit.shadowHit(this, t);
}

Vector Rect::normal(const Point&, const HitInfo&)
{
    return n;
}

void Rect::softshadow_intersect(Light* light, Ray& ray,
				HitInfo& hit, double dist, Color& atten,
				DepthStats* st, PerProcessorContext*)
{
    st->rect_light_isect++;
    Vector dir(ray.direction());
    Point orig(ray.origin());
    double dt=Dot(dir, n);
    if(dt < 1.e-4 && dt > -1.e-4)
	return;
    double t=(d-Dot(n, orig))/dt;


    if(t<=1.e-6 || t>dist)
	return;

    Point p(orig+dir*t);
    Vector pv(p-cen);

    double delta=light->radius*t/dist;

    double a1=Dot(un, pv);
    if(a1 > du+delta || a1 < -(du+delta))
	return;
    double a2=Dot(vn, pv);
    if(a2 > dv+delta || a2 < -(dv+delta))
	return;
    if(a1<du && a1>-du && a2<dv && a2>-dv){
	atten=Color(0,0,0);
	st->rect_light_hit++;
	hit.hit(this, 1);
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
    uv.set((Dot(un, p)/(2*du)+.5)*tex_scale.x(),
           (Dot(vn, p)/(2*dv)+.5)*tex_scale.y());
}


const int RECT_VERSION = 1;

void 
Rect::io(SCIRun::Piostream &str)
{
  str.begin_class("Rect", RECT_VERSION);
  Object::io(str);
  UVMapping::io(str);
  SCIRun::Pio(str, cen);
  SCIRun::Pio(str, u);
  SCIRun::Pio(str, v);
  SCIRun::Pio(str, n);
  SCIRun::Pio(str, d);
  SCIRun::Pio(str, d1);
  SCIRun::Pio(str, d2);
  SCIRun::Pio(str, un);
  SCIRun::Pio(str, vn);
  SCIRun::Pio(str, du);
  SCIRun::Pio(str, dv);
  SCIRun::Pio(str, tex_scale);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Rect*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Rect::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Rect*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
