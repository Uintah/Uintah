
#include <Packages/rtrt/Core/Parallelogram.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Core/Math/MiscMath.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/UV.h>

using namespace rtrt;
using namespace SCIRun;

SCIRun::Persistent* Parallelogram_maker() {
  return new Parallelogram;
}

// initialize the static member type_id
SCIRun::PersistentTypeID Parallelogram::type_id("Parallelogram", "Object", 
						Parallelogram_maker);

Parallelogram::Parallelogram(Material* matl, const Point& anchor, 
			     const Vector& u, const Vector& v) : 
  Object(matl, this), 
  anchor(anchor), 
  u(u), 
  v(v)
{
    n=Cross(u, v);
    n.normalize();
    d=Dot(n, anchor);
    //double l1=1./u.length2();
    //double l2=1./v.length2();

    un=u;
    du=un.normalize();
    vn=v;
    dv=vn.normalize();

    //this->u*=l1;
    //this->v*=l2;
    d1=Dot(this->u, anchor);
    d2=Dot(this->v, anchor);
}

Parallelogram::~Parallelogram()
{
}

void Parallelogram::intersect(Ray& ray, HitInfo& hit, DepthStats* st,
		     PerProcessorContext*)
{
    st->parallelogram_isect++;
    Vector dir(ray.direction());
    Point orig(ray.origin());
    double dt=Dot(dir, n);
    if(dt < 1.e-6 && dt > -1.e-6)
	return;
    double t=(d-Dot(n, orig))/dt;
    if(hit.was_hit && t>hit.min_t)
	return;
    Point p(orig+dir*t);
#if 0
    double a1=Dot(u, p)-d1;
    if(a1 > 1 || a1 < -1)
	return;
    double a2=Dot(v, p)-d2;
    if(a2 > 1 || a2 < -1)
	return;
#else
    Vector vi(p-anchor);
    double a1 = Dot(un, vi);
    if (a1 < 0 || a1 > du)
      return;
    double a2 = Dot(vn, vi);
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

void Parallelogram::light_intersect(Ray& ray, HitInfo& hit, Color&,
				    DepthStats* st, PerProcessorContext*)
{
    st->rect_light_isect++;
    Vector dir(ray.direction());
    Point orig(ray.origin());
    double dt=Dot(dir, n);
    if(dt < 1.e-6 && dt > -1.e-6)
	return;
    double t=(d-Dot(n, orig))/dt;

    if(t<=1.e-6 || t>hit.min_t)
	return;

    Point p(orig+dir*t);

    Vector vi(p-anchor);
    double a1 = Dot(un, vi);
    if (a1 < 0 || a1 > du)
	return; // miss
    double a2 = Dot(vn, vi);
    if (a2 < 0 || a2 > dv)
	return; // miss
    hit.shadowHit(this, t);
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
    double uu=(Dot(un, p)*dv+1)*0.5;
    double vv=(Dot(vn, p)*du+1)*0.5;
#else
    double uu=(Dot(un, p)/(du));
    double vv=(Dot(vn, p)/(dv));
#endif
    uv.set(uu,vv);
}

const int PARALLELOGRAM_VERSION = 1;

void 
Parallelogram::io(SCIRun::Piostream &str)
{
  str.begin_class("Parallelogram", PARALLELOGRAM_VERSION);
  Object::io(str);
  UVMapping::io(str);
  Pio(str, anchor);
  Pio(str, u);
  Pio(str, v);
  Pio(str, n);
  Pio(str, d);
  Pio(str, d1);
  Pio(str, d2);
  Pio(str, un);
  Pio(str, vn);
  Pio(str, du);
  Pio(str, dv);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Parallelogram*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Parallelogram::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Parallelogram*>(pobj);
    //ASSERT(obj != 0);
  }
}
} // end namespace SCIRun
