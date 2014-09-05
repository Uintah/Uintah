
#include <Packages/rtrt/Core/Ring.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Core/Math/MiscMath.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/UV.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* ring_maker() {
  return new Ring();
}

// initialize the static member type_id
PersistentTypeID Ring::type_id("Ring", "Object", ring_maker);

Ring::Ring(Material* matl, const Point& cen, const Vector& n,
	   double radius, double thickness)
  : Object(matl,this), cen(cen), n(n), radius(radius), thickness(thickness)
{
    this->n.normalize();
    d=Dot(this->n, cen);
}

Ring::~Ring()
{
}

void Ring::uv(UV& uv, const Point& hitpos, const HitInfo&)  
{
  uv.set(hitpos.x(),hitpos.y());
}

void Ring::intersect(Ray& ray, HitInfo& hit, DepthStats*,
		     PerProcessorContext*)
{
    Vector dir(ray.direction());
    Point orig(ray.origin());
    double dt=Dot(dir, n);
    if(dt < 1.e-6 && dt > -1.e-6)
	return;
    double t=(d-Dot(n, orig))/dt;
    if(hit.was_hit && t>hit.min_t)
	return;
    Point p(orig+dir*t);
    double l=(p-cen).length2();
    double outer_radius=radius+thickness;
    if(l > radius*radius && l < outer_radius*outer_radius)
	hit.hit(this, t);
}

Vector Ring::normal(const Point&, const HitInfo&)
{
    return n;
}

void Ring::light_intersect(Ray& ray, HitInfo& hit, Color&,
			   DepthStats*, PerProcessorContext*)
{
  Vector dir(ray.direction());
  Point orig(ray.origin());
  double dt=Dot(dir, n);
  if(dt < 1.e-6 && dt > -1.e-6)
    return;
  double t=(d-Dot(n, orig))/dt;
  if(t>hit.min_t)
    return;
  Point p(orig+dir*t);
  double l=(p-cen).length2();
  double outer_radius=radius+thickness;
  if(l > radius*radius && l < outer_radius*outer_radius)
    hit.shadowHit(this, t);
}

void Ring::compute_bounds(BBox& bbox, double offset)
{
    bbox.extend(cen-Vector(1,1,1)*(offset+radius+thickness));
    bbox.extend(cen+Vector(1,1,1)*(offset+radius+thickness));
}

const int RING_VERSION = 1;

void 
Ring::io(SCIRun::Piostream &str)
{
  str.begin_class("Ring", RING_VERSION);
  Object::io(str);
  Pio(str, cen);
  Pio(str, n);
  Pio(str, d);
  Pio(str, radius);
  Pio(str, thickness);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Ring*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Ring::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Ring*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun

