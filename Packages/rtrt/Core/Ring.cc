
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

Ring::Ring(Material* matl, const Point& cen, const Vector& n,
	   double radius, double thickness)
  : Object(matl), cen(cen), n(n), radius(radius), thickness(thickness)
{
    this->n.normalize();
    d=Dot(this->n, cen);
}

Ring::~Ring()
{
}

void Ring::intersect(const Ray& ray, HitInfo& hit, DepthStats*,
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
    if(l < radius*radius)
	hit.hit(this, t);
}

Vector Ring::normal(const Point&, const HitInfo&)
{
    return n;
}

void Ring::light_intersect(const Ray& ray, HitInfo& hit, Color&,
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

