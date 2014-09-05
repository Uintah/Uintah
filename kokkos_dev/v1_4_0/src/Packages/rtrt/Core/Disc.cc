
#include <Packages/rtrt/Core/Disc.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/MiscMath.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/UV.h>

using namespace rtrt;

Disc::Disc(Material* matl, const Point& cen, const Vector& n,
	   double radius)
    : Object(matl), cen(cen), n(n), radius(radius)
{
    this->n.normalize();
    d=Dot(this->n, cen);
}

Disc::~Disc()
{
}

void Disc::intersect(const Ray& ray, HitInfo& hit, DepthStats*,
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

Vector Disc::normal(const Point&, const HitInfo&)
{
    return n;
}

void Disc::light_intersect(Light*, const Ray& ray,
			   HitInfo&, double, Color& atten,
			   DepthStats*, PerProcessorContext*)
{
    Vector dir(ray.direction());
    Point orig(ray.origin());
    double dt=Dot(dir, n);
    if(dt < 1.e-6 && dt > -1.e-6)
	return;
    double t=(d-Dot(n, orig))/dt;
    Point p(orig+dir*t);
    double l=(p-cen).length2();
    if(l < radius*radius)
	atten=Color(0,0,0);
}

void Disc::compute_bounds(BBox& bbox, double offset)
{
    bbox.extend(cen-Vector(1,1,1)*(offset+radius));
    bbox.extend(cen+Vector(1,1,1)*(offset+radius));
}

