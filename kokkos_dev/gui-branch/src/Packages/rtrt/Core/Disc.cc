
#include "Disc.h"
#include "Ray.h"
#include "Light.h"
#include "HitInfo.h"
#include "BBox.h"
#include "MiscMath.h"
#include "Stats.h"
#include "UV.h"

using namespace rtrt;

Disc::Disc(Material* matl, const Point& cen, const Vector& n,
	   double radius)
    : Object(matl), cen(cen), n(n), radius(radius)
{
    this->n.normalize();
    d=this->n.dot(cen);
}

Disc::~Disc()
{
}

void Disc::intersect(const Ray& ray, HitInfo& hit, DepthStats*,
		     PerProcessorContext*)
{
    Vector dir(ray.direction());
    Point orig(ray.origin());
    double dt=dir.dot(n);
    if(dt < 1.e-6 && dt > -1.e-6)
	return;
    double t=(d-n.dot(orig))/dt;
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
    double dt=dir.dot(n);
    if(dt < 1.e-6 && dt > -1.e-6)
	return;
    double t=(d-n.dot(orig))/dt;
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

