
#include <Packages/rtrt/Core/DiscArc.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Core/Math/MiscMath.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/UV.h>

using namespace rtrt;
using namespace SCIRun;

DiscArc::DiscArc(Material* matl, const Point& cen, const Vector& n,
		 double radius)
  : Disc(matl, cen, n, radius), theta0(0), theta1(6.283185)
{
}

DiscArc::~DiscArc()
{
}

void DiscArc::intersect(Ray& ray, HitInfo& hit, DepthStats*,
			PerProcessorContext*)
{
  Vector xdir(xform.unproject(ray.direction()));
  Point xorig(xform.unproject(ray.origin()));
  double dt=xdir.z();
  if(dt < 1.e-6 && dt > -1.e-6)
    return;
  double t=-xorig.z()/dt;
  if(hit.was_hit && t>hit.min_t)
    return;
  Point xp(xorig+xdir*t);
  double l=xp.y()*xp.y()+xp.x()*xp.x();
  double theta=atan2(xp.x(), xp.y());
  if (theta<0) theta+=2*M_PI;
  if(l < radius*radius && (theta > theta0) && (theta < theta1))
    hit.hit(this, t);
}

void DiscArc::light_intersect(Ray& ray, HitInfo& hit, Color&,
				DepthStats*, PerProcessorContext*)
{
  Vector xdir(xform.unproject(ray.direction()));
  Point xorig(xform.unproject(ray.origin()));
  double dt=xdir.z();
  if(dt < 1.e-6 && dt > -1.e-6)
    return;
  double t=-xorig.z()/dt;
  if(hit.was_hit && t>hit.min_t)
    return;
  Point xp(xorig+xdir*t);
  double l=xp.y()*xp.y()+xp.x()*xp.x();
  double theta=atan2(xp.x(), xp.y());
  if (theta<0) theta+=2*M_PI;
  if(l < radius*radius && (theta > theta0) && (theta < theta1))
    hit.shadowHit(this, t);
}
