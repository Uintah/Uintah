
#include <Packages/rtrt/Core/Shadows/UncachedHardShadows.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>
#include <Packages/rtrt/Core/Stats.h>
using namespace rtrt;


UncachedHardShadows::UncachedHardShadows()
{
}

UncachedHardShadows::~UncachedHardShadows()
{
}

bool UncachedHardShadows::lit(const Point& hitpos, Light*,
		    const Vector& light_dir, double dist, Color& atten,
		    int depth, Context* cx)
{
  HitInfo hit;
  hit.min_t = dist;
  Ray lightray(hitpos, light_dir);
  Object* obj=cx->scene->get_shadow_object();
  obj->light_intersect(lightray, hit, atten, &cx->stats->ds[depth], cx->ppc);
  return !hit.was_hit;
}

