
#include <Packages/rtrt/Core/Shadows/HardShadows.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>
#include <Packages/rtrt/Core/Stats.h>
using namespace rtrt;

#define MAXDEPTH 200

HardShadows::HardShadows()
{
}

HardShadows::~HardShadows()
{
}

void HardShadows::preprocess(Scene* scene, int& pp_offset, int&)
{
  shadow_cache_offset = pp_offset;
  pp_offset += sizeof(Object*)*MAXDEPTH*(1<<scene->nlightBits());
}

bool HardShadows::lit(const Point& hitpos, Light* light,
		      const Vector& light_dir, double dist, Color& atten,
		      int depth, Context* cx)
{
  HitInfo hit;
  hit.min_t = dist;
  Ray lightray(hitpos, light_dir);
  Object* obj=cx->scene->get_shadow_object();
  Object** shadow_cache = (Object**)cx->ppc->get(shadow_cache_offset,
						 sizeof(Object*)*MAXDEPTH);
  int scindex = (depth<<cx->scene->nlightBits())+light->getIndex();
  if(shadow_cache[scindex]){
    shadow_cache[scindex]->light_intersect(lightray, hit, atten,
					 &cx->stats->ds[depth], cx->ppc);
    cx->stats->ds[depth].shadow_cache_try++;
    if(hit.was_hit){
      return false;
    }
    shadow_cache[scindex]=0;
    cx->stats->ds[depth].shadow_cache_miss++;
  }
  obj->light_intersect(lightray, hit, atten, &cx->stats->ds[depth], cx->ppc);

  if(hit.was_hit){
    shadow_cache[scindex]=hit.hit_obj;
    return false;
  }
  shadow_cache[scindex]=0;
  return true;
}
