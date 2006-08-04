
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

using namespace SCIRun;

Persistent* hardShadows_maker() {
  return new HardShadows();
}

// initialize the static member type_id
PersistentTypeID HardShadows::type_id("HardShadows", "ShadowBase", 
				      hardShadows_maker);

// I have no idea why this is set so high.  There should acutally be a
// global version that everyone could use and keep consistent.
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
  // This code looks so crazy, but here's the deal.  Rather than
  // sizing an array based on the total number of lights they use
  // scene->nlightBits() where nlights()+nPerMatlLights() <=
  // 2^nlightBits().  This allows the use of bit shift operations
  // instead of simple multiplication.  Steve says that bit shifts are
  // a lot faster on the SGI than multiplication and that this should
  // be a win for smaller number of lights.  This code actually hasn't
  // been profiled though.
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
  // Compute the index of the occluder.
  int scindex = (depth<<cx->scene->nlightBits())+light->getIndex();
  // If this entry is 0 it signifies there is no cached occluder.  The
  // memory is zeroed out by the PerProcessorContext constructor.
  // It's OK that the cache persists from one frame to the next, we'll
  // just get more misses as the worker moves around the scene.
  if(shadow_cache[scindex]){
    shadow_cache[scindex]->light_intersect(lightray, hit, atten,
					 &cx->stats->ds[depth], cx->ppc);
    cx->stats->ds[depth].shadow_cache_try++;
    if(hit.was_hit){
      // Hit the cached occluder, so this point is not lit by this light.
      return false;
    }
    // This was a cache miss, so reset the cache.
    shadow_cache[scindex]=0;
    cx->stats->ds[depth].shadow_cache_miss++;
  }
  // We have to check the whole scene now for an occluder.
  obj->light_intersect(lightray, hit, atten, &cx->stats->ds[depth], cx->ppc);

  if(hit.was_hit){
    // Found an occluder.  Cache it.
    shadow_cache[scindex]=hit.hit_obj;
    return false;
  }

  // Found no occluder for this shadow ray.
  shadow_cache[scindex]=0;
  return true;
}

const int HARDSHADOWS_VERSION = 1;

void 
HardShadows::io(SCIRun::Piostream &str)
{
  str.begin_class("HardShadows", HARDSHADOWS_VERSION);
  ShadowBase::io(str);
  SCIRun::Pio(str, shadow_cache_offset);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::HardShadows*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::HardShadows::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::HardShadows*>(pobj);
    ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
