
#include <Packages/rtrt/Core/Shadows/ScrewyShadows.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/PhongMaterial.h>
#include <Packages/rtrt/Core/InvisibleMaterial.h>
#include <Packages/rtrt/Core/CycleMaterial.h>
#include <Packages/rtrt/Core/PhongMaterial.h>
using namespace rtrt;
using namespace SCIRun;

Persistent* screwyShadows_maker() {
  return new ScrewyShadows();
}

// initialize the static member type_id
PersistentTypeID ScrewyShadows::type_id("ScrewyShadows", "ShadowBase", 
				      screwyShadows_maker);


ScrewyShadows::ScrewyShadows()
{
}

ScrewyShadows::~ScrewyShadows()
{
}

void ScrewyShadows::preprocess(Scene*, int& pp_offset, int&)
{
  shadow_cache_offset = pp_offset;
  pp_offset += sizeof(Object*)*MAXDEPTH;
}

bool ScrewyShadows::lit(const Point& hitpos, Light*,
		    const Vector& light_dir, double dist, Color& atten,
		    int depth, Context* cx)
{
  HitInfo hit;
  Ray lightray(hitpos, light_dir);
  Object* obj=cx->scene->get_shadow_object();
  Object** shadow_cache = (Object**)cx->ppc->get(shadow_cache_offset,
						 sizeof(Object*)*MAXDEPTH);
  if(shadow_cache[depth]){
    shadow_cache[depth]->light_intersect(lightray, hit, atten,
					 &cx->stats->ds[depth], cx->ppc);
    cx->stats->ds[depth].shadow_cache_try++;
    //if(hit.was_hit && hit.min_t < dist || atten.luminance() < 1.e-6){
    if(hit.was_hit){
      return false;
    }
    shadow_cache[depth]=0;
    cx->stats->ds[depth].shadow_cache_miss++;
  }
    int done=0;
    Point start_origin(lightray.origin());
    double t=0;
    while (!done) {
      obj->intersect(lightray, hit, &cx->stats->ds[depth], cx->ppc);
      if (hit.was_hit && hit.min_t < dist) {
	Material *m = hit.hit_obj->get_matl();
	int see_through = 0;
	if (dynamic_cast<PhongMaterial*>(m) && 
	    dynamic_cast<PhongMaterial*>(m)->get_opacity() < .5) 
	  see_through=1;
	else if (dynamic_cast<InvisibleMaterial*>(m)) see_through=1;
	else if (dynamic_cast<CycleMaterial*>(m) &&
		 dynamic_cast<InvisibleMaterial*>(dynamic_cast<CycleMaterial*>(m)->curr())) see_through=1;
	else if (dynamic_cast<CycleMaterial*>(m) &&
		 dynamic_cast<PhongMaterial*>(dynamic_cast<CycleMaterial*>(m)->curr()) &&
		 (dynamic_cast<PhongMaterial*>(dynamic_cast<CycleMaterial*>(m)->curr()))->get_opacity() < 0.5)
		 see_through=1;
	if (see_through) {
	  lightray.set_origin(lightray.origin() + hit.min_t * lightray.direction());
	  t += hit.min_t;
	  hit.was_hit = false;
	} else
	  done=1;
      } else done=1;
    }
    if (hit.was_hit) {
      hit.min_t = t;
    }
    lightray.set_origin(start_origin);
  if(hit.was_hit){
    shadow_cache[depth]=hit.hit_obj;
    return false;
  }
  shadow_cache[depth]=0;
  return true;
}

const int SCREWYSHADOWS_VERSION = 1;

void 
ScrewyShadows::io(SCIRun::Piostream &str)
{
  str.begin_class("ScrewyShadows", SCREWYSHADOWS_VERSION);
  ShadowBase::io(str);
  SCIRun::Pio(str, shadow_cache_offset);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::ScrewyShadows*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::ScrewyShadows::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::ScrewyShadows*>(pobj);
    ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
