
#include <Packages/rtrt/Core/Shadows/UncachedHardShadows.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>
#include <Packages/rtrt/Core/Stats.h>
using namespace rtrt;
using namespace SCIRun;

Persistent* uncachedHardShadows_maker() {
  return new UncachedHardShadows();
}

// initialize the static member type_id
PersistentTypeID UncachedHardShadows::type_id("UncachedHardShadows", "ShadowBase", 
				      uncachedHardShadows_maker);


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

const int UNCACHEDHARDSHADOWS_VERSION = 1;

void 
UncachedHardShadows::io(SCIRun::Piostream &str)
{
  str.begin_class("UncachedHardShadows", UNCACHEDHARDSHADOWS_VERSION);
  ShadowBase::io(str);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::UncachedHardShadows*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::UncachedHardShadows::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::UncachedHardShadows*>(pobj);
    ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
