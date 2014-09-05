#include <Packages/rtrt/Core/InvisibleMaterial.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/Context.h>
#include <math.h>
#include <iostream>

using namespace rtrt;
using namespace SCIRun;

Persistent* invisibleMaterial_maker() {
  return new InvisibleMaterial();
}

// initialize the static member type_id
PersistentTypeID InvisibleMaterial::type_id("InvisibleMaterial", "Material", invisibleMaterial_maker);

InvisibleMaterial::InvisibleMaterial()
{
}

InvisibleMaterial::~InvisibleMaterial()
{
}

void InvisibleMaterial::shade(Color& result, const Ray& ray,
		  const HitInfo& hit, int depth,
		  double atten, const Color& accumcolor,
		  Context* cx)
{
    double nearest=hit.min_t;
    //Object* obj=hit.hit_obj;
    Point hitpos(ray.origin()+ray.direction()*nearest);
    Ray rray(hitpos, ray.direction());
    cx->worker->traceRay(result, rray, depth, atten,
			 accumcolor, cx);
}

const int INVISIBLEMATERIAL_VERSION = 1;

void 
InvisibleMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("InvisibleMaterial", INVISIBLEMATERIAL_VERSION);
  Material::io(str);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::InvisibleMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::InvisibleMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::InvisibleMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
