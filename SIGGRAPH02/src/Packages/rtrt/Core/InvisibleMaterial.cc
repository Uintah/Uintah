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
