#include <Packages/rtrt/Core/CycleMaterial.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Context.h>
#include <math.h>

using namespace rtrt;

CycleMaterial::CycleMaterial()
    : current(0)
{
}

CycleMaterial::~CycleMaterial()
{
}

void CycleMaterial::next() {
  if (!members.size()) return;
  current++;
  if (current==members.size()) current=0;
}

void CycleMaterial::prev() {
  if (!members.size()) return;
  current--;
  if (current==-1) current=members.size()-1;
}

void CycleMaterial::shade(Color& result, const Ray& ray,
		  const HitInfo& hit, int depth, 
		  double atten, const Color& accumcolor,
		  Context* cx)
{
  members[current]->shade(result, ray, hit, depth, atten, accumcolor, cx);
}
