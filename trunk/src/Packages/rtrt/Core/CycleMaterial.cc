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
using namespace SCIRun;

Persistent* cycleMaterial_maker() {
  return new CycleMaterial();
}

// initialize the static member type_id
PersistentTypeID CycleMaterial::type_id("CycleMaterial", "Material", 
					cycleMaterial_maker);


CycleMaterial::CycleMaterial()
    : current(0)
{
}

CycleMaterial::~CycleMaterial()
{
}

void CycleMaterial::next() {
  if (members.size() == 0) ASSERTFAIL("Cycle material has no members");
  if (current+1 == members.size())
    current=0;
  else
    current++;
}

void CycleMaterial::prev() {
  if (members.size() == 0) ASSERTFAIL("Cycle material has no members");
  if (current-1 < 0)
    current=members.size()-1;
  else
    current--;
}

void CycleMaterial::shade(Color& result, const Ray& ray,
		  const HitInfo& hit, int depth, 
		  double atten, const Color& accumcolor,
		  Context* cx)
{
  members[current]->shade(result, ray, hit, depth, atten, accumcolor, cx);
}

const int CYCLEMATERIAL_VERSION = 1;

void 
CycleMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("CycleMaterial", CYCLEMATERIAL_VERSION);
  Material::io(str);
  SCIRun::Pio(str, current);
  SCIRun::Pio(str, members);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::CycleMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::CycleMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::CycleMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
