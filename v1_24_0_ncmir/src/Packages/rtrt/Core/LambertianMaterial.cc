#include <Packages/rtrt/Core/LambertianMaterial.h>
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

Persistent* lambertianMaterial_maker() {
  return new LambertianMaterial();
}

// initialize the static member type_id
PersistentTypeID LambertianMaterial::type_id("LambertianMaterial", "Material", 
					     lambertianMaterial_maker);

LambertianMaterial::LambertianMaterial(const Color& R)
    : R(R)
{
}

LambertianMaterial::~LambertianMaterial()
{
}

void LambertianMaterial::shade(Color& result, const Ray& ray,
		  const HitInfo& hit, int depth,
		  double , const Color& ,
		  Context* cx)
{
  lambertianshade(result, R, ray, hit, depth, cx);
}


const int LAMBERTIANMATERIAL_VERSION = 1;

void 
LambertianMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("LambertianMaterial", LAMBERTIANMATERIAL_VERSION);
  Material::io(str);
  SCIRun::Pio(str, R);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::LambertianMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::LambertianMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::LambertianMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
