
#include <Packages/rtrt/Core/LightMaterial.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* lightMaterial_maker() {
  return new LightMaterial();
}

// initialize the static member type_id
PersistentTypeID LightMaterial::type_id("LightMaterial", "Material", 
					lightMaterial_maker);

LightMaterial::LightMaterial( const Color & color ) :
  color_( color )
{
}

LightMaterial::~LightMaterial()
{
}

void
LightMaterial::shade(Color& result, const Ray & /*ray*/,
		     const HitInfo & /*hit*/, int /*depth*/,
		     double , const Color& ,
		     Context* /*cx*/)
{
  result = color_;
}

const int LIGHTMATERIAL_VERSION = 1;

void 
LightMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("LightMaterial", LIGHTMATERIAL_VERSION);
  Material::io(str);
  SCIRun::Pio(str, color_);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::LightMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::LightMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::LightMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
