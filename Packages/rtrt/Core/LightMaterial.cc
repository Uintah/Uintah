
#include <Packages/rtrt/Core/LightMaterial.h>

using namespace rtrt;

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
