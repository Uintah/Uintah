#include "Phong.h"

using namespace rtrt;

Phong::Phong(const Color& ambient, const Color& diffuse,
	     const Color& specular, double specpow, double refl)
    : ambient(ambient), diffuse(diffuse), specular(specular),
      specpow(specpow), refl(refl), transp(0)
{
}



Phong::~Phong()
{
}

void Phong::shade(Color& result, const Ray& ray,
		  const HitInfo& hit, int depth, 
		  double atten, const Color& accumcolor,
		  Context* cx)
{
    phongshade(result, ambient, diffuse, specular, specpow, refl,
	       ray, hit, depth, atten, accumcolor, cx);
}
