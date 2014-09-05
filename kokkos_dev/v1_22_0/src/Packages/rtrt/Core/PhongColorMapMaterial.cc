#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>
#include <Packages/rtrt/Core/PhongColorMapMaterial.h>

using namespace rtrt;

PhongColorMapMaterial::PhongColorMapMaterial(Object *value_source,
			ScalarTransform1D<float,Color> *diffuse_transform,
			ScalarTransform1D<float,float> *opacity_transform,
					     double spec_coeff,
					     double reflectance):
  value_source(value_source),
  diffuse_transform(diffuse_transform), opacity_transform(opacity_transform),
  spec_coeff(spec_coeff), reflectance(reflectance)
{
  // Let's just make sure that the reflectance is bounded by 0 and 1.
  if (reflectance < 0)
    this->reflectance = 0;
  else if (reflectance > 1)
    this->reflectance = 1;
}

PhongColorMapMaterial::~PhongColorMapMaterial() {
  // We don't need deallocate diffuse_transform or opacity_transform,
  // because these should belong to someone else.
  // Same for value_source.
}  

// This is where all the magic happens.  OK, here's what happens.
// 1. We get the hit location from HitInfo.
// 2. We use this Point in the call to value_source->interior_value() and
//    get the value there.
// 3. Use this value to index into diffuse_transform and opacity_transform.
// 4. If the opacity is less than 1, shoot the ray out and get the color there.
// 5. If the opacity is greater than 0, shade the surface using
//    Material::phongshade().
// 6. Return the resulting color.
void PhongColorMapMaterial::shade(Color& result, const Ray& ray,
				  const HitInfo& hit, int depth, 
				  double atten, const Color& accumcolor,
				  Context* cx) {
  double value;
  float opacity;
  if (value_source->interior_value(value, ray, hit.min_t)) {
    // found a good value
    // If we can guarantee that value will not exceed the min and max of
    // the transfer function we call call just lookup() which does no
    // bounds checking (and thus boosting performance).
    opacity = opacity_transform->lookup_bound((float)value);
  } else {
    // Hmm...no value.  Set the opacity to 0, so that the ray just continues.
    opacity = 0;
  }

  Color surface_color(0,0,0);
  
  if (opacity > 0) {
    // compute the surface color
    Color rcolor;
    phongshade(rcolor, diffuse_transform->lookup_bound((float)value),
	       Color(1,1,1), spec_coeff, reflectance,
	       ray, hit, depth, atten, accumcolor, cx);
    surface_color += rcolor * opacity;
  }
  if (opacity < 1) {
    // compute the transmitted ray
    Ray tray(ray.eval(hit.min_t+1e-6), ray.direction());
    Color tcolor;
    cx->worker->traceRay(tcolor, tray, depth+1,  atten,
			 accumcolor, cx);
    surface_color += tcolor * (1-opacity);
    cx->stats->ds[depth].nrefl++;
  }
  
  result = surface_color;
}
