/*
  PhongColorMapMaterial.h

    Shades a surface based on the colors and opacities obtained from
    looking up the value (value_source->interior_value()) in diffuse_terms
    and opacity_terms.

 Author: James Bigler (bigler@cs.utah.edu)
 Date: July 11, 2002
 
 */
#ifndef PHONG_COLOR_MAP_MATERIAL_H
#define PHONG_COLOR_MAP_MATERIAL_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>

namespace rtrt {

class PhongColorMapMaterial : public Material {
  // diffuse_transform and opacity_transform should belong to someone
  // else, preferably to someone who can edit them at runtime.
  ScalarTransform1D<float,Color> *diffuse_transform;
  ScalarTransform1D<float,float> *opacity_transform;
  double spec_coeff;
  double reflectance; // Goes from 0 to 1

  // This object must define a meaningful interior_value funtion, in order
  // for any results to work.
  Object *value_source;
public:
  PhongColorMapMaterial(Object *value_source,
			ScalarTransform1D<float,Color> *diffuse_transform,
			ScalarTransform1D<float,float> *opacity_transform,
			double spec_coeff = 100, double reflectance = 0);
  virtual ~PhongColorMapMaterial();

  virtual void io(SCIRun::Piostream &/*str*/) { ASSERTFAIL("not implemented");}
  // This function is used by some shadow routines to determine intersections
  // for shadow feelers.  Because we need the HitInfo to determine the
  // opacity, we should always return 1.
  inline double get_opacity() { return 1; }

  // This is where all the magic happens.
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
