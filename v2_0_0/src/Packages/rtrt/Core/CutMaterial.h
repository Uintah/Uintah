#ifndef CUTMATERIAL_H
#define CUTMATERIAL_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/CutPlaneDpy.h>
#include <Packages/rtrt/Core/ColorMap.h>

/*
New Material for Cutting Planes.
If the object returns true and a value from color_interior,
this will color the interior of an object according to a ColorMap.
Otherwise it will just call surfmat to color as normal.
*/

namespace rtrt {

class CutMaterial : public Material {
  Material *surfmat; //if not outside, use this to color instead
  CutPlaneDpy *dpy;
  ColorMap *cmap;
public:
  CutMaterial(Material *surfmat, ColorMap *cmap=0, CutPlaneDpy *dpy=0);
  CutMaterial(Material *surfmat, CutPlaneDpy *dpy=0, ColorMap *cmap=0);
  virtual ~CutMaterial() {};
  virtual void io(SCIRun::Piostream &/*str*/) { ASSERTFAIL("not implemented");}
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx);
};
 
} // end namespace rtrt
#endif
