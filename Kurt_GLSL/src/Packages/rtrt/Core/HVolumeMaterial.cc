#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/VolumeDpy.h>
#include <Packages/rtrt/Core/HVolumeMaterial.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>

using namespace rtrt;
using namespace SCIRun;

HVolumeMaterial::HVolumeMaterial(VolumeDpy *dpy,
				 ScalarTransform1D<float,float> *f1_to_f2,
				 ScalarTransform1D<float,Material*> *f2_to_material):
  vdpy(dpy), f1_to_f2(f1_to_f2), f2_to_material(f2_to_material) {

}

void HVolumeMaterial::shade(Color& result, const Ray& ray,
			    const HitInfo& hit, int depth,
			    double atten, const Color& accumcolor,
			    Context* cx) {
  // get the current value from vdpy
  float isoval = vdpy->isoval;
  // lookup into f1_to_f2 if it exists
  float f2;
  if (f1_to_f2)
    f2 = f1_to_f2->lookup_bound(isoval);
  else
    f2 = isoval;
  // use this value to get the material
  Material *matl = f2_to_material->lookup(f2);
  matl->shade(result, ray, hit, depth, atten, accumcolor, cx);
}
