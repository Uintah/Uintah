
#ifndef MATERIAL_H
#define MATERIAL_H 1

#include <Core/Persistent/Persistent.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Array1.h>
#include <math.h>

namespace rtrt {
  class Material;
}
namespace SCIRun {
  void Pio(Piostream&, rtrt::Material*&);
}

namespace rtrt {

struct Context;
class  HitInfo;
class  Ray;
class  Stats;
class  Worker;

class Material : public virtual SCIRun::Persistent {
protected:

  double uscale;
  double vscale;

  // For a simple implementation of material, use this function.  Just
  // pass in diffuse, specular colors, as well as the specular
  // exponent (spec_coeff), the reflectivity (refl),
  // The other arguments should just be forwarded from the shade
  // parameter block.
  void phongshade(Color& result,
		  const Color& diffuse, const Color& specular,
		  int spec_coeff, double refl,
		  const Ray& ray, const HitInfo& hit,
		  int depth, 
		  double atten, const Color& accumcolor,
		  Context* cx);
  // This is like phongshade, but for lambertian surfaces
  void lambertianshade(Color& result,  const Color& diffuse,
                       const Ray& ray, const HitInfo& hit,
                       int depth, Context* cx);
  // This one takes an ambient term
  void lambertianshade(Color& result,  const Color& diffuse,
                       const Color& amb,
                       const Ray& ray, const HitInfo& hit,
                       int depth, Context* cx);
public:
  Material();
  virtual ~Material();
  Array1<Light *> my_lights;
  AmbientType local_ambient_mode;
  
  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, rtrt::Material*&);
  
  //ambient color (irradiance/pi) at position with surface normal
  Color ambient(Scene* scene, const SCIRun::Vector& normal) const;

  void SetScale(double u, double v) { uscale = u; vscale = v; }

  // reflection of v with respect to n
  SCIRun::Vector reflection(const SCIRun::Vector& v,
                            const SCIRun::Vector n) const;

  // gives the phong term without color of light or kh
  double phong_term( const SCIRun::Vector& e, const SCIRun::Vector& l,
                     const SCIRun::Vector& n, int exponent) const;

  // This allows a material to be animated at the frame change.
  
  // By principal, this function should be fast in the common case.
  // In other words, if you are doing something every frame it needs
  // to be quick, but if you do something very infreqently (such as a
  // user generated event) then it may be ok for the function to
  // selectively perform the longer operation.  It will cause a hickup
  // in the framerate, but because it happens with a user event, the
  // user will be anticipating it.  In general this should be as fast
  // an operation as possible, though.
  virtual void animate(double t, bool& changed);

  // To implement a new material, you must override this method.
  // It should compute a resulting color (result), for the ray.
  // Parameters are:
  // result - resultant color
  // ray    - incoming ray
  // hit    - Contains the hit record for the intersection point.  You
  //		should use hit.min_t for the distance along the ray where
  // 		the intersection occurred.  The object hit is in hit.hit_obj.
  //		When calling the normal() method on the hit object, you
  //		should pass in *this* hit record, but NOT when calling other
  // 		intersect, light_intersect or multi_light_intersect methods
  //		(i.e. when computing a shadow ray, reflection ray or
  // 		transparency ray).
  // depth  - The depth of the ray.  depth==0 is an eye ray.  You should
  //          stop any recursive rays after (depth > cx->scene->maxdepth).
  // atten  - An accumulated attenuation factor.  This is 1.0 for the
  //          primary ray, and is diminished by reflection and transparency
  //          rays.  This can also be used to cull the ray tree.
  // accumcolor - The accumulated color of the intersection point, as
  //              passed down the ray tree.  This is an approxmation
  //              of the final surface color.
  // cx     - The context of the ray.  Context contains pointers to the
  //          scene, the worker, and the stats objects.  cx->worker should
  //          be used to trace any subsequent rays (normal rays using
  //          cx->worker->traceRay, and shadow rays using cx->worker->lit).
  //          The cx object should passed to these methods as well.
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx)=0;
};

} // end namespace rtrt

#endif
