
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

inline double ipow(double x, int p)
{
  double result=1;
  while(p){
    if(p&1)
      result*=x;
    x*=x;
    p>>=1;
  }
  return result;
}

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
  inline Color ambient(Scene* scene, const SCIRun::Vector& normal) const {
    int a_mode = (local_ambient_mode==Global_Ambient) ? scene->ambient_mode : 
      local_ambient_mode;
    // in this next line, a_mode should never be Global_Ambient
    // .. but just in case someone sets someone sets it wrong
    // we'll just return the constant ambient color
    if (a_mode == Constant_Ambient || a_mode == Global_Ambient)
      return scene->getAmbientColor();

    if (a_mode == Arc_Ambient) {
      float cosine = scene->get_groundplane().cos_angle( normal );
#ifdef __sgi
      float sine = fsqrt ( 1.F - cosine*cosine );
#else
      float sine = sqrt(1.-cosine*cosine);
#endif
      //double w = (cosine > 0)? sine/2 : (1 -  sine/2);
      float w0, w1;
      if(cosine > 0){
	w0= sine/2.F;
	w1= (1.F -  sine/2.F);
      } else {
	w1= sine/2.F;
	w0= (1.F -  sine/2.F);
      }
      return scene->get_cup()*w1 + scene->get_cdown()*w0;
    } 

    // must be Sphere_Ambient
    Color c;
    scene->get_ambient_environment_map_color(normal, c);
    return c;
  }

  void SetScale(double u, double v) { uscale = u; vscale = v; }

  // reflection of v with respect to n
  SCIRun::Vector reflection(const SCIRun::Vector& v, const SCIRun::Vector n) const;

  // gives the phong term without color of light or kh
  double phong_term( const SCIRun::Vector& e, const SCIRun::Vector& l, const SCIRun::Vector& n, int exponent) const;

  //    virtual int get_scratchsize() {
  //      return 0;
  //    }

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
