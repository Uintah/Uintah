
#ifndef MATERIAL_H
#define MATERIAL_H 1

#include <Packages/rtrt/Core/Color.h>

namespace SCIRun {
  class Point;
  class Vector;
}

namespace rtrt {

  using namespace SCIRun;
  
struct Context;
class HitInfo;
class Scene;
class Ray;
class Stats;
class Worker;

class Material {
protected:
    // For a simple implementation of material, use this function.  Just
    // pass in ambient, diffuse, specular colors, as well as the specular
    // exponent (spec_coeff), the reflectivity (refl),
    // The other arguments should just be forwarded from the shade
    // parameter block.
    void phongshade(Color& result, const Color& ambient,
		    const Color& diffuse, const Color& specular,
		    double spec_coeff, double refl,
		    const Ray& ray, const HitInfo& hit,
		    int depth, 
		    double atten, const Color& accumcolor,
		    Context* cx);
public:
    Material();
    virtual ~Material();

    //ambient color (irradiance/pi) at position with surface normal
    Color ambient_hack(Scene* scene, const Point& position, const Vector& normal) const;

    // reflection of v with respect to n
    Vector reflection(const Vector& v, const Vector n) const;

    // gives the phong term without color of light or kh
    double phong_term( const Vector& e, const Vector& l, const Vector& n, double exponent) const;

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
