#include <Packages/rtrt/Core/PerlinBumpMaterial.h>
#include <Packages/rtrt/Core/DummyObject.h>

using namespace rtrt;
using namespace SCIRun;

void PerlinBumpMaterial::shade(Color& result, const Ray& ray,
			 const HitInfo& hit, int depth,
			 double atten, const Color& accumcolor,
			 Context* cx)
{
  double nearest=hit.min_t;
  Object* obj = hit.hit_obj;
    
  Point hitpos(ray.origin()+ray.direction()*nearest);
  Vector n = obj->normal(hitpos,hit);
  double c = .2;
  HitInfo tmp_hit = hit;
    
  n += c*noise.vectorTurbulence(Point(hitpos.x()*64,
                                      hitpos.y()*64,
                                      hitpos.z()*64),2);
  n.normalize();

  // Create dummy's memory on the stack.  Please, please, please don't
  // allocate memory in rendertime code!
  DummyObject dummy(obj,m);
  dummy.SetNormal(n);
  tmp_hit.hit_obj = &dummy;
  
  m->shade(result, ray, tmp_hit, depth, atten, accumcolor, cx);
}

