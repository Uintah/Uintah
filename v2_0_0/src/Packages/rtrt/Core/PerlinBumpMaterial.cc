#include <Packages/rtrt/Core/PerlinBumpMaterial.h>
#include <Packages/rtrt/Core/DummyObject.h>
#include <iostream.h>

using namespace rtrt;
using namespace SCIRun;

void PerlinBumpMaterial::shade(Color& result, const Ray& ray,
			 const HitInfo& hit, int depth,
			 double atten, const Color& accumcolor,
			 Context* cx)
{
    double nearest=hit.min_t;
    Object* obj = hit.hit_obj;
    
    DummyObject* dummy = new DummyObject(obj,m);
    Point hitpos(ray.origin()+ray.direction()*nearest);
    Vector n = obj->normal(hitpos,hit);
    double c = .2;
    HitInfo tmp_hit = hit;
    
   n += c*noise.vectorTurbulence(Point(hitpos.x()*64,
					hitpos.y()*64,
					hitpos.z()*64),2);
//      n += c*bubble.Perturbation(Point(hitpos.x()*16,
//  				     hitpos.y()*16,
//  				     hitpos.z()*16));
//     n += c*bubble.Perturbation(hitpos);
    
	
//     cout << "Normal: " << n << endl;

//     n += Vector(1,1,1);
    n.normalize();

//     cout << "Normal: " << n << endl;
    
    dummy->SetNormal(n);

    tmp_hit.hit_obj = dummy;
    m->shade(result, ray, tmp_hit, depth, atten, accumcolor, cx);
}

