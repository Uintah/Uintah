#include <Packages/rtrt/Core/Wood.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/Light.h>
#include <Core/Math/MiscMath.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* wood_maker() {
  return new Wood();
}

// initialize the static member type_id
PersistentTypeID Wood::type_id("Wood", "Material", wood_maker);

Wood::Wood(const Color&  c1,const Color&  c2, double ringscale)
    : ringscale(ringscale), lightwood(c1), darkwood(c2)
{
}

Wood::~Wood()
{
}

void Wood::shade(Color& result, const Ray& ray,
		  const HitInfo& hit, int depth, 
		  double, const Color&,
		  Context* cx)
{
    double nearest=hit.min_t;
    Object* obj=hit.hit_obj;
    Point hitpos(ray.origin()+ray.direction()*nearest);

    double n=noise(hitpos.asVector());
    hitpos+=Vector(n,n,n);
    double y=hitpos.y();
    double z=hitpos.z();
    double r=sqrt(y*y+z*z);
    r*=ringscale;
    r+=fabs(noise(r));
    r-=(int)r; // mod r,1
    r=SmoothStep(r, 0, 0.8) - SmoothStep(r, 0.83, 1.0);
    Color Rd=Interpolate(lightwood, darkwood, r);

    Vector normal(obj->normal(hitpos, hit));
    double cos_prime=-Dot(normal, ray.direction());
    if(cos_prime<0){
	cos_prime=-cos_prime;
	normal=-normal;
    }
    
    Color difflight(0,0,0);
    int nlights=cx->scene->nlights();
    cx->stats->ds[depth].nshadow+=nlights;
    for(int i=0;i<nlights;i++){
	Light* light=cx->scene->light(i);
	Vector light_dir=light->get_pos()-hitpos;
	double dist=light_dir.normalize();
	Color shadowfactor(1,1,1);
	if(cx->scene->lit(hitpos, light, light_dir, dist, shadowfactor, depth, cx) ){
            double cos_theta=Dot(light_dir, normal);
	    if(cos_theta < 0){
		cos_theta=-cos_theta;
		light_dir=-light_dir;
	    }
	    difflight+=light->get_color()*shadowfactor;
	} else {
	    cx->stats->ds[depth].inshadow++;
	}
    }
    
    result = Rd * (difflight + ambient(cx->scene, normal));
}

const int WOOD_VERSION = 1;

void 
Wood::io(SCIRun::Piostream &str)
{
  str.begin_class("Wood", WOOD_VERSION);
  Material::io(str);
  SCIRun::Pio(str, ringscale);
  SCIRun::Pio(str, lightwood);
  SCIRun::Pio(str, darkwood);
  SCIRun::Pio(str, noise);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Wood*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Wood::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Wood*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
