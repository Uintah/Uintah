#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Context.h>
#include <math.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* lambertianMaterial_maker() {
  return new LambertianMaterial();
}

// initialize the static member type_id
PersistentTypeID LambertianMaterial::type_id("LambertianMaterial", "Material", 
					     lambertianMaterial_maker);

LambertianMaterial::LambertianMaterial(const Color& R)
    : R(R)
{
}

LambertianMaterial::~LambertianMaterial()
{
}

void LambertianMaterial::shade(Color& result, const Ray& ray,
		  const HitInfo& hit, int depth,
		  double , const Color& ,
		  Context* cx)
{

    double nearest=hit.min_t;
    Object* obj=hit.hit_obj;
    Point hitpos(ray.origin()+ray.direction()*nearest);
    Vector normal(obj->normal(hitpos, hit));
    double incident_angle=-Dot(normal, ray.direction());
    if(incident_angle<0){
	incident_angle=-incident_angle;
	normal=-normal;
    }
    double ray_objnormal_dot(Dot(ray.direction(),normal));
    
    result = R * ambient(cx->scene, normal);
    int ngloblights=cx->scene->nlights();
    int nloclights=my_lights.size();
    int nlights=ngloblights+nloclights;
    cx->stats->ds[depth].nshadow+=nlights;
    for(int i=0;i<nlights;i++){
        Light* light;
        if (i<ngloblights)
	  light=cx->scene->light(i);
        else 
	  light=my_lights[i-ngloblights];

	if( !light->isOn() )
	  continue;

        Vector light_dir=light->get_pos()-hitpos;
	if (ray_objnormal_dot*Dot(normal,light_dir)>0) {
	  cx->stats->ds[depth].inshadow++;
	  continue;
	}
	double dist=light_dir.normalize();
	Color shadowfactor(1,1,1);
	if(cx->scene->lit(hitpos, light, light_dir, dist, shadowfactor, depth, cx) ){
	    double cos_theta=Dot(light_dir, normal);
	    if(cos_theta < 0){
		cos_theta=-cos_theta;
		light_dir=-light_dir;
	    }
	    result+=light->get_color(light_dir)*R*(cos_theta*shadowfactor);
	} else {
	    cx->stats->ds[depth].inshadow++;
	}
    }
}


const int LAMBERTIANMATERIAL_VERSION = 1;

void 
LambertianMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("LambertianMaterial", LAMBERTIANMATERIAL_VERSION);
  Material::io(str);
  SCIRun::Pio(str, R);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::LambertianMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::LambertianMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::LambertianMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
