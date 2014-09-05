#include <Packages/rtrt/Core/PhongMaterial.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/Context.h>
#include <math.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* phongMaterial_maker() {
  return new PhongMaterial();
}

// initialize the static member type_id
PersistentTypeID PhongMaterial::type_id("PhongMaterial", "Material", 
				     phongMaterial_maker);


PhongMaterial::PhongMaterial(const Color& Rd, double opacity, double Rphong, 
			     double phong_exponent)
  : Rd(Rd), opacity(opacity), Rphong(Rphong), phong_exponent(phong_exponent)
{
}

PhongMaterial::~PhongMaterial()
{
}

void PhongMaterial::shade(Color& result, const Ray& ray,
		  const HitInfo& hit, int depth, 
		  double atten, const Color& accumcolor,
		  Context* cx)
{
    double nearest=hit.min_t;
    Object* obj=hit.hit_obj;
    Point hitpos(ray.origin()+ray.direction()*nearest);
    Vector normal(obj->normal(hitpos, hit));
    double cos_prime=-Dot(normal, ray.direction());
    if(cos_prime<0){
	cos_prime=-cos_prime;
	normal=-normal;
    }
    double ray_objnormal_dot(Dot(ray.direction(),normal));

    double opac = opacity + (1-opacity)*(1-cos_prime)*(1-cos_prime);/*+(depth-2)*.1;*/
#if 0
    if(opac>1)
	opac=1;
    else if(opac<0)
	opac=0;
#endif
    double transp = 1 - opac;
    
    Color difflight(0,0,0);
    Color speclight(0,0,0);
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
	    difflight+=light->get_color() * shadowfactor;
            if (Rphong > 0)
               speclight+=light->get_color() * shadowfactor * phong_term( ray.direction(), light_dir,
                                 normal, phong_exponent);
	} else {
	    cx->stats->ds[depth].inshadow++;
	}
    }
    
    Color surfcolor=Rd * opac * (difflight + ambient(cx->scene, normal)) + Rphong*speclight;

    // fire the reflection ray
    if (Rphong && depth < cx->scene->maxdepth && 
	atten > 0.02 && (1-transp) > 0.02){
      Vector refl_dir = reflection( ray.direction(), normal );
      Ray rray(hitpos, refl_dir);
      Color rcolor;
      cx->worker->traceRay(rcolor, rray, depth+1,  atten,
			   accumcolor+surfcolor*atten, cx);
      surfcolor += rcolor * (1.-transp) * Rphong;
      cx->stats->ds[depth].nrefl++;
    }

    // fire the transparency ray
    if (depth < cx->scene->maxdepth && 
	atten > 0.02 && transp > 0.02){
            Ray tray(hitpos, ray.direction());
            Color tcolor;
            cx->worker->traceRay(tcolor, tray, depth+1,  atten,
                                 accumcolor+difflight+speclight, cx);
            surfcolor+= tcolor * transp;
            cx->stats->ds[depth].nrefl++;
    }

    result=surfcolor;
}

const int PHONGMATERIAL_VERSION = 1;

void 
PhongMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("PhongMaterial", PHONGMATERIAL_VERSION);
  Material::io(str);
  SCIRun::Pio(str, Rd);
  SCIRun::Pio(str, opacity);
  SCIRun::Pio(str, Rphong);
  SCIRun::Pio(str, phong_exponent);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::PhongMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::PhongMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::PhongMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
}
