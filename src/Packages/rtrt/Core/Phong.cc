#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/Stats.h>

using namespace rtrt;
using namespace SCIRun;

Phong::Phong(const Color& ambient, const Color& diffuse,
	     const Color& specular, int specpow, double refl)
    : ambient(ambient), diffuse(diffuse), specular(specular),
      specpow(specpow), refl(refl)
{
}



Phong::~Phong()
{
}

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

void Phong::shade(Color& result, const Ray& ray,
		  const HitInfo& hit, int depth, 
		  double atten, const Color& accumcolor,
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
    
  Color difflight(0,0,0);
  Color speclight(0,0,0);
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
      difflight+=light->get_color()*(cos_theta*shadowfactor);
      if(specpow > 0.0){
	Vector H=light_dir-ray.direction();
	H.normalize();
	double cos_alpha= Dot(H, normal);
	if ( cos_alpha > 0 )
	  speclight+=light->get_color() * shadowfactor * ipow( cos_alpha, specpow);
      }
    } else {
      cx->stats->ds[depth].inshadow++;
    }
  }
    
#if 0
    if(cx->scene->ambient_hack){
      result=diffuse*(difflight+ambient_hack(cx->scene, normal))
	    +specular*speclight;
    } else {
#endif
      result=ambient+diffuse*difflight+specular*speclight;
#if 0
    }
#endif
  if (depth < cx->scene->maxdepth && (refl>0 )){
    double thresh=cx->scene->base_threshold;
    double ar=atten*refl;
    if(ar>thresh){
      Vector refl_dir = ray.direction() + normal*(2*incident_angle);
      Ray rray(hitpos, refl_dir);
      Color rcolor;
      cx->worker->traceRay(rcolor, rray, depth+1, ar,
			   accumcolor+result*atten, cx);
      result+=rcolor*refl;
      cx->stats->ds[depth].nrefl++;
    }
  }
}
