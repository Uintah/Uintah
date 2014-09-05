#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/SeaLambertian.h>
#include <Core/Thread/Time.h>
#include <math.h>

using namespace rtrt;

SeaLambertianMaterial::SeaLambertianMaterial(const Color& R, 
					     TimeVaryingCheapCaustics *caustics)
  : R(R), caustics(caustics)
{
}

SeaLambertianMaterial::~SeaLambertianMaterial()
{
}

void SeaLambertianMaterial::shade(Color& result, const Ray& ray,
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
    int nlights=cx->scene->nlights();
    cx->stats->ds[depth].nshadow+=nlights;
    for(int i=0;i<nlights;i++){
	Light* light=cx->scene->light(i);
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
	    result+=light->get_color()*R*(cos_theta*shadowfactor);
	} else {
	    cx->stats->ds[depth].inshadow++;
	}
    }

    if (caustics)
      {
	Color shadowfactor(1,1,1);
	Vector light_dir = caustics->GetProjectionAxis();
	if(cx->scene->lit(hitpos, NULL, light_dir,
			   MAXDOUBLE, shadowfactor, depth, cx) )
	  {
	    double cos_theta=Dot(light_dir,normal);
	    result += caustics->GetCausticColor( hitpos, 
						 SCIRun::Time::currentSeconds() ) * 
	      (cos_theta*shadowfactor);
	  }
      }

}
