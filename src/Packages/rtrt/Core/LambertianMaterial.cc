#include "LambertianMaterial.h"
#include "HitInfo.h"
#include "Light.h"
#include "Ray.h"
#include "Scene.h"
#include "Stats.h"
#include "Object.h"
#include "Worker.h"
#include "Context.h"
#include <math.h>

using namespace rtrt;

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
    double incident_angle=-normal.dot(ray.direction());
    if(incident_angle<0){
	incident_angle=-incident_angle;
	normal=-normal;
    }
    
    result = R * ambient_hack(cx->scene, hitpos, normal);
    int nlights=cx->scene->nlights();
    cx->stats->ds[depth].nshadow+=nlights;
    for(int i=0;i<nlights;i++){
	Light* light=cx->scene->light(i);
	Vector light_dir=light->get_pos()-hitpos;
	double dist=light_dir.normalize();
	Color shadowfactor(1,1,1);
	if(cx->worker->lit(hitpos, light, light_dir, dist, shadowfactor, depth, cx) ){
	    double cos_theta=light_dir.dot(normal);
	    if(cos_theta < 0){
		cos_theta=-cos_theta;
		light_dir=-light_dir;
	    }
	    result+=light->get_color()*R*(cos_theta*shadowfactor);
	} else {
	    cx->stats->ds[depth].inshadow++;
	}
    }
}
