#include "PhongMaterial.h"
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

PhongMaterial::PhongMaterial(const Color& Rd, double opacity, double Rphong, double phong_exponent)
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
    double cos_prime=-normal.dot(ray.direction());
    if(cos_prime<0){
	cos_prime=-cos_prime;
	normal=-normal;
    }

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
	    difflight+=light->get_color() * shadowfactor;
            if (Rphong > 0)
               speclight+=light->get_color() * shadowfactor * phong_term( ray.direction(), light_dir,
                                 normal, phong_exponent);
	} else {
	    cx->stats->ds[depth].inshadow++;
	}
    }
    
    Color surfcolor=Rd * opac * (difflight + ambient_hack(cx->scene, hitpos, normal)) + Rphong*speclight;

    if (depth < cx->scene->maxdepth && atten > 0.02 && transp > 0.02){
            Ray tray(hitpos, ray.direction());
            Color tcolor;
            cx->worker->traceRay(tcolor, tray, depth+1,  atten,
                                 accumcolor+difflight+speclight, cx);
            surfcolor+= tcolor * transp;
            cx->stats->ds[depth].nrefl++;
    }
    

    result=surfcolor;
}
