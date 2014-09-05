#include "Speckle.h"
#include "Point.h"
#include "Vector.h"
#include "Ray.h"
#include "HitInfo.h"
#include "Scene.h"
#include "Stats.h"
#include "Object.h"
#include "Worker.h"
#include "Context.h"
#include "Light.h"

using namespace rtrt;

Speckle::Speckle(double scale,
               const Color&  c1,
               const Color&  c2)
    : scale(scale), c1(c1), c2(c2)
{
}

Speckle::~Speckle()
{
}

void Speckle::shade(Color& result, const Ray& ray,
		  const HitInfo& hit, int depth, 
		  double, const Color&,
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

    double pixelsize=0.001;
    Point p(scale*hitpos.x(), scale*hitpos.y(), scale*hitpos.z());
    double noise=turbulence(p, pixelsize);
    double w=noise/1.3;
    Color Rd= w*c1 + (1-w)*c2;
    
    Color difflight(0,0,0);
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
	    difflight+=light->get_color()*shadowfactor;
	} else {
	    cx->stats->ds[depth].inshadow++;
	}
    }
    
    result = Rd * (difflight + ambient_hack(cx->scene, hitpos, normal));

}
