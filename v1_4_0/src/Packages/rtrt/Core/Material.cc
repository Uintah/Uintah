
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/Context.h>
#include <math.h>
#include <iostream>

using namespace rtrt;

Material::Material()
{
}

Material::~Material()
{
}

Color Material::ambient_hack(Scene* scene, const Point& /*p*/, const Vector& normal) const
{
    if(scene->ambient_hack){
#if 0
	Color B(scene->get_cup() );
	Color C(scene->get_cdown());
#endif

        float cosine = scene->get_groundplane().cos_angle( normal );
        float sine = fsqrt ( 1.F - cosine*cosine );
        //double w = (cosine > 0)? sine/2 : (1 -  sine/2);
        float w0, w1;
	if(cosine > 0){
             w0= sine/2.F;
	     w1= (1.F -  sine/2.F);
	} else {
             w1= sine/2.F;
	     w0= (1.F -  sine/2.F);
	}
/*
        double cons =  scene->get_groundplane().scaled_distance( p );
        cons += 0.5*(1+cosine);
        if (cons > 1) cons = 1;
*/
#if 0
        double cons = 1;
        Color D = B*C;
        Color E = B*(1.F-w) + C*(w);
        return cons*E + (1-cons)*D;
#else
        return scene->get_cup()*w1 + scene->get_cdown()*w0;
#endif
    } else {
	return scene->get_average_bg( ) ;
    }
} 


Vector Material::reflection(const Vector& v, const Vector n) const {
     return v - n * (2*Dot(v, n));
}


double Material::phong_term( const Vector& e, const Vector& l, const Vector& n, double ex) const {
    Vector L = l;
    L.normalize();
    Vector E = e;
    E.normalize();
    Vector H= L - E;
    H.normalize();
    double cos_alpha= Dot(H, n );
    if ( cos_alpha > 0 )
        return pow( cos_alpha, ex );
    else
        return 0;
}



void Material::phongshade(Color& result,
			  const Color& amb,
			  const Color& diffuse,
			  const Color& specular,
			  double spec_coeff,
			  double refl,
			  const Ray& ray,
			  const HitInfo& hit,
                          int depth,
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
	if(cx->worker->lit(hitpos, light, light_dir, dist, shadowfactor, depth, cx) ){
	    double cos_theta=Dot(light_dir, normal);
	    if(cos_theta < 0){
		cos_theta=-cos_theta;
		light_dir=-light_dir;
	    }
	    difflight+=light->get_color()*(cos_theta*shadowfactor);
	    if(spec_coeff > 0.0)
		speclight+=light->get_color() * shadowfactor * phong_term( ray.direction(), light_dir,
                                 normal, spec_coeff);
	} else {
	    cx->stats->ds[depth].inshadow++;
	}
    }
    
    Color surfcolor;
    if(cx->scene->ambient_hack){
	surfcolor=diffuse*(difflight+ambient_hack(cx->scene, hitpos, normal))
	    +specular*speclight;
    } else {
	surfcolor=amb+diffuse*difflight+specular*speclight;
    }
    if (depth < cx->scene->maxdepth && (refl>0 )){

	double thresh=cx->scene->base_threshold;
	double ar=atten*refl;
	if(ar>thresh){
	    Vector refl_dir = reflection( ray.direction(), normal );
	    Ray rray(hitpos, refl_dir);
	    Color rcolor;
	    cx->worker->traceRay(rcolor, rray, depth+1, ar,
				 accumcolor+surfcolor*atten, cx);
	    surfcolor+=rcolor*refl;
	    cx->stats->ds[depth].nrefl++;
	}
    }
    result=surfcolor;
}
