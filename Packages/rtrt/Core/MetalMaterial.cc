#include <Packages/rtrt/Core/MetalMaterial.h>
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

MetalMaterial::MetalMaterial(const Color& specular_reflectance)
    : specular_reflectance(specular_reflectance)
{
     phong_exponent = 100.0;
}

MetalMaterial::MetalMaterial(const Color& specular_reflectance, double phong_exponent)
    : specular_reflectance(specular_reflectance), phong_exponent(phong_exponent)
{
}

MetalMaterial::~MetalMaterial()
{
}

void MetalMaterial::shade(Color& result, const Ray& ray,
		  const HitInfo& hit, int depth, 
		  double atten, const Color& accumcolor,
		  Context* cx)
{
    result = Color(0,0,0);
    double nearest=hit.min_t;
    Object* obj=hit.hit_obj;
    Point hitpos(ray.origin()+ray.direction()*nearest);
    Vector normal(obj->normal(hitpos, hit));
    double cosine = -Dot(normal, ray.direction());

    if(cosine<0){
	cosine=-cosine;
	normal=-normal;
    }
    int nlights=cx->scene->nlights();

    for(int i=0;i<nlights;i++){
	Light* light=cx->scene->light(i);
	Vector light_dir=light->get_pos()-hitpos;
	light_dir.normalize();
	result+=light->get_color() * specular_reflectance *
                   phong_term( ray.direction(), light_dir, normal, phong_exponent);
    }

    if (depth < cx->scene->maxdepth ){
            Vector refl_dir = reflection( ray.direction(), normal );
            float k = 1 - cosine;
            k *= k*k*k*k;
            Color R = specular_reflectance * (1-k) + Color(1,1,1)*k;
            Ray rray(hitpos, refl_dir);
            Color rcolor;
            cx->worker->traceRay(rcolor, rray, depth+1,  atten*R.luminance(),
                                 accumcolor, cx);
            result+= R * rcolor;
            cx->stats->ds[depth].nrefl++;
    }
}
    
