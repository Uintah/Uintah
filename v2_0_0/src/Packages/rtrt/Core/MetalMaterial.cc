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
using namespace SCIRun;

Persistent* metalMaterial_maker() {
  return new MetalMaterial();
}

// initialize the static member type_id
PersistentTypeID MetalMaterial::type_id("MetalMaterial", "Material", 
					metalMaterial_maker);


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

    double ray_objnormal_dot(Dot(ray.direction(),normal));
    int ngloblights=cx->scene->nlights();
    int nloclights=my_lights.size();
    int nlights=ngloblights+nloclights;
    
    for(int i=0;i<nlights;i++){
        Light* light;
	if (i<ngloblights)
	  light=cx->scene->light(i);
	else 
	  light=my_lights[i-ngloblights];

	if( !light->isOn() )
	  continue;

	Vector light_dir=light->get_pos()-hitpos;
	light_dir.normalize();
//	if (ray_objnormal_dot*Dot(normal,light_dir)>0) continue;
	result+=light->get_color(light_dir) * specular_reflectance *
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
    
const int METALMATERIAL_VERSION = 1;

void 
MetalMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("MetalMaterial", METALMATERIAL_VERSION);
  Material::io(str);
  SCIRun::Pio(str, specular_reflectance);
  SCIRun::Pio(str, phong_exponent);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::MetalMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::MetalMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::MetalMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
