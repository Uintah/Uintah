
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

//#ifndef HAVE_FASTM
//#define fsqrt sqrt
//#endif

using namespace rtrt;

// initialize the static member type_id
SCIRun::PersistentTypeID Material::type_id("Material", "Persistent", 0);


Material::Material() : 
    uscale(1), 
    vscale(1),
    local_ambient_mode(Global_Ambient)
{
}

Material::~Material()
{
}

Vector Material::reflection(const Vector& v, const Vector n) const {
     return v - n * (2*Dot(v, n));
}


double Material::phong_term( const Vector& E, const Vector& L, const Vector& n, int ex) const {
  Vector H= L - E;
  H.normalize();
  double cos_alpha= Dot(H, n );
  if ( cos_alpha > 0 )
    return ipow( cos_alpha, ex );
  else
    return 0;
}

void Material::phongshade(Color& result,
			  const Color& diffuse,
			  const Color& specular,
			  int spec_coeff,
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
  double ray_objnormal_dot(Dot(ray.direction(),normal));
    
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
      //difflight+=light->get_color()*(cos_theta*shadowfactor);
      difflight+=light->get_color(light_dir)*cos_theta;

      if(spec_coeff > 0.0){
	Vector H=light_dir-ray.direction();
	H.normalize();
	double cos_alpha= Dot(H, normal);
	if ( cos_alpha > 0 )
	  speclight+=light->get_color(light_dir) * /*shadowfactor * */ipow( cos_alpha, spec_coeff);
      }
    } else {
      cx->stats->ds[depth].inshadow++;
    }
  }
    
  const Color & amb = ambient( cx->scene, normal );

  Color surfcolor = diffuse*(amb+difflight) + (specular*speclight);
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

const int MATERIAL_VERSION = 1;

void 
Material::io(SCIRun::Piostream &str)
{
  str.begin_class("Material", MATERIAL_VERSION);
  SCIRun::Pio(str, my_lights);
  AmbientType &tmp = local_ambient_mode;
  SCIRun::Pio(str, (unsigned int&)tmp);
  SCIRun::Pio(str, uscale);
  SCIRun::Pio(str, vscale);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Material*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Material::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Material*>(pobj);
    //ASSERT(obj != 0);
  }
}
} // end namespace SCIRun
