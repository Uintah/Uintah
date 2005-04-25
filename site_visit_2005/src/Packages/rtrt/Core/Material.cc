
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/Context.h>

#include <Core/Math/Expon.h>

#include <math.h>

//#ifndef HAVE_FASTM
//#define fsqrt sqrt
//#endif

using namespace rtrt;
using namespace SCIRun;

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

//ambient color (irradiance/pi) at position with surface normal
Color Material::ambient(Scene* scene, const SCIRun::Vector& normal) const {
  int a_mode = (local_ambient_mode==Global_Ambient) ? scene->ambient_mode : 
    local_ambient_mode;
  // In this next line, a_mode should never be Global_Ambient ... but
  // just in case someone sets it wrong we'll just return the constant
  // ambient color.
  if (a_mode == Constant_Ambient || a_mode == Global_Ambient)
    return scene->getAmbientColor();

  if (a_mode == Arc_Ambient) {
    float cosine = scene->get_groundplane().cos_angle( normal );
#ifdef __sgi
    float sine = fsqrt ( 1.F - cosine*cosine );
#else
    float sine = sqrt(1.-cosine*cosine);
#endif
    float w0, w1;
    if(cosine > 0){
      w0= sine/2.F;
      w1= (1.F -  sine/2.F);
    } else {
      w1= sine/2.F;
      w0= (1.F -  sine/2.F);
    }
    return scene->get_cup()*w1 + scene->get_cdown()*w0;
  } 

  // must be Sphere_Ambient
  Color c;
  scene->get_ambient_environment_map_color(normal, c);
  return c;
}


Vector Material::reflection(const Vector& v, const Vector n) const {
     return v - n * (2*Dot(v, n));
}


double Material::phong_term( const Vector& E, const Vector& L, const Vector& n, int ex) const {
  Vector H= L - E;
  H.normalize();
  double cos_alpha= Dot(H, n );
  if ( cos_alpha > 0 )
    return Pow( cos_alpha, ex );
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

    if(cx->scene->lit(hitpos, light, light_dir, dist, shadowfactor, depth, cx))
      {
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
            speclight+=light->get_color(light_dir) * /*shadowfactor * */
                       Pow( cos_alpha, spec_coeff);
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
      Worker::traceRay(rcolor, rray, depth+1, ar,
                       accumcolor+surfcolor*atten, cx);
      surfcolor+=rcolor*refl;
      cx->stats->ds[depth].nrefl++;
    }
  }
  result=surfcolor;
}

void Material::lambertianshade(Color& result,  const Color& diffuse,
                               const Ray& ray, const HitInfo& hit,
                               int depth, Context* cx)
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
    
  result = diffuse * ambient(cx->scene, normal);
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
    if(cx->scene->lit(hitpos, light, light_dir, dist, shadowfactor, depth, cx))
      {
        double cos_theta=Dot(light_dir, normal);
        if(cos_theta < 0){
          cos_theta=-cos_theta;
          light_dir=-light_dir;
        }
        result+=light->get_color(light_dir)*diffuse*(cos_theta*shadowfactor);
      } else {
        cx->stats->ds[depth].inshadow++;
      }
  }
}

void Material::lambertianshade(Color& result,  const Color& diffuse,
                               const Color& amb,
                               const Ray& ray, const HitInfo& hit,
                               int depth, Context* cx)
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
    
  result = diffuse * amb * ambient(cx->scene, normal);
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
    if(cx->scene->lit(hitpos, light, light_dir, dist, shadowfactor, depth, cx))
      {
        double cos_theta=Dot(light_dir, normal);
        if(cos_theta < 0){
          cos_theta=-cos_theta;
          light_dir=-light_dir;
        }
        result+=light->get_color(light_dir)*diffuse*(cos_theta*shadowfactor);
      } else {
        cx->stats->ds[depth].inshadow++;
      }
  }
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

void Material::animate(double, bool&) {}

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
