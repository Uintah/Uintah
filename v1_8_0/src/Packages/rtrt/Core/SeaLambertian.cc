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
using namespace SCIRun;

Persistent* seaLambertian_maker() {
  return new SeaLambertianMaterial();
}

// initialize the static member type_id
PersistentTypeID SeaLambertianMaterial::type_id("SeaLambertianMaterial", "Material", 
					seaLambertian_maker);

SeaLambertianMaterial::SeaLambertianMaterial(const Color& R, 
					     TimeVaryingCheapCaustics *caustics)
  : Object(this), R(R), caustics(caustics)
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
      result += caustics->GetCausticColor( hitpos, currentTime ) * 
	(cos_theta*shadowfactor);
    }
  }

}

void SeaLambertianMaterial::intersect(Ray&, HitInfo&, DepthStats*,
				      PerProcessorContext*)
{
  cerr << "SeaLambertianMaterial should not be added to scene!\n";
}

Vector SeaLambertianMaterial::normal(const Point&, const HitInfo&)
{
  return Vector(0,0,0);
}
void SeaLambertianMaterial::animate(double t, bool& changed)
{
  currentTime=t;
  changed=true;
}

void SeaLambertianMaterial::compute_bounds(BBox&, double)
{
}

const int SEALAMBERTIAN_VERSION = 1;

void 
SeaLambertianMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("SeaLambertianMaterial", SEALAMBERTIAN_VERSION);
  Material::io(str);
  SCIRun::Pio(str, R);
  SCIRun::Pio(str, caustics);
  str.end_class();
}


namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::SeaLambertianMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::SeaLambertianMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::SeaLambertianMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
