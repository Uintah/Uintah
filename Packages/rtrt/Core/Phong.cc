/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/Stats.h>

#include <Core/Math/Expon.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* phong_maker() {
  return new Phong();
}

// initialize the static member type_id
PersistentTypeID Phong::type_id("Phong", "Material", phong_maker);


Phong::Phong(const Color& diffuse, const Color& specular,
	     int specpow, double refl) :
  diffuse(diffuse), specular(specular), refl(refl), specpow(specpow)
{
}



Phong::~Phong()
{
}

void Phong::shade(Color& result, const Ray& ray,
		  const HitInfo& hit, int depth, 
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

    if( !light->isOn() ) continue;

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
      difflight+=light->get_color(light_dir)*(cos_theta*shadowfactor);
      if(specpow > 0.0){
	Vector H=light_dir-ray.direction();
	H.normalize();
	double cos_alpha= Dot(H, normal);
	if ( cos_alpha > 0 )
	  speclight+=light->get_color(light_dir) * shadowfactor * Pow( cos_alpha, specpow);
      }
    } else {
      cx->stats->ds[depth].inshadow++;
    }
  }

  result=(ambient(cx->scene, normal)+difflight)*diffuse+specular*speclight;

  if (depth < cx->scene->maxdepth && (refl>0 )){
    double thresh=cx->scene->base_threshold;
    double ar=atten*refl;
    if(ar>thresh){
      Vector refl_dir = ray.direction() + normal*(2*incident_angle);
      Ray rray(hitpos, refl_dir);
      Color rcolor;
      Worker::traceRay(rcolor, rray, depth+1, ar, accumcolor+result*atten, cx);
      result+=rcolor*refl;
      cx->stats->ds[depth].nrefl++;
    }
  }
}

const int PHONG_VERSION = 1;

void 
Phong::io(SCIRun::Piostream &str)
{
  str.begin_class("Phong", PHONG_VERSION);
  Material::io(str);
  SCIRun::Pio(str, diffuse);
  SCIRun::Pio(str, specular);
  SCIRun::Pio(str, refl);
  SCIRun::Pio(str, specpow);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Phong*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Phong::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Phong*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
