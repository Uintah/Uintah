#include <Packages/rtrt/Core/DielectricMaterial.h>
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
using namespace SCIRun;

// when a ray travels through the medium it loses
// intensity according to dI = -CI dx where dx is distance
// This means DI/dx = -CI.  This is solved by the exponential
// I = k exp(-Cx) + k'.  Putting in boundary conditions, we know
// that I(0) = I0, and I(1) = I(0) * extinction.  The first implies
// I(x) = I0 exp(-Cx).  The second implies
// I0 * extinction = I0 exp(-C) -> -C = log(extinction)
// Color extinction_constant stores -C for each channel

// The above is crap - Steve

#define COLOR_EPS .02

Persistent* dielectricMaterial_maker() {
  return new DielectricMaterial();
}

// initialize the static member type_id
PersistentTypeID DielectricMaterial::type_id("DielectricMaterial", "Material", 
					     dielectricMaterial_maker);

DielectricMaterial::DielectricMaterial(double n_in, double n_out, bool nothing_inside) :
    n_in( n_in ), n_out( n_out ), nothing_inside(nothing_inside)
{
    R0 = (n_in-n_out)/ (n_in + n_out);
    R0 *= R0;
    phong_exponent = 128;
    extinction_in = Color(1,1,1);
    extinction_out = Color(1,1,1);

    bg_in=Color(1,1,1);
    bg_out=Color(1,1,1);
}


DielectricMaterial::DielectricMaterial(double n_in, double n_out,
				       double R0, int phong_exponent,
				       const Color& extinction_in,
				       const Color& extinction_out,
				       bool nothing_inside,
				       double extinction_scale)
: n_in( n_in ), n_out( n_out ), R0(R0), phong_exponent(phong_exponent),
  extinction_in(extinction_in), extinction_out(extinction_out),
  nothing_inside(nothing_inside), extinction_scale(extinction_scale)
{
    bg_in=Color(extinction_in.red()==1, 
		extinction_in.green()==1, 
		extinction_in.blue()==1);
    bg_out=Color(extinction_out.red()==1, 
		 extinction_out.green()==1, 
		 extinction_out.blue()==1);
}

DielectricMaterial::~DielectricMaterial()
{
}




void DielectricMaterial::shade(Color& result, const Ray& ray,
			       const HitInfo& hit, int depth,
			       double atten, const Color& accumcolor,
			       Context* cx)
{
  double nearest=hit.min_t;
  Object* obj=hit.hit_obj;
  Point hitpos(ray.origin()+ray.direction()*nearest);
  Vector normal(obj->normal(hitpos, hit));
  double ray_objnormal_dot(Dot(ray.direction(),normal));
  double cosine = -Dot(normal, ray.direction());
  bool incoming;

  if(cosine<0){
    cosine=-cosine;
    normal=-normal;
    incoming = false;
  }  else {
    incoming = true;
  }

  // compute Phong highlights
  int ngloblights=cx->scene->nlights();
  int nloclights=my_lights.size();
  int nlights=ngloblights+nloclights;
  result = Color(0,0,0);
  for(int i=0;i<nlights;i++){
    Light* light;
    if (i<ngloblights)
      light=cx->scene->light(i);
    else 
      light=my_lights[i-ngloblights];

    if( !light->isOn() )
      continue;

    Vector light_dir=light->get_pos()-hitpos;
    double n_dot_light = Dot(normal, light_dir);
    if (ray_objnormal_dot*n_dot_light>0)
      continue;
    double light_dist = light_dir.normalize();
    Color color;
    double scaled_dist = light_dist*extinction_scale;
    if(n_dot_light > 0){
      // Light is outside of the surface, use extinction_out
      color=light->get_color(light_dir)*Color(powf(extinction_out.red(), scaled_dist),
				     powf(extinction_out.green(), scaled_dist),
				     powf(extinction_out.blue(), scaled_dist));
    } else {
      // Light is inside of the surface, use exctinction_in
      color=light->get_color(light_dir)*Color(powf(extinction_in.red(), scaled_dist),
				     powf(extinction_in.green(), scaled_dist),
				     powf(extinction_in.blue(), scaled_dist));
    }
    result += color * phong_term( ray.direction(), light_dir,
				  normal, phong_exponent);
  }

    
  // Snell's Law: n sin t = n' sin t'
  // so n^2 sin^2 t =  n'^2 sin ^2 t'
  // so n^2 (1 - cos^2 t)  =  n'^2 (1 - cos ^2 t')
  // cos^2 t' = [ n'^2 - n^2 (1 - cos^2 t) ] /  n'^2
  //          = 1 - (n^2 / n'^2) (1 - cos^2 t)
  // refracted ray, geometry
  //
  //            ^
  //            | N
  //     \      |
  //     V \    |
  //         \  |
  //           \|
  //             --------> U
  //             \ 
  //              \ 
  //               \  
  //                \ V'
  //
  //     V = Usint - Ncost
  //     U = (V +  Ncost) / sint
  //     V'= Usint'- Ncost'
  //       = (V + Ncost) (n/n') -  Ncost'
  //
  if (depth < cx->scene->maxdepth){
    double n;
    double nPrime;
    Object* obj = cx->scene->get_object();
    if (incoming) { // cosine is associated with n_out
      n = n_out;
      nPrime = n_in;
      if(nothing_inside)
	obj = hit.hit_obj;
    } else {
      n = n_in;
      nPrime = n_out;
    }
    double cosinePrimeSquared = 1 - ((n*n) / (nPrime*nPrime)) *
      (1 - cosine*cosine);
    if (cosinePrimeSquared <= 0) { // total internal reflection
      cx->stats->ds[depth].nrefl++;
      Ray rray(hitpos, reflection( ray.direction(), normal ));
      HitInfo rhit;
      obj->intersect(rray, rhit, &cx->stats->ds[depth], cx->ppc);
      if(rhit.was_hit){
	double scaled_dist = rhit.min_t*extinction_scale;
	Color filter;
	if(incoming){
	  filter = Color(powf(extinction_in.red(), scaled_dist),
			 powf(extinction_in.green(), scaled_dist),
			 powf(extinction_in.blue(), scaled_dist));
	} else {
	  filter = Color(powf(extinction_out.red(), scaled_dist),
			 powf(extinction_out.green(), scaled_dist),
			 powf(extinction_out.blue(), scaled_dist));
	}
	double ratten = atten * filter.max_component();
	Color rcolor;
	rhit.hit_obj->get_matl()->shade(rcolor, rray, rhit, depth+1, ratten,
				       accumcolor, cx);
	result += filter*rcolor;
      } else {
	// Attenuate to background at infinity
	// Someday, perhaps we should have a background distance, so that
	// You could actually see the background if you wanted...
	cx->stats->ds[depth].nbg++;
	Color bg;
	cx->scene->get_bgcolor( ray.direction(), bg );
	if(incoming)
	  result += bg_in; // I'm changing this so the water is blue
//	  result += bg_in*bg;
	else
	  result += bg_out; // changing this for the same reason
//	  result += bg_out*bg;
      }

      cx->stats->ds[depth].nrefl++;
      return;
    }
    double cosinePrime = sqrt( cosinePrimeSquared );
    double smallCosine = cosine < cosinePrime? cosine : cosinePrime;
    double k = 1 - smallCosine;
    k *= (k*k)*(k*k);
    double R = R0 * (1-k) + k;
    double Ratten = R*atten;
    if(Ratten > COLOR_EPS) {
      cx->stats->ds[depth].nrefl++;
      Ray rray(hitpos, reflection( ray.direction(), normal ));
      HitInfo rhit;
      obj->intersect(rray, rhit, &cx->stats->ds[depth], cx->ppc);
      if(rhit.was_hit){
	double scaled_dist = rhit.min_t*extinction_scale;
	Color filter;
	if(incoming){
	  filter = Color(powf(extinction_in.red(), scaled_dist),
			 powf(extinction_in.green(), scaled_dist),
			 powf(extinction_in.blue(), scaled_dist));
	} else {
	  filter = Color(powf(extinction_out.red(), scaled_dist),
			 powf(extinction_out.green(), scaled_dist),
			 powf(extinction_out.blue(), scaled_dist));
	}
	Ratten *= filter.max_component();
	// Could do another if(Ratten > COLOR_EPS) here, but I suspect
	// That is will just slow us down - Steve
	Color rcolor;
	rhit.hit_obj->get_matl()->shade(rcolor, rray, rhit, depth+1, Ratten,
					accumcolor, cx);
	result += filter*rcolor*R;
      } else {
	// Attenuate to background at infinity
	// Someday, perhaps we should have a background distance, so that
	// You could actually see the background if you wanted...
	cx->stats->ds[depth].nbg++;
	Color bg;
	cx->scene->get_bgcolor( ray.direction(), bg );
	if(incoming)
	  result += bg_in*bg*R;
	else
	  result += bg_out*bg*R;
      }
    }
    double Tatten = (1.-R)*atten;
    if(Tatten > COLOR_EPS) {
      cx->stats->ds[depth].ntrans++;
      Vector transmittedDirection = ( n / nPrime) * 
	(ray.direction() + normal * cosine) - normal*cosinePrime;
      Ray tray(hitpos, transmittedDirection);
      HitInfo thit;
      obj->intersect(tray, thit, &cx->stats->ds[depth], cx->ppc);
      if(thit.was_hit){
	double scaled_dist = thit.min_t*extinction_scale;
	Color filter;
	if(incoming){
	  filter = Color(powf(extinction_out.red(), scaled_dist),
			 powf(extinction_out.green(), scaled_dist),
			 powf(extinction_out.blue(), scaled_dist));
	} else {
	  filter = Color(powf(extinction_in.red(), scaled_dist),
			 powf(extinction_in.green(), scaled_dist),
			 powf(extinction_in.blue(), scaled_dist));
	}
	Tatten *= filter.max_component();
	// Could do another if(Ratten > COLOR_EPS) here, but I suspect
	// That is will just slow us down - Steve
	Color tcolor;
	thit.hit_obj->get_matl()->shade(tcolor, tray, thit, depth+1, Tatten,
					accumcolor, cx);
	result += filter*tcolor*(1-R);
      } else {
	// Attenuate to background at infinity
	// Someday, perhaps we should have a background distance, so that
	// You could actually see the background if you wanted...
	cx->stats->ds[depth].nbg++;
//	Color bg;
//	cx->scene->get_bgcolor( ray.direction(), bg );
	if(incoming)
	  result += bg_out*(1-R)*.2;
	else
	  result += bg_in*(1-R)*.2;
      }
    }
  }
}
    
const int DIELECTRICMATERIAL_VERSION = 1;

void 
DielectricMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("DielectricMaterial", DIELECTRICMATERIAL_VERSION);
  Material::io(str);
  SCIRun::Pio(str, R0);
  SCIRun::Pio(str, n_in);
  SCIRun::Pio(str, n_out);
  SCIRun::Pio(str, phong_exponent);
  SCIRun::Pio(str, extinction_in);
  SCIRun::Pio(str, extinction_out);
  SCIRun::Pio(str, nothing_inside);
  SCIRun::Pio(str, extinction_scale);
  str.end_class();
  if(str.reading()){
    bg_in=Color(extinction_in.red()==1, 
		extinction_in.green()==1, 
		extinction_in.blue()==1);
    bg_out=Color(extinction_out.red()==1, 
		 extinction_out.green()==1, 
		 extinction_out.blue()==1);
  }
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::DielectricMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::DielectricMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::DielectricMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
