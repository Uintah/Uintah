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

Persistent* dielectricMaterial_maker() {
  return new DielectricMaterial();
}

// initialize the static member type_id
PersistentTypeID DielectricMaterial::type_id("DielectricMaterial", "Material", 
					     dielectricMaterial_maker);

DielectricMaterial::DielectricMaterial(double n_in, double n_out, bool nothing_inside) :
    n_in( n_in ), n_out( n_out ), nothing_inside(nothing_inside)
{
    double er, eg, eb;
    R0 = (n_in-n_out)/ (n_in + n_out);
    R0 *= R0;
    phong_exponent = 100.0;
    extinction_in = Color(1,1,1);
    extinction_out = Color(1,1,1);

    er = log( extinction_in.red() );
    eg = log( extinction_in.green() );
    eb = log( extinction_in.blue() );
    extinction_constant_in = Color(er, eg, eb);

    er = log( extinction_out.red() );
    eg = log( extinction_out.green() );
    eb = log( extinction_out.blue() );
    extinction_constant_out = Color(er, eg, eb);

    bg_in=Color(1,1,1);
    bg_out=Color(1,1,1);
}


DielectricMaterial::DielectricMaterial( double n_in, double n_out, double R0,
           double phong_exponent, const Color& extinction_in, const Color& extinction_out, bool nothing_inside, double extinction_scale) :
           n_in( n_in ), n_out( n_out ), R0(R0),
           phong_exponent(phong_exponent), extinction_in(extinction_in), extinction_out(extinction_out), nothing_inside(nothing_inside), extinction_scale(extinction_scale)
{
    double er, eg, eb;
    er = log( extinction_in.red() );
    eg = log( extinction_in.green() );
    eb = log( extinction_in.blue() );
    extinction_constant_in = Color(er, eg, eb);

    er = log( extinction_out.red() );
    eg = log( extinction_out.green() );
    eb = log( extinction_out.blue() );
    extinction_constant_out = Color(er, eg, eb);

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
    result = Color(0,0,0);
    double nearest=hit.min_t;
    Object* obj=hit.hit_obj;
    Point hitpos(ray.origin()+ray.direction()*nearest);
    Vector normal(obj->normal(hitpos, hit));
    double ray_objnormal_dot(Dot(ray.direction(),normal));
    double cosine = -Dot(normal, ray.direction());
    bool incoming = true;

#if 0
    Color filter;
#endif
    if(cosine<0){
	cosine=-cosine;
	normal=-normal;
        incoming = false;
#if 0
        filter = Color ( exp( extinction_constant_in.red() ), 
                         exp( extinction_constant_in.green() ), 
                         exp( extinction_constant_in.blue() ) );
#endif
    }
    else {
#if 0
        filter = Color ( exp( extinction_constant_out.red() ),
                         exp( extinction_constant_out.green() ),
                         exp( extinction_constant_out.blue() ) );
#endif
    }

#if 0
    atten *= filter.max_component();
#endif
    

    // compute Phong highlights
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
	if (ray_objnormal_dot*Dot(normal,light_dir)>0) continue;
	result+=light->get_color() * phong_term( ray.direction(), light_dir, normal, phong_exponent);
//	result+=filter*light->get_color() * phong_term( ray.direction(), light_dir, normal, phong_exponent);
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
            Ray rray(hitpos, reflection( ray.direction(), normal ));

            if (incoming) { // cosine is associated with n_out
                 n = n_out;
                 nPrime = n_in;
            }
            else {
                 n = n_in;
                 nPrime = n_out;
            }
            double cosinePrimeSquared = 1 - ((n*n) / (nPrime*nPrime)) *
                         (1 - cosine*cosine);
            if (cosinePrimeSquared <= 0) { // total internal reflection
               Color rcolor;
	       double dist;
	       if(!incoming && nothing_inside){
		   cx->worker->traceRay(rcolor, rray, depth+1, atten,
					accumcolor, cx, hit.hit_obj, dist);
	       } else {
		   cx->worker->traceRay(rcolor, rray, depth+1, atten,
					accumcolor, cx, dist);
	       }
#if 0
	       if (dist == MAXDOUBLE) {
		 if (incoming) filter=bg_in;
		 else filter=bg_in;
	       } else {
		 double scaled_t = dist * extinction_scale;
		 if (incoming) {
		   filter = Color(exp(extinction_constant_out.red()*scaled_t),
				  exp(extinction_constant_out.green()*scaled_t),
				  exp(extinction_constant_out.blue()*scaled_t));
		 } else {
		   filter = Color(exp(extinction_constant_in.red()*scaled_t),
				  exp(extinction_constant_in.green()*scaled_t),
				  exp(extinction_constant_in.blue()*scaled_t));
		 }		 
	       }
	       result+= filter*rcolor;
#else
	       result += rcolor;
#endif
	       cx->stats->ds[depth].nrefl++;
               return;
            }
            double cosinePrime = sqrt( cosinePrimeSquared );
            Vector transmittedDirection = ( n / nPrime) * 
                  (ray.direction() + normal * cosine) - normal*cosinePrime;
            Ray tray(hitpos, transmittedDirection);
            double smallCosine = cosine < cosinePrime? cosine : cosinePrime;
            double k = 1 - smallCosine;
            k *= (k*k)*(k*k);
            double R = R0 * (1-k) + k;
            Color rcolor(0,0,0), tcolor(0,0,0);
	    double dist;
            if(R*atten > 0.02) {
		if(!incoming && nothing_inside){
		    cx->worker->traceRay(rcolor, rray, depth+1, R*atten,
					 accumcolor, cx, hit.hit_obj, dist);
		} else {
		    cx->worker->traceRay(rcolor, rray, depth+1, R*atten,
					 accumcolor, cx, dist);
		}
		double scaled_t = dist * extinction_scale;
#if 0
		if (incoming) {
		  filter = Color(exp(extinction_constant_out.red()*scaled_t),
			 exp(extinction_constant_out.green()*scaled_t),
			 exp(extinction_constant_out.blue()*scaled_t));
		} else {
		  filter = Color(exp(extinction_constant_in.red()*scaled_t),
				 exp(extinction_constant_in.green()*scaled_t),
				 exp(extinction_constant_in.blue()*scaled_t));
		}		 
		result+= R*(filter*rcolor);
#else
		result += R*rcolor;
#endif
		cx->stats->ds[depth].nrefl++;
            }
            if((1-R)*atten > 0.02) {
		if(incoming && nothing_inside){
		    cx->worker->traceRay(tcolor, tray, depth+1, (1-R)*atten,
					 accumcolor, cx, hit.hit_obj, dist);
		} else {
		    cx->worker->traceRay(tcolor, tray, depth+1, (1-R)*atten,
					 accumcolor, cx, dist);
		}
		double scaled_t = dist * extinction_scale;
#if 0
		if (incoming) {
		  filter = Color(exp(extinction_constant_out.red()*scaled_t),
			 exp(extinction_constant_out.green()*scaled_t),
			 exp(extinction_constant_out.blue()*scaled_t));
		} else {
		  filter = Color(exp(extinction_constant_in.red()*scaled_t),
				 exp(extinction_constant_in.green()*scaled_t),
				 exp(extinction_constant_in.blue()*scaled_t));
		}		 
		result+= (1-R)*(filter*tcolor);
#else
		result+= (1-R)*tcolor;
#endif
		cx->stats->ds[depth].ntrans++;
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
  SCIRun::Pio(str, extinction_constant_in);
  SCIRun::Pio(str, extinction_constant_out);
  SCIRun::Pio(str, nothing_inside);
  SCIRun::Pio(str, extinction_scale);
  str.end_class();
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
