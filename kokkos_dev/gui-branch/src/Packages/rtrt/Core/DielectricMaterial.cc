#include "DielectricMaterial.h"
#include "HitInfo.h"
#include "Light.h"
#include "Ray.h"
#include "Scene.h"
#include "Stats.h"
#include "Object.h"
#include "Worker.h"
#include "Context.h"
#include <math.h>
#include <iostream>

using namespace rtrt;

// when a ray travels through the medium it loses
// intensity according to dI = -CI dx where dx is distance
// This means DI/dx = -CI.  This is solved by the exponential
// I = k exp(-Cx) + k'.  Putting in boundary conditions, we know
// that I(0) = I0, and I(1) = I(0) * extinction.  The first implies
// I(x) = I0 exp(-Cx).  The second implies
// I0 * extinction = I0 exp(-C) -> -C = log(extinction)
// Color extinction_constant stores -C for each channel

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
}


DielectricMaterial::DielectricMaterial( double n_in, double n_out, double R0,
           double phong_exponent, const Color& extinction_in, const Color& extinction_out, bool nothing_inside) :
           n_in( n_in ), n_out( n_out ), R0(R0),
           phong_exponent(phong_exponent), extinction_in(extinction_in), extinction_out(extinction_out), nothing_inside(nothing_inside)
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
    double cosine = -normal.dot(ray.direction());
    bool incoming = true;

    Color filter;
    if(cosine<0){
	cosine=-cosine;
	normal=-normal;
        incoming = false;
        filter = Color ( exp( extinction_constant_in.red() ), 
                         exp( extinction_constant_in.green() ), 
                         exp( extinction_constant_in.blue() ) );
    }
    else {
        filter = Color ( exp( extinction_constant_out.red() ),
                         exp( extinction_constant_out.green() ),
                         exp( extinction_constant_out.blue() ) );

    }

    atten *= filter.max_component();
    

    // compute Phong highlights
    int nlights=cx->scene->nlights();
    for(int i=0;i<nlights;i++){
	Light* light=cx->scene->light(i);
	Vector light_dir=light->get_pos()-hitpos;
	result+=filter*light->get_color() * phong_term( ray.direction(), light_dir, normal, phong_exponent);
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
	       if(!incoming && nothing_inside){
		   cx->worker->traceRay(rcolor, rray, depth+1, atten,
					accumcolor, cx, hit.hit_obj);
	       } else {
		   cx->worker->traceRay(rcolor, rray, depth+1, atten,
					accumcolor, cx);
	       }
	       result+= filter*rcolor;
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
            if(R*atten > 0.02) {
		if(!incoming && nothing_inside){
		    cx->worker->traceRay(rcolor, rray, depth+1, R*atten,
					 accumcolor, cx, hit.hit_obj);
		} else {
		    cx->worker->traceRay(rcolor, rray, depth+1, R*atten,
					 accumcolor, cx);
		}
		cx->stats->ds[depth].nrefl++;
		result+= R*(filter*rcolor);
            }
            if((1-R)*atten > 0.02) {
		if(incoming && nothing_inside){
		    cx->worker->traceRay(tcolor, tray, depth+1, (1-R)*atten,
					 accumcolor, cx, hit.hit_obj);
		} else {
		    cx->worker->traceRay(tcolor, tray, depth+1, (1-R)*atten,
					 accumcolor, cx);
		}
		cx->stats->ds[depth].ntrans++;
		result+= (1-R)*(filter*tcolor);
            }
    }
}
    
