#include <Packages/rtrt/Core/VolumeVis.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/MiscMath.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/BrickArray3.cc>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <Packages/rtrt/Core/Point.h>
#include <Packages/rtrt/Core/Vector.h>
#include <float.h>
#include <iostream>

using namespace std;
using namespace rtrt;

static float t_inc = 1;

VolumeVis::VolumeVis(BrickArray3<float>& _data, float data_min, float data_max,
		     int nx, int ny, int nz,
		     Point min, Point max, const Array1<Color*> &matls,
		     int nmatls, const Array1<float> &alphas, int nalphas,
		     double spec_coeff, double ambient, double diffuse,
		     double specular, float _t_inc):
  Object(this), diag(max - min),
  data_min(data_min), data_max(data_max),
  nx(nx), ny(ny), nz(nz),
  min(min), max(max), matls(matls), nmatls(nmatls),
  alphas(alphas), nalphas(nalphas), spec_coeff(spec_coeff),
  ambient(ambient), diffuse(diffuse), specular(specular)
{
  if (data_max < data_min) {
    float temp = data_max;
    data_max = data_min;
    data_min = temp;
  }
  if (data_min != data_max)
    data_diff_inv = 1/(data_max - data_min);
  else
    data_diff_inv = 0;

  data.share(_data);
  delta_x2 = 2 * (max.x() - min.x())/nx;
  delta_y2 = 2 * (max.y() - min.y())/ny;
  delta_z2 = 2 * (max.z() - min.z())/nz;

  if (diag.x() == 0)
    inv_diag.x(0);
  else
    inv_diag.x(1.0/diag.x());

  if (diag.y() == 0)
    inv_diag.y(0);
  else
    inv_diag.y(1.0/diag.y());

  if (diag.z() == 0)
    inv_diag.z(0);
  else
    inv_diag.z(1.0/diag.z());

  t_inc = _t_inc;
  // modify the alpha matrix by the t_inc
  // we are assuming the base delta_t is 1 and that the new delta_t is t_inc
  // the formula:
  //    a_1 : original opacity
  //    a_2 : resulting opacity
  //    d_1 : original sampling distance
  //    d_2 : new sampling distance
  // a_2 = 1 - (1 - a_1)^(d_2/d_1)
  for(unsigned int i = 0; i < alphas.size(); i++) {
    alphas[i] = 1 - powf(1 - alphas[i], t_inc);
    cout << "ALPHAS[i="<<i<<"] = "<<alphas[i]<<endl;
  }
}

VolumeVis::~VolumeVis() {
}

void VolumeVis::intersect(const Ray& ray, HitInfo& hit, DepthStats*,
			  PerProcessorContext*) {
  // determines the min and max t of the intersections with the boundaries
   double t1, t2, tx1, tx2, ty1, ty2, tz1, tz2;

   if (ray.direction().x() > 0) {
     tx1 = (min.x() - ray.origin().x()) / ray.direction().x();
     tx2 = (max.x() - ray.origin().x()) / ray.direction().x();
   }
   else {
     tx1 = (max.x() - ray.origin().x()) / ray.direction().x();
     tx2 = (min.x() - ray.origin().x()) / ray.direction().x();
   }
   
   if (ray.direction().y() > 0) {
     ty1 = (min.y() - ray.origin().y()) / ray.direction().y();
     ty2 = (max.y() - ray.origin().y()) / ray.direction().y();
   }
   else {
     ty1 = (max.y() - ray.origin().y()) / ray.direction().y();
     ty2 = (min.y() - ray.origin().y()) / ray.direction().y();
   }
   
   if (ray.direction().z() > 0) {
     tz1 = (min.z() - ray.origin().z()) / ray.direction().z();
     tz2 = (max.z() - ray.origin().z()) / ray.direction().z();
   }
   else {
     tz1 = (max.z() - ray.origin().z()) / ray.direction().z();
     tz2 = (min.z() - ray.origin().z()) / ray.direction().z();
   }
   
   t1 =  DBL_MIN; 
   t2 =  DBL_MAX;
   
   if (tx1 > t1) t1 = tx1;
   if (ty1 > t1) t1 = ty1;
   if (tz1 > t1) t1 = tz1;
   
   if (tx2 < t2) t2 = tx2;
   if (ty2 < t2) t2 = ty2;
   if (tz2 < t2) t2 = tz2;

   // t1 is t_min and t2 is t_max
   if (t2 > t1) {
     if (t1 > FLT_EPSILON) {
       hit.hit(this, t1);
       float* tmax=(float*)hit.scratchpad;
       *tmax = t2;
     }
     //else if (t2 > FLT_EPSILON)
     //hit.hit(this, t2);
   }
   
}

void VolumeVis::light_intersect(Light* , const Ray& lightray,
				HitInfo& hit, double , Color& ,
				DepthStats* ds, PerProcessorContext* ppc) {
  intersect(lightray, hit, ds, ppc);
}

// All incoming vectors should be normalized

// Parameters:
//   N, objnorm - the normal of the surface
//   V, viewnorm - direction from the point to the eye
//   L, lightnorm - direction from the point to the light
//   point - the location of intersection
//   object_color - the unlit color of the surface
/*
  R = 2.0*(N.L)*N - L
  
  I found this vector also needs to be normalized. Then, we feed these
  vectors into the illumination equation for the phong lighting model, which
  is
  
  I = Ia*ka*Oda + fatt*Ip[kd*Od(N.L) + ks(R.V)^n]
  
  Here, the variables are:
  
  * Ia is the ambient intensity
  * ka is the ambient co-efficient
  * Oda is the colour for the ambient
  * fatt is the atmospheric attenuation factor, ie depth shading
  * Ip is the intensity of the point light source
  * kd is the diffuse co-efficient
  * Od is the objects colour
  * ks is the specular co-efficient
  * n is the objects shinyness
  * N is the normal vector
  * L is the lighting vector
  * R is the reflection vector
  * V is the viewing vector
  */
Color VolumeVis::color(const Vector &N, const Vector &V, const Vector &L, 
		       const Color &object_color, const Color &light_color) {

  Color result; // the resulting color

  double L_N_dot = L.dot(N);

  // the dot product is negative then the objects face points
  // away from the light and should be shaded.
  if (L_N_dot > 0) {
    // do the ambient, diffuse, and specular calculations
    Vector R;
    double attinuation = 1;

    //    R = vector_norm(vector_sub(vector_mult(N, 2* L_N_dot),L));
    R = (2.0 * N.dot(L) * N - L).normal();
    double spec = attinuation * specular * pow(Max(R.dot(V),0.0), spec_coeff);

    result = light_color * (object_color *(ambient+attinuation*diffuse*L_N_dot)
			    + Color(spec, spec, spec));
  }
  else {
    // do only the ambient calculations
    result = light_color * object_color * ambient;
  }

  return result;
}
	
Vector VolumeVis::normal(const Point&, const HitInfo& hit) {
  // the normal should be placed in the scratchpad
  Vector* norm = (Vector*)hit.scratchpad;
  return *norm;
}

void VolumeVis::compute_bounds(BBox& bbox, double offset) {
    bbox.extend( min - Vector(offset, offset, offset) );
    bbox.extend( max + Vector(offset, offset, offset) );
}

void VolumeVis::print(ostream& out) {
    out << "VolumeVis: min=" << min << ", max=" << max << '\n';
}

int VolumeVis::bound(const int val, const int min, const int max) {
  return (val>min?(val<max?val:max):min);
}

#define RAY_TERMINATION_THRESHOLD 0.98

void VolumeVis::shade(Color& result, const Ray& ray,
		      const HitInfo& hit, int depth,
		      double atten, const Color& accumcolor,
		      Context* cx) {
  float t_min = hit.min_t;
  float* t_maxp = (float*)hit.scratchpad;
  float t_max = *t_maxp;

  // alpha is the accumulating opacities
  // alphas are in levels of opacity: 1 - completly opaque
  //                                  0 - completly transparent
  float alpha = 0;
  Color total(0,0,0);
  Vector p;

  //  float t_inc = cx->scene->rtrt_engine->t_inc;
  for(float t = t_min; t < t_max; t += t_inc) {
    // opaque values are 0, so terminate the ray at alpha values close to zero
    if (alpha < RAY_TERMINATION_THRESHOLD) {
      // get the point to interpolate
      p = ray.origin() + ray.direction() * t - min;

      ////////////////////////////////////////////////////////////
      // interpolate the point

      // get the indices and weights for the indicies
      float norm = p.x() * inv_diag.x();
      float step = norm * (nx - 1);
      int x_low = bound((int)step, 0, data.dim1()-2);
      int x_high = x_low+1;
      float x_weight_low = x_high - step;

      norm = p.y() * inv_diag.y();
      step = norm * (ny - 1);
      int y_low = bound((int)step, 0, data.dim2()-2);
      int y_high = y_low+1;
      float y_weight_low = y_high - step;

      norm = p.z() * inv_diag.z();
      step = norm * (nz - 1);
      int z_low = bound((int)step, 0, data.dim3()-2);
      int z_high = z_low+1;
      float z_weight_low = z_high - step;

      ////////////////////////////////////////////////////////////
      // do the interpolation

      float a,b,c,d,e,f,g,h;
      a = data(x_low,  y_low,  z_low);
      b = data(x_low,  y_low,  z_high);
      c = data(x_low,  y_high, z_low);
      d = data(x_low,  y_high, z_high);
      e = data(x_high, y_low,  z_low);
      f = data(x_high, y_low,  z_high);
      g = data(x_high, y_high, z_low);
      h = data(x_high, y_high, z_high);

      float lz1, lz2, lz3, lz4, ly1, ly2, value;
      lz1 = a * z_weight_low + b * (1 - z_weight_low);
      lz2 = c * z_weight_low + d * (1 - z_weight_low);
      lz3 = e * z_weight_low + f * (1 - z_weight_low);
      lz4 = g * z_weight_low + h * (1 - z_weight_low);

      ly1 = lz1 * y_weight_low + lz2 * (1 - y_weight_low);
      ly2 = lz3 * y_weight_low + lz4 * (1 - y_weight_low);

      value = ly1 * x_weight_low + ly2 * (1 - x_weight_low);
      
      //cout << "value = " << value << endl;
      float normalized = (value - data_min) * data_diff_inv;
      int alpha_idx = bound((int)(normalized*(nalphas -1 )), 0, nalphas - 1);
      if ((alphas[alpha_idx] * (1-alpha)) > 0.001) {
      //      if (true) {
	// the point is contributing, so compute the color

	// compute the gradient
	Vector gradient;
	float dx = ly2 - ly1;
	
	float dy, dy1, dy2;
	dy1 = lz2 - lz1;
	dy2 = lz4 - lz3;
	dy = dy1 * x_weight_low + dy2 * (1 - x_weight_low);
	
	float dz, dz1, dz2, dz3, dz4, dzly1, dzly2;
	dz1 = b - a;
	dz2 = d - c;
	dz3 = f - e;
	dz4 = h - g;
	dzly1 = dz1 * y_weight_low + dz2 * (1 - y_weight_low);
	dzly2 = dz3 * y_weight_low + dz4 * (1 - y_weight_low);
	dz = dzly1 * x_weight_low + dzly2 * (1 - x_weight_low);
	if (dx || dy || dz)
	  gradient = (Vector(dx,dy,dz)).normal();
	else
	  gradient = Vector(0,0,0);

	int idx=bound((int)(normalized*(nmatls-1)), 0, nmatls - 1);
	Light* light=cx->scene->light(0);
	Vector light_dir;
	light_dir = light->get_pos()-p;
#if 1
	Color temp = color(gradient, ray.direction(), light_dir.normal(), 
			   *(matls[idx]), light->get_color());
#else
	Color temp = *(matls[idx]);
#endif
	total += temp * (alphas[alpha_idx] * (1-alpha));
	alpha += alphas[alpha_idx] * (1-alpha);
      }
      //cout << "total = " << total << ", temp = " << temp << ", alpha = " << alpha;
      //cout << ", new alpha = " << alpha << endl;
      //cout << "result = " << result << ", atten = " << atten << ", accumcolor = " << accumcolor << endl;
    } else {
      break;
    }
  }
  if (alpha < RAY_TERMINATION_THRESHOLD) {
    Color bgcolor;
    Point origin(p.x(),p.y(),p.z());
    cx->worker->traceRay(bgcolor, Ray(origin,ray.direction()), depth+1, atten,
			 accumcolor, cx);
    total += bgcolor * (1-alpha);
  }
  result = total;
}

