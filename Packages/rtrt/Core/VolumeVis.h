#ifndef __RTRT_VOLUMEVIS_H__
#define __RTRT_VOLUMEVIS_H__

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/VolumeVisDpy.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/VolumeVisBase.h>
#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/BBox.h>
//#include <Packages/rtrt/Core/Stats.h>
//#include <Core/Math/MiscMath.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <stdlib.h>
#include <float.h>
#include <iostream>

namespace rtrt {

#ifndef MAXUNSIGNEDSHORT
#define MAXUNSIGNEDSHORT 65535
#endif

template<class DataType>
class VolumeVis : public VolumeVisBase {
protected:
  Vector diag;
  Vector inv_diag;
  BrickArray3<DataType> data;
  DataType data_min, data_max;
  int nx, ny, nz;
  Point min, max;
  double spec_coeff, ambient, diffuse, specular;
  float delta_x2, delta_y2, delta_z2;
  
  Color color(const Vector &N, const Vector &V, const Vector &L, 
	      const Color &object_color, const Color &light_color) const;
public:
  VolumeVis(BrickArray3<DataType>& data, DataType data_min, DataType data_max,
	    int nx, int ny, int nz, Point min, Point max,
	    double spec_coeff, double ambient,
	    double diffuse, double specular, VolumeVisDpy *dpy);
  virtual ~VolumeVis() {}

  ///////////////////////////////////////
  // From SCIRun::Persistent
  virtual void io(SCIRun::Piostream &) {
    ASSERTFAIL("Pio not implemented for VolumeVis");
  }

  ///////////////////////////////////////
  // From Object
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual Vector normal(const Point&, const HitInfo&) {
    // There really isn't a normal suitable for this object
    return Vector(1,0,0);
  }
  virtual void compute_bounds(BBox& bbox, double offset)  {
    bbox.extend( min - Vector(offset, offset, offset) );
    bbox.extend( max + Vector(offset, offset, offset) );
  }
  virtual void print(ostream& out) {
    out << "VolumeVis: min=" << min << ", max=" << max << '\n';
  }

  ///////////////////////////////////////
  // From Material
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);

  ///////////////////////////////////////
  // From VolumeVisBase
  virtual void compute_hist(int nhist, int* hist,
			    float hdatamin, float hdatamax);

  virtual void get_minmax(float& min, float& max) {
    min = data_min;
    max = data_max;
  }
};


  ////////////////////////////////////////////////////////////
  // Member function implementations
  ////////////////////////////////////////////////////////////

template<class DataType>
VolumeVis<DataType>::VolumeVis(BrickArray3<DataType>& _data,
			       DataType data_min, DataType data_max,
			       int nx, int ny, int nz,
			       Point min, Point max,
			       double spec_coeff, double ambient,
			       double diffuse, double specular,
			       VolumeVisDpy *dpy):
  VolumeVisBase(dpy), diag(max - min),
  data_min(data_min), data_max(data_max),
  nx(nx), ny(ny), nz(nz),
  min(min), max(max), spec_coeff(spec_coeff),
  ambient(ambient), diffuse(diffuse), specular(specular)
{
  if (data_max < data_min) {
    DataType temp = data_max;
    data_max = data_min;
    data_min = temp;
  }

  cerr << "VolumeVis::data_min = "<<data_min<<", data_max = "<<data_max<<endl;
  
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

  dpy->attach(this);
}

template<class DataType>
void VolumeVis<DataType>::intersect(Ray& ray, HitInfo& hit, DepthStats*,
				    PerProcessorContext*) {
  // determines the min and max t of the intersections with the boundaries
  double t1, t2, tx1, tx2, ty1, ty2, tz1, tz2;
  
  Point sub_min = Max(min, min);
  Point sub_max = Max(max, max);
  
  if (ray.direction().x() > 0) {
    tx1 = (sub_min.x() - ray.origin().x()) / ray.direction().x();
    tx2 = (sub_max.x() - ray.origin().x()) / ray.direction().x();
  }
  else {
    tx1 = (sub_max.x() - ray.origin().x()) / ray.direction().x();
    tx2 = (sub_min.x() - ray.origin().x()) / ray.direction().x();
  }
   
  if (ray.direction().y() > 0) {
    ty1 = (sub_min.y() - ray.origin().y()) / ray.direction().y();
    ty2 = (sub_max.y() - ray.origin().y()) / ray.direction().y();
  }
  else {
    ty1 = (sub_max.y() - ray.origin().y()) / ray.direction().y();
    ty2 = (sub_min.y() - ray.origin().y()) / ray.direction().y();
  }
   
  if (ray.direction().z() > 0) {
    tz1 = (sub_min.z() - ray.origin().z()) / ray.direction().z();
    tz2 = (sub_max.z() - ray.origin().z()) / ray.direction().z();
  }
  else {
    tz1 = (sub_max.z() - ray.origin().z()) / ray.direction().z();
    tz2 = (sub_min.z() - ray.origin().z()) / ray.direction().z();
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
      if (hit.hit(this, t1)) {
	float* tmax=(float*)hit.scratchpad;
	*tmax = t2;
      }
    }
    //else if (t2 > FLT_EPSILON)
    //hit.hit(this, t2);
  }
   
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
template<class DataType>
Color VolumeVis<DataType>::color(const Vector &N, const Vector &V,
				 const Vector &L, const Color &object_color,
				 const Color &light_color) const {

  Color result; // the resulting color

  double L_N_dot = Dot(L, N);

#if 1 // Double Sided shading
  double attenuation = 1;
  Vector L_use;

  // the dot product is negative then the objects face points
  // away from the light and the normal should be reversed.
  if (L_N_dot >= 0) {
    L_use = L;
  } else {
    L_N_dot = -L_N_dot;
    L_use = -L;
  }

  // do the ambient, diffuse, and specular calculations
  double exponent;
#if 0 // Use Halfway vector instead of reflection vector
  //  Vector H = (L + V) * 0.5f;
  Vector H = (L_use + V) * 0.5f;
  exponent = Dot(N, H);
#else
  Vector R = N * (2.0 * L_N_dot) - L_use;
  exponent = Dot(R, V);
#endif
  double spec;
  if (exponent > 0) {
    spec = attenuation * specular * pow(exponent, spec_coeff*0.5);
  } else {
    spec = 0;
  }
  
  result = light_color * (object_color *(ambient+attenuation*diffuse*L_N_dot)
			  + Color(spec, spec, spec));
#else
  // the dot product is negative then the objects face points
  // away from the light and should only contribute an ambient term
  if (L_N_dot > 0) {
    // do the ambient, diffuse, and specular calculations
    double attenuation = 1;

    Vector R = N * (2.0 * L_N_dot) - L;
    double spec = attenuation * specular * pow(Max(Dot(R, V),0.0), spec_coeff);

    result = light_color * (object_color *(ambient+attenuation*diffuse*L_N_dot)
			    + Color(spec, spec, spec));
  }
  else {
    // do only the ambient calculations
    result = light_color * object_color * ambient;
  }
#endif
  
  return result;
}
  
#define RAY_TERMINATION_THRESHOLD 0.98

template<class DataType>
void VolumeVis<DataType>::shade(Color& result, const Ray& ray,
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
  Point current_p;

  // This is precomputed stuff for the fast rendering mode
  double x_weight_high, y_weight_high, z_weight_high;
  double tx, ty, tz;
  int x_low, y_low, z_low;
  int dxdx = 1, dydy = 1, dzdz = 1;

  float t_inc = dpy->t_inc;
  bool fast_render_mode = dpy->fast_render_mode;
  
  if (fast_render_mode) {
    // This is the start
    current_p = ray.origin() + ray.direction() * t_min;
    // Compute tx, ty, tz
    tx = ray.direction().x() * t_inc * (nx-1) * inv_diag.x();
    ty = ray.direction().y() * t_inc * (ny-1) * inv_diag.y();
    tz = ray.direction().z() * t_inc * (nz-1) * inv_diag.z();
    // Do stuff for x
    if (tx >= 0) {
      dxdx = 1;
      double norm = (current_p.x() - min.x()) * inv_diag.x();
      double step = norm * (nx - 1);
      x_low = clamp(0, (int)step, nx - 2);
      x_weight_high = step - x_low;
    } else {
      dxdx = -1;
      double norm = (max.x() - current_p.x()) * inv_diag.x();
      double step = norm * (nx - 1);
      x_low = clamp(0, (int)step, nx - 2);
      x_weight_high = step - x_low;
      x_low = nx - 1 - x_low;
      tx *= -1;
    }
    // Do stuff for y
    if (ty >= 0) {
      dydy = 1;
      double norm = (current_p.y() - min.y()) * inv_diag.y();
      double step = norm * (ny - 1);
      y_low = clamp(0, (int)step, ny - 2);
      y_weight_high = step - y_low;
    } else {
      dydy = -1;
      double norm = (max.y() - current_p.y()) * inv_diag.y();
      double step = norm * (ny - 1);
      y_low = clamp(0, (int)step, ny - 2);
      y_weight_high = step - y_low;
      y_low = ny - 1 - y_low;
      ty *= -1;
    }
    // Do stuff for z
    if (tz >= 0) {
      dzdz = 1;
      double norm = (current_p.z() - min.z()) * inv_diag.z();
      double step = norm * (nz - 1);
      z_low = clamp(0, (int)step, nz - 2);
      z_weight_high = step - z_low;
    } else {
      dzdz = -1;
      double norm = (max.z() - current_p.z()) * inv_diag.z();
      double step = norm * (nz - 1);
      z_low = clamp(0, (int)step, nz - 2);
      z_weight_high = step - z_low;
      z_low = nz - 1 - z_low;
      tz *= -1;
    }
  }
  
  for(float t = t_min; t < t_max; t += t_inc) {
    // opaque values are 0, so terminate the ray at alpha values close to zero
    if (alpha < RAY_TERMINATION_THRESHOLD) {
      int x_high, y_high, z_high;
      if (fast_render_mode) {
	// update all of our values
	x_high = x_low + dxdx;
	y_high = y_low + dydy;
	z_high = z_low + dzdz;

	if (x_low < 0 || x_low >= nx) {
	  //	  cerr << "x_low bad: "<<x_low<<endl;
	  continue;
	}
	if (x_high < 0 || x_high >= nx) {
	  //	  cerr << "x_high bad: "<<x_high<<endl;
	  continue;
	}
	if (y_low < 0 || y_low >= ny) {
	  //	  cerr << "y_low bad: "<<y_low<<endl;
	  continue;
	}
	if (y_high < 0 || y_high >= ny) {
	  //	  cerr << "y_high bad: "<<y_high<<endl;
	  continue;
	}
	if (z_low < 0 || z_low >= nz) {
	  //	  cerr << "z_low bad: "<<z_low<<endl;
	  continue;
	}
	if (z_high < 0 || z_high >= nz) {
	  //	  cerr << "z_high bad: "<<z_high<<endl;
	  continue;
	}
      } else {
	// get the point to interpolate
	current_p = ray.origin() + ray.direction() * t - min.vector();
	
	////////////////////////////////////////////////////////////
	// interpolate the point
	
	// get the indices and weights for the indicies
	double norm = current_p.x() * inv_diag.x();
	double step = norm * (nx - 1);
	x_low = clamp(0, (int)step, data.dim1()-2);
	x_high = x_low+1;
	//      float x_weight_low = x_high - step;
	x_weight_high = step - x_low;
	
	norm = current_p.y() * inv_diag.y();
	step = norm * (ny - 1);
	y_low = clamp(0, (int)step, data.dim2()-2);
	y_high = y_low+1;
	//      float y_weight_low = y_high - step;
	y_weight_high = step - y_low;
	
	norm = current_p.z() * inv_diag.z();
	step = norm * (nz - 1);
	z_low = clamp(0, (int)step, data.dim3()-2);
	z_high = z_low+1;
	//      float z_weight_low = z_high - step;
	z_weight_high = step - z_low;
      }

      ////////////////////////////////////////////////////////////
      // do the interpolation

      DataType a,b,c,d,e,f,g,h;
      a = data(x_low,  y_low,  z_low);
      b = data(x_low,  y_low,  z_high);
      c = data(x_low,  y_high, z_low);
      d = data(x_low,  y_high, z_high);
      e = data(x_high, y_low,  z_low);
      f = data(x_high, y_low,  z_high);
      g = data(x_high, y_high, z_low);
      h = data(x_high, y_high, z_high);
      
      float lz1, lz2, lz3, lz4, ly1, ly2, value;
      lz1 = a * (1 - z_weight_high) + b * z_weight_high;
      lz2 = c * (1 - z_weight_high) + d * z_weight_high;
      lz3 = e * (1 - z_weight_high) + f * z_weight_high;
      lz4 = g * (1 - z_weight_high) + h * z_weight_high;

      ly1 = lz1 * (1 - y_weight_high) + lz2 * y_weight_high;
      ly2 = lz3 * (1 - y_weight_high) + lz4 * y_weight_high;

      value = ly1 * (1 - x_weight_high) + ly2 * x_weight_high;
      
      //cout << "value = " << value << endl;

#if 0
      // One thing to note is that this bit of code indicated that there were
      // occasions when value was close to 0, but on the negative side.  This
      // is OK, because rounding schemes would basically round that number to
      // 0 rather than -1 which would be bad.
      //
      // The moral of the story is that negative numbers of very small
      // magnitude are OK, and don't need to be clamped.
      if (value < data_min || value > data_max) {
	cerr << "value is bad!! value = "<<value<<", data_min = "<<data_min<<", data_max = "<<data_max<<endl;
	flush(cerr);
      }
#endif
      float alpha_factor = dpy->lookup_alpha(value) * (1-alpha);
      if (alpha_factor > 0.001) {
	//      if (true) {
	// the point is contributing, so compute the color

	// compute the gradient
	Vector gradient;
	float dx = ly2 - ly1;
	
	float dy, dy1, dy2;
	dy1 = lz2 - lz1;
	dy2 = lz4 - lz3;
	dy = dy1 * (1 - x_weight_high) + dy2 * x_weight_high;
	
	float dz, dz1, dz2, dz3, dz4, dzly1, dzly2;
	dz1 = b - a;
	dz2 = d - c;
	dz3 = f - e;
	dz4 = h - g;
	dzly1 = dz1 * (1 - y_weight_high) + dz2 * y_weight_high;
	dzly2 = dz3 * (1 - y_weight_high) + dz4 * y_weight_high;
	dz = dzly1 * (1 - x_weight_high) + dzly2 * x_weight_high;
	if (dx || dy || dz){
	  float length2 = dx*dx+dy*dy+dz*dz;
	  // this lets the compiler use a special 1/sqrt() operation
	  float ilength2 = 1/sqrtf(length2);
	  gradient = Vector(dx*ilength2*dxdx, dy*ilength2*dydy,
			    dz*ilength2*dzdz);
	} else {
	  gradient = Vector(0,0,0);
	}

	Light* light=cx->scene->light(0);
	Vector light_dir;
	light_dir = light->get_pos()-current_p;

	Color temp = color(gradient, ray.direction(), light_dir.normal(), 
			   *(dpy->lookup_color(value)),
			   light->get_color());
	total += temp * alpha_factor;
	alpha += alpha_factor;
      }
      if (fast_render_mode) {
	x_weight_high += tx;
	if (x_weight_high > 1) {
	  int xinc = (int)x_weight_high;
	  x_weight_high = x_weight_high - xinc;
	  x_low += xinc * dxdx;
	}
	y_weight_high += ty;
	if (y_weight_high > 1) {
	  int yinc = (int)y_weight_high;
	  y_weight_high = y_weight_high - yinc;
	  y_low += yinc * dydy;
	}
	z_weight_high += tz;
	if (z_weight_high > 1) {
	  int zinc = (int)z_weight_high;
	  z_weight_high = z_weight_high - zinc;
	  z_low += zinc * dzdz;
	}
      }	
    } else {
      break;
    }
  }
  if (alpha < RAY_TERMINATION_THRESHOLD) {
    Color bgcolor;
    Ray r(current_p,ray.direction());
    cx->worker->traceRay(bgcolor, r, depth+1, atten,
			 accumcolor, cx);
    total += bgcolor * (1-alpha);
  }
  result = total;
}

template<class DataType>
void VolumeVis<DataType>::compute_hist(int nhist, int* hist,
				       float hdatamin, float hdatamax) {
  float scale=(nhist-1)/(hdatamax-hdatamin);
  int nx1=nx-1;
  int ny1=ny-1;
  int nz1=nz-1;
  int nynz=ny*nz;
  //cerr << "scale = " << scale << "\tnx1 = " << nx1 << "\tny1 = " << ny1 << "\tnz1 = " << nz1 << "\tnynz = " << nynz << endl;
  for(int ix=0;ix<nx1;ix++){
    for(int iy=0;iy<ny1;iy++){
      int idx=ix*nynz+iy*nz;
      for(int iz=0;iz<nz1;iz++){
	DataType p000=data(ix,iy,iz);
	DataType p001=data(ix,iy,iz+1);
	DataType p010=data(ix,iy+1,iz);
	DataType p011=data(ix,iy+1,iz+1);
	DataType p100=data(ix+1,iy,iz);
	DataType p101=data(ix+1,iy,iz+1);
	DataType p110=data(ix+1,iy+1,iz);
	DataType p111=data(ix+1,iy+1,iz+1);
	DataType min=Min(Min(Min(p000, p001), Min(p010, p011)), Min(Min(p100, p101), Min(p110, p111)));
	DataType max=Max(Max(Max(p000, p001), Max(p010, p011)), Max(Max(p100, p101), Max(p110, p111)));
	int nmin=(int)((min-hdatamin)*scale);
	int nmax=(int)((max-hdatamin)*scale+.999999);
	if(nmax>=nhist)
	  nmax=nhist-1;
	if(nmin<0)
	  nmin=0;
	//if ((nmin != 0) || (nmax != 0))
	//  cerr << "nmin = " << nmin << "\tnmax = " << nmax << endl;
	for(int i=nmin;i<nmax;i++){
	  hist[i]++;
	}
	idx++;
      }
    }
  }
}

} // end namespace rtrt

#endif
