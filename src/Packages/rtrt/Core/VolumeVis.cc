#include "VolumeVis.h"
#include "HitInfo.h"
#include "Ray.h"
#include "Light.h"
#include "BBox.h"
#include "Stats.h"
#include "MiscMath.h"
#include "Material.h"
#include "Color.h"
#include "Array3.cc"
#include "HVolumeBrick.h"
#include "Context.h"
#include "Worker.h"
#include <float.h>
#include <iostream>



using namespace rtrt;


VolumeVis::VolumeVis(Array3<float>& _data, float data_min, float data_max,
		     int nx, int ny, int nz,
		     Point min, Point max, Material** matls, int nmatls):
  Object(this),
  data_min(data_min), data_max(data_max), data_diff(data_max - data_min),
  nx(nx), ny(ny), nz(nz),
  min(min), max(max), matls(matls), nmatls(nmatls)
{
  data.share(_data);
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

Vector VolumeVis::normal(const Point& hitpos, const HitInfo&) {
    if (Abs(hitpos.x() - min.x()) < 0.0001)
         return Vector(-1, 0, 0 );
    else if (Abs(hitpos.x() - max.x()) < 0.0001)
         return Vector( 1, 0, 0 );
    else if (Abs(hitpos.y() - min.y()) < 0.0001)
         return Vector( 0,-1, 0 );
    else if (Abs(hitpos.y() - max.y()) < 0.0001)
         return Vector( 0, 1, 0 );
    else if (Abs(hitpos.z() - min.z()) < 0.0001)
         return Vector( 0, 0,-1 );
    else 
         return Vector( 0, 0, 1 );
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

void VolumeVis::shade(Color& result, const Ray& ray,
		      const HitInfo& hit, int depth,
		      double atten, const Color& accumcolor,
		      Context* cx) {
  double t_min = hit.min_t;
  float* t_maxp = (float*)hit.scratchpad;
  double t_max = *t_maxp;

  //cout << "result = " << result << ", atten = " << atten << ", accumcolor = " << accumcolor << endl;
  float t_inc = 0.01;
  //cout << "t_max = " << t_max << ", t_min = " << t_min << ", t_iterations = " << (t_max - t_min)/t_inc << endl;
#define NALPHAS 10
  int nalphas = NALPHAS;
  float alphas[NALPHAS] = { 0.01, 0.5, 0.1, 0.001, 0.01,
			    0.1, 0.25, 0.375, 0.5, 0.9 };
  float alpha = 1;
  Color total(0,0,0);
  Vector diff = max - min;
  //cout << "data.x = " << data.dim1() << ", data.y = " << data.dim2() << ", data.z = " << data.dim3() << endl;
  Vector p;
  for(float t = t_min; t < t_max; t += t_inc) {
    // opaque values are 1, so terminate the ray at alpha values close to one
    if (alpha > 0.02) {
      // get the point to interpolate
      p = ray.origin() + ray.direction() * t - min;
      //cout << "p = " << p << ", diff = " << diff << endl;
      // interpolate the point
      float norm = p.x() / diff.x();
      float step = norm * (nx - 1);
      int x_low = bound((int)floorf(step), 0, data.dim1()-1);
      float x_index_high = ceilf(step);
      int x_high = bound((int)x_index_high, 0, data.dim1()-1);
      float x_weight_low = x_index_high - step;
      float x_weight_high = 1 - x_weight_low; // step - index_high
      //cout << "norm = " << norm << ", step = " << step << ", x_low = " << x_low << ", x_high = " << x_high << ", x_weight_low = " << x_weight_low << ", x_weight_high = " << x_weight_high << endl;
      norm = p.y() / diff.y();
      step = norm * (ny - 1);
      int y_low = bound((int)floorf(step), 0, data.dim2()-1);
      float y_index_high = ceilf(step);
      int y_high = bound((int)y_index_high, 0, data.dim2()-1);
      float y_weight_low = y_index_high - step;
      float y_weight_high = 1 - y_weight_low; // step - index_high
      //cout << "norm = " << norm << ", step = " << step << ", y_low = " << y_low << ", y_high = " << y_high << ", y_weight_low = " << y_weight_low << ", y_weight_high = " << y_weight_high << endl;
      norm = p.z() / diff.z();
      step = norm * (nz - 1);
      int z_low = bound((int)floorf(step), 0, data.dim3()-1);
      float z_index_high = ceilf(step);
      int z_high = bound((int)z_index_high, 0, data.dim3()-1);
      float z_weight_low = z_index_high - step;
      float z_weight_high = 1 - z_weight_low; // step - index_high
      //cout << "norm = " << norm << ", step = " << step << ", z_low = " << z_low << ", z_high = " << z_high << ", z_weight_low = " << z_weight_low << ", z_weight_high = " << z_weight_high << endl;
      //cout << "data(0,0,0) = "; flush(cout); cout << data(0,0,0) << endl;
#if 0
      float x1 = data(x_low, y_low, z_low) * x_weight_low;
      float x2 = data(x_high,y_low, z_low) * x_weight_high;
      float x3 = data(x_low, y_high,z_low) * x_weight_low;
      float x4 = data(x_high,y_high,z_low) * x_weight_high;
      float x5 = data(x_low ,y_low, z_high) * x_weight_low;
      float x6 = data(x_high,y_low, z_high) * x_weight_high;
      float x7 = data(x_low ,y_high,z_high) * x_weight_low;
      float x8 = data(x_high,y_high,z_high) * x_weight_high;
      float y1 = (x1 + x2) * y_weight_low;
      float y2 = (x3 + x4) * y_weight_high;
      float y3 = (x5 + x6) * y_weight_low;
      float y4 = (x7 + x8) * y_weight_high;
      float z1 = (y1 + y2) * z_weight_low;
      float z2 = (y3 + y4) * z_weight_high;
      float value2 = z1 + z2;
      cout << "value2 = " << value2 << endl;
#endif
#if 1
      if (x_low < 0 || x_low >= data.dim1())
	cerr << "x_low = " << x_low << ", Must be [0,"<<data.dim1()<<")"<<endl;
      if (x_high < 0 || x_high >= data.dim1())
	cerr << "x_high = "<< x_high<< ", Must be [0,"<<data.dim1()<<")"<<endl;
      if (y_low < 0 || y_low >= data.dim2())
	cerr << "y_low = " << y_low << ", Must be [0,"<<data.dim2()<<")"<<endl;
      if (y_high < 0 || y_high >= data.dim2())
	cerr << "y_high = "<< y_high<< ", Must be [0,"<<data.dim2()<<")"<<endl;
      if (z_low < 0 || z_low >= data.dim3())
	cerr << "z_low = " << z_low << ", Must be [0,"<<data.dim3()<<")"<<endl;
      if (z_high < 0 || z_high >= data.dim3())
	cerr << "z_high = "<< z_high<< ", Must be [0,"<<data.dim3()<<")"<<endl;
#endif
      float value =
	((data(x_low, y_low, z_low) * x_weight_low +
	  data(x_high,y_low, z_low) * x_weight_high) * y_weight_low +
	 (data(x_low, y_high,z_low) * x_weight_low +
	  data(x_high,y_high,z_low) * x_weight_high) * y_weight_high) *
	z_weight_low +
	((data(x_low ,y_low, z_high) * x_weight_low +
	  data(x_high,y_low, z_high) * x_weight_high) * y_weight_low +
	 (data(x_low ,y_high,z_high) * x_weight_low +
	  data(x_high,y_high,z_high) * x_weight_high) * y_weight_high) *
	z_weight_high;
      //cout << "value = " << value << endl;
      float normalized = (value - data_min)/data_diff;
      int idx=bound((int)(normalized*(nmatls-1)), 0, nmatls - 1);
      Color temp;
      matls[idx]->shade(temp, ray, hit, depth, atten, accumcolor, cx);
      idx = bound((int)(normalized*(nalphas -1 )), 0, nalphas - 1);
      total += temp * (alphas[idx] * alpha);
      //cout << "total = " << total << ", temp = " << temp << ", alpha = " << alpha;
      alpha = alpha * (1 - alphas[idx]);
      //cout << ", new alpha = " << alpha << endl;
      //cout << "result = " << result << ", atten = " << atten << ", accumcolor = " << accumcolor << endl;
    } else {
      break;
    }
  }
  if (alpha > 0.02) {
    Color bgcolor;
    Point origin(p.x(),p.y(),p.z());
    cx->worker->traceRay(bgcolor, Ray(origin,ray.direction()), depth+1, atten,
			 accumcolor, cx);
    total += bgcolor * alpha;
  }  
  result = total;
}

