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
#include <float.h>
#include <iostream>

using namespace std;
using namespace rtrt;

static bool vectors_initialized = false;
static Vector Vectors[MAXUNSIGNEDSHORT];

#define RTRT_VOLUME_VIS_USE_TRUE_NORM 1

VolumeVis::VolumeVis(BrickArray3<Voxel>& _data, float data_min, float data_max,
		     int nx, int ny, int nz,
		     Point min, Point max, Material** matls, int nmatls,
		     float *alphas, int nalphas):
  Object(this), diag(max - min),
  data_min(data_min), data_max(data_max),
  data_diff_inv(1/(data_max - data_min)),
  nx(nx), ny(ny), nz(nz),
  min(min), max(max), matls(matls), nmatls(nmatls),
  alphas(alphas), nalphas(nalphas)
{
  data.share(_data);
  delta_x2 = 2 * (max.x() - min.x())/nx;
  delta_y2 = 2 * (max.y() - min.y())/ny;
  delta_z2 = 2 * (max.z() - min.z())/nz;
  // initialize the Vector array if needed
  if (!vectors_initialized) {
    vectors_initialized = true;
    cout << "Initializing vector array\n"; flush(cout);
    for (unsigned short i = 0; i < MAXUNSIGNEDSHORT; i++)
      Vectors[i] = get_vector(i);
    cout << "Done initializing vector array\n"; flush(cout);
  }
#if 0
  // compute the vectors for each voxel and then assign it
  cout << "Computing gradient for each voxel\n"; flush(cout);
  for (int x = 0; x < nx; x++)
    for (int y = 0; y < ny; y++)
      for (int z = 0; z < nz; z++)
	data(x,y,z).gradient_index = get_index(compute_gradient(x,y,z));
  cout << "Done computing gradient for each voxel\n"; flush(cout);
#endif
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

Vector VolumeVis::normal(const Point&, const HitInfo& hit) {
#if 1
  Vector* norm = (Vector*)hit.scratchpad;
  return *norm;
#else
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
#endif
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

Vector VolumeVis::gradient(const int x, const int y, const int z) {
  return compute_gradient(x,y,z);
}

Vector VolumeVis::compute_gradient(const int x, const int y, const int z) {
#if 1
  float xf = (data(bound(x-1,0,x),y,z).val - data(bound(x+1,x,nx-1),y,z).val)/
    delta_x2;
  float yf = (data(x,bound(y-1,0,y),z).val - data(x,bound(y+1,y,ny-1),z).val)/
    delta_y2;
  float zf = (data(x,y,bound(z-1,0,z)).val - data(x,y,bound(z+1,z,nz-1)).val)/
    delta_z2;
#else
  float xf = 0;
  float yf = 0;
  float zf = 0;
#endif
  if ((xf == 0) && (yf == 0) && (zf == 0))
    return Vector(0,0,0);
  else
    return Vector(xf,yf,zf).normal();
}

unsigned short VolumeVis::get_index(const Vector &v) {
  return 0;
}

Vector VolumeVis::get_vector(const unsigned short index) {
  return Vector(1,0,0);
}

void VolumeVis::shade(Color& result, const Ray& ray,
		      const HitInfo& hit, int depth,
		      double atten, const Color& accumcolor,
		      Context* cx) {
  float t_min = hit.min_t;
  float* t_maxp = (float*)hit.scratchpad;
  float t_max = *t_maxp;

  //cout << "result = " << result << ", atten = " << atten << ", accumcolor = " << accumcolor << endl;
  float t_inc = 1;
  //cout << "t_max = " << t_max << ", t_min = " << t_min << ", t_iterations = " << (t_max - t_min)/t_inc << endl;

  float alpha = 1;
  Color total(0,0,0);
  //cout << "data.x = " << data.dim1() << ", data.y = " << data.dim2() << ", data.z = " << data.dim3() << endl;
  Vector p;
  HitInfo new_hit = hit;
  for(float t = t_min; t < t_max; t += t_inc) {
    // opaque values are 0, so terminate the ray at alpha values close to zero
    if (alpha > 0.02) {
      // get the point to interpolate
      p = ray.origin() + ray.direction() * t - min;
      //cout << "p = " << p << ", diag = " << diag << endl;
      // interpolate the point
      float norm = p.x() / diag.x();
      float step = norm * (nx - 1);
      int x_low = bound((int)step, 0, data.dim1()-1);
      //      float x_index_high = ceilf(step);
      int x_high = bound(x_low+1, 0, data.dim1()-1);
      float x_weight_low = x_high - step;
      float x_weight_high = 1 - x_weight_low; // step - index_high
      //cout << "norm = " << norm << ", step = " << step << ", x_low = " << x_low << ", x_high = " << x_high << ", x_weight_low = " << x_weight_low << ", x_weight_high = " << x_weight_high << endl;
      norm = p.y() / diag.y();
      step = norm * (ny - 1);
      int y_low = bound((int)step, 0, data.dim2()-1);
      //      float y_index_high = ceilf(step);
      int y_high = bound(y_low+1, 0, data.dim2()-1);
      float y_weight_low = y_high - step;
      float y_weight_high = 1 - y_weight_low; // step - index_high
      //cout << "norm = " << norm << ", step = " << step << ", y_low = " << y_low << ", y_high = " << y_high << ", y_weight_low = " << y_weight_low << ", y_weight_high = " << y_weight_high << endl;
      norm = p.z() / diag.z();
      step = norm * (nz - 1);
      int z_low = bound((int)step, 0, data.dim3()-1);
      //      float z_index_high = ceilf(step);
      int z_high = bound(z_low+1, 0, data.dim3()-1);
      float z_weight_low = z_high - step;
      float z_weight_high = 1 - z_weight_low; // step - index_high
      //cout << "norm = " << norm << ", step = " << step << ", z_low = " << z_low << ", z_high = " << z_high << ", z_weight_low = " << z_weight_low << ", z_weight_high = " << z_weight_high << endl;
      //cout << "data(0,0,0) = "; flush(cout); cout << data(0,0,0) << endl;
//#ifdef RTRT_VOLUME_VIS_USE_TRUE_NORM
#if 1
      float a,b,c,d,e,f,g,h;
      
      a = data(x_low,  y_low,  z_low).val;
      b = data(x_low,  y_low,  z_high).val;
      c = data(x_low,  y_high, z_low).val;
      d = data(x_low,  y_high, z_high).val;
      e = data(x_high, y_low,  z_low).val;
      f = data(x_high, y_low,  z_high).val;
      g = data(x_high, y_high, z_low).val;
      h = data(x_high, y_high, z_high).val;

      float lz1, lz2, lz3, lz4, ly1, ly2, value;

      lz1 = a * z_weight_low + b * z_weight_high;
      lz2 = c * z_weight_low + d * z_weight_high;
      lz3 = e * z_weight_low + f * z_weight_high;
      lz4 = g * z_weight_low + h * z_weight_high;

      ly1 = lz1 * y_weight_low + lz2 * y_weight_high;
      ly2 = lz3 * y_weight_low + lz4 * y_weight_high;

      value = ly1 * x_weight_low + ly2 * x_weight_high;
      
#else
#if 0
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
      float value =
	((data(x_low ,y_low, z_low).val  * z_weight_low +
	  data(x_low ,y_low, z_high).val * z_weight_high) * y_weight_low +
	 (data(x_low ,y_high,z_low).val  * z_weight_low +
	  data(x_low ,y_high,z_high).val * z_weight_high) * y_weight_high) *
	x_weight_low +
	((data(x_high,y_low, z_low).val  * z_weight_low +
	  data(x_high,y_low, z_high).val * z_weight_high) * y_weight_low +
	 (data(x_high,y_high,z_low).val  * z_weight_low +
	  data(x_high,y_high,z_high).val * z_weight_high) * y_weight_high) *
	(1 - x_weight_low);
#else
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
#endif
#endif // ifdef RTRT_VOLUME_VIS_USE_TRUE_NORM
      //cout << "value = " << value << endl;
      float normalized = (value - data_min) * data_diff_inv;
      int alpha_idx = bound((int)(normalized*(nalphas -1 )), 0, nalphas - 1);
      if ((alphas[alpha_idx] * alpha) > 0.001) {
	// the point is contributing, so compute the color

	// compute the gradient and tuck it away for the normal function to get
	Vector* p_vector = (Vector*)new_hit.scratchpad;
#ifdef RTRT_VOLUME_VIS_USE_TRUE_NORM
	float dx = ly2 - ly1;
	
	float dy, dy1, dy2;
	dy1 = lz2 - lz1;
	dy2 = lz4 - lz3;
	dy = dy1 * x_weight_low + dy2 * x_weight_high;
	
	float dz, dz1, dz2, dz3, dz4, dzly1, dzly2;
	dz1 = b - a;
	dz2 = d - c;
	dz3 = f - e;
	dz4 = h - g;
	dzly1 = dz1 * y_weight_low + dz2 * y_weight_high;
	dzly2 = dz3 * y_weight_low + dz4 * y_weight_high;
	dz = dzly1 * x_weight_low + dzly2 * x_weight_high;
	if (dx || dy || dz)
	  *p_vector = (Vector(dx,dy,dz)).normal();
	else
	  *p_vector = Vector(0,0,0);
#else
	*p_vector =
	  ((gradient(x_low ,y_low, z_low)  * z_weight_low +
	    gradient(x_low ,y_low, z_high) * z_weight_high) * y_weight_low +
	   (gradient(x_low ,y_high,z_low)  * z_weight_low +
	    gradient(x_low ,y_high,z_high) * z_weight_high) * y_weight_high) *
	  x_weight_low +
	  ((gradient(x_high,y_low, z_low)  * z_weight_low +
	    gradient(x_high,y_low, z_high) * z_weight_high) * y_weight_low +
	   (gradient(x_high,y_high,z_low)  * z_weight_low +
	    gradient(x_high,y_high,z_high) * z_weight_high) * y_weight_high) *
	  (1 - x_weight_low);
#endif
	int idx=bound((int)(normalized*(nmatls-1)), 0, nmatls - 1);
	Color temp;
	new_hit.min_t = t;
	matls[idx]->shade(temp, ray, new_hit, depth, atten, accumcolor, cx);
	if(temp.red()>1){
	  //cerr << "+";
	}
	total += temp * (alphas[alpha_idx] * alpha);
	alpha = alpha * (1 - alphas[alpha_idx]);
      }
      //cout << "total = " << total << ", temp = " << temp << ", alpha = " << alpha;
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

