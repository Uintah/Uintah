#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/VolumeVis2D.h>
#include <Packages/rtrt/Core/Volvis2DDpy.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Core/Math/MiscMath.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <float.h>
#include <iostream>

using namespace std;
using namespace rtrt;

// Persistent* vv_maker() {
//   return new VolumeVis2D();
// }

// // initialize the static member type_id
// PersistentTypeID VolumeVis2D::type_id("VolumeVis2D", "Object", vv_maker);

// template<class T>
VolumeVis2D::VolumeVis2D( BrickArray3<Voxel2D<float> >& _data,
			     Voxel2D<float> data_min, Voxel2D<float> data_max,
			     int nx, int ny, int nz,
			     Point min, Point max,
			     double spec_coeff, double ambient, double diffuse,
			     double specular, Volvis2DDpy *dpy ):
  Object(this), dpy(dpy), cdpy(0), cutplane_active(false),
  data_min(data_min), data_max(data_max), diag(max - min),
  nx(nx), ny(ny), nz(nz), use_cutplane_material(false),
  min(min), max(max), spec_coeff(spec_coeff),
  ambient(ambient), diffuse(diffuse), specular(specular)
{
  if (data_max.v() < data_min.v()) {
    float temp = data_max.v();
    data_max.vref() = data_min.v();
    data_min.vref() = temp;
  }
  if (data_max.g() < data_min.g()) {
    float temp = data_max.g();
    data_max.gref() = data_min.g();
    data_min.gref() = temp;
  }

  cerr << "VolumeVis2D::data_min = "<<data_min<<", data_max = "<<data_max<<"\n";
  
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

  norm_step_x = inv_diag.x() * (nx - 1 );
  norm_step_y = inv_diag.y() * (ny - 1 );
  norm_step_z = inv_diag.z() * (nz - 1 );

  dpy->attach(this);
}

// template<class T>
void
VolumeVis2D::initialize_cuttingPlane( PlaneDpy *cdpy ) {
  if(cdpy) {
    this->cdpy = cdpy;
    cutplane_normal = cdpy->n;
    cutplane_displacement = cdpy->d;
    cutplane_active = true;
    use_cutplane_material = true;
    // set_material( cdpy );
  }
}

// template<class T>
VolumeVis2D::~VolumeVis2D() {
}

// template<class T>
void
VolumeVis2D::mouseDown(int x, int y, const Ray& ray, const HitInfo& hit)
{
  cerr << "mouseDown was called\n";

  Point p = ray.origin() + ray.direction() * hit.min_t - min.vector();

  float norm_step_x = inv_diag.x() * (nx - 1 );
  float norm_step_y = inv_diag.y() * (ny - 1 );
  float norm_step_z = inv_diag.z() * (nz - 1 );

  // get the indices and weights for the indices
  float step = p.x() * norm_step_x;
  int x_low = bound((int)step, 0, data.dim1()-2);
  int x_high = x_low+1;
  float x_weight_low = x_high - step;
  
  step = p.y() * norm_step_y;
  int y_low = bound((int)step, 0, data.dim2()-2);
  int y_high = y_low+1;
  float y_weight_low = y_high - step;
  
  step = p.z() * norm_step_z;
  int z_low = bound((int)step, 0, data.dim3()-2);
  int z_high = z_low+1;
  float z_weight_low = z_high - step;
  
  ////////////////////////////////////////////////////////////
  // do the interpolation
  
  Voxel2D<float> a, b, c, d, e, f, g, h;
  a = data(x_low,  y_low,  z_low);
  b = data(x_low,  y_low,  z_high);
  c = data(x_low,  y_high, z_low);
  d = data(x_low,  y_high, z_high);
  e = data(x_high, y_low,  z_low);
  f = data(x_high, y_low,  z_high);
  g = data(x_high, y_high, z_low);
  h = data(x_high, y_high, z_high);
  
  Voxel2D<float> lz1, lz2, lz3, lz4, ly1, ly2, value;
  lz1 = a * z_weight_low + b * (1 - z_weight_low);
  lz2 = c * z_weight_low + d * (1 - z_weight_low);
  lz3 = e * z_weight_low + f * (1 - z_weight_low);
  lz4 = g * z_weight_low + h * (1 - z_weight_low);
  
  ly1 = lz1 * y_weight_low + lz2 * (1 - y_weight_low);
  ly2 = lz3 * y_weight_low + lz4 * (1 - y_weight_low);
  
  value = ly1 * x_weight_low + ly2 * (1 - x_weight_low);

  dpy->store_voxel( a );
  dpy->store_voxel( b );
  dpy->store_voxel( c );
  dpy->store_voxel( d );
  dpy->store_voxel( e );
  dpy->store_voxel( f );
  dpy->store_voxel( g );
  dpy->store_voxel( h );
  dpy->store_voxel( value );
}

void
VolumeVis2D::mouseUp( int x, int y, const Ray& ray, const HitInfo& hit )
{
  cerr << "mouseUp was called\n";
  dpy->delete_voxel_storage();
}

void
VolumeVis2D::mouseMotion( int x, int y, const Ray& ray, const HitInfo& hit )
{
  cerr << "mouseMotion was called\n";
  dpy->delete_voxel_storage();
  
  cerr << dpy->cp_voxels.size();
  Point p = ray.origin() + ray.direction() * hit.min_t - min.vector();

  float norm_step_x = inv_diag.x() * (nx - 1 );
  float norm_step_y = inv_diag.y() * (ny - 1 );
  float norm_step_z = inv_diag.z() * (nz - 1 );

  // get the indices and weights for the indices
  float step = p.x() * norm_step_x;
  int x_low = bound((int)step, 0, data.dim1()-2);
  int x_high = x_low+1;
  float x_weight_low = x_high - step;
  
  step = p.y() * norm_step_y;
  int y_low = bound((int)step, 0, data.dim2()-2);
  int y_high = y_low+1;
  float y_weight_low = y_high - step;
  
  step = p.z() * norm_step_z;
  int z_low = bound((int)step, 0, data.dim3()-2);
  int z_high = z_low+1;
  float z_weight_low = z_high - step;
  
  ////////////////////////////////////////////////////////////
  // do the interpolation
  
  Voxel2D<float> a, b, c, d, e, f, g, h;
  a = data(x_low,  y_low,  z_low);
  b = data(x_low,  y_low,  z_high);
  c = data(x_low,  y_high, z_low);
  d = data(x_low,  y_high, z_high);
  e = data(x_high, y_low,  z_low);
  f = data(x_high, y_low,  z_high);
  g = data(x_high, y_high, z_low);
  h = data(x_high, y_high, z_high);
  
  Voxel2D<float> lz1, lz2, lz3, lz4, ly1, ly2, value;
  lz1 = a * z_weight_low + b * (1 - z_weight_low);
  lz2 = c * z_weight_low + d * (1 - z_weight_low);
  lz3 = e * z_weight_low + f * (1 - z_weight_low);
  lz4 = g * z_weight_low + h * (1 - z_weight_low);
  
  ly1 = lz1 * y_weight_low + lz2 * (1 - y_weight_low);
  ly2 = lz3 * y_weight_low + lz4 * (1 - y_weight_low);
  
  value = ly1 * x_weight_low + ly2 * (1 - x_weight_low);

  dpy->store_voxel( a );
  dpy->store_voxel( b );
  dpy->store_voxel( c );
  dpy->store_voxel( d );
  dpy->store_voxel( e );
  dpy->store_voxel( f );
  dpy->store_voxel( g );
  dpy->store_voxel( h );
  dpy->store_voxel( value );
}


// template<class T>
void VolumeVis2D::intersect(Ray& ray, HitInfo& hit, DepthStats* ,
			       PerProcessorContext* ) {
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
   
  //     t1 =  DBL_MIN; 
  //     t2 =  DBL_MAX;
   
  //     if (tx1 > t1) t1 = tx1;
  t1 = tx1;
  if (ty1 > t1) t1 = ty1;
  if (tz1 > t1) t1 = tz1;
   
  //     if (tx2 < t2) t2 = tx2;
  t2 = tx2;
  if (ty2 < t2) t2 = ty2;
  if (tz2 < t2) t2 = tz2;
  
  if (t2 > t1) {
    if (cutplane_active) {
      VolumeVis2D_scratchpad vs;
      vs.coe = Neither;
      // Compute cutting plane intersection (t_cp)
      double dt = Dot( ray.direction(), cutplane_normal );
      double plane = Dot(cutplane_normal, ray.origin())-cutplane_displacement;
      double t_cp = -plane/dt;
      if( plane > 0 || t_cp < t2 ) {
	if( dt > 1.e-6 || dt < -1.e-6 ) {
	  // and determine what to do with t_cp
	  if( t_cp >= t1 && t_cp <= t2 ) {
	    // Determine which direction the normal is facing
	    if( dt < 0 ) {
	      vs.coe = OverwroteTMax;
	      t2 = t_cp;
	    } else if( dt > 0 ) {
	      vs.coe = OverwroteTMin;
	      t1 = t_cp;
	    }
	  } else if( t_cp < t1 ) {
	    if( dt < 0 ) {
	      return;
	    }
	  } else if( t_cp > t2 ) {
	    if( dt > 0 ) {
	      return;
	    }
	  }
	}
	// Check to see which side of the plane the origin of the ray is
      } else if( plane <= 0 ) {
	// on the back side of the plane, so there is no intersection
	return;
      }
      // We need to just compute the intersection with the object
      if( t1 > FLT_EPSILON ) {
	if( hit.hit( this, t1 ) ) {
	  vs.tmax = t2;
	  VolumeVis2D_scratchpad* vsp=(VolumeVis2D_scratchpad*)hit.scratchpad;
	  *vsp = vs;
	}
      } // if active cutting plane
      
      
    } else 
      // if cutting plane is not being used in computation
      if( t1 > FLT_EPSILON ) {
	if( hit.hit( this, t1 ) ) {
	  float* tmax = (float*)hit.scratchpad;
	  *tmax = t2;
	}
      }
  }
} // intersect()



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
// template<class T>
Color VolumeVis2D::color(const Vector &N, const Vector &V, const Vector &L, 
			 const Color &object_color, const Color &light_color) {

  Color result; // the resulting color

  double L_N_dot = Dot(L, N);

#if 1 // Double Sided shading
  //  double attenuation = 1;
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
  // double attenuation = 1;
  if (exponent > 0) {
    //    spec = attenuation * specular * pow(exponent, spec_coeff*0.5);
    spec = specular*pow(exponent,spec_coeff*0.5);
  } else {
    spec = 0;
  }
  
  result = light_color * (object_color *(ambient+diffuse*L_N_dot)
			  + Color(spec, spec, spec));
#else
  // the dot product is negative then the objects face points
  // away from the light and should only contribute an ambient term
  if (L_N_dot > 0) {
    // do the ambient, diffuse, and specular calculations
    //    double attenuation = 1;

    Vector R = N * (2.0 * L_N_dot) - L;
    //    double spec = attenuation * specular * pow(Max(Dot(R, V),0.0), spec_coeff);
    double spec = specular * pow( Max( Dot( R, V ), 0.0 ), spec_coeff );
    //    result = light_color * (object_color *(ambient+attenuation*diffuse*L_N_dot)
    //		    + Color(spec, spec, spec));
    result = light_color * (object_color * (ambient + diffuse*L_N_dot )
			    + Color( spec, spec, spec ) );
  }
  else {
    // do only the ambient calculations
    result = light_color * object_color * ambient;
  }
#endif
  
  return result;
}
	
// template<class T>
Vector VolumeVis2D::normal(const Point&, const HitInfo& hit) {
  // the normal should be placed in the scratchpad
  Vector* norm = (Vector*)hit.scratchpad;
  return *norm;
}

// template<class T>
void VolumeVis2D::compute_bounds(BBox& bbox, double offset) {
  bbox.extend( min - Vector(offset, offset, offset) );
  bbox.extend( max + Vector(offset, offset, offset) );
}

// template<class T>
void VolumeVis2D::print(ostream& out) {
  out << "VolumeVis2D: min=" << min << ", max=" << max << '\n';
}

#define RAY_TERMINATION_THRESHOLD 0.98

// template<class T>
void VolumeVis2D::shade(Color& result, const Ray& ray,
			const HitInfo& hit, int depth,
			double atten, const Color& accumcolor,
			Context* cx) {

  //  cerr <<__LINE__<<"Number of lights in the scene is "<<cx->scene->nlights()<<"\n";

  // deal with whether or not a cutting plane is being used
  VolumeVis2D_scratchpad* vsp;
  float t_min = hit.min_t;
  float* t_maxp;
  float t_max;
  if(cutplane_active ) {
    vsp = (VolumeVis2D_scratchpad*)hit.scratchpad;
    t_max = vsp->tmax;
  } else {
    t_maxp = (float*)hit.scratchpad;
    t_max = *t_maxp;
  }

  // opacity is the accumulating opacities of each voxel sample
  // values of opacities: 1 - completly opaque
  //                      0 - completly transparent
  float opacity = 0;
  Color total(0,0,0);

  Vector p_inc = dpy->t_inc*ray.direction();
  Point p = ray.origin() + ray.direction() * t_min - min.vector();

  float norm_step_x = inv_diag.x() * (nx - 1 );
  float norm_step_y = inv_diag.y() * (ny - 1 );
  float norm_step_z = inv_diag.z() * (nz - 1 );


  //  cerr <<__LINE__<<"("<<cx->scene->nlights()<<")Number of lights in the scene\n"; cerr.flush();

  float opacity_factor;
  Color value_color;
  // If the cutting plane is in front add the cutting plane's color
  Color plane_color(1,0,1);
  if(cutplane_active ) {
    if (vsp->coe == OverwroteTMin) {
      Voxel2D<float> value;
      lookup_value( p, value, false );
      float sample_opacity;
      Color sample_color;
      dpy->voxel_lookup(value, sample_color, sample_opacity);
      opacity = dpy->cp_opacity;
      float gray_scale = (value.v() -data_min.v())/(data_max.v()-data_min.v());
      Color composite_plane_color = plane_color * (1-dpy->cp_gs) +
	Color(gray_scale,gray_scale,gray_scale) * dpy->cp_gs;
      total = (composite_plane_color * (1-sample_opacity) +
	       sample_color * sample_opacity) * opacity;
    }
  } // if cutplane is active
  p -= p_inc;

  for(float t = t_min; t < t_max; t += dpy->t_inc) {
    // opaque values are 1, so terminate the ray at opacity values close to one
    if( opacity >= RAY_TERMINATION_THRESHOLD )
      break;
    // get the point to interpolate
    p += p_inc;

    ////////////////////////////////////////////////////////////
    // interpolate the point
    
    // get the indices and weights for the indices
    float step = p.x() * norm_step_x;
    int x_low = bound((int)step, 0, data.dim1()-2);
    int x_high = x_low+1;
    float x_weight_low = x_high - step;
    
    step = p.y() * norm_step_y;
    int y_low = bound((int)step, 0, data.dim2()-2);
    int y_high = y_low+1;
    float y_weight_low = y_high - step;
    
    step = p.z() * norm_step_z;
    int z_low = bound((int)step, 0, data.dim3()-2);
    int z_high = z_low+1;
    float z_weight_low = z_high - step;
    
    ////////////////////////////////////////////////////////////
    // do the interpolation
    
    Voxel2D<float> a, b, c, d, e, f, g, h;
    a = data(x_low,  y_low,  z_low);
    d = data(x_low,  y_high, z_high);
    f = data(x_high, y_low,  z_high);
    g = data(x_high, y_high, z_low);
    // user-selectable acceleration method
    // (works best when widget areas are larger)
    if( dpy->render_mode && dpy->skip_opacity( a, d, f, g ) )
      continue;
    b = data(x_low,  y_low,  z_high);
    c = data(x_low,  y_high, z_low);
    e = data(x_high, y_low,  z_low);
    h = data(x_high, y_high, z_high);
    
    Voxel2D<float> lz1, lz2, lz3, lz4, ly1, ly2, value;
    lz1 = a * z_weight_low + b * (1 - z_weight_low);
    lz2 = c * z_weight_low + d * (1 - z_weight_low);
    lz3 = e * z_weight_low + f * (1 - z_weight_low);
    lz4 = g * z_weight_low + h * (1 - z_weight_low);
    
    ly1 = lz1 * y_weight_low + lz2 * (1 - y_weight_low);
    ly2 = lz3 * y_weight_low + lz4 * (1 - y_weight_low);
      
    value = ly1 * x_weight_low + ly2 * (1 - x_weight_low);
    if( value.v() < dpy->current_vmin || value.v() > dpy->current_vmax ||
	value.g() < dpy->current_gmin || value.g() > dpy->current_gmax )
      continue;

    //cout << "value = " << value << endl;
#if 0
    // One thing to note is that this bit of code indicated that there were
    // occasions when value was close to 0, but on the negative side.  This
    // is OK, because rounding schemes would basically round that number to
    // 0 rather than -1 which would be bad.
    //
    // The moral of the story is that negative numbers of very small
    // magnitude are OK, and don't need to be clamped.
    if (value.v() < data_min.v() || value.v() > data_max.v()) {
      cerr << "value.v is bad!! value.v = "<<value.v()<<", data_min.v = "<<data_min.v()<<", data_max.v = "<<data_max.v()<<endl;
      flush(cerr);
    }
    if (value.g() < data_min.g() || value.g() > data_max.g()) {
      cerr << "value.g is bad!! value.g = "<<value.g()<<", data_min.g = "<<data_min.g()<<", data_max.g = "<<data_max.g()<<endl;
      flush(cerr);
    }
#endif
    dpy->voxel_lookup(value, value_color, opacity_factor);
    opacity_factor *= 1-opacity;
    if (opacity_factor > 0.001) {
      //if (true) {
      // the point is contributing, so compute the color
      
      // compute the gradient This should probably take into
      // consideration the other value of the voxel, but I'm not
      // sure how to compute that just yet.
      float dx = ly2.v() - ly1.v();
      
      float dy1, dy2, dy;
      dy1 = lz2.v() - lz1.v();
      dy2 = lz4.v() - lz3.v();
      dy = dy1 * x_weight_low + dy2 * (1 - x_weight_low);
      
      float dz1, dz2, dz3, dz4, dzly1, dzly2, dz;
      dz1 = b.v() - a.v();
      dz2 = d.v() - c.v();
      dz3 = f.v() - e.v();
      dz4 = h.v() - g.v();
      dzly1 = dz1 * y_weight_low + dz2 * (1 - y_weight_low);
      dzly2 = dz3 * y_weight_low + dz4 * (1 - y_weight_low);
      dz = dzly1 * x_weight_low + dzly2 * (1 - x_weight_low);
      
      Vector gradient;
      if (dx || dy || dz){
	float length2 = dx*dx+dy*dy+dz*dz;
	// this lets the compiler use a special 1/sqrt() operation
	float ilength2 = 1/sqrtf(length2);
	gradient = Vector(dx*ilength2, dy*ilength2, dz*ilength2);
      } else {
	gradient = Vector(0,0,0);
      }
      
      Light* light=cx->scene->light(0);
      Vector light_dir;
      light_dir = light->get_pos()-p;
      
      Color temp = color(gradient, ray.direction(), light_dir.normal(), 
			 value_color,light->get_color());
      total += temp * opacity_factor;
      opacity += opacity_factor;
    }
  }

  if (opacity < RAY_TERMINATION_THRESHOLD) {
    if (vsp->coe == OverwroteTMax ) {
      Voxel2D<float> value;
      lookup_value( p, value, false );
      float sample_opacity;
      Color sample_color;
      dpy->voxel_lookup(value, sample_color, sample_opacity);
      float gray_scale = (value.v() -data_min.v())/(data_max.v()-data_min.v());
      Color composite_plane_color = plane_color * (1-dpy->cp_gs) +
  	Color(gray_scale,gray_scale,gray_scale) * dpy->cp_gs;
      total += (composite_plane_color * (1-sample_opacity) +
		sample_color * sample_opacity) * (Max(0.0f,dpy->cp_opacity -
						      opacity));
      opacity += dpy->cp_opacity * (1 - opacity);
    }
  }
  
  if (opacity < RAY_TERMINATION_THRESHOLD) {
    Color bgcolor;
    Point origin(p.x(),p.y(),p.z());
    Ray r(origin,ray.direction());
    cx->worker->traceRay(bgcolor, r, depth+1, atten,
			 accumcolor, cx);
    total += bgcolor * (1-opacity);
  }
  result = total;
} // shade()

void VolumeVis2D::compute_grad( Ray ray, Point p, Vector gradient,
				float &opacity, Color value_color,
				Color &total, Context* cx )
{
  Light* light=cx->scene->light(0);
  Vector light_dir;
  light_dir = light->get_pos()-p;
  
  Color temp = color(gradient, ray.direction(), light_dir.normal(), 
		     value_color,light->get_color());
  total += temp * (1 - opacity);
  opacity += opacity;//_factor;
}

// template<class T>
void VolumeVis2D::animate(double, bool& changed)
{
  if( cdpy ) {
    if( cdpy->n != cutplane_normal ||
	cdpy->d != cutplane_displacement ||
	cdpy->active != cutplane_active ||
	cdpy->use_material != use_cutplane_material ) {
      changed = true;
      cutplane_normal = cdpy->n;
      cutplane_displacement = cdpy->d;
      cutplane_active = cdpy->active;
      use_cutplane_material = cdpy->use_material;
    }
  }
  dpy->animate(changed);
}

void VolumeVis2D::point2indexspace(Point &p,
		      int& x_low, int& x_high, float& x_weight_low,
		      int& y_low, int& y_high, float& y_weight_low,
		      int& z_low, int& z_high, float& z_weight_low) {
  float step = p.x() * norm_step_x;
  x_low = bound((int)step, 0, data.dim1()-2);
  x_high = x_low+1;
  x_weight_low = x_high - step;
  
  step = p.y() * norm_step_y;
  y_low = bound((int)step, 0, data.dim2()-2);
  y_high = y_low+1;
  y_weight_low = y_high - step;
  
  step = p.z() * norm_step_z;
  z_low = bound((int)step, 0, data.dim3()-2);
  z_high = z_low+1;
  z_weight_low = z_high - step;
}

// Should return true if the value was completely computed
bool VolumeVis2D::lookup_value(Voxel2D<float>& return_value,
			       bool exit_early,
		  int x_low, int x_high, float x_weight_low,
		  int y_low, int y_high, float y_weight_low,
		  int z_low, int z_high, float z_weight_low)
{
  Voxel2D<float> a, b, c, d, e, f, g, h;
  a = data(x_low,  y_low,  z_low);
  d = data(x_low,  y_high, z_high);
  f = data(x_high, y_low,  z_high);
  g = data(x_high, y_high, z_low);
  // user-selectable acceleration method
  // (works best when widget areas are larger)
  if( exit_early && dpy->skip_opacity( a, d, f, g ) )
    return false;
  b = data(x_low,  y_low,  z_high);
  c = data(x_low,  y_high, z_low);
  e = data(x_high, y_low,  z_low);
  h = data(x_high, y_high, z_high);
  Voxel2D<float> lz1, lz2, lz3, lz4, ly1, ly2, value;
  lz1 = a * z_weight_low + b * (1 - z_weight_low);
  lz2 = c * z_weight_low + d * (1 - z_weight_low);
  lz3 = e * z_weight_low + f * (1 - z_weight_low);
  lz4 = g * z_weight_low + h * (1 - z_weight_low);
  
  ly1 = lz1 * y_weight_low + lz2 * (1 - y_weight_low);
  ly2 = lz3 * y_weight_low + lz4 * (1 - y_weight_low);
  
  return_value = ly1 * x_weight_low + ly2 * (1 - x_weight_low);

  return true;
}

// Returns true if the value was completely computed
bool VolumeVis2D::lookup_value(Point &p,
			       Voxel2D<float>& return_value,
			       bool exit_early) {
  int x_low, x_high;
  float x_weight_low;
  int y_low, y_high;
  float y_weight_low;
  int z_low, z_high;
  float z_weight_low;
  point2indexspace(p,
		   x_low, x_high, x_weight_low,
		   y_low, y_high, y_weight_low,
		   z_low, z_high, z_weight_low);
  return lookup_value(return_value, exit_early,
		      x_low, x_high, x_weight_low,
		      y_low, y_high, y_weight_low,
		      z_low, z_high, z_weight_low);
}

//const int VVIS2D_VERSION = 1;

// template<class T>
void VolumeVis2D::cblookup( Object* /*obj*/ )
{

}

// template<class T>
void 
VolumeVis2D::io(SCIRun::Piostream &str)
{
  ASSERTFAIL("Pio not implemented for VolumeVis2D");
}

// namespace SCIRun {
// void SCIRun::Pio(SCIRun::Piostream& stream, rtrt::VolumeVis2D*& obj)
// {
//   SCIRun::Persistent* pobj=obj;
//   stream.io(pobj, rtrt::VolumeVis2D::type_id);
//   if(stream.reading()) {
//     obj=dynamic_cast<rtrt::VolumeVis2D*>(pobj);
//     ASSERT(obj != 0)
//   }
// }
// } // end namespace SCIRun
