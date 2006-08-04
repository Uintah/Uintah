
#include <Packages/rtrt/Core/TextureGridSpheres.h>

#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/UV.h>
#include <Packages/rtrt/Core/RegularColorMap.h>

#include <Core/Thread/Time.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <stdlib.h>

using namespace rtrt;
using namespace SCIRun;
using namespace std;


#define one_over_255 .00392156862745098039
//#define USE_MINMAX_FOR_RENDERING 1

TextureGridSpheres::TextureGridSpheres(float* spheres, size_t nspheres, int ndata,
				       float radius,
				       int *tex_indices,
				       unsigned char* tex_data, size_t ntextures,
				       int tex_res,
				       int nsides, int depth,
                                       RegularColorMap* cmap,
				       const Color& color)
  : GridSpheres(spheres, 0, 0, nspheres, ndata, nsides,  depth,
		radius, cmap),
    tex_indices(tex_indices), tex_data(tex_data), ntextures(ntextures),
    tex_res(tex_res), color(color)
{
}

TextureGridSpheres::~TextureGridSpheres()
{
}

void 
TextureGridSpheres::io(SCIRun::Piostream&)
{
  ASSERTFAIL("Pio for TextureGridSpheres not implemented");
}

void TextureGridSpheres::shade(Color& result, const Ray& ray,
			      const HitInfo& hit, int depth,
			      double /*atten*/, const Color& /*accumcolor*/,
			      Context* cx)
{
  if (dpy->shade_method == 0) {
    // do diffuse shading
    lambertianshade(result, surface_color(hit), ray, hit, depth, cx);
    return;
  } else if (dpy->shade_method == 1 ) {
    lambertianshade(result, color, ray, hit, depth, cx);
    return;
  }

  // cell is the index of the sphere which was intersected.  To get to
  // the actual data you need to simply just add cell to spheres.  To
  // get the number of the sphere which was intersected you need to
  // divide by the number of data items.
  int cell=*(int*)hit.scratchpad;
  int sphere_index = cell / ndata;

  // Get the texture index
  int tex_index;
  if (tex_indices)
    tex_index = *(tex_indices + sphere_index);
  else
    tex_index = sphere_index;

  if (tex_index < 0) {
    // Solid black texture
    result=Color(0, 0, 0);
    return;
  } else if (tex_index >= ntextures) {
    // Bad index
    result = Color(1,0,1);
    return;
  }

  // Get the hitpos
  Point hitpos(ray.origin()+ray.direction()*hit.min_t);

  // Get the UV coordinates
  UV uv;
  get_uv(uv, hitpos, hit);

  // Do the uv lookup stuff.  Here we are only clamping
  double u=uv.u()*uscale;
  if(u>1)
    u=1;
  else if(u<0)
    u=0;

  double v=uv.v()*vscale;
  if(v>1)
    v=1;
  else if(v<0)
    v=0;

  // Get the pointer into the texture
  unsigned char *texture = tex_data + (tex_index * tex_res * tex_res);

  float luminance = interp_luminance(texture, u, v);
  if (cmap && dpy->shade_method == 2) {
    result = surface_color(hit) * luminance;
  } else if (dpy->shade_method == 3) {
    result = color * luminance;
  } else if (dpy->shade_method == 4) {
    lambertianshade(result, surface_color(hit),
                    Color(luminance, luminance, luminance),
                    ray, hit, depth, cx);
  } else if (dpy->shade_method == 5) {
    lambertianshade(result, color,
                    Color(luminance, luminance, luminance),
                    ray, hit, depth, cx);
  }
}

void TextureGridSpheres::get_uv(UV& uv, const Point& hitpos,
                                const HitInfo& hit)
{
  // Get point on unit sphere
  Vector point_on_unit_sphere(normal(hitpos, hit));
  double uu,vv,theta,phi;  
  theta = acos(-point_on_unit_sphere.y());
  phi = atan2(point_on_unit_sphere.z(), point_on_unit_sphere.x());
  if (phi < 0)
    phi += 2*M_PI;
  uu = phi * 0.5 * M_1_PI;
  vv = (M_PI - theta) * M_1_PI;
  uv.set( uu,vv);
}

float TextureGridSpheres::interp_luminance(unsigned char *image,
                                           double u, double v)
{
#if 0
  u *= tex_res;
  int iu = (int)u;
  if (iu == tex_res)
    iu = tex_res - 1;
  
  v *= tex_res;
  int iv = (int)v;
  if (iv == tex_res)
    iv = tex_res - 1;

  float lum=*(image + (iv * tex_res + iu));

  return lum*one_over_255;
#else
#if 1
  u *= tex_res;
  int iu = (int)u;
  int iu_high;
  if (iu < tex_res-1) {
    iu_high = iu + 1;
  } else {
    iu = tex_res - 1;
    iu_high = 0;
  }
#else
  u *= tex_res-1;
  int iu = (int)u;
  if (iu > tex_res - 2)
    iu = tex_res - 2;
  int iu_high = iu+1;
#endif
  double u_weight_high = u-iu;

  v *= tex_res-1;
  int iv = (int)v;
  if (iv > tex_res - 2)
    iv = tex_res - 2;
  double v_weight_high = v-iv;

  float lum00 = *(image + (iv * tex_res + iu));
  float lum01 = *(image + (iv * tex_res + iu_high));
  float lum10 = *(image + ((iv+1) * tex_res + iu));
  float lum11 = *(image + ((iv+1) * tex_res + iu_high));
  
  float lum = 
    lum00*(1-u_weight_high)*(1-v_weight_high)+
    lum01*   u_weight_high *(1-v_weight_high)+
    lum10*(1-u_weight_high)*   v_weight_high +
    lum11*   u_weight_high *   v_weight_high;

  return lum*one_over_255;
#endif
}
