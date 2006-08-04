
#include <Packages/rtrt/Core/PCAGridSpheres.h>

#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/UV.h>
#include <Packages/rtrt/Core/RegularColorMap.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <stdlib.h>

using namespace rtrt;
using namespace SCIRun;
using namespace std;

#define one_over_255 .00392156862745098039

PCAGridSpheres::PCAGridSpheres(float* spheres, size_t nspheres, int ndata,
			       float radius,
			       int *tex_indices,
			       unsigned char* tex_data, int nbases,
                               int tex_res, unsigned char* coeff,
                               unsigned char *mean, int nvecs,
			       float tex_min, float tex_max,
                               float coeff_min, float coeff_max,
			       int nsides, int depth, RegularColorMap* cmap,
			       const Color& color) :
 TextureGridSpheres(spheres, nspheres, ndata, radius, tex_indices, tex_data,
		    nbases, tex_res, nsides, depth, cmap, color),
 coeff(coeff), mean(mean), nbases(nbases), nvecs(nvecs),
 tex_min(tex_min), coeff_min(coeff_min)
{
  tex_diff = (tex_max - tex_min)*one_over_255;
  coeff_diff = (coeff_max - coeff_min)*one_over_255;
}

PCAGridSpheres::~PCAGridSpheres()
{
}

void 
PCAGridSpheres::io(SCIRun::Piostream&)
{
  ASSERTFAIL("Pio for PCAGridSpheres not implemented");
}

void PCAGridSpheres::shade(Color& result, const Ray& ray,
			      const HitInfo& hit, int /*depth*/,
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
  } else if (tex_index >= nvecs) {
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
  if (u>1)
    u=1;
  else if (u<0)
    u=0;

  double v=uv.v()*vscale;
  if (v>1)
    v=1;
  else if (v<0)
    v=0;

  float luminance = interp_luminance(u, v, tex_index);
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

// Given the pixel and texture index compute the pixel's luminance
float PCAGridSpheres::get_pixel(int x, int y, int tex_index) {
  float outdata = 0;
  unsigned char* btdata = tex_data + (y*tex_res+x);
  unsigned char* coeff_data = coeff + (tex_index*nbases);

  // Multiply basis vectors by coefficients
  for (int b=0; b<nbases; b++)
    outdata+=(coeff_data[b]*coeff_diff+coeff_min) *
      (btdata[b*tex_res*tex_res]*tex_diff+tex_min);
  
  // Add mean vector
  outdata+=mean[y*tex_res+x];
  
  return outdata;
}

float PCAGridSpheres::interp_luminance(double u, double v, int index)
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
  
  float lum = get_pixel(iu, iv, index);

  return lum*one_over_255;
#else
  u *= tex_res;
  int iu = (int)u;
  int iu_high;
  if (iu < tex_res-1) {
    iu_high = iu + 1;
  } else {
    iu = tex_res - 1;
    iu_high = 0;
  }
  double u_weight_high = u-iu;

  v *= tex_res-1;
  int iv = (int)v;
  if (iv > tex_res - 2)
    iv = tex_res - 2;
  double v_weight_high = v-iv;

  float lum00 = get_pixel(iu,      iv,   index);
  float lum01 = get_pixel(iu_high, iv,   index);
  float lum10 = get_pixel(iu,      iv+1, index);
  float lum11 = get_pixel(iu_high, iv+1, index);
  
  float lum = 
    lum00*(1-u_weight_high)*(1-v_weight_high)+
    lum01*   u_weight_high *(1-v_weight_high)+
    lum10*(1-u_weight_high)*   v_weight_high +
    lum11*   u_weight_high *   v_weight_high;

  return lum*one_over_255;
#endif
}
