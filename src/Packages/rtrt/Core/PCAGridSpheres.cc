
#include <Packages/rtrt/Core/PCAGridSpheres.h>

#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/UV.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <stdlib.h>

using namespace rtrt;
using namespace SCIRun;
using namespace std;

#define one_over_255 .00392156862745098039

PCAGridSpheres::PCAGridSpheres(float* spheres, size_t nspheres,
			       float radius,
			       int *tex_indices,
			       unsigned char *tex_data, int nbases,
			       int tex_res,
			       float *xform, float *mean, int nchannels,
			       float tex_min, float tex_max,
			       int nsides, int depth):
 TextureGridSpheres(spheres, nspheres, radius, tex_indices, tex_data,
		    nbases, tex_res, nsides, depth),
 xform(xform), mean(mean), nbases(nbases), nchannels(nchannels),
 tex_min(tex_min)
{
  tex_diff = (tex_max - tex_min)/255.0f;
  cout << "PCAGridSpheres::tex_diff = "<<tex_diff<<"\n";
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
			      Context* /*cx*/)
{
  // cell is the index of the sphere which was intersected.  To get to
  // the actuall data you need to simply just add cell to spheres.  To
  // get the number of the sphere which was intersected you need to
  // divide by the number of data items.
  int cell=*(int*)hit.scratchpad;
  // Because our sphere index will get multiplied by 3 (RGB) to
  // actually index into the texture data, it seems silly to divide by
  // 3 (ndata) and then multiply by 3.  I will add an assert to make
  // sure that this condition hold though.
  ASSERT(ndata == 3);
  int sphere_index = cell;

  // Get the texture index
  int red_index, green_index, blue_index;
  if (tex_indices) {
    red_index = *(tex_indices + sphere_index);
    green_index = *(tex_indices + sphere_index+1);
    blue_index = *(tex_indices + sphere_index+2);
  } else {
    red_index = sphere_index;
    green_index = sphere_index + 1;
    blue_index = sphere_index + 2;
  }

  if (red_index >= nchannels || green_index >= nchannels ||
      blue_index >= nchannels)
    {
      // bad index
      result = Color(1,0,1);
      return;
    }

  // Get the hitpos
  Point hitpos(ray.origin()+ray.direction()*hit.min_t);

  // Get the center
  float* p=spheres+cell;
  Point cen(p[0], p[1], p[2]);
  
  // Get the UV coordinates
  UV uv;
  get_uv(uv, hitpos, cen);

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

  Color surface_color = interp_color(u, v, red_index, green_index, blue_index);

  result = surface_color;
}

// Given the pixel and texture index compute the pixel's color
float PCAGridSpheres::getPixel(int x, int y, int channel_index) {
  float outdata = 0;
  unsigned char *btdata = tex_data + (y*tex_res+x)*nbases;
  float *tdata = xform + (channel_index*nbases);

  // Compute the dot produce between the column vector of the tranform
  // and the pixel of the basis texture.
  for(int base = 0; base < nbases; base++)
    outdata += tdata[base] * (btdata[base]*tex_diff+tex_min);
    
  // Add the mean
  outdata += mean[channel_index];

  return outdata;
}

Color PCAGridSpheres::interp_color(double u, double v, int red_index,
				   int green_index, int blue_index)
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
  
  Color c(getPixel(iu, iv, red_index),
	  getPixel(iu, iv, green_index),
	  getPixel(iu, iv, blue_index));

  return c*one_over_255;
#else
  
  // u & v *= dimensions minus the slop(2) and the zero base difference (1)
  // for a total of 3
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

  Color c00(getPixel(iu,      iv,   red_index),
	    getPixel(iu,      iv,   green_index),
	    getPixel(iu,      iv,   blue_index));
  Color c01(getPixel(iu_high, iv,   red_index),
	    getPixel(iu_high, iv,   green_index),
	    getPixel(iu_high, iv,   blue_index));
  Color c10(getPixel(iu,      iv+1, red_index),
	    getPixel(iu,      iv+1, green_index),
	    getPixel(iu,      iv+1, blue_index));
  Color c11(getPixel(iu_high, iv+1, red_index),
	    getPixel(iu_high, iv+1, green_index),
	    getPixel(iu_high, iv+1, blue_index));

  Color c =
    c00*(1-u_weight_high)*(1-v_weight_high)+
    c01*   u_weight_high *(1-v_weight_high)+
    c10*(1-u_weight_high)*   v_weight_high +
    c11*   u_weight_high *   v_weight_high;

  return c*one_over_255;
#endif
}
