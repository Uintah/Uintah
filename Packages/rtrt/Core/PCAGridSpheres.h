
#ifndef PCAGridSpheres_H
#define PCAGridSpheres_H 1

#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/TextureGridSpheres.h>

namespace rtrt {

class PCAGridSpheres : public TextureGridSpheres {
  float *xform; // size is nbasis * nchannels
  float *mean; // length of nchannels
  int nbases; // represents the number of textures in tex_data;
  int nchannels;
  float tex_min, tex_diff; // used to unquantize the basis texture

  // From TextureGridSpheres
  
  // int *tex_indices; // length of nspheres*3, output [0..nchannels-1]
  // unsigned char *tex_data; // size = nbases * tex_res * tex_res;
  // size_t ntextures; // not used directly

  float getPixel(int x, int y, int channel_index);
  Color interp_color(double u, double v, int index);
public:
  PCAGridSpheres(float* spheres, size_t nspheres, int ndata,
		 float radius,
		 int *tex_indices,
		 unsigned char* tex_data, int nbases, int tex_res,
		 float *xform, float *mean, int nchannels,
		 float tex_min, float tex_max,
		 int nsides, int depth,
		 const Color& color = Color(1.0, 1.0, 1.0));

  virtual ~PCAGridSpheres();
  virtual void io(SCIRun::Piostream &stream);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
