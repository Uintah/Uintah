
#ifndef TextureGridSpheres_H
#define TextureGridSpheres_H 1

#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/GridSpheres.h>

namespace rtrt {

  class UV;
  
class TextureGridSpheres : public GridSpheres {
protected:
  int *tex_indices; // length of spheres
  unsigned char *tex_data;
  size_t ntextures;
  int tex_res;
  Color color;
  
  void get_uv(UV& uv, const Point& hitpos, const HitInfo& hit);
  float interp_luminance(unsigned char *image, double u, double v);

public:
  TextureGridSpheres(float* spheres, size_t nspheres, int ndata,
		     float radius,
		     int *tex_indices,
		     unsigned char* tex_data, size_t ntextures, int tex_res,
		     int nsides, int depth, RegularColorMap* cmap = 0,
		     const Color& color = Color(1.0, 1.0, 1.0));
  virtual ~TextureGridSpheres();
  virtual void io(SCIRun::Piostream &stream);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
