
#ifndef TextureGridSpheres_H
#define TextureGridSpheres_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Material.h>
#include <string>

namespace rtrt {
using std::string;

  struct MCell;
  class UV;
  
class TextureGridSpheres : public Object, public Material {
protected:
  float* spheres;
  size_t nspheres;
  float radius;
  int *tex_indices;
  unsigned char *tex_data;
  size_t ntextures;
  int tex_res;


  // Number of data items per sphere, currently defaults to 3 for x,y,z
  int ndata;
  float* min;
  float* max;
  int* counts;
  int* cells;
  MCell** macrocells;
  int totalsize;
  int cellsize;
  int map_idx(int, int, int, int);
  int depth;
  float iradius;
  BBox bbox;

  double icellsize;
  bool preprocessed; // indicates if it has been preprocessed or not
  
  void isect(int depth, double t,
	     double dtdx, double dtdy, double dtdz,
	     double next_x, double next_y, double next_z,
	     int idx, int ix, int iy, int iz,
	     int dix_dx, int diy_dy, int diz_dz,
	     int didx_dx, int didx_dy, int didx_dz,
	     const Vector& cellcorner, const Vector& celldir,
	     const Ray& ray, HitInfo& hit,
	     DepthStats* st, PerProcessorContext* ppc);
  void intersect_print(Ray& ray,
		       HitInfo& hit, DepthStats* st,
		       PerProcessorContext*);
  void calc_mcell(int depth, int idx, MCell& mcell);

  void get_uv(UV& uv, const Point& hitpos, const Point& cen);
  Color interp_color(unsigned char *image, double u, double v);
public:
  TextureGridSpheres(float* spheres, size_t nspheres,
		    float radius,
		    int *tex_indices,
		    unsigned char *tex_data, size_t ntextures, int tex_res,
		    int nsides, int depth);
  virtual ~TextureGridSpheres();
  virtual void io(SCIRun::Piostream &stream);

  virtual void intersect(Ray& ray,
			 HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void animate(double t, bool& changed);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void compute_bounds(BBox&, double offset);
  virtual void collect_prims(Array1<Object*>& prims);
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
