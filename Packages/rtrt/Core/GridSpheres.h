
#ifndef GridSpheres_H
#define GridSpheres_H 1

#include "Object.h"
#include "BBox.h"
#include "Color.h"
#include "Material.h"
#include "GridSpheresDpy.h"
#include <string>

using std::string;

namespace rtrt {

struct GridSpheresTree;
struct BoundedObject;
struct MCell;

class GridSpheres : public Object, public Material {
  float* spheres;
  int* counts;
  int* cells;
  MCell** macrocells;
  float* min;
  float* max;
  int totalsize;
  int nspheres;
  int ndata;
  int cellsize;
  int map_idx(int, int, int, int);
  int depth;
  float radius;
  float iradius;
  int nmatls;
  Material** matls;
  string *var_names;
  double specpow;
  double refl;
  double transp;
  double ior;
  BBox bbox;
  friend class GridSpheresMatl;
  friend class GridSpheresDpy;
  GridSpheresDpy* dpy;
  double icellsize;
  
  void isect(int depth, double t,
	     double dtdx, double dtdy, double dtdz,
	     double next_x, double next_y, double next_z,
	     int idx, int ix, int iy, int iz,
	     int dix_dx, int diy_dy, int diz_dz,
	     int didx_dx, int didx_dy, int didx_dz,
	     const Vector& cellcorner, const Vector& celldir,
	     const Ray& ray, HitInfo& hit,
	     DepthStats* st, PerProcessorContext* ppc);
  void intersect_print(const Ray& ray,
		       HitInfo& hit, DepthStats* st,
		       PerProcessorContext*);
  void calc_mcell(int depth, int idx, MCell& mcell);
public:
  GridSpheres(float* spheres, float* min, float* max,
	      int nspheres, int ndata, int nsides, int depth,
	      float radius, int nmatls, Material** matls,
	      string *var_names = 0);
  virtual ~GridSpheres();
  virtual void intersect(const Ray& ray,
			 HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void light_intersect(Light* light, const Ray& ray,
			       HitInfo& hit, double dist, Color& atten,
			       DepthStats* st, PerProcessorContext*);
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
