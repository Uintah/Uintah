
#ifndef HTVOLUMEBRICK_H
#define HTVOLUMEBRICK_H 1

#include <Core/Thread/WorkQueue.h>
#include <Packages/rtrt/Core/VolumeBase.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/BrickArray3.h>
#include <stdlib.h>

namespace rtrt {

using SCIRun::WorkQueue;

struct VMCell;

class HTVolumeBrick : public VolumeBase {
protected:
  Point min;
  Vector datadiag;
  Vector idatadiag;
  Vector sdiag;
  int nx,ny,nz,npts,ntetra;
  float datamin, datamax;
  float *points;
  int *tetra;
  int *cells;
  int *lists;
  int* xidx;
  int* yidx;
  int* zidx;
  int depth;
  int* xsize;
  int* ysize;
  int* zsize;
  double* ixsize;
  double* iysize;
  double* izsize;
  VMCell** macrocells;
  BrickArray3<VMCell> minmax;
  int** macrocell_xidx;
  int** macrocell_yidx;
  int** macrocell_zidx;
  WorkQueue work;
  void parallel_calc_mcell(int);
  char* filebase;
  void calc_mcell(int depth, int ix, int iy, int iz, VMCell& mcell);
  void isect(int depth, float isoval, double t,
	     double dtdx, double dtdy, double dtdz,
	     double next_x, double next_y, double next_z,
	     int ix, int iy, int iz,
	     int dix_dx, int diy_dy, int diz_dz,
	     int startx, int starty, int startz,
	     const Vector& cellcorner, const Vector& celldir,
	     const Ray& ray, HitInfo& hit,
	     DepthStats* st, PerProcessorContext* ppc);
  // do not EVER inline the following 4 functions doing so will
  // cause hours of debugging just to find that the optimizer
  // decided to use different floating point instruction sequences
  // each time which causes minor floating point round-off error
  // which will cause the whole world to break in a violent manner!
  bool intersect_voxel_tetra(int x, int y, int z, int* nodes);
  bool tetra_edge_in_box(const Point&  min, const Point&  max,
			 const Point& orig, const Vector& dir);
  bool vertex_in_tetra(const Point&  v, const Point& p0, const Point& p1,
		       const Point& p2, const Point& p3);
  void tetra_bounds(int *nodes, int *sx, int *ex, int *sy, int *ey,
		    int *sz, int *ez);
public:
  HTVolumeBrick(Material* matl, VolumeDpy* dpy,
		char* filebase, int depth, int np, double density);
  virtual ~HTVolumeBrick();
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void compute_bounds(BBox&, double offset);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void compute_hist(int nhist, int* hist,
			    float datamin, float datamax);
  virtual void get_minmax(float& min, float& max);
};

} // end namespace rtrt

#endif
