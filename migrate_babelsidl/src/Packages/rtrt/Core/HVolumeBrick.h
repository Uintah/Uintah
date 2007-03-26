
#ifndef HVOLUMEBRICK_H
#define HVOLUMEBRICK_H 1

#include <Core/Thread/WorkQueue.h>
#include <Packages/rtrt/Core/VolumeBase.h>
#include <Core/Geometry/Point.h>
#include <stdlib.h>

namespace rtrt {

using SCIRun::WorkQueue;

struct VMCellfloat;

class HVolumeBrick : public VolumeBase {
protected:
  Point min;
  Vector datadiag;
  Vector hierdiag;
  Vector ihierdiag;
  Vector sdiag;
  int nx,ny,nz;
  float* indata;
  float* blockdata;
  float datamin, datamax;
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
  VMCellfloat** macrocells;
  int** macrocell_xidx;
  int** macrocell_yidx;
  int** macrocell_zidx;
  WorkQueue work;
  void brickit(int);
  void parallel_calc_mcell(int);
  char* filebase;
  void calc_mcell(int depth, int ix, int iy, int iz, VMCellfloat& mcell);
  void isect(int depth, float isoval, double t,
	     double dtdx, double dtdy, double dtdz,
	     double next_x, double next_y, double next_z,
	     int ix, int iy, int iz,
	     int dix_dx, int diy_dy, int diz_dz,
	     int startx, int starty, int startz,
	     const Vector& cellcorner, const Vector& celldir,
	     const Ray& ray, HitInfo& hit,
	     DepthStats* st, PerProcessorContext* ppc);
public:
  HVolumeBrick(Material* matl, VolumeDpy* dpy,
	       char* filebase, int depth, int np);
  HVolumeBrick(Material* matl, VolumeDpy* dpy,
	       int depth, int np,
	       int _nx, int _ny, int _nz,
	       Point min, Point max,
	       float _datamin, float _datamax, float* _indata);
  virtual ~HVolumeBrick();
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void compute_bounds(BBox&, double offset);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void compute_hist(int nhist, int* hist,
			    float datamin, float datamax);
  virtual void get_minmax(float& min, float& max);
  inline int get_nx() {
    return nx;
  }
  inline int get_ny() {
    return ny;
  }
  inline int get_nz() {
    return nz;
  }
};
  
} // end namespace rtrt

#endif
