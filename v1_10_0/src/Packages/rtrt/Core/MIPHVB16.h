
#ifndef MIPHVB16_H
#define MIPHVB16_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Material.h>
#include <Core/Thread/WorkQueue.h>
#include <Core/Geometry/Point.h>
#include <stdlib.h>

namespace rtrt {

using SCIRun::WorkQueue;

struct MIPVMCell;

class MIPHVB16 : public Object, public Material {
protected:
  Point min;
  Vector datadiag;
  Vector hierdiag;
  Vector ihierdiag;
  Vector sdiag;
  int nx,ny,nz;
  short* indata;
  short* blockdata;
  short datamin, datamax;
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
  Vector* cellscale;
  Vector* icellscale;
  Vector isdatadiag;
  MIPVMCell** macrocells;
  int** macrocell_xidx;
  int** macrocell_yidx;
  int** macrocell_zidx;
  WorkQueue work;
  int offset;
  void brickit(int);
  void parallel_calc_mcell(int);
  char* filebase;
  void calc_mcell(int depth, int ix, int iy, int iz, MIPVMCell& mcell);
  void isect(int depth, float isoval,double t,
	     double dtdx, double dtdy, double dtdz,
	     double next_x, double next_y, double next_z,
	     int ix, int iy, int iz,
	     int dix_dx, int diy_dy, int diz_dz,
	     int startx, int starty, int startz,
	     const Vector& cellcorner, const Vector& celldir,
	     const Ray& ray, HitInfo& hit,
	     DepthStats* st, PerProcessorContext* ppc);
public:
  MIPHVB16(char* filebase, int depth, int np);
  MIPHVB16(MIPHVB16* share);

  virtual void io(SCIRun::Piostream &stream);

  virtual ~MIPHVB16();
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void compute_bounds(BBox&, double offset);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
