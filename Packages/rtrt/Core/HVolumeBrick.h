/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#ifndef HVOLUMEBRICK_H
#define HVOLUMEBRICK_H 1

#include <Core/Thread/WorkQueue.h>
#include <Packages/rtrt/Core/VolumeBase.h>
#include <Core/Geometry/Point.h>
#include <cstdlib>

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
