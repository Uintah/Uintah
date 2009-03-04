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



#ifndef RTRT_HVolumeVG_H
#define RTRT_HVolumeVG_H 1

#include <Packages/rtrt/Core/VolumeVGBase.h>
#include <Packages/rtrt/Core/HVolume.h> // For VMCell
#include <Packages/rtrt/Core/Array3.h>

#include <Core/Geometry/Point.h>

#include <cstdlib>

namespace rtrt {

using SCIRun::WorkQueue;

class ImplicitLine;

template<class V, class G>
struct VG {
  V v;
  G g;
  inline VG() {}
  inline VG(V v, G g) : v(v), g(g) {}
  void setmin();
  void setmax();
};

  // Forward declarations of template specialization
  template<> void VG<unsigned short, unsigned short>::setmin();
  template<> void VG<unsigned short, unsigned short>::setmax();
  template<> void VG<unsigned char, unsigned char>::setmin();
  template<> void VG<unsigned char, unsigned char>::setmax();

template<class V, class G> inline VG<V,G> Min(VG<V,G> v1, VG<V,G> v2)
{
  return VG<V,G>(Min(v1.v, v2.v), Min(v1.g, v2.g));
}

template<class V, class G> inline VG<V,G> Max(VG<V,G> v1, VG<V,G> v2)
{
  return VG<V,G>(Max(v1.v, v2.v), Max(v1.g, v2.g));
}


template<class T, class A, class B>
class HVolumeVG : public VolumeVGBase {
protected:
public:
  Point min;
  Vector datadiag;
  Vector hierdiag;
  Vector ihierdiag;
  Vector sdiag;
  int nx,ny,nz;
  Array3<T> indata;
  A blockdata;
  T datamin, datamax;
  int depth;
  int* xsize;
  int* ysize;
  int* zsize;
  double* ixsize;
  double* iysize;
  double* izsize;
  B* macrocells;
  WorkQueue* work;
  void brickit(int);
  void parallel_calc_mcell(int);
  char* filebase;
  void calc_mcell(int depth, int ix, int iy, int iz, VMCell<T>& mcell);
  void isect(int depth, const ImplicitLine& isoline, double t,
	     double dtdx, double dtdy, double dtdz,
	     double next_x, double next_y, double next_z,
	     int ix, int iy, int iz,
	     int dix_dx, int diy_dy, int diz_dz,
	     int startx, int starty, int startz,
	     const Vector& cellcorner, const Vector& celldir,
	     const Ray& ray, HitInfo& hit,
	     DepthStats* st, PerProcessorContext* ppc);
  HVolumeVG(Material* matl, Hist2DDpy* dpy,
	    char* filebase, int depth, int np);
  HVolumeVG(Material* matl, Hist2DDpy* dpy, HVolumeVG<T,A,B>* share);
  virtual ~HVolumeVG();
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void compute_bounds(BBox&, double offset);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void compute_hist(int nvhist, int nghist, int** hist,
			    float vdatamin, float vdatamax,
			    float gdatamin, float gdatamax);
  virtual void get_minmax(float& vmin, float& vmax,
			  float& gmin, float& gmax);
};

} // end namespace rtrt

#endif // #ifndef RTRT_HVolumeVG_H
