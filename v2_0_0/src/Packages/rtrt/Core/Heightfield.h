
#ifndef HEIGHTFIELD_H
#define HEIGHTFIELD_H 1

#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Array2.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <sci_config.h>
#include <stdlib.h>

namespace rtrt {
template<class A, class B> class Heightfield;
template<class T> struct HMCell;
}

namespace SCIRun {
//template<class A, class B> void Pio(Piostream&, rtrt::Heightfield<A,B>*&);
class WorkQueue;
template<class T> void Pio(Piostream&, rtrt::HMCell<T>&);
}

namespace rtrt {

using SCIRun::WorkQueue;

template<class T>
struct HMCell {
  T max;
  T min;
};

#define GRIDFLOAT 0
#define ELEV 1

template<class A, class B>
class Heightfield : public Object, public UVMapping {
protected:
public:
  Point min;
  Vector datadiag;
  Vector hierdiag;
  Vector ihierdiag;
  Vector sdiag;
  int nx,ny;
  double x1,y1;
  double x2,y2;
  double maxx,minx;
  double maxy,miny;
  Array2<typename A::data_type> indata;
  A blockdata;
  typename A::data_type datamin, datamax;
  int depth;
  int* xsize;
  int* ysize;
  double* ixsize;
  double* iysize;
  B* macrocells;
  WorkQueue* work;
  int np_;
  void brickit(int);
  void parallel_calc_mcell(int);
  char* filebase;
  void calc_mcell(int depth, int ix, int iy, HMCell<typename A::data_type>& mcell);
  void isect_up(int depth, double t,
		double dtdx, double dtdy,
		double next_x, double next_y,
		int ix, int iy,
		int dix_dx, int diy_dy,
		int startx, int starty,
		const Vector& cellcorner, const Vector& celldir,
		Ray& ray, HitInfo& hit,
		DepthStats* st, PerProcessorContext* ppc);
  void isect_down(int depth, double t,
		  double dtdx, double dtdy,
		  double next_x, double next_y,
		  int ix, int iy,
		  int dix_dx, int diy_dy,
		  int startx, int starty,
		  const Vector& cellcorner, const Vector& celldir,
		  Ray& ray, HitInfo& hit,
		  DepthStats* st, PerProcessorContext* ppc);
  Heightfield(Material* matl,
	      char* filebase, int depth, int np);
  Heightfield(Material* matl, Heightfield<A,B>* share);
  Heightfield() : Object(0) {};
  virtual ~Heightfield();



  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void compute_bounds(BBox&, double offset);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void uv(UV& uv, const Point&, const HitInfo& hit);

  static  SCIRun::PersistentTypeID type_id;
  static SCIRun::Persistent* maker();
  virtual void io(SCIRun::Piostream &stream);
  //friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, Heightfield<A,B>*&);
  static const string type_name();
};
}

#include <Packages/rtrt/Core/Heightfield.cc>

#endif
