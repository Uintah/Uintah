
#ifndef WORKER_H
#define WORKER_H 1

#include <Core/Thread/Runnable.h>
#include <Core/Thread/Barrier.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/params.h>

namespace SCIRun {
  class Vector;
  class Point;
}

namespace rtrt {

  using namespace SCIRun;
  
//class Barrier;
class Color;
class Dpy;
class HitInfo;
class Ray;
class Scene;
class Light;
class Object;
class Stats;
class PerProcessorContext;
class Context;
class Counters;

class Worker : public Runnable {
  Dpy* dpy;
  Barrier* barrier;
  int num;
  Stats* stats[2];
  Scene* scene;
  Object* shadow_cache[MAXDEPTH];
  PerProcessorContext* ppc;
  Array1<Color> attens;
  Counters* counters;
  int ncounters;
  int c0, c1;
  
public:
  Worker(Dpy* dpy, Scene* scene, int num, int pp_size, int scratchsize,
	 int ncounters, int c0, int c1);
  virtual ~Worker();
  virtual void run();
  void traceRay(Color& result, const Ray& ray, int depth,
		double atten, const Color& accum,
		Context* cx);
  void traceRay(Color& result, const Ray& ray, int depth,
		double atten, const Color& accum,
		Context* cx, Object* obj);
  void traceRay(Color& result, const Ray& ray,
		Point& hitpos, Object*& hitobj);
  bool lit(const Point& hitpos, Light* light,
	   const Vector& light_dir, double dist, Color& shadow_factor,
	   int depth, Context* cx);
  Stats* get_stats(int i);
  inline Counters* get_counters() {
    return counters;
  }
};

} // end namespace rtrt

#endif
