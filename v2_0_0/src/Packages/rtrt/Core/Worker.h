
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

using SCIRun::Runnable;
using SCIRun::Barrier;
using SCIRun::Point;
using SCIRun::Vector;

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
  Barrier* addSubThreads_;
  int num;
  Stats* stats[2];
  Scene* scene;
  PerProcessorContext* ppc;
  Counters* counters;
  int ncounters;
  int c0, c1;

  bool stop_;
  bool useAddSubBarrier_;
  int  oldNumWorkers_;
  
public:
  int rendering_scene;

  Worker(Dpy* dpy, Scene* scene, int num, int pp_size, int scratchsize,
	 int ncounters, int c0, int c1);
  virtual ~Worker();

  virtual void run();

  // If stop is true, this thread will stop running.
  void syncForNumThreadChange( int oldNumWorkers, bool stop = false );

  void traceRay(Color& result, Ray& ray, int depth,
		double atten, const Color& accum,
		Context* cx);
  void traceRay(Color& result, Ray& ray, int depth,
		double atten, const Color& accum,
		Context* cx, double &dist);
  void traceRay(Color& result, Ray& ray, int depth,
		double atten, const Color& accum,
		Context* cx, Object* obj);
  void traceRay(Color& result, Ray& ray, int depth,
		double atten, const Color& accum,
		Context* cx, Object* obj, double &dist);
  void traceRay(Color& result, Ray& ray,
		Point& hitpos, Object*& hitobj);
  Stats* get_stats(int i);
  inline Counters* get_counters() {
    return counters;
  }
  PerProcessorContext* get_ppc() {return ppc;}

};

} // end namespace rtrt

#endif
