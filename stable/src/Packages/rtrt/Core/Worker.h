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



#ifndef WORKER_H
#define WORKER_H 1

#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/params.h>

#include <Core/Thread/Runnable.h>
#include <Core/Thread/Barrier.h>

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
  
  int rendering_scene;

  // This reallocates the work distribution for the workers when doing
  // frameless rendering.
  void fill_frameless_work(Array1<int>& xpos, Array1<int>& ypos,
                           int xres, int yres, int& nwork);
  void renderFrameless();
public:
  Worker(Dpy* dpy, Scene* scene, int num, int pp_size, int scratchsize,
	 int ncounters, int c0, int c1);
  virtual ~Worker();

  virtual void run();

  // If stop is true, this thread will stop running.
  void syncForNumThreadChange( int oldNumWorkers, bool stop = false );

  static void traceRay(Color& result, Ray& ray, int depth,
                       double atten, const Color& accum,
                       Context* cx);
  static void traceRay(Color& result, Ray& ray, int depth,
                       double atten, const Color& accum,
                       Context* cx, double &dist);
  static void traceRay(Color& result, Ray& ray, int depth,
                       double atten, const Color& accum,
                       Context* cx, Object* obj);
  static void traceRay(Color& result, Ray& ray, int depth,
                       double atten, const Color& accum,
                       Context* cx, Object* obj, double &dist);
  Stats* get_stats(int i);
  inline Counters* get_counters() {
    return counters;
  }
  PerProcessorContext* get_ppc() {return ppc;}

  // This is probably a bad idea to use this as I can't enforce num to
  // be between 0 and the number of workers or be unique, but for now
  // it is.  --James Bigler
  inline int rank() const { return num; }

  // Alows the Dpy to set the rendering_scene parameter before the
  // thread starts running.  Setting this during rendering could be a
  // "bad thing" (TM).
  void set_rendering_scene(const int new_rendering_scene) {
    // Be sure to do bounds checks
    if (new_rendering_scene >= 0 && new_rendering_scene <= 1)
      rendering_scene = new_rendering_scene;
  }

};

} // end namespace rtrt

#endif
