
#ifndef DPY_H
#define DPY_H 1

#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Barrier.h>

namespace rtrt {

using SCIRun::Barrier;
using SCIRun::Mutex;
using SCIRun::Runnable;

//class Barrier;
class Scene;
class Stats;
class Worker;
//class Mutex;
class Counters;

extern Mutex xlock;
  //extern Mutex cameralock; // to synchronize camera...

struct Dpy_private;

class Dpy : public Runnable {
  Scene* scene;
  char* criteria1;
  char* criteria2;
  Barrier* barrier;
  int nworkers;
  bool bench;
  Worker** workers;
  Stats* drawstats[2];
  Counters* counters;
  int ncounters;
  int c0, c1;
  
  Dpy_private* priv; // allows cleaner integration of frameless stuff
  
  float xScale,yScale; // for using pixelzoom 
  
  int frameless;       // wether it's frameless or not...
  int synch_frameless; // 1 if frameless rendering is supposed to synchronize..

  bool display_frames; // whether or not to display the rendering
  
  void get_input();         // some common code for frameless vs. not frameless
  
  
  
public:
  Dpy(Scene* scene, char* criteria1, char* criteria2,
      int nworkers, bool bench, int ncounters, int c0, int c1,
      float xScale,float yScale, bool display_frames, int frameless=0);
  virtual ~Dpy();
  virtual void run();
  Barrier* get_barrier();

  // <<<< bigler >>>>
  int get_num_procs();
  
  int doing_frameless() { return frameless; }
  int synching_frameless() { return synch_frameless; }
  
  void register_worker(int i, Worker* worker);
};

} // end namespace rtrt

#endif
