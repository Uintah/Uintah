
#ifndef DPY_H
#define DPY_H 1

#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Barrier.h>

#include <Packages/rtrt/Core/Scene.h> // for ShadowType

#include <X11/Xlib.h>

namespace rtrt {

using SCIRun::Barrier;
using SCIRun::Mutex;
using SCIRun::Runnable;

//class Barrier;
//class Mutex;

class  Scene;
class  Stats;
class  PerProcessorContext;
class  Worker;
class  Gui;
class  Stealth;
class  Counters;
class  Camera;
struct DpyPrivate;

extern Mutex xlock;

//////////////////////////////////////////////////////////
// WARNING: Currently there can only be ONE Dpy at a time!
//////////////////////////////////////////////////////////

class Dpy : public Runnable {

  friend class Gui;

  // Begin Gui Interaction Flags:
  Image    * showImage_;
  bool       doAutoJitter_; // Jitter when not moving
  bool       doJitter_;     // Jitter on/off
  int        shadowMode_;   // Must be an int so GLUI can write to it.

  bool       showLights_;    // Display lights as spheres  
  bool       lightsShowing_; // Lights are being displayed as spheres.

  bool       turnOffAllLights_; 
  bool       turnOnAllLights_; 
  Light    * turnOffLight_;
  Light    * turnOnLight_;

  // End Gui Interaction Flags.

  Camera   * guiCam_;
  Stealth  * stealth_;

  Scene    * scene;
  char     * criteria1;
  char     * criteria2;
  Barrier  * barrier;
  int        nworkers;
  bool       bench;
  Worker  ** workers;
  Stats    * drawstats[2];
  Counters * counters;
  int        ncounters;
  int        c0, c1;

  PerProcessorContext* ppc;

  DpyPrivate* priv; // allows cleaner integration of frameless stuff
  
  float xScale,yScale; // for using pixelzoom 
  
  int frameless;       // wether it's frameless or not...
  int synch_frameless; // 1 if frameless rendering is supposed to synchronize..

  bool display_frames; // whether or not to display the rendering
  
  static void idleFunc();
  void renderFrame();
  void renderFrameless();

public:
  Dpy(Scene* scene, char* criteria1, char* criteria2,
      int nworkers, bool bench, int ncounters, int c0, int c1,
      float xScale,float yScale, bool display_frames, 
      int pp_size, int scratchsize, int frameless=0);
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
