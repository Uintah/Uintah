
#ifndef DPY_H
#define DPY_H 1

#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Thread.h>

#include <Packages/rtrt/Core/Scene.h> // for ShadowType
#include <Packages/rtrt/Core/DynamicInstance.h>
#include <Packages/rtrt/Core/DpyBase.h>

#include <vector>

#include <X11/Xlib.h>

namespace rtrt {

using SCIRun::Barrier;
using SCIRun::Mutex;
using SCIRun::Runnable;
using SCIRun::Semaphore;
using SCIRun::Thread;

using std::vector;

//class Barrier;
//class Mutex;

class  Scene;
class  Stats;
class  PerProcessorContext;
class  Worker;
class  Gui;
  class RServer;
class  Stealth;
class  Counters;
class  Camera;
struct DpyPrivate;

extern Mutex xlock;

//////////////////////////////////////////////////////////
// WARNING: Currently there can only be ONE Dpy at a time!
//////////////////////////////////////////////////////////

  class Dpy : public DpyBase {

  friend class Gui;

  bool       fullScreenMode_;
    Window parentWindow;
    int nstreams;
    Semaphore releaseSema;

  // Begin Gui Interaction Flags:
  bool       doAutoJitter_; // Jitter when not moving
  int        shadowMode_;  // Must be an int so GLUI can write to it.
  int        ambientMode_; // Must be an int so GLUI can write to it.

  bool       showLights_;    // Display lights as spheres  
  bool       lightsShowing_; // Lights are being displayed as spheres.

  bool       turnOnAllLights_; 
  bool       turnOffAllLights_; 
  Light    * turnOnLight_;
  Light    * turnOffLight_;

  bool       toggleRenderWindowSize_;
  int        renderWindowSize_; // 0 == full, 1 == med

  bool       turnOnTransmissionMode_;

  int        numThreadsRequested_;
  bool       changeNumThreads_;

  Camera   * guiCam_;
  Stealth  * objectStealth_;

  Stealth  * stealth_;

  DynamicInstance * attachedObject_;
  float             objectRotationMatrix_[4][4];

  bool     holoToggle_;

  // End Gui Interaction Flags.

  Scene    * scene;
  char     * criteria1;
  char     * criteria2;
  Barrier  * barrier;
  Barrier  * addSubThreads_;
  int        nworkers;
  int        pp_size_;
  int        scratchsize_;
  bool       bench;
  RServer* rserver;

  Stats    * drawstats[2];
  Counters * counters;
  int        ncounters;
  int        c0, c1;

  vector<Worker*> workers_;

  PerProcessorContext* ppc;

  DpyPrivate* priv; // allows cleaner integration of frameless stuff
  
  float xScale,yScale; // for using pixelzoom 
  
  bool frameless;       // wether it's frameless or not...
  int synch_frameless; // 1 if frameless rendering is supposed to synchronize..

  bool display_frames; // whether or not to display the rendering
  
  static void idleFunc();
  void renderFrame();
  void renderFrameless();

  bool checkGuiFlags();

public:
  Dpy(Scene* scene, char* criteria1, char* criteria2,
      int nworkers, bool bench, int ncounters, int c0, int c1,
      float xScale,float yScale, bool display_frames, 
      int pp_size, int scratchsize, bool fullscreen, bool frameless,
      bool rserver);
  virtual ~Dpy();
    void release(Window win);

  virtual void run();
  void get_barriers(Barrier *& mainBarrier, Barrier *& addSubThreads);

  const Camera * getGuiCam() { return guiCam_; }

  // <<<< bigler >>>>
  int get_num_procs();
  
  int doing_frameless() { return frameless; }
  int synching_frameless() { return synch_frameless; }
  
  void register_worker(int i, Worker* worker);
};

} // end namespace rtrt

#endif
