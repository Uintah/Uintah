
#ifndef DPY_H
#define DPY_H 1

#include <Packages/rtrt/Core/Scene.h> // for ShadowType
#include <Packages/rtrt/Core/DynamicInstance.h>
#include <Packages/rtrt/Core/DpyBase.h>

#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Thread.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <X11/Xlib.h>

namespace rtrt {

using SCIRun::Barrier;
using SCIRun::Mutex;
using SCIRun::Runnable;
using SCIRun::Semaphore;
using SCIRun::Thread;

//class Barrier;
//class Mutex;

class Scene;
class Stats;
class PerProcessorContext;
class Worker;
class DpyGui;
class GGT;
class RServer;
class RTRT;
class Stealth;
class Counters;
class Camera;
struct DpyPrivate;

//////////////////////////////////////////////////////////
// WARNING: Currently there can only be ONE Dpy at a time!
//////////////////////////////////////////////////////////

class Dpy : public DpyBase {

  friend class GGT;
  friend class DpyGui;
  friend class Worker;

  bool       fullScreenMode_;
  Window parentWindow;
  int nstreams;
  Semaphore parentSema;
  
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

  // If you want to change the number of threads externally use
  // numThreadsRequested_new or change_nworkers().  Modifying
  // numThreadsRequested_ externally could cause race conditions that
  // lock up the renderer.
  int        numThreadsRequested_;
  int        numThreadsRequested_new;
  bool       changeNumThreads_;

  Camera   * guiCam_;
  Stealth  * objectStealth_;

  Stealth  * stealth_;

  DynamicInstance * attachedObject_;
  float             objectRotationMatrix_[4][4];

  bool     holoToggle_;

  // End Gui Interaction Flags.

  Scene    * scene;
  RTRT     * rtrt_engine;
  char     * criteria1;
  char     * criteria2;
  Barrier  * barrier;
  Barrier  * addSubThreads_;
  int        pp_size_;
  int        scratchsize_;
  bool       bench;
  // Number of frames to warmup with before starting the bench mark.
  int bench_warmup;
  // This is the number of frames to bench + warmup frames.
  int bench_num_frames;
  // Please exit rtrt after doing the bench mark
  bool exit_after_bench;
  RServer* rserver;

  Stats    * drawstats[2];
  Counters * counters;
  int        ncounters;
  int        c0, c1;

  std::vector<Worker*> workers_;

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

  // Call this function when you want to start a benchmark
  void start_bench(int num_frames=100, int warmup=5);
  // Call this to stop the currently in progress benchmark
  void stop_bench();

  // Inherited from DpyBase
  virtual bool should_close();

  // Displays the framerate in the corner of the window.
  void display_frame_rate(double framerate);

  // This saves an image or a frame of a movie.
  void save_frame(Image* image);
public:
  Dpy(Scene* scene, RTRT* rtrt_endinge, char* criteria1, char* criteria2,
      bool bench, int ncounters, int c0, int c1,
      float xScale,float yScale, bool display_frames, 
      int pp_size, int scratchsize, bool fullscreen, bool frameless,
      bool rserver, bool stereo=false);
  virtual ~Dpy();
  void release(Window win);

  virtual void run();
  void get_barriers(Barrier *& mainBarrier, Barrier *& addSubThreads);

  const Camera * getGuiCam() { return guiCam_; }

  int get_num_procs();
  
  int doing_frameless() { return frameless; }
  int synching_frameless() { return synch_frameless; }
  
  void register_worker(int i, Worker* worker);

  void wait_on_close();

  void change_nworkers(int num);
};

} // end namespace rtrt

#endif
