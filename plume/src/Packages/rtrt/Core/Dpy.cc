//
// Dpy.cc
//
// Key Board User Interface Key Mapping:
//
// ------------------------------------------------------------------
// Key Pad -- Used to move Stealth
//
//                  / (Down)          * (Up)           - (decelerate)
// 7 (move left)    8 (Pitch Forward) 9 (move right)   + (accelerate)
// 4 (turn left)    5 (Stop Rotates)  6 (turn right)   
// 1 (nothing yet)  2 (Pitch Back/Up) 3 (nothing yet)  Enter (0 pitch/roll)
// 0 (stop all movement)              . (slow down all)
//
// OTHER STEALTH COMMANDS
//
// x    follow path
// /    Clear Stealth Path
// '    Load Stealth Path from file "..."
// ,    Save Stealth Path to file "temp"
// .    Add a path "frame" (current location and look at point) to path list.
// 
// ------------------------------------------------------------------
//
// a      toggle animate
// c      display (c)amera position
// f      decrease framerate
// F      increase framerate (was 'g' key.)
// g      toggle gravity on/off
// h      cycle ambient_mode
// j      toggle on/off jitter sampling
// J      toggle on/off AUTO jitter (jitters if not moving)
// m      scale eyesep to 1.1
// n      scale eyesep to 0.9
// o      cycle through materials
// p      toggle pstats
// q      quit
// r      toggle rstats
// s      cycle through shadow modes
// t      toggle hotspots
// v      recenter view so that whole scene is displayed
// w      (W)rite picture of screen in images/image.raw 
// x      see STEALTH COMMANDS
// y      Synchronize Mode for Frameless Rendering
// 
// Esc    quit
// 2      toggle stereo on/off
// -      decrease max depth (keyboard minus, NOT KEYPAD)
// +      increase max depth (actually the '=' key.  NOT KEYPAD)
// 
// HOME   Galpha + 1
// END    Galpha - 1
// ------------------------------------------------------------------
//
// ARROW KEYS (not keypad)
//
// UP     up + 4    ???
// DOWN   up - 4    ???
// LEFT   left + 4  ???
// RIGHT  right + 4 ???
// ------------------------------------------------------------------
//
//

#include <Packages/rtrt/Core/Dpy.h>
#include <Packages/rtrt/Core/DpyPrivate.h>

#include <Packages/rtrt/Core/rtrt.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/CycleMaterial.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Stealth.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>
#include <Packages/rtrt/Core/Image.h>
#include <Packages/rtrt/Core/rtrt.h>

#include <Packages/rtrt/Core/Ball.h>
#include <Packages/rtrt/Core/BallMath.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Grid2.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/MusilRNG.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/RServer.h>

#include <Packages/rtrt/visinfo/visinfo.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Transform.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Core/Math/MiscMath.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <unistd.h>
#include <strings.h>

#include <sys/time.h>

/* #include <SpeedShop/api.h> */

using namespace rtrt;
using namespace SCIRun;
using namespace std;

extern bool pin;

Mutex io_lock_("io lock");

//////////////////////////////////////////////////////////////////

float Galpha=1.0; // doh!  global variable...  probably should be moved...

/* Lost code... didn't know what key to bind these to...
   after I moved them from the keypad...

    float   & base_threshold = priv->base_threshold;
    float   & full_threshold = priv->full_threshold;

    base_threshold*=2;
    cerr << "base_threshold=" << base_threshold << '\n';
    base_threshold/=2;
    cerr << "base_threshold=" << base_threshold << '\n';
    full_threshold*=2;
    cerr << "full_threshold=" << full_threshold << '\n';
    full_threshold/=2;
    cerr << "full_threshold=" << full_threshold << '\n';
*/


//static double    eye_dist = 0;
//static double    prev_time[3]; // history for quaternions and time
//static HVect     prev_quat[3];


/////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////
///
///
/// NONE OF THESE SHOULD BE STATIC.  Each Dpy should get their own set
/// of these variables.
///
///
static Transform prev_trans;

static double   last_frame = SCIRun::Time::currentSeconds();
static int      frame = 0;
static int      rendering_scene = 0;
static MusilRNG rng;
//static Object * obj;

static double   lightoff_frame = -1.0;
/////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////

namespace rtrt {
  double _HOLO_STATE_=1;
}

//static float float_identity[4][4] = { {1,0,0,0}, {0,1,0,0},
//	 			        {0,0,1,0}, {0,0,0,1} };

//////////////////////////////////////////////////////////////////

Dpy::Dpy( Scene* scene, RTRT* rtrt_engine, char* criteria1, char* criteria2,
	  bool bench, int ncounters, int c0, int c1,
	  float, float, bool display_frames, 
	  int pp_size, int scratchsize, bool fullscreen, bool frameless,
	  bool rserver, bool stereo):
  DpyBase("Real-time Ray Tracer", DoubleBuffered, false),
  fullScreenMode_( fullscreen ),
  parentSema("Dpy window wait", 0),
  doAutoJitter_( false ),
  showLights_( false ), lightsShowing_( false ),
  turnOnAllLights_( false ), turnOffAllLights_( false ),
  turnOnLight_( false ), turnOffLight_( false ),
  toggleRenderWindowSize_(fullscreen), renderWindowSize_(1),
  turnOnTransmissionMode_(false), 
  numThreadsRequested_(rtrt_engine->nworkers),
  numThreadsRequested_new(rtrt_engine->nworkers),
  changeNumThreads_(false),
  stealth_(NULL), attachedObject_(NULL), holoToggle_(false),
  scene(scene), rtrt_engine(rtrt_engine),
  criteria1(criteria1), criteria2(criteria2),
  pp_size_(pp_size), scratchsize_(scratchsize),
  bench(bench), bench_warmup(10), bench_num_frames(110),
  exit_after_bench(true), ncounters(ncounters),
  c0(c0), c1(c1),
  frameless(frameless), synch_frameless(0), display_frames(display_frames)
{
  if(rserver)
    this->rserver = new RServer();
  else
    this->rserver = 0;
  nstreams=1;
  ppc = new PerProcessorContext( pp_size, scratchsize );

  for( int i = 0; i < 4; i++ )
    for( int j = 0; j < 4; j++ )
      if( i == j )
	objectRotationMatrix_[i][j] = 1;
      else
	objectRotationMatrix_[i][j] = 0;

  barrier        = new Barrier("Frame end");
  addSubThreads_ = new Barrier("Change Number of Threads");

  drawstats[0]=new Stats(1000);
  drawstats[1]=new Stats(1000);
  priv = new DpyPrivate;

  priv->waitDisplay = new Mutex( "wait for display" );
  priv->waitDisplay->lock();
  priv->xres = scene->get_image(0)->get_xres();
  priv->yres = scene->get_image(0)->get_yres();
  xres = priv->xres;
  yres = priv->yres;
  
  priv->followPath = false;

  priv->show_frame_rate = true;

  shadowMode_ = scene->shadow_mode;
  ambientMode_ = scene->ambient_mode;

  //obj=scene->get_object();
  priv->maxdepth=scene->maxdepth;

  workers_.resize( rtrt_engine->nworkers );

  guiCam_ = new Camera;

  *guiCam_ = *(scene->get_camera( rendering_scene ));

  // Get the size of the area of interest and use it to initialize 
  // the stealth.  This will effect how fast the stealth moves.
  //
  //     haven't quite figured a good heuristic out for this yet...
  //     hard coding for stadium.
  //BBox bb;
  //scene->get_object()->compute_bounds( bb, 1e-3 );

  double translate_scale   = 100.0;
  double rotate_scale      = 4.0;
  double gravity           = 10.0;

  stealth_       = new Stealth( translate_scale, rotate_scale, gravity );
  objectStealth_ = new Stealth( translate_scale, rotate_scale, gravity );

  priv->stereo=stereo;

  // Initialize for frameless
  priv->last_changed = true;
  priv->accum_count = 0;
}

Dpy::~Dpy()
{
  delete stealth_;
  delete objectStealth_;
}

void
Dpy::register_worker(int i, Worker* worker)
{
  io_lock_.lock();
  cout << "registering worker " << i << ", PID: " << worker << "\n";
  io_lock_.unlock();
  workers_[i] = worker;
}

int Dpy::get_num_procs() {
  return rtrt_engine->nworkers;
}

void Dpy::release(Window win)
{
  parentWindow=win;
  parentSema.up();
}

void
Dpy::run()
{
  if(pin)
    Thread::self()->migrate(0);
  io_lock_.lock();
  cerr << "display is pid " << getpid() << '\n';
  //  cerr << "xres = "<<xres<<", yres = "<<yres<<'\n';
  //  cerr << "priv->xres = "<<priv->xres<<", priv->yres = "<<priv->yres<<'\n';
  io_lock_.unlock();

  parentSema.down();
  if(rserver){
    rserver->openWindow(parentWindow);
    rserver->resize(priv->xres, priv->yres);
  } else {
    resize(priv->xres, priv->yres);
    open_display(parentWindow, false);

    // Don't close this if you are a child of another window.
    dont_close();

    init();
  }

#if 0
  io_lock_.lock();
  cerr << "After init.\n";
  cerr << "xres = "<<xres<<", yres = "<<yres<<'\n';
  cerr << "priv->xres = "<<priv->xres<<", priv->yres = "<<priv->yres<<'\n';
  io_lock_.unlock();
#endif

  if(ncounters)
    counters=new Counters(ncounters, c0, c1);
  else
    counters=0;

  priv->ball = new BallData();
  priv->ball->Init();

  priv->last_time = 0;

  priv->showing_scene=1;
  priv->animate=true;
  priv->base_threshold=0.005;
  priv->full_threshold=0.01;
  priv->left=0;
  priv->up=0;

  priv->exposed=true;
  priv->FrameRate = 0;

  double benchstart=0;
  for(;;)
    {
      if (frameless) { renderFrameless(); }
      else           { renderFrame();     }
      frame++;

      //cerr << "xres = "<<xres<<", yres = "<<yres<<'\n';
      //cerr << "priv->xres = "<<priv->xres<<", priv->yres = "<<priv->yres<<'\n';

      // Exit if you are supposed to.
      if (should_close()) {
//         cerr << "Dpy is closing\n";
        cleanup();
//         cerr << "Dpy::cleanup finished\n";

        // This can't proceed until someone calls wait_on_close which
        // calls down on the sema.  This must be placed after
        // cleanup() to ensure the window has closed.
	parentSema.up();
//         cerr << "parentSema.up finished\n";
        return;
      }
  
      if(bench){
	if(frame==bench_warmup){
	  cerr << "Warmup done, starting bench\n";
	  benchstart=SCIRun::Time::currentSeconds();
	} else if(frame == bench_num_frames){
	  double dt=SCIRun::Time::currentSeconds()-benchstart;
	  cerr << "Benchmark completed in " <<  dt << " seconds ("
	       << (frame-bench_warmup)/dt << " frames/second)\n";
          if (exit_after_bench)
            rtrt_engine->stop_engine();//Thread::exitAll(0);
          else
            bench = false;
	}
      }

      // dump the frame and quit for now
      if (frame == 3) {
	if (!display_frames) {
	  scene->get_image(priv->showing_scene)->save_ppm("displayless");
          rtrt_engine->stop_engine();//Thread::exitAll(0);
	}
      }

      // Slurp up X events....  We don't need these as they are
      // handled by the parent window.
      if(!rserver){
	while (XEventsQueued(dpy, QueuedAfterReading)){
	  XEvent e;
	  XNextEvent(dpy, &e);
	}
      }
    }
} // end run()

bool
Dpy::checkGuiFlags()
{
  bool  & animate= priv->animate;
  int   & maxdepth= priv->maxdepth;
  float & base_threshold= priv->base_threshold;
  float & full_threshold= priv->full_threshold;

  bool changed = true;

  // Display image as a "transmission".  Ie: turn off every other scan line.
  if( turnOnTransmissionMode_ ){
    rtrt_engine->frameMode = RTRT::OddRows;
  } else {
    rtrt_engine->frameMode = RTRT::FullFrame;
  }

  if( showLights_ && !lightsShowing_ ){
    scene->renderLights( true );
    lightsShowing_ = true;
  } else if( !showLights_ && lightsShowing_ ){
    scene->renderLights( false );
    lightsShowing_ = false;
  }

  if ( _HOLO_STATE_<1 && !holoToggle_) {
    _HOLO_STATE_ += SCIRun::Time::currentSeconds()*.0025;
    if (_HOLO_STATE_>1) _HOLO_STATE_=1;
  } else if ( _HOLO_STATE_>0 && holoToggle_ ) {
    _HOLO_STATE_ -= SCIRun::Time::currentSeconds()*.00025;
    if (_HOLO_STATE_<0) _HOLO_STATE_=0;
  }

  if( turnOffAllLights_ ){
    double left;
    if( lightoff_frame < 0 ) {
      lightoff_frame = SCIRun::Time::currentSeconds();
      left = 1.0;
    } else {
      left = 1.0 - (SCIRun::Time::currentSeconds()-lightoff_frame)*0.2;
    }

    scene->turnOffAllLights(left);
    if (left <= 0.0) {
      lightoff_frame = -1.0;
      turnOffAllLights_ = false;
    }
  }
  if( turnOnAllLights_ ){
    lightoff_frame = -1;
    scene->turnOnAllLights();
    turnOnAllLights_ = false;
  }
  if( turnOnLight_ ) {
    turnOnLight_->turnOn();
    turnOnLight_ = NULL;
  }
  if( turnOffLight_ ) {
    turnOffLight_->turnOff();
    turnOffLight_ = NULL;
  }

  if( toggleRenderWindowSize_ ){ 
    toggleRenderWindowSize_ = false;
    if( renderWindowSize_ == 0 ) { // was full, go to medium
      priv->xres = 512;
      priv->yres = 288;
      renderWindowSize_ = 1;
    } else {
      priv->xres = 1024;
      priv->yres = 600;
      renderWindowSize_ = 0;
    }
  }

  if(animate && scene->animate) {
    // Do all the regular animated objects.
    Array1<Object*> & objects = scene->animateObjects_;
    double current_seconds = SCIRun::Time::currentSeconds();
    for( int num = 0; num < objects.size(); num++ ) {
      objects[num]->animate(current_seconds, changed);
    }
    Array1<Material*>& materials = scene->animateMaterials_;
    for( int num = 0; num < materials.size(); num++ ) {
      materials[num]->animate(current_seconds, changed);
    }
    // Do the special objects that require bounding box mojo.
    Array1<Object*> & dobjects = scene->dynamicBBoxObjects_;
    BBox bbox1,bbox2;
    for( int num = 0; num < dobjects.size(); num++ ) {
      bbox1.reset();
      dobjects[num]->compute_bounds(bbox1,1E-5);
      dobjects[num]->animate(current_seconds, changed);
      Grid2 *anim_grid = dobjects[num]->get_anim_grid();
      if (anim_grid) {
        bbox2.reset();
        dobjects[num]->compute_bounds(bbox2, 1E-5);
        anim_grid->remove(dobjects[num],bbox1);
        anim_grid->insert(dobjects[num],bbox2);
      }
    }
  }

  if( attachedObject_ ) {
    attachedObject_->useNewTransform();
  }

  if(scene->shadow_mode != shadowMode_){
    scene->shadow_mode = shadowMode_;
    changed=true;
  }
  if(scene->ambient_mode != ambientMode_){
    scene->ambient_mode = ambientMode_;
    changed=true;
  }
  if(scene->maxdepth != maxdepth){
    scene->maxdepth=maxdepth;
    changed=true;
  }
  if(scene->base_threshold != base_threshold){
    cerr << "Old: " << scene->base_threshold << '\n';
    cerr << "New: " << base_threshold << '\n';
    scene->base_threshold=base_threshold;
    changed=true;
  }
  if(scene->full_threshold != full_threshold){
    scene->full_threshold=full_threshold;
    changed=true;
  }
  if(frame<5){
    changed=true;
  }
  if(priv->exposed){
    changed=true;
    priv->exposed=false;
  }

  if( !priv->followPath ) {
    guiCam_->updatePosition( *stealth_, scene, ppc );
  } else {
    guiCam_->followPath( *stealth_ );
  }
  
  if (numThreadsRequested_new != numThreadsRequested_)
    numThreadsRequested_ = numThreadsRequested_new;
  
  // If we've been told to exit go ahead and check this.  This code
  // needs to come after the check for numThreadsRequested_new,
  // because it could override this for us.
  if (rtrt_engine->exit_engine) {
    numThreadsRequested_ = 0;
  }

  { // Check to see if we are going to change the number of workers
    int nworkers = rtrt_engine->nworkers;
    if( nworkers != numThreadsRequested_ ) {

      changeNumThreads_ = true;
      // remove excess threads if need be;
      int numToRemove = 0;

      if( nworkers > numThreadsRequested_ )
        numToRemove = nworkers - numThreadsRequested_;

      int cnt;
      // Tell the first bunch to just sync up
      for( cnt = 0; cnt < nworkers-numToRemove; cnt++ ) {
        workers_[cnt]->syncForNumThreadChange( nworkers );
      }

      // Tell the rest to exit after syncing up
      for( ; cnt < nworkers; cnt++ ) {
        Worker * worker = workers_[ cnt ];
        workers_.pop_back();
        cout << "worker " << cnt << " told to stop!\n";
        // Tell the worker to stop.
        worker->syncForNumThreadChange( nworkers, true );
      }

      if (numToRemove) cout << "Done asking threads to die.\n";
    }
  }

  return changed;
}

// Call this function when you want to start a benchmark
void
Dpy::start_bench(int num_frames, int warmup) {
  // Since this is most likely being called during rendering, we don't
  // want to kill rtrt.
  exit_after_bench = false;

  // Reset the number of frames.  We must do this so, we know how many
  // frames were rendered. (simply adding frame to bench_num_frames
  // would lose the num_frames).
  frame = 0;
  
  // Set the bench flags
  bench_warmup = warmup;
  bench_num_frames = warmup + num_frames;

  // Turn on bench marking
  bench = true;
}

// Call this to stop the currently in progress benchmark
void
Dpy::stop_bench() {
  bench = false;
}

bool Dpy::should_close() {
  // The only time you should close is if the number of workers equals zero
  return rtrt_engine->nworkers == 0;
}

void
Dpy::renderFrameless() {

  int   & showing_scene = priv->showing_scene;
  // only 1 buffer for frameless...
  if (showing_scene != rendering_scene)
    showing_scene = rendering_scene;

  // If we need to change the number of worker threads:
  if( changeNumThreads_ ) {
    //    cout << "changeNumThreads\n";
    int nworkers = rtrt_engine->nworkers;
    int oldNumWorkers = nworkers;
    rtrt_engine->nworkers = nworkers = numThreadsRequested_;

    if( oldNumWorkers < nworkers ) { // Create more workers
      int numNeeded     = nworkers - oldNumWorkers;
      int stopAt        = oldNumWorkers + numNeeded;

      workers_.resize( rtrt_engine->nworkers );
      for( int cnt = oldNumWorkers; cnt < stopAt; cnt++ ) {
	char buf[100];
	sprintf(buf, "worker %d", cnt);
	Worker * worker = new Worker(this, scene, cnt,
				     pp_size_, scratchsize_,
				     ncounters, c0, c1);
        worker->set_rendering_scene(rendering_scene);
        //	cout << "created worker: " << cnt << ", " << worker << "\n";
	Thread * thread = new Thread( worker, buf);
	thread->detach();
      }

      //// THIS IS FOR DEBUGGING:
      //cout << "workers are:\n";
      //for( int cnt = 0; cnt < numThreadsRequested_; cnt++ )
      //{
      //cout << "worker " << cnt << ": " << workers_[cnt] << "\n";
      //}
    }
    //    cout << "sync with workers for change: " << oldNumWorkers+1 << "\n";
    cout << "Number of workers is now "<<rtrt_engine->nworkers<<"\n";
    addSubThreads_->wait( oldNumWorkers + 1 );
    changeNumThreads_ = false;
  }

  // Exit if you are supposed to.
  if (rtrt_engine->nworkers == 0) {
    cout << "Dpy has no more workers, going away\n";
    return;
  }
  
  Image * displayedImage = scene->get_image(showing_scene);

  // This block of code needs to go after the changeNumThreads_ block,
  // because we need to make sure we clear any waits before we cause
  // another one to happen.
  if( displayedImage->get_xres() != priv->xres ||
      displayedImage->get_yres() != priv->yres ||
      displayedImage->get_stereo() != priv->stereo) {
//     cerr << "Dpy::rendering_scene: showing_scene = "<<showing_scene<<", rendering_scene = "<<rendering_scene<<"\n";
//     cerr << "Dpy::renderFrameless: changing image resolution from ("<<displayedImage->get_xres()<<", "<<displayedImage->get_yres()<<") to ("<<priv->xres<<", "<<priv->yres<<")\n";

    // We actually can't delete this buffer, because it is being used
    // by the worker threads as we speak.  Intead delete the other
    // buffer not being used and change showing_scene to point to that
    // one when the workers and dpy sync up.
    Image* otherBuffer = scene->get_image(1-showing_scene);
    delete otherBuffer;
    otherBuffer = new Image(priv->xres, priv->yres, priv->stereo);
    scene->set_image(1-showing_scene, otherBuffer);
    
    if(rserver){
      rserver->resize(priv->xres, priv->yres);
    } else {
      //      if (display_frames) XResizeWindow(dpy, win, priv->xres, priv->yres);
      XResizeWindow(dpy, win, priv->xres, priv->yres);
      resize(priv->xres, priv->yres);
    }
    
    // Tell the workers to sync up.
    int nworkers = rtrt_engine->nworkers;
    if (frameless && frame > 5) {
      for( int cnt = 0; cnt < nworkers; cnt++ ) {
        workers_[cnt]->syncForNumThreadChange( nworkers );
      }
    }

    // Ok, now switch the buffer
    showing_scene = 1 - showing_scene;
    rendering_scene = 1 - rendering_scene;

    // Now wait until the workers are done blocking
    addSubThreads_->wait( nworkers + 1 );
  }

  // Time how long it takes to do the display, only if we are not
  // synched with the workers.
  double starttime = 0;

  // We'll go ahead and set the desired framerate we want to see
  // updates.  The display thread will wait for the remaining time.
  priv->FrameRate = 1.0/15.0;
  // I'm not sure what this is used for, because an assignment to
  // synch_frameless is the only place it's used.
  // priv->doing_frameless = 0;
  double& FrameRate = priv->FrameRate;

  // Just to make sure this whole pass works from the same value of
  //synch_frameless we'll copy it over and do it.
  int do_synch = synch_frameless;
  //int do_some_type_of_synch=0;
  if (do_synch) { // synchronize stuff...
#if 0
    io.lock();
    cerr << "Display on first\n";
    io.unlock();
#endif
    // wait here for all of the procs...
    barrier->wait(rtrt_engine->nworkers+1);
#if 0
    io.lock();
    cerr << "Display out of first\n";
    io.unlock();
#endif
    //do_some_type_of_synch=1; // do the right thing for barriers...
  } else {
    // Start timing how long it takes to display the frame.
    starttime = SCIRun::Time::currentSeconds();
  }

  // Should this actually be false??
  bool changed = true;
  {
    // Hmmm... should this be locked here?
    rtrt_engine->cameralock.lock();
    Camera * cam1 = scene->get_camera(rendering_scene);
    
    if( *cam1 != *guiCam_ ) {
      *cam1 = *guiCam_;
      changed = true;
    }
    rtrt_engine->cameralock.unlock();
    
    changed |= checkGuiFlags();
    
    if(!changed && !scene->no_aa){
      double x1, x2, w;
      do {
        x1 = 2.0 * rng() - 1.0;
        x2 = 2.0 * rng() - 1.0;
        w = x1 * x1 + x2 * x2;
      } while ( w >= 1.0 );
      
      w = sqrt( (-2.0 * log( w ) ) / w );
      scene->xoffset = x1 * w * 0.5;
      scene->yoffset = x2 * w * 0.5;
    } else {
      scene->xoffset=0;
      scene->yoffset=0;
    }
  }

  // no stats for frameless stuff for now...

#if 0
  drawstats[showing_scene]->add(SCIRun::Time::currentSeconds(), Color(0,1,0));

  // This is the last stat for the rendering scene (cyan)
  drawstats[showing_scene]->add(SCIRun::Time::currentSeconds(), Color(0,1,1));
  counters->end_frame();
      
  Stats* st=drawstats[rendering_scene];
  st->reset();
#endif
  double tnow=SCIRun::Time::currentSeconds();
  //st->add(tnow, Color(1,0,0));
  double dt=tnow-last_frame;
  double framerate=1./dt;
  last_frame=tnow;

  if(ncounters){
    fprintf(stderr, "%2d: %12lld", c0, counters->count0());	
    for(size_t i=0; i < workers_.size(); i++) {
      fprintf(stderr, "%12lld", workers_[i]->get_counters()->count0());
    }
    fprintf(stderr, "\n");
    if(ncounters>1){
      fprintf(stderr, "%2d: %12lld", c1, counters->count1());	
      for(size_t i=0; i < workers_.size(); i++) {
	fprintf(stderr, "%12lld", workers_[i]->get_counters()->count1());
      }
      fprintf(stderr, "\n\n");
    }
  }

  bool swap=true;

#if 1
  if(display_frames && !bench){
    if(rserver){
      rserver->sendImage(displayedImage, nstreams);
    } else {
      
      // Set up the projection matrix for pixel writes
      glViewport(0, 0, priv->xres, priv->yres);
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluOrtho2D(0, priv->xres, 0, priv->yres);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      glTranslatef(0.375, 0.375, 0.0);
      
#if 0
      // zoom stuff to fit to the window...
      if ((xScale != 1.0) || (yScale != 1.0)) {
        glPixelZoom(1.0/xScale,1.0/yScale);
      }
#endif

      if(scene->no_aa || priv->last_changed){
        priv->accum_count=0;
        displayedImage->draw( renderWindowSize_, fullScreenMode_ );
      } else {
        if(priv->accum_count==0){
          // Load last image...
          glReadBuffer(GL_FRONT);
          glAccum(GL_LOAD, 1.0);
          glReadBuffer(GL_BACK);
          priv->accum_count=1;
        }
        priv->accum_count++;
        glAccum(GL_MULT, 1.-1./priv->accum_count);
        displayedImage->draw( renderWindowSize_, fullScreenMode_ );
        glAccum(GL_ACCUM, 1.0/priv->accum_count);
        if(priv->accum_count>=4){
          /* The picture jumps around quite a bit when we show one
           * that doesn't have very many samples, so we don't show the
           * accumulated picture until we have at least 4 samples
           */
          glAccum(GL_RETURN, 1.0);
        } else {
          swap = false;
        }
      }
      if (priv->show_frame_rate)
        display_frame_rate(framerate);
      
      //st->add(SCIRun::Time::currentSeconds(), Color(0,0,1));

      // Draw stats removed
  
      //st->add(SCIRun::Time::currentSeconds(), Color(1,0,1));
      glFinish();
      if (swap && ((window_mode & BufferModeMask) == DoubleBuffered))
        glXSwapBuffers(dpy, win);
      XFlush(dpy);
      //st->add(SCIRun::Time::currentSeconds(), Color(1,1,0));

      save_frame(displayedImage);
    }
  } // if (display_frames && !bench)
#else
  glClearColor(.3,.3,.3, 1);
  glClear(GL_COLOR_BUFFER_BIT);
#endif
  priv->last_changed = changed;


  // Moved the locking before the frame draw
//   cameralock.lock(); // lock this - for synchronization stuff...
//   priv->camera=scene->get_camera(showing_scene);
      
//   get_input(); // this does all of the x stuff...
      
//   cameralock.unlock();

  if (do_synch) {
#if 0
    io.lock();
    cerr << "Display on second\n";
    io.unlock();
#endif
    barrier->wait(rtrt_engine->nworkers+1);
#if 0
    io.lock();
    cerr << "Display out of second\n";
    io.unlock();
#endif
  }

#if 0
  if (do_some_type_of_synch) {
    // just block in the begining???
    barrier->wait(rtrt_engine->nworkers+1); // block - this lets them get camera params...
    synch_frameless = priv->doing_frameless; // new ones always catch on 1st barrier...
  }
#endif

  //st->add(SCIRun::Time::currentSeconds(), Color(1,0,0));
  //rendering_scene=1-rendering_scene;
  //showing_scene=1-showing_scene;

  double endtime = SCIRun::Time::currentSeconds();
  if (!do_synch)
    SCIRun::Time::waitFor(FrameRate - (endtime-starttime));
} // end renderFrameless()

void
Dpy::renderFrame() {

  bool  & stereo        = priv->stereo;  
  int   & showing_scene = priv->showing_scene;

  // If we need to change the number of worker threads:
  if( changeNumThreads_ ) {
    //    cout << "changeNumThreads\n";
    int nworkers = rtrt_engine->nworkers;
    int oldNumWorkers = nworkers;
    rtrt_engine->nworkers = nworkers = numThreadsRequested_;

    if( oldNumWorkers < nworkers ) { // Create more workers
      int numNeeded     = nworkers - oldNumWorkers;
      int stopAt        = oldNumWorkers + numNeeded;

      workers_.resize( nworkers );
      for( int cnt = oldNumWorkers; cnt < stopAt; cnt++ ) {
	char buf[100];
	sprintf(buf, "worker %d", cnt);
	Worker * worker = new Worker(this, scene, cnt,
				     pp_size_, scratchsize_,
				     ncounters, c0, c1);
        worker->set_rendering_scene(rendering_scene);
        //	cout << "created worker: " << cnt << ", " << worker << "\n";
	Thread * thread = new Thread( worker, buf);
	thread->detach();
      }

      //// THIS IS FOR DEBUGGING:
      //cout << "workers are:\n";
      //for( int cnt = 0; cnt < numThreadsRequested_; cnt++ )
      //{
      //cout << "worker " << cnt << ": " << workers_[cnt] << "\n";
      //}
    }
    //    cout << "sync with workers for change: " << oldNumWorkers+1 << "\n";
    cout << "Number of workers is now "<<nworkers<<"\n";
    addSubThreads_->wait( oldNumWorkers + 1 );
    changeNumThreads_ = false;
  }

  barrier->wait(rtrt_engine->nworkers+1);

  // Exit if you are supposed to.
  if (rtrt_engine->nworkers == 0) {
    cout << "Dpy has no more workers, going away\n";
    return;
  }
  
  drawstats[showing_scene]->add(SCIRun::Time::currentSeconds(),Color(1,0,0));

  // Should this actually be false??
  bool changed = true;

  Camera * cam1 = scene->get_camera(rendering_scene);

  if( *cam1 != *guiCam_ ) {
    *cam1 = *guiCam_;
    changed = true;
  }

  changed |= checkGuiFlags();

  scene->refill_work(rendering_scene, rtrt_engine->nworkers);

  if(!changed && !scene->no_aa){
    double x1, x2, w;
    do {
      x1 = 2.0 * rng() - 1.0;
      x2 = 2.0 * rng() - 1.0;
      w = x1 * x1 + x2 * x2;
    } while ( w >= 1.0 );
    
    w = sqrt( (-2.0 * log( w ) ) / w );
    scene->xoffset = x1 * w * 0.5;
    scene->yoffset = x2 * w * 0.5;
  } else {
    scene->xoffset=0;
    scene->yoffset=0;
  }

  drawstats[showing_scene]->add(SCIRun::Time::currentSeconds(),Color(0,1,0));

  barrier->wait(rtrt_engine->nworkers+1);

  Image * displayedImage = scene->get_image(showing_scene);

  if(display_frames && !bench){
    if(rserver){
      rserver->sendImage(displayedImage, nstreams);
    } else {

      glViewport(0, 0, priv->xres, priv->yres);
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluOrtho2D(0, priv->xres, 0, priv->yres);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      glTranslatef(0.375, 0.375, 0.0);
      // Draw the pixels
      if (!scene->display_depth) {
        if (scene->display_sils)
          displayedImage->draw_sils_on_image( scene->max_depth );
        displayedImage->draw( renderWindowSize_, fullScreenMode_ );
      } else {
        displayedImage->draw_depth( scene->max_depth );
      }
      if (priv->show_frame_rate)
        display_frame_rate(priv->FrameRate);
      
      display();
#if 0
      if(priv->displayRStats_ )
	drawrstats(dpy->nworkers, dpy->workers_,
		   priv->showing_scene, fontbase2, 
		   priv->xres, priv->yres,
		   fontInfo2, priv->left, priv->up,
		   0.0 /* dt */);
      if(displayPStats_ ){
	Stats * mystats = dpy->drawstats[!priv->showing_scene];
	drawpstats(mystats, dpy->nworkers, dpy->workers_, 
		   /*draw_framerate*/true, priv->showing_scene,
		   fontbase, lasttime, cum_ttime,
		   cum_dt);
      }
#endif
      save_frame(displayedImage);
    }
  } else {
    // Do stuff that the benchmark will notice
    if (!scene->display_depth && scene->display_sils)
      displayedImage->draw_sils_on_image( scene->max_depth );
  }

  if( displayedImage->get_xres() != priv->xres ||
      displayedImage->get_yres() != priv->yres ||
      displayedImage->get_stereo() != stereo) {
    delete displayedImage;
    displayedImage = new Image(priv->xres, priv->yres, stereo);
    scene->set_image(showing_scene, displayedImage);
    if(rserver){
      rserver->resize(priv->xres, priv->yres);
    } else {
      //      if (display_frames) XResizeWindow(dpy, win, priv->xres, priv->yres);
      XResizeWindow(dpy, win, priv->xres, priv->yres);
      resize(priv->xres, priv->yres);
    }
  }

  // This is the last stat for the rendering scene (cyan)
  drawstats[showing_scene]->add(SCIRun::Time::currentSeconds(),Color(0,1,1));
  counters->end_frame();
	
  Stats* st=drawstats[rendering_scene];
  st->reset();
  double tnow=SCIRun::Time::currentSeconds();
  st->add(tnow, Color(1,0,0));
  double dt=tnow-last_frame;
  double framerate=1./dt;

  priv->FrameRate = framerate;

  last_frame=tnow;
  if(ncounters){
    fprintf(stderr, "%2d: %12lld", c0, counters->count0());	
    for(size_t i=0; i < workers_.size(); i++) {
      fprintf(stderr, "%12lld", workers_[i]->get_counters()->count0());
    }
    fprintf(stderr, "\n");
    if(ncounters>1){
      fprintf(stderr, "%2d: %12lld", c1, counters->count1());	
      for(size_t i=0; i < workers_.size(); i++) {
	fprintf(stderr, "%12lld", workers_[i]->get_counters()->count1());
      }
      fprintf(stderr, "\n\n");
    }
  }

  // color blue
  st->add(SCIRun::Time::currentSeconds(), Color(0,0,1));
  //      st->add(SCIRun::Time::currentSeconds(), Color(0.4,0.2,1));

  if (display_frames) {
      
    double curtime=SCIRun::Time::currentSeconds();
    //double dt=curtime-scene->lasttime;

    scene->lasttime=curtime;

    // If keypad is attached to an object.
    if( attachedObject_ ) {
//      attachedObject_->updatePosition( objectStealth_, guiCam_ );

      Point  origin;
      Vector lookdir, up, side;
      double fov;
      cam1->getParams( origin, lookdir, up, side, fov );

      lookdir.normalize();
      up.normalize();
      side.normalize();

      Transform viewpoint( Point(0,0,0), -lookdir, side, up );

      attachedObject_->updateNewTransform( objectRotationMatrix_,
					   viewpoint );
    }

    st->add(SCIRun::Time::currentSeconds(), Color(1,1,0));
    if( !priv->followPath ) {
      guiCam_->updatePosition( *stealth_, scene, ppc );
    } else {
      guiCam_->followPath( *stealth_ );
    }
    st->add(SCIRun::Time::currentSeconds(), Color(0,1,1));
  }

  // These need to change even if we are not displaying frames,
  // because the workers are changing these.
  rendering_scene=1-rendering_scene;
  showing_scene=1-showing_scene;
} // end renderFrame()

void
Dpy::get_barriers( Barrier *& mainBarrier, Barrier *& addSubThreads )
{
  mainBarrier   = barrier;
  addSubThreads = addSubThreads_;
}

void Dpy::wait_on_close() {
  parentSema.down();
  //  cerr << "Dpy::wait_on_close::parentSema.down()\n";
  
  // I don't think we actually need to wait for the thread to
  // shutdown, just to make sure that it has finished calling
  // cleanup() (which it should have by the tie parentSema.up() is
  // called.
#if 0
  // Now wait for the thread to have exited
  unsigned int i =0;
  while(my_thread_ != 0) {
    i++;
//     if (i %10000 == 0)
//       cerr << "+";
  }
#endif
}

void Dpy::change_nworkers(int num) {
  int newnum = numThreadsRequested_new+num;
  if (newnum >= 1 && newnum <= SCIRun::Thread::numProcessors())
    numThreadsRequested_new += num;
  else
    cerr << "Dpy::change_nworkers:: Number of threads requested ("<<num<<") is outside the acceptable range [1, "<<SCIRun::Thread::numProcessors()<<"].\n";
}

void Dpy::display_frame_rate(double framerate) {
  // Display textual information on the screen:
  char buf[200];
  if (framerate > 1)
    sprintf( buf, "%3.1lf fps", framerate);
  else
    sprintf( buf, "%2.2lf fps - %3.1lf spf", framerate , 1.0f/framerate);
  // Figure out how wide the string is
  int width = calc_width(fontInfo, buf);
  // Now we want to draw a gray box beneth the font using blending. :)
  glEnable(GL_BLEND);
  glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
  glColor4f(0.5,0.5,0.5,0.5);
  glRecti(8,3-fontInfo->descent-2,12+width,fontInfo->ascent+3);
  glDisable(GL_BLEND);
  printString(fontbase, 10, 3, buf, Color(1,1,1));
}

void Dpy::save_frame(Image* image) {
  switch (priv->dumpFrame) {
  case 1:
    // Dump a single frame
    if (!scene->display_depth)
      image->save_ppm( "images/image" );
    else
      image->save_depth( "images/depth" );
    priv->dumpFrame = 0;
    break;
  case -1:
    // Initialize the stream and dump a frame
    if (image->start_ppm_stream( "movies/movie" )) {
      image->save_to_ppm_stream();
      priv->dumpFrame = -2;
    } else {
      // Something went wrong, so set dumpFrame to stop
      priv->dumpFrame = 0;
    }
    break;
  case -2:
    // Dump the next frame
    image->save_to_ppm_stream();
    break;
  case -3:
    // Stop recording
    image->save_to_ppm_stream();
    image->end_ppm_stream();
    priv->dumpFrame = 0;
    break;
  }
}
#if 0 /////////////////////////////////////////////////////////
// Looks like this displays the string with a shadow on it...
void
GGT::displayShadowText(GLuint fontbase,
		       double x, double y, char *s, const Color& c)
{
  Color b(0,0,0);
  printString(fontbase, x-1, y-1, s, b);
  printString(fontbase, x, y, s, c);
}

int
calc_width(XFontStruct* font_struct, char* str)
{
  XCharStruct overall;
  int ascent, descent;
  int dir;
  XTextExtents(font_struct, str, (int)strlen(str), &dir,
	       &ascent, &descent, &overall);
  if (overall.width < 20) return 50;
  else return overall.width;
}

#define PS(str) \
displayShadowText(fontbase, x, y, str, c); y-=dy; \
width=calc_width(font_struct, str); \
maxwidth=width>maxwidth?width:maxwidth;

void
GGT::draw_labels(XFontStruct* font_struct, GLuint fontbase,
		 int& column, int dy, int top)
{
  int x=column;
  int y=top;
  int maxwidth=0;
  int width;
  Color c(0,1,0);
  PS("");
  PS("Number of Rays");
  PS("Hit background");
  PS("Reflection rays");
  PS("Tranparency rays");
  PS("Shadow rays");
  PS("Rays in shadow");
  PS("Shadow cache tries");
  PS("Shadow cache misses");
  PS("BV intersections");
  PS("BV primitive intersections");
  PS("Light BV intersections");
  PS("Light BV prim intersections");
  PS("Sphere intersections");
  PS("Sphere hits");
  PS("Sphere light intersections");
  PS("Sphere light hits");
  PS("Sphere light hit penumbra");
  PS("Tri intersections");
  PS("Tri hits");
  PS("Tri light intersections");
  PS("Tri light hits");
  PS("Tri light hit penumbra");
  PS("Rect intersections");
  PS("Rect hits");
  PS("Rect light intersections");
  PS("Rect light hits");
  PS("Rect light hit penumbra");
  y-=dy;
  PS("Rays/second");
  PS("Rays/second/processor");
  PS("Rays/pixel");
  column+=maxwidth;
} // end draw_labels()

#define PN(n) \
sprintf(buf, "%d", n); \
width=calc_width(font_struct, buf); \
displayShadowText(fontbase, x-width-w2, y, buf, c); y-=dy;

#define PD(n) \
sprintf(buf, "%g", n); \
width=calc_width(font_struct, buf); \
displayShadowText(fontbase, x-width-w2, y, buf, c); y-=dy;

#define PP(n, den) \
if(den==0) \
percent=0; \
else \
percent=100.*n/den; \
sprintf(buf, "%d", n); \
width=calc_width(font_struct, buf); \
displayShadowText(fontbase, x-width-w2, y, buf, c); \
sprintf(buf, " (%4.1f%%)", percent); \
displayShadowText(fontbase, x-w2, y, buf, c); \
y-=dy;
  
void
GGT::draw_column(XFontStruct* font_struct,
		 GLuint fontbase, char* heading, DepthStats& sum,
		 int x, int w2, int dy, int top,
		 bool first/*=false*/, double dt/*=1*/, int nworkers/*=0*/,
		 int npixels/*=0*/)
{
  char buf[100];
  int y=top;
  Color c(0,1,0);
  double percent;

  int width=calc_width(font_struct, heading);
  displayShadowText(fontbase, x-width-w2, y, heading, Color(1,0,0)); y-=dy;
  
  PN(sum.nrays);
  PP(sum.nbg, sum.nrays);
  PP(sum.nrefl, sum.nrays);
  PP(sum.ntrans, sum.nrays);
  PN(sum.nshadow);
  PP(sum.inshadow, sum.nshadow);
  PP(sum.shadow_cache_try, sum.nshadow);
  PP(sum.shadow_cache_miss, sum.shadow_cache_try);
  PN(sum.bv_total_isect);
  PP(sum.bv_prim_isect, sum.bv_total_isect);
  PN(sum.bv_total_isect_light);
  PP(sum.bv_prim_isect_light, sum.bv_total_isect_light);
  
  PN(sum.sphere_isect);
  PP(sum.sphere_hit, sum.sphere_isect);
  PN(sum.sphere_light_isect);
  PP(sum.sphere_light_hit, sum.sphere_light_isect);
  PP(sum.sphere_light_penumbra, sum.sphere_light_isect);
  
  PN(sum.tri_isect);
  PP(sum.tri_hit, sum.tri_isect);
  PN(sum.tri_light_isect);
  PP(sum.tri_light_hit, sum.tri_light_isect);
  PP(sum.tri_light_penumbra, sum.tri_light_isect);
  
  PN(sum.rect_isect);
  PP(sum.rect_hit, sum.rect_isect);
  PN(sum.rect_light_isect);
  PP(sum.rect_light_hit, sum.rect_light_isect);
  PP(sum.rect_light_penumbra, sum.rect_light_isect);
  if(first){
    y-=dy;
    double rps=sum.nrays/dt;
    PD(rps);
    double rpspp=rps/nworkers;
    PD(rpspp);
    double rpp=sum.nrays/(double)npixels;
    PD(rpp);
  }
} // end draw_column()


void
GGT::drawpstats(Stats* mystats, int nworkers, vector<Worker*> & workers,
		bool draw_framerate, int showing_scene,
		GLuint fontbase, double& lasttime,
		double& cum_ttime, double& cum_dt)
{
  double thickness=.3;
  double border=.5;

  double mintime=1.e99;
  double maxtime=0;
  for(int i=0;i<nworkers;i++){
    Stats* stats=workers[i]->get_stats(showing_scene);
    int nstats=stats->nstats();
    if(stats->time(0)<mintime)
      mintime=stats->time(0);
    if(stats->time(nstats-1)>maxtime)
      maxtime=stats->time(nstats-1);
  }
  double maxworker=maxtime;
  if(mystats->time(0)<mintime)
    mintime=mystats->time(0);
  int nstats=mystats->nstats();
  if(mystats->time(nstats-1)>maxtime)
    maxtime=mystats->time(nstats-1);
    
  if(draw_framerate){
    char buf[100];
    double total_dt=0;
    for(int i=0;i<nworkers;i++){
      Stats* stats=workers[i]->get_stats(showing_scene);
      int nstats=stats->nstats();
      double dt=maxtime-stats->time(nstats-1);
      total_dt+=dt;
    }
    double ttime=(maxtime-lasttime)*nworkers;
    double imbalance=total_dt/ttime*100;
    cum_ttime+=ttime;
    cum_dt+=total_dt;
    double cum_imbalance=cum_dt/cum_ttime*100;
    sprintf(buf, "%5.1fms  %5.1f%% %5.1f%%", (maxtime-maxworker)*1000,
	    imbalance, cum_imbalance);
    printString(fontbase, 80, 3, buf, Color(0,1,0));
    lasttime=maxtime;
  }
  //cerr << mintime << " " << maxworker << " " << maxtime << " " << (maxtime-maxworker)*1000 << '\n';

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(mintime, maxtime, -border, nworkers+2+border);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
	    
  for(int i=0;i<nworkers;i++){
    Stats* stats=workers[i]->get_stats(showing_scene);
    int nstats=stats->nstats();
    double tlast=stats->time(0);
    for(int j=1;j<nstats;j++){
      double tnow=stats->time(j);
      glColor3dv(stats->color(j));
      glRectd(tlast, i+1, tnow, i+thickness+1);
      tlast=tnow;
    }
  }

  double tlast=mystats->time(0);
  int i=nworkers+1;
  for(int j=1;j<nstats;j++){
    double tnow=mystats->time(j);
    glColor3dv(mystats->color(j));
    glRectd(tlast, i, tnow, i+thickness);
    tlast=tnow;
  }	
} // end drawpstats()

void
GGT::drawrstats(int nworkers, vector<Worker*> & workers,
		int showing_scene,
		GLuint fontbase, int xres, int yres,
		XFontStruct* font_struct,
		int left, int up,
		double dt)
{
  DepthStats sums[MAXDEPTH];
  DepthStats sum;
  bzero(sums, sizeof(sums));
  bzero(&sum, sizeof(sum));
  int md=0;
  for(int i=0;i<nworkers;i++){
    int depth=md;
    while(depth<MAXDEPTH){
      Stats* st=workers[i]->get_stats(showing_scene);
      if(st->ds[depth].nrays==0)
	break;
      depth++;
    }
    md=depth;
  }

  for(int i=0;i<nworkers;i++){
    for(int depth=0;depth<md;depth++){
      Stats* st=workers[i]->get_stats(showing_scene);
      sums[depth].addto(st->ds[depth]);
    }
  }
  for(int depth=0;depth<md;depth++){
    sum.addto(sums[depth]);
  }

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, xres, 0, yres);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0.375, 0.375, 0.0);

  XCharStruct overall;
  int ascent, descent;
  char* str="123456789 (100%)";
  int dir;
  XTextExtents(font_struct, str, (int)strlen(str), &dir, &ascent, &descent, &overall);
  int dy=ascent+descent;
  if (dy == 0) dy=15;
  int dx=overall.width;
  if (dx == 0) dx=175;
  int column=3-left;
  int top=yres-3-dy+up;
  char* str2="(100%)";
  XTextExtents(font_struct, str2, (int)strlen(str2), &dir, &ascent, &descent, &overall);
  int w2=overall.width;
  draw_labels(font_struct, fontbase, column, dy, top);
  column+=dx;
  draw_column(font_struct, fontbase, "Total", sum, column, w2, dy, top,
  	      true, dt, nworkers, xres*yres);
  column+=dx;
  for(int depth=0;depth<md;depth++){
    char buf[20];
    sprintf(buf, "%d", depth);
    draw_column(font_struct, fontbase, buf, sums[depth], column, w2, dy, top);
    column+=dx;
  }
} // end draw_rstats()

#endif ////////////  Block of code from Gui.cc that needs to be here
