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

#include <Packages/rtrt/Core/Ball.h>
#include <Packages/rtrt/Core/BallMath.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Grid2.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/MusilRNG.h>
#include <Packages/rtrt/Core/Scene.h>

#include <Packages/rtrt/visinfo/visinfo.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Transform.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Core/Math/MiscMath.h>

#include <iostream>

#include <unistd.h>
#include <strings.h>

#include <sys/time.h>

/* #include <SpeedShop/api.h> */

using namespace rtrt;
using namespace SCIRun;

using std::endl;
using std::cerr;

//Mutex rtrt::cameralock("Frameless Camera Synch lock");

Mutex rtrt::xlock("X windows startup lock");

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
static Transform prev_trans;

static double   last_frame = SCIRun::Time::currentSeconds();
static int      frame = 0;
static int      rendering_scene = 0;
static MusilRNG rng;
//static Object * obj;

static double   lightoff_frame = -1.0;

double _HOLO_STATE_=1;

//static float float_identity[4][4] = { {1,0,0,0}, {0,1,0,0},
//	 			        {0,0,1,0}, {0,0,0,1} };

//////////////////////////////////////////////////////////////////

Dpy::Dpy( Scene* scene, char* criteria1, char* criteria2,
	  int nworkers, bool bench, int ncounters, int c0, int c1,
	  float, float, bool display_frames, 
	  int pp_size, int scratchsize, bool fullscreen, int frameless )
  : scene(scene), criteria1(criteria1), criteria2(criteria2),
    nworkers(nworkers), bench(bench), ncounters(ncounters),
    c0(c0), c1(c1), frameless(frameless),synch_frameless(0),
    display_frames(display_frames), stealth_(NULL),
    showImage_(NULL), doAutoJitter_(false), doJitter_(false),
    showLights_( false ), lightsShowing_( false ),
    turnOffAllLights_( false ), turnOnAllLights_( false ),
    turnOnLight_( false ), turnOffLight_( false ),
    attachedObject_(NULL), turnOnTransmissionMode_(false), 
    numThreadsRequested_(nworkers), changeNumThreads_(false),
    pp_size_(pp_size), scratchsize_(scratchsize),
    toggleRenderWindowSize_(false), renderWindowSize_(1),
    fullScreenMode_( fullscreen ), holoToggle_(false)
{
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

  priv->followPath = false;

  shadowMode_ = scene->shadow_mode;
  ambientMode_ = scene->ambient_mode;

  //obj=scene->get_object();
  priv->maxdepth=scene->maxdepth;

  workers_.resize( nworkers );

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
}

Dpy::~Dpy()
{
  delete stealth_;
  delete objectStealth_;
}

void
Dpy::register_worker(int i, Worker* worker)
{
  cout << "registering worker " << i << "-" << worker << "\n";
  workers_[i] = worker;
}

int Dpy::get_num_procs() {
  return nworkers;
}

void
Dpy::run()
{
  io_lock_.lock();
  cerr << "display is pid " << getpid() << '\n';
  io_lock_.unlock();

  if(ncounters)
    counters=new Counters(ncounters, c0, c1);
  else
    counters=0;

  ////////////////////////////////////////////////////////////
  // open the display
  priv->xres = scene->get_image(0)->get_xres();
  priv->yres = scene->get_image(0)->get_yres();

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
  priv->stereo=false;

  double benchstart=0;
  int frame=0;
  for(;;)
    {
      if (frameless) { renderFrameless(); }
      else           { renderFrame();     }

      if(bench){
	if(frame==10){
	  cerr << "Warmup done, starting bench\n";
	  benchstart=SCIRun::Time::currentSeconds();
	} else if(frame == 110){
	  double dt=SCIRun::Time::currentSeconds()-benchstart;
	  cerr << "Benchmark completed in " <<  dt << " seconds ("
	       << (frame-10)/dt << " frames/second)\n";
	  Thread::exitAll(0);
	}
      }

      // Exit if you are supposed to.
      if (scene->get_rtrt_engine()->stop_execution()) {
	cout << "Dpy going down\n";
	Thread::exit();
      }
      frame++;
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
    scene->transmissionMode_ = true;
  } else {
    scene->transmissionMode_ = false;
  }

  if( showLights_ && !lightsShowing_ ){
    scene->renderLights( true );
    lightsShowing_ = true;
  } else if( !showLights_ && lightsShowing_ ){
    scene->renderLights( false );
    lightsShowing_ = false;
  }

  if ( _HOLO_STATE_<1 && !holoToggle_) {
    _HOLO_STATE_ += .1;
    if (_HOLO_STATE_>1) _HOLO_STATE_=1;
  } else if ( _HOLO_STATE_>0 && holoToggle_ ) {
    _HOLO_STATE_ -= .1;
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

  if( doJitter_ ) { scene->rtrt_engine->do_jitter = true; }
  else            { scene->rtrt_engine->do_jitter = false;  }

  if(animate && scene->animate) {
    Array1<Object*> & objects = scene->animateObjects_;
    BBox bbox1,bbox2;
    for( int num = 0; num < objects.size(); num++ ) {
      bbox1.reset();
      bbox2.reset();
      objects[num]->compute_bounds(bbox1,1E-5);
      objects[num]->animate(SCIRun::Time::currentSeconds(), changed);
      Grid2 *anim_grid = objects[num]->get_anim_grid();
      if (anim_grid) {
        objects[num]->compute_bounds(bbox2, 1E-5);
        anim_grid->remove(objects[num],bbox1);
        anim_grid->insert(objects[num],bbox2);
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

  if( priv->showing_scene == 0 )
    {
      // Only create new threads if showing_scene is 0.  This will
      // create them in sync with the Dpy thread.
      if( nworkers != numThreadsRequested_ ) {

	changeNumThreads_ = true;
	for( int cnt = 0; cnt < nworkers; cnt++ ){ // Tell all workers sync up
	  workers_[cnt]->syncForNumThreadChange( nworkers );
	}

	if( nworkers > numThreadsRequested_ ) {

	  int numToRemove = nworkers - numThreadsRequested_;

	  for( int cnt = 1; cnt <= numToRemove; cnt++ )
	    {
	      Worker * worker = workers_[ nworkers - cnt ];
	      workers_.pop_back();
	      cout << "worker " << nworkers - cnt << " told to stop!\n";
	      // Tell the worker to stop.
	      workers_[nworkers-cnt]->syncForNumThreadChange( nworkers, true );
	    }
	  // remove excess threads
	  cout << "done removing threads\n";
	}
      }
    } // end if showing_scene == 0 (thread number change code)

  return changed;
}

void
Dpy::renderFrameless() {
  cout << "can't do frameless right now.\n";
  return;
} // end renderFrameless()

void
Dpy::renderFrame() {

  bool  & stereo        = priv->stereo;  
  int   & showing_scene = priv->showing_scene;
  int     counter       = 1;

  frame++;
    
  // If we need to change the number of worker threads:
  if( changeNumThreads_ ) {
    cout << "changeNumThreads\n";
    int oldNumWorkers = nworkers;
    nworkers = numThreadsRequested_;

    if( oldNumWorkers < numThreadsRequested_ ) { // Create more workers
      int numNeeded     = numThreadsRequested_ - oldNumWorkers;
      int stopAt        = oldNumWorkers + numNeeded;

      workers_.resize( numThreadsRequested_ );
      for( int cnt = oldNumWorkers; cnt < stopAt; cnt++ ) {
	char buf[100];
	sprintf(buf, "worker %d", cnt);
	Worker * worker = new Worker(this, scene, cnt,
				     pp_size_, scratchsize_,
				     ncounters, c0, c1);

	cout << "created worker: " << cnt << ", " << worker << "\n";
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
    cout << "sync with workers for change: " << oldNumWorkers+1 << "\n";
    addSubThreads_->wait( oldNumWorkers + 1 );
    changeNumThreads_ = false;
  }

  barrier->wait(nworkers+1);

  drawstats[showing_scene]->add(SCIRun::Time::currentSeconds(),Color(1,0,0));

  bool changed = true;

  Camera * cam1 = scene->get_camera(rendering_scene);

  if( *cam1 != *guiCam_ ){ *cam1 = *guiCam_; }

  changed = checkGuiFlags();

  scene->refill_work(rendering_scene, numThreadsRequested_);

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

  barrier->wait(nworkers+1);

  Image * displayedImage = scene->get_image(showing_scene);
  // Tell Gui thread to display the image showImage_

  //if( showImage_ != NULL ){
  //  cout << "Warning, gui has not displayed previous frame!\n";
  //}

  if(!bench)
    showImage_ = displayedImage;

  // dump the frame and quit for now
  if (counter == 0) {
    if (!display_frames) {
      displayedImage->save("displayless.raw");
      cerr <<"Wrote frame to displayless.raw\n";
    }
  }
  counter--;

  // Wait until the Gui (main) thread has displayed this image...
  //priv->waitDisplay->lock();

  if( displayedImage->get_xres() != priv->xres ||
      displayedImage->get_yres() != priv->yres ) {
    delete displayedImage;
    displayedImage = new Image(priv->xres, priv->yres, stereo);
    scene->set_image(showing_scene, displayedImage);
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
    for(int i=0;i<nworkers;i++){
      fprintf(stderr, "%12lld", workers_[i]->get_counters()->count0());
    }
    fprintf(stderr, "\n");
    if(ncounters>1){
      fprintf(stderr, "%2d: %12lld", c1, counters->count1());	
      for(int i=0;i<nworkers;i++){
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
    if (display_frames) {
      if( !priv->followPath ) {
	guiCam_->updatePosition( *stealth_, scene, ppc );
      } else {
	guiCam_->followPath( *stealth_ );
      }
    }
    st->add(SCIRun::Time::currentSeconds(), Color(0,1,1));

    rendering_scene=1-rendering_scene;
    showing_scene=1-showing_scene;
  }
} // end renderFrame()

void
Dpy::get_barriers( Barrier *& mainBarrier, Barrier *& addSubThreads )
{
  mainBarrier   = barrier;
  addSubThreads = addSubThreads_;
}

