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

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

#include "FontString.h"
/* #include <SpeedShop/api.h> */

using namespace rtrt;
using namespace SCIRun;

using std::endl;
using std::cerr;

//Mutex rtrt::cameralock("Frameless Camera Synch lock");

Mutex rtrt::xlock("X windows startup lock");

Mutex io("io lock");

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


static double    eye_dist = 0;
static double    prev_time[3]; // history for quaternions and time
static HVect     prev_quat[3];
static Transform prev_trans;

static double   last_frame = SCIRun::Time::currentSeconds();
static int      frame = 0;
static int      rendering_scene = 0;
static MusilRNG rng;
static Object * obj;

//static float float_identity[4][4] = { {1,0,0,0}, {0,1,0,0},
//	 			        {0,0,1,0}, {0,0,0,1} };

//////////////////////////////////////////////////////////////////

Dpy::Dpy(Scene* scene, char* criteria1, char* criteria2,
	 int nworkers, bool bench, int ncounters, int c0, int c1,
	 float, float, bool display_frames, 
	 int pp_size, int scratchsize, int frameless)
  : scene(scene), criteria1(criteria1), criteria2(criteria2),
    nworkers(nworkers), bench(bench), ncounters(ncounters),
    c0(c0), c1(c1), frameless(frameless),synch_frameless(0),
    display_frames(display_frames), stealth_(NULL),
    showImage_(NULL), doAutoJitter_(false), doJitter_(false),
    showLights_( false ), lightsShowing_( false ),
    turnOffAllLights_( false ), turnOnAllLights_( false ),
    turnOnLight_( false ), turnOffLight_( false ),
    attachedObject_(NULL)
{
  ppc = new PerProcessorContext( pp_size, scratchsize );

  for( int i = 0; i < 4; i++ )
    for( int j = 0; j < 4; j++ )
      if( i == j )
	objectRotationMatrix_[i][j] = 1;
      else
	objectRotationMatrix_[i][j] = 0;

  //  barrier=new Barrier("Frame end", nworkers+1);
  barrier=new Barrier("Frame end");
  workers=new Worker*[nworkers];
  drawstats[0]=new Stats(1000);
  drawstats[1]=new Stats(1000);
  priv = new DpyPrivate;

  //priv->waitDisplay = new Mutex( "wait for display" );
  //priv->waitDisplay->lock();

  priv->followPath = false;

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

void Dpy::register_worker(int i, Worker* worker)
{
  workers[i]=worker;
}

int Dpy::get_num_procs() {
  return nworkers;
}

void
Dpy::run()
{
  if( bench ) {
    cerr << "Bench is currently not supported.\n";
    Thread::exitAll(0);    
  }

  io.lock();
  cerr << "display is pid " << getpid() << '\n';
  io.unlock();

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

  priv->draw_pstats=false;
  priv->draw_rstats=false;
  shadowMode_ = scene->shadow_mode;
  ambientMode_ = scene->ambient_mode;
  obj=scene->get_object();

  priv->showing_scene=1;
  priv->animate=true;
  priv->maxdepth=scene->maxdepth;
  priv->base_threshold=0.005;
  priv->full_threshold=0.01;
  priv->left=0;
  priv->up=0;

  priv->exposed=true;
  priv->stereo=false;

  for(;;)
    {
      if (frameless) { renderFrameless(); }
      else           { renderFrame();     }

      // Exit if you are supposed to.
      if (scene->get_rtrt_engine()->stop_execution()) {
	cout << "Dpy going down\n";
	Thread::exit();
      }
    }
} // end run()

void
Dpy::renderFrameless() {
  cout << "can't do frameless right now.\n";
  return;
} // end renderFrameless()

void
Dpy::renderFrame() {

  bool  & stereo= priv->stereo;  
  bool  & animate= priv->animate;
  int   & maxdepth= priv->maxdepth;
  float & base_threshold= priv->base_threshold;
  float & full_threshold= priv->full_threshold;
  bool  & draw_pstats= priv->draw_pstats;
  bool  & draw_rstats= priv->draw_rstats;
  int   & left = priv->left;
  int   & up   = priv->up;
  int   & showing_scene= priv->showing_scene;
  int     counter = 1;

  frame++;
    
  barrier->wait(nworkers+1);

  //drawstats[showing_scene]->add(SCIRun::Time::currentSeconds(), Color(1,0,0));

  if( doJitter_ ) { scene->rtrt_engine->do_jitter = true; }
  else            { scene->rtrt_engine->do_jitter = false;  }

  scene->refill_work(rendering_scene, nworkers);
  //bool changed=false;
  bool changed=true;

  Camera* cam1=scene->get_camera(rendering_scene);

  if( *cam1 != *guiCam_ ){
    //cout << "updating cam " << cam1 << "\n";
    *cam1 = *guiCam_;
  }

  if(animate && scene->animate) {
    Array1<Object*> & objects = scene->animateObjects_;
    for( int num = 0; num < objects.size(); num++ )
      obj->animate(SCIRun::Time::currentSeconds(), changed);
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

  //drawstats[showing_scene]->add(SCIRun::Time::currentSeconds(),Color(0,1,0));

  if( showLights_ && !lightsShowing_ ){
    cout << "show lights\n";
    scene->renderLights( true );
    lightsShowing_ = true;
  } else if( !showLights_ && lightsShowing_ ){
    cout << "don't show lights\n";
    scene->renderLights( false );
    lightsShowing_ = false;
  }

  if( turnOffAllLights_ ){
    scene->turnOffAllLights();
    turnOffAllLights_ = false;
  }
  if( turnOnAllLights_ ){
    scene->turnOnAllLights();
    turnOnAllLights_ = false;
  }
  if( turnOnLight_ ) {
    scene->turnOnLight( turnOnLight_ );
    turnOnLight_ = NULL;
  }
  if( turnOffLight_ ) {
    scene->turnOffLight( turnOffLight_ );
    turnOffLight_ = NULL;
  }

  barrier->wait(nworkers+1);

  Image * displayedImage = scene->get_image(showing_scene);
  // Tell Gui thread to display the image showImage_

  //if( showImage_ != NULL ){
  //  cout << "Warning, gui has not displayed previous frame!\n";
  //}

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
  //drawstats[showing_scene]->add(SCIRun::Time::currentSeconds(),Color(0,1,1));
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
      fprintf(stderr, "%12lld", workers[i]->get_counters()->count0());
    }
    fprintf(stderr, "\n");
    if(ncounters>1){
      fprintf(stderr, "%2d: %12lld", c1, counters->count1());	
      for(int i=0;i<nworkers;i++){
	fprintf(stderr, "%12lld", workers[i]->get_counters()->count1());
      }
      fprintf(stderr, "\n\n");
    }
  }

  // color blue
  st->add(SCIRun::Time::currentSeconds(), Color(0,0,1));
  //      st->add(SCIRun::Time::currentSeconds(), Color(0.4,0.2,1));

  if (display_frames) {
    // color 
    st->add(SCIRun::Time::currentSeconds(), Color(1,0.5,0));
    // color light blue
    st->add(SCIRun::Time::currentSeconds(), Color(0.5,0.5,1));
    // color grey
    st->add(SCIRun::Time::currentSeconds(), Color(0.5,0.5,0.5));
    st->add(SCIRun::Time::currentSeconds(), Color(1,0,1));
    st->add(SCIRun::Time::currentSeconds(), Color(1,0.5,0.3));
      
    double curtime=SCIRun::Time::currentSeconds();
    double dt=curtime-scene->lasttime;

    scene->lasttime=curtime;

    // If keypad is attached to an object.
    if( attachedObject_ ) {
//      attachedObject_->updatePosition( objectStealth_, guiCam_ );
      attachedObject_->updateNewTransform( objectRotationMatrix_ );
    }

    if (display_frames) {
      if( !priv->followPath ) {
	guiCam_->updatePosition( *stealth_, scene, ppc );
      } else {
	guiCam_->followPath( *stealth_ );
      }

    }

    st->add(SCIRun::Time::currentSeconds(), Color(1,0,0));
    rendering_scene=1-rendering_scene;
    showing_scene=1-showing_scene;
  }
} // end renderFrame()

Barrier*
Dpy::get_barrier()
{
  return barrier;
}

