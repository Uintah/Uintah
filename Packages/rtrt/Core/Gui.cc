
#include <sci_defs/ogl_defs.h> // For HAVE_OOGL
#if defined(HAVE_OOGL)
#undef HAVE_OOGL
#endif
#include <Packages/rtrt/Core/Gui.h>
#include <Packages/rtrt/Core/Dpy.h>
#include <Packages/rtrt/Core/DpyPrivate.h>
#include <Packages/rtrt/Core/DpyGui.h>
#include <Packages/rtrt/Core/Stealth.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Ball.h>
#include <Packages/rtrt/Core/BallMath.h>
#include <Packages/rtrt/Core/Stealth.h>
#include <Packages/rtrt/Core/Names.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Image.h>
#include <Packages/rtrt/Core/CycleMaterial.h>
#include <Packages/rtrt/Core/SketchMaterialBase.h>
#include <Packages/rtrt/Core/FontString.h>
#include <Packages/rtrt/Core/DynamicInstance.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/PPMImage.h>
#include <Packages/rtrt/Core/params.h>
#include <Packages/rtrt/Core/SelectableGroup.h>
#include <Packages/rtrt/Core/SpinningInstance.h>
#include <Packages/rtrt/Core/CutGroup.h>
#include <Packages/rtrt/Core/PPMImage.h>
#include <Packages/rtrt/Core/Trigger.h>
#include <Packages/rtrt/Core/VolumeVis2D.h>
#include <Packages/rtrt/Core/MouseCallBack.h>
#include <Packages/rtrt/Core/rtrt.h>

#if !defined(linux) && !defined(__APPLE__)
#  include <Packages/rtrt/Sound/SoundThread.h>
#  include <Packages/rtrt/Sound/Sound.h>
#endif

#include <Core/Math/Trig.h>
#include <Core/Thread/Time.h>
#include <Core/Geometry/Transform.h>

#include <GL/glut.h>
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1430
#pragma set woff 3201
#pragma set woff 1375
#endif
#include <glui.h>
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1430
#pragma reset woff 3201
#pragma reset woff 1375
#endif

#include <unistd.h>  // for sleep
#include <strings.h> // for bzero
#include <errno.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

// From Glyph.cc
namespace rtrt {
  extern float glyph_threshold;
}
  
//oogl
ShadedPrim   * backgroundTexQuad; // from rtrt.cc
BasicTexture * backgroundTex;     // from rtrt.cc

////////////////////////////////////////////

extern "C" Display *__glutDisplay;
extern "C" Window** __glutWindowList;

namespace rtrt {
  double ORBIT_SPEED  = 0;
  double ROTATE_SPEED = 1;
}

using namespace rtrt;
using namespace SCIRun;
using namespace std;

static GGT * activeGGT = 0;

PPMImage * livingRoomImage = NULL;
PPMImage * scienceRoomImage = NULL;
PPMImage * museumRoomImage = NULL;
PPMImage * underwaterRoomImage = NULL;
PPMImage * galaxyRoomImage = NULL;

static double    prev_time[3]; // history for quaternions and time
static HVect     prev_quat[3];
static Transform prev_trans;

// GLUI Component IDs
#define TURN_ON_JITTER_BTN        50

#define WHITE_BG_BUTTON           52
#define BLACK_BG_BUTTON           53
#define ORIG_BG_BUTTON            54

#define SENSITIVITY_SPINNER_ID    60
#define FOV_SPINNER_ID            61
#define DEPTH_SPINNER_ID          62

#define LIGHT_LIST_ID            100
#define LIGHTS_BUTTON_ID         101
#define TOGGLE_LIGHT_SWITCHES_ID 102
#define TOGGLE_SHOW_LIGHTS_ID    103
#define LIGHT_X_POS_ID           104
#define LIGHT_Y_POS_ID           105
#define LIGHT_Z_POS_ID           106

#define ROUTE_LIST_ID            110
#define ROUTE_BUTTON_ID          111
#define SAVE_ROUTE_BUTTON_ID     112
#define LOAD_ROUTE_BUTTON_ID     113
#define ADD_TO_ROUTE_BTN_ID      114
#define CLEAR_ROUTE_BTN_ID       115
#define GO_TO_NEXT_MARK_BTN_ID   116
#define GO_TO_PREV_MARK_BTN_ID   117
#define TRAVERSE_ROUTE_BTN_ID    118
#define GO_TO_RTE_BEGIN_BTN_ID   119
#define NEW_ROUTE_BUTTON_ID      120
#define DELETE_MARKER_BTN_ID     121

#define CLOSE_GETSTRING_BTN      130

#define OBJECT_LIST_ID           140
#define OBJECTS_BUTTON_ID        141
#define MATERIALS_BUTTON_ID      142
#define SOUNDS_BUTTON_ID         143
#define ATTACH_KEYPAD_BTN_ID     144
#define SICYCLE_BTN_ID           145

#define SOUND_LIST_ID            150

#define TOGGLE_HOTSPOTS_ID          190
#define TOGGLE_TRANSMISSION_MODE_ID 191
#define SOUND_VOLUME_SPINNER_ID     192
#define NUM_THREADS_SPINNER_ID      193

#define START_BENCH_BTN 200
#define STOP_BENCH_BTN 201

// GLUT MENU ITEM IDS

#define TOGGLE_GUI                 1
#define TOGGLE_HOT_SPOTS           2
#define TOGGLE_RIGHT_BUTTON_MENU   3
#define QUIT_MENU_ID               4

// Defines how much you can manually increase/decrease movement
// controls sensitivity.
#define MIN_SENSITIVITY      0.1
#define MAX_SENSITIVITY    200.0

#define MIN_FOV   0.1
#define MAX_FOV  160.0

// Used for loading in routes.  (Is monotonicly increasing)
static int routeNumber = 0;

// Callback global variable
  
// These are to help callbacks.
namespace rtrt {
  
class CallbackInfo {
public:
  int callback_id;
};

class SGCallbackInfo: public CallbackInfo {
public:
  SelectableGroup *sg;
  int current_frame;
  GLUI_Spinner *current_frame_spinner;
};

} // end namespace rtrt

static std::vector<CallbackInfo*> callback_info_list;


// If someone accidentally types in a huge number to the number of
// threads spinner, the machine goes into a fit.  If possible, this
// number should be set dynamically based on the number of processors
// on the machine that rtrt is running on.
#define MAX_NUM_THREADS 120

GGT::GGT() :
  on_death_row(false), opened(false), mainWindowID(-1),
  activeMTT_(NULL), queuedMTT_(NULL),
  bottomGraphicTrig_(NULL), leftGraphicTrig_(NULL),
  visWomanTrig_(NULL), csafeTrig_(NULL), geophysicsTrig_(NULL),
  recheckBackgroundCnt_(10),
  selectedLightId_(0), selectedRouteId_(0), selectedObjectId_(0),
  selectedSoundId_(0), selectedTriggerId_(-1),
  routeWindowVisible(false), lightsWindowVisible(false), 
  objectsWindowVisible(false), materialsWindowVisible(false),
  soundsWindowVisible(false), triggersWindowVisible(false),
  mainWindowVisible(true),
  enableSounds_(false),
  mouseDown_(0), rightButtonMenuActive_(false), beQuiet_(true),
  displayRStats_(false), displayPStats_(false),
  lightsOn_(true), lightsBeingRendered_(false),
  lightList(NULL),
  r_color_spin(NULL), g_color_spin(NULL), b_color_spin(NULL), 
  lightIntensity_(NULL), 
  lightBrightness_(1.0),
  routeList(NULL),
  soundList_(NULL)
{
  inputString_[0] = 0;

  setActiveGGT(this);
}

GGT::~GGT()
{
  cleanup();
  setActiveGGT(0);
}

void GGT::run() {
  DpyBase::xlock();
  printf("before glutInit\n");
  char *argv = "GGT Thread run";
  int argc = 1;
  glutInit( &argc, &argv );
  printf("after glutInit\n");
    
  // Initialize GLUT and GLUI stuff.
  printf("start glut inits\n");
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
  glutInitWindowSize(400, 545 );
  glutInitWindowPosition( 400, 0 );
    
  mainWindowID = glutCreateWindow("GG Controls");

  glut_dpy = __glutDisplay;
  // This is an ugly, cheaty way of getting the window id out of glut...
  glut_win = __glutWindowList[mainWindowID-1][1];
  cerr << "initial win = "<<glut_win<<"\n";

  // Setup callback functions
  glutDisplayFunc( GGT::displayCB );

#if 0
  //////////////////////////////////////////////////////////////////
    
  cout << "sb: " << glutDeviceGet( GLUT_HAS_SPACEBALL ) << "\n";
    
  glutKeyboardFunc( Gui::handleKeyPressCB );
  glutSpecialFunc( Gui::handleSpecialKeyCB );
  glutMouseFunc( Gui::handleMouseCB );
  glutMotionFunc( Gui::handleMouseMotionCB );
  glutSpaceballMotionFunc( Gui::handleSpaceballMotionCB );
  glutSpaceballRotateFunc( Gui::handleSpaceballRotateCB );
  glutSpaceballButtonFunc( Gui::handleSpaceballButtonCB );
    
  glutReshapeFunc( Gui::handleWindowResizeCB );
  glutDisplayFunc( Gui::redrawBackgroundCB );
#endif

  // Must do this after glut is initialized.
  createMenus(mainWindowID);

  opened = true;

  addSceneLights();
  
  DpyBase::xunlock();
  printf("end glut inits\n");
    
  glutMainLoop();
}

void
GGT::cleanup() {
  if (!opened) return;
  else opened = false;
  
  DpyBase::xlock();
  GLUI_Master.close_all();
  cerr << "GLUI_Master.close_all finished\n";
  glutDestroyWindow(mainWindowID);
  cerr << "glutDestroyWindow finished\n";
  XCloseDisplay(glut_dpy);
  cerr << "XCloseDisplay for GGT finished\n";
  DpyBase::xunlock();
}

void
GGT::stop() {
  cerr << "GGT::stop called\n";
  on_death_row = true;
}

void
GGT::quit(int all)
{
  if (all) {
    cerr << "Quitting rtrt.\n";
    // Stop threads...This will eventually call stop().
    activeGGT->rtrt_dpy->rtrt_engine->stop_engine();
  } else {
    // Remove yourself from the DpyGui list.
    dpygui->removeExternalUIInterface(this);
    stop();
  }
}

void
GGT::setDpyGui(DpyGui* new_gui) {
  dpygui = new_gui;
  dpygui->addExternalUIInterface(this);
}

void
GGT::handleMenuCB( int item )
{
  switch( item ) {
  case TOGGLE_HOT_SPOTS:
    activeGGT->toggleHotspotsCB( -1 );
    break;
  case TOGGLE_GUI:
    activeGGT->toggleGui();
    break;
  case TOGGLE_RIGHT_BUTTON_MENU:
    if( activeGGT->rightButtonMenuActive_ )
      {
	activeGGT->rightButtonMenuActive_ = false;
	glutDetachMenu(GLUT_RIGHT_BUTTON);
      }
    else
      {
	activeGGT->rightButtonMenuActive_ = true;
	glutAttachMenu(GLUT_RIGHT_BUTTON);
      }
    break;
  case QUIT_MENU_ID:
    activeGGT->quit();
    break;
  }
}

void
GGT::setActiveGGT( GGT * gui )
{
  activeGGT = gui;
}

GGT*
GGT::getActiveGGT()
{
  return activeGGT;
}

void
GGT::setStealth( Stealth * stealth )
{
  stealth_ = stealth;
}

void
GGT::setDpy( Dpy * dpy )
{
  rtrt_dpy = dpy;
  priv     = dpy->priv;
  stealth_ = dpy->stealth_;
  camera_  = dpy->guiCam_;
}

void
GGT::idleFunc() {
  // Check to see if we need to go bye bye
  if (activeGGT->on_death_row) {
    cerr << "GGT::idleFunc:: calling Thread::exit()\n";
    Thread::exit();
  }
  else
    usleep(1000);
}

void
GGT::displayCB() {
  // Do nothing
}

void
GGT::handleTriggers()
{
  // Handle Active Trigger
  if( activeGGT->activeMTT_ )
    {
      Trigger * next = NULL;
      // next is NULL if no next trigger associated with this trigger.
      bool result = activeGGT->activeMTT_->advance( next );
      if( result == false ) // done, remove from active list.
	{
	  if( activeGGT->queuedMTT_ )
	    {
	      if( next )
		{
		  double quedPriority = activeGGT->queuedMTT_->getPriority();
		  double nextPriority = next->getPriority();
		  if( quedPriority < nextPriority )
		    {
		      cout << "using 'next' trigger: " <<next->getName()<<"\n";
		      cout << " priorities: " << nextPriority << ", "
			   << quedPriority << "\n";
		      activeGGT->activeMTT_ = next;
		    }
		  else
		    {
		      activeGGT->activeMTT_ = activeGGT->queuedMTT_;
		      activeGGT->queuedMTT_ = NULL;
		    }
		}
	      else
		{
		  cout << "moving in queued trigger: " << 
		    activeGGT->queuedMTT_->getName() << "\n";
		  activeGGT->activeMTT_ = activeGGT->queuedMTT_;
		  activeGGT->queuedMTT_ = NULL;
		}
	    }
	  else
	    {
	      activeGGT->activeMTT_ = next;
	    }
	  if( activeGGT->activeMTT_ ) 
	    {
	      cout << "using next trigger: " << next->getName() << "\n";
	      activeGGT->activeMTT_->activate();
	    }
	}
    }

  // Check all triggers.
  vector<Trigger*> & triggers = rtrt_dpy->scene->getTriggers();
  for( unsigned int cnt = 0; cnt < triggers.size(); cnt++ )
    {
      Trigger * trigger = triggers[cnt];
      bool result = trigger->check( activeGGT->camera_->eye );
      if( result == true && trigger != activeGGT->activeMTT_ )
	{
	  // The trigger is in range, so determine what to do with it
	  // based on its priority and the active triggers priority.
	  if( activeGGT->activeMTT_ )
	    {
	      if( trigger == activeGGT->queuedMTT_ ) // already queued.
		continue;
	      double trigPriority = trigger->getPriority();
	      double currPriority = activeGGT->activeMTT_->getPriority();
	      if( currPriority <= trigPriority )
		{ // Tell current to stop and queue up new trigger.
		  cout << "deactivating current trigger: " <<
		    activeGGT->activeMTT_->getName() << " to start " <<
		    trigger->getName() << "\n";
		  activeGGT->activeMTT_->deactivate();
		  activeGGT->queuedMTT_ = trigger;
		}
	      else if( activeGGT->queuedMTT_ )
		{
		  double quedPriority = activeGGT->queuedMTT_->getPriority();
		  if( trigPriority > quedPriority )
		    {
		      activeGGT->queuedMTT_ = trigger;
		    }
		  
		}
	    }
	  else
	    {
	      cout << "starting " << trigger->getName() << "\n";
	      activeGGT->activeMTT_ = trigger;
	      activeGGT->activeMTT_->activate();
	    }
	}
    }

  // Deal with bottom graphic trigger
  if( activeGGT->bottomGraphicTrig_ )
    {
      Trigger * next = NULL;
      activeGGT->bottomGraphicTrig_->advance( next );
      if( next )
	{
	  activeGGT->bottomGraphicTrig_ = next;
	  next->activate();
	}
      else
	{
	  // Calling check() just to advance the time of the trigger.
	  activeGGT->bottomGraphicTrig_->check( Point(0,0,0) );
	}
    }

  // Deal with left graphic trigger
  if( activeGGT->leftGraphicTrig_ )
    {
      Trigger * next = NULL;
      activeGGT->leftGraphicTrig_->advance( next );
      if( next )
	{
	  activeGGT->leftGraphicTrig_ = next;
	  next->activate();
	}
      else
	{
	  // Calling check() just to advance the time of the trigger.
	  activeGGT->leftGraphicTrig_->check( Point(0,0,0) );
	}
    }
} // end handleTriggers()

void
GGT::drawBackground()
{
#if defined(HAVE_OOGL)
  glutSetWindow( mainWindowID );

  glViewport(0, 0, 1280, 1024);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, 1280, 0, 1024);
  glDisable( GL_DEPTH_TEST );
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0.375, 0.375, 0.0);

  backgroundTex->reset( GL_FLOAT, &((*(activeGGT->backgroundImage_))(0,0)) );
  backgroundTexQuad->draw();
#if 0   // this was for two-screen mode for the demo at SIGGRAPH
  // the following prevents flickering on the second channel
  glViewport(1280,0,1280,1024);
  backgroundTexQuad->draw();
  // reset the viewport after the flickering is fixed
  glViewport(0, 0, 1280, 1024);
#endif
#endif
}

void
GGT::redrawBackgroundCB()
{
  if( activeGGT->rtrt_dpy->fullScreenMode_ ) {
    if( !activeGGT->backgroundImage_ ) return;
    // update loop will check for a new background, see it is different,
    // and update correctly.
    activeGGT->recheckBackgroundCnt_ = 10; // forces a redraw
    activeGGT->backgroundImage_ = NULL;
  }
}

bool
GGT::setBackgroundImage( int room )
{
  PPMImage * current = backgroundImage_;

  switch( room ) {
  case 0:
    backgroundImage_ = scienceRoomImage;
    break;
  case 1:
    backgroundImage_ = livingRoomImage;
    break;
  case 2:
    backgroundImage_ = galaxyRoomImage;
    break;
  case 3:
    backgroundImage_ = museumRoomImage;
    break;
  default:
    backgroundImage_ = underwaterRoomImage;
    break;
  }

  return (current != backgroundImage_);
}

#if 0
void
GGT::idleFunc()
{
  Dpy               * dpy = activeGGT->rtrt_dpy;
  struct DpyPrivate * priv = activeGGT->priv;

  // Hacking these vars for now:
  static double lasttime  = SCIRun::Time::currentSeconds();
  static double cum_ttime = 0;
  static double cum_dt    = 0;

  if( activeGGT->enableSounds_ )
    {
      // Sound Thread has finished loading sounds and is now active...
      // ...so turn on the sound GUIs.
      activeGGT->enableSounds_ = false; 
      activeGGT->openSoundPanelBtn_->enable();
      activeGGT->soundVolumeSpinner_->enable();
      activeGGT->startSoundThreadBtn_->set_name( "Sounds Started" );
    }

  glutSetWindow( activeGGT->mainWindowID );
  activeGGT->handleTriggers();

  // I know this is a hack... 
  if( dpy->showImage_ ){

    // Display textual information on the screen:
    char buf[100];
    sprintf( buf, "%3.1lf fps", (activeGGT->priv->FrameRate) );

    bool redrawBG = false;
    if( dpy->fullScreenMode_ )
      {
	redrawBG = activeGGT->checkBackgroundWindow();
	if( redrawBG ) activeGGT->drawBackground();
      }

    dpy->showImage_->draw( dpy->renderWindowSize_, dpy->fullScreenMode_ );
    if( dpy->fullScreenMode_ )
      activeGGT->printString(fontbase, 133, 333, buf, Color(1,1,1));
    else
      activeGGT->printString(fontbase, 5, 5, buf, Color(1,1,1));

    if( dpy->fullScreenMode_ ) {
      glutSwapBuffers();
      // Need to draw into other buffer so that as we "continuously"
      // flip them, it doesn't look bad.
      if( redrawBG ) activeGGT->drawBackground();
      dpy->showImage_->draw( dpy->renderWindowSize_, dpy->fullScreenMode_ );
      activeGGT->printString(fontbase, 133, 333, buf, Color(1,1,1));
    }

    if( activeGGT->displayRStats_ )
      {
	activeGGT->drawrstats(dpy->nworkers, dpy->workers_,
			      priv->showing_scene, fontbase2, 
			      priv->xres, priv->yres,
			      fontInfo2, priv->left, priv->up,
			      0.0 /* dt */);
      }
    if( activeGGT->displayPStats_ )
      {
	Stats * mystats = dpy->drawstats[!priv->showing_scene];
	activeGGT->drawpstats(mystats, dpy->nworkers, dpy->workers_, 
			      /*draw_framerate*/true, priv->showing_scene,
			      fontbase, lasttime, cum_ttime,
			      cum_dt);
      }

    dpy->showImage_ = NULL;

    // Let the Dpy thread start drawing the next image.
    //activeGGT->priv->waitDisplay->unlock();
    if( activeGGT->mainWindowVisible ) {
      activeGGT->update(); // update the gui each time a frame is finished.
    }
    if( !dpy->fullScreenMode_ )
      glutSwapBuffers(); 
  } else {
    if( dpy->fullScreenMode_ )
      glutSwapBuffers(); 
  }
}
#endif

void
GGT::handleKeyPressCB( unsigned char key, int /*mouse_x*/, int /*mouse_y*/ )
{
  // static double FPS = 15;

  DpyPrivate * priv = activeGGT->priv;

  // int     & maxdepth       = priv->maxdepth;
  // bool    & stereo         = priv->stereo;  
  // bool    & animate        = priv->animate;
  // double  & FrameRate      = priv->FrameRate;
  // bool    & draw_pstats    = priv->draw_pstats;
  // bool    & draw_rstats    = priv->draw_rstats;
  int     & showing_scene  = priv->showing_scene;

  // int     & left           = priv->left;
  // int     & up             = priv->up;

  int mods   = glutGetModifiers();
  activeGGT->shiftDown_ = mods & GLUT_ACTIVE_SHIFT;
  activeGGT->altDown_   = mods & GLUT_ACTIVE_ALT;
  activeGGT->ctrlDown_  = mods & GLUT_ACTIVE_CTRL;

  switch( key ){

  // KEYPAD KEYS USED FOR MOVEMENT

  case '+':
    if (activeGGT->shiftDown_) {
      // increase planet orbit speed
      if (ORBIT_SPEED<.02) ORBIT_SPEED=1;
      else ORBIT_SPEED*=1.9;
      cerr << "orbit speed: " << ORBIT_SPEED << endl;
    } else if (activeGGT->ctrlDown_) {
      // increase planet rotate speed
      ROTATE_SPEED*=1.1;
    } else {
      // SPEED up or slow down
      activeGGT->stealth_->accelerate();
    }
    break;
  case '-':
    if (activeGGT->shiftDown_) {
      // decrease planet orbit speed
      if (ORBIT_SPEED<.1) ORBIT_SPEED=0;
      else ORBIT_SPEED*=.6;
      cerr << "orbit speed: " << ORBIT_SPEED << endl;
    } else if (activeGGT->ctrlDown_) {
      // decrease planet rotate speed
      ROTATE_SPEED*=.6;
    } else {
      activeGGT->stealth_->decelerate();
    }
    break;
  // PITCH up and down
  case '8':
    cout << "pitchdown\n";
    activeGGT->stealth_->pitchDown();
    break;
  case '2':
    cout << "pitchup\n";
    activeGGT->stealth_->pitchUp();
    break;
  // SLIDE left and right
  case '9':
    activeGGT->stealth_->slideRight();
    break;
  case '7':
    activeGGT->stealth_->slideLeft();
    break;
  // TURN left and right
  case '4':
    activeGGT->stealth_->turnLeft();
    break;
  case '5':    // STOP rotations (pitch/turn)
    activeGGT->stealth_->stopPitchAndRotate();
    break;
  case '6':
    activeGGT->stealth_->turnRight();
    break;
  // SLOW down and STOP
  case '.':
    activeGGT->stealth_->slowDown();
    break;
  case ' ':
  case '0':
    activeGGT->stealth_->stopAllMovement();
    break;
  // ACCELERATE UPWARDS or DOWNWARDS
  case '*': 
    activeGGT->stealth_->goUp();   // Accelerate UP
    break;
  case '/': 
    activeGGT->stealth_->goDown(); // Accelerate DOWN
    break;
  case 'q':
    activeGGT->quit();
    break;
  case 'G': // Toggle Display of Gui
    activeGGT->toggleGui();
    break;
  case 'g':
    cout << "Toggling Gravity.  If you want to increase framerate, use 'F'\n";
    activeGGT->stealth_->toggleGravity();
    break;
  case 't':
    if( activeGGT->rtrt_dpy->rtrt_engine->hotSpotsMode != RTRT::HotSpotsOff)
      activeGGT->rtrt_dpy->rtrt_engine->hotSpotsMode = RTRT::HotSpotsOff;
    else
      activeGGT->rtrt_dpy->rtrt_engine->hotSpotsMode = RTRT::HotSpotsOn;
    break;
  case 'T':
    if( activeGGT->rtrt_dpy->rtrt_engine->hotSpotsMode != RTRT::HotSpotsOff)
      activeGGT->rtrt_dpy->rtrt_engine->hotSpotsMode = RTRT::HotSpotsOff;
    else
      activeGGT->rtrt_dpy->rtrt_engine->hotSpotsMode = RTRT::HotSpotsHalfScreen;
    break;
  case 'Q':
    activeGGT->beQuiet_ = !activeGGT->beQuiet_;
    break;
  case 's':
    activeGGT->cycleShadowMode();
    break;
  case 'h':
    activeGGT->cycleAmbientMode();
    break;
  case 'z':
    handleMenuCB( TOGGLE_RIGHT_BUTTON_MENU );
    break;
  case 'v':
    {
      if(activeGGT->priv->followPath) { activeGGT->priv->followPath = false; }
      activeGGT->stealth_->stopAllMovement();

      // Animate lookat point to center of BBox...
      Object* obj= activeGGT->rtrt_dpy->scene->get_object();
      BBox bbox;
      obj->compute_bounds(bbox, 0);
      if(bbox.valid()){
	activeGGT->camera_->set_lookat(bbox.center());
        
	// Move forward/backwards until entire view is in scene...
	// change this a little, make it so that the FOV must
	// be 60 deg...
	// 60 degrees sucks - try 40...
        // Let user specify using gui.

	const double FOVtry = activeGGT->fovValue_;

	Vector diag(bbox.diagonal());
	double w=diag.length();
	Vector lookdir(activeGGT->camera_->get_lookat() -
		       activeGGT->camera_->get_eye()); 
	lookdir.normalize();
	const double scale = 1.0/(2*tan(DtoR(FOVtry/2.0)));
	double length = w*scale;
	activeGGT->camera_->set_fov(FOVtry);
	activeGGT->camera_->set_eye( activeGGT->camera_->get_lookat() -
				    lookdir*length );
	activeGGT->camera_->setup();
	activeGGT->fovSpinner_->set_float_val( FOVtry );

	Point origin;
	Vector lookdir2;
	Vector up;
	Vector side;
	double fov;
	activeGGT->camera_->getParams(origin, lookdir2, up, side, fov);
	lookdir2.normalize();
	up.normalize();
	side.normalize();
	// Move the lights that are fixed to the eye
	for(int i = 0; i < activeGGT->rtrt_dpy->scene->nlights(); i++) {
	  Light *light = activeGGT->rtrt_dpy->scene->light(i);
	  if (light->fixed_to_eye) {
	    //	    light->updatePosition(light->get_pos() + dir*scl);
	    light->updatePosition(origin, 
				  Vector(side*light->eye_offset_basis.x()+
					 up*light->eye_offset_basis.y()+
					 lookdir2*light->eye_offset_basis.z()),
				  lookdir2);
	  }
	}
      }
    }
    break;
  case 13: // Enter
    if (activeGGT->shiftDown_) {
      // toggle holo room on/off
      activeGGT->rtrt_dpy->holoToggle_ = !activeGGT->rtrt_dpy->holoToggle_;
      cout << "holo room is now " << activeGGT->rtrt_dpy->holoToggle_ << endl;
    } else {
      activeGGT->camera_->flatten(); // Right yourself (0 pitch, 0 roll)
    }
    break;
  case 'x':
    traverseRouteCB(-1);
    break;
  case 'a':
    activeGGT->priv->animate =! activeGGT->priv->animate;
    cout << "animate is now " << activeGGT->priv->animate << "\n";
    break;
  case 'c':
    activeGGT->camera_->print();
    break;
  case 'n':
    activeGGT->camera_->scale_eyesep(0.9);
    cerr << "camera->eyesep="<<activeGGT->camera_->get_eyesep()<<"\n";
    break;
  case 'm':
    activeGGT->camera_->scale_eyesep(1.1);
    cerr << "camera->eyesep="<<activeGGT->camera_->get_eyesep()<<"\n";
    break;
  case 'o':
    printf("Number materials: %d\n",activeGGT->rtrt_dpy->scene->nmaterials());
    for (int m=0; m<activeGGT->rtrt_dpy->scene->nmaterials(); m++) {
      CycleMaterial * cm =
	dynamic_cast<CycleMaterial*>(activeGGT->rtrt_dpy->scene->get_material(m));
      if (cm) { cm->next(); printf("Got a cycle material!\n");}
    }
    break;

  case 'J': // toggle on/off "Jitter On Stop" moded...
    toggleAutoJitterCB( -1 );
    break;
  case 'j': // toggle on/off continuous jittered sampling...
    toggleJitterCB( -1 );
    break;

  case 'e':
    activeGGT->rtrt_dpy->nstreams++;
    break;
  case 'E':
    if(activeGGT->rtrt_dpy->nstreams > 1)
      activeGGT->rtrt_dpy->nstreams--;
    break;

  case 'r':
    activeGGT->displayRStats_ = !activeGGT->displayRStats_;
    break;
  case 'p':
    activeGGT->displayPStats_ = !activeGGT->displayPStats_;
    break;

  case 'f':
#if 0
    if( activeGGT->rtrt_dpy->fullScreenMode_ )
      activeGGT->rtrt_dpy->toggleRenderWindowSize_ = true;
    else
      cout << "Can't toggle to full res on non-full screen mode.\n";
#else
    activeGGT->priv->show_frame_rate = !activeGGT->priv->show_frame_rate;
#endif
    break;

  case 27: // Escape key... need to find a symbolic name for this...
    activeGGT->quit();
    break;
  case 'S':
    activeGGT->priv->stereo=!(activeGGT->priv->stereo);
    break;
#if 0
    // below is for blending "pixels" in
    // frameless rendering...

  case 'y': // sychronizing mode for frameless...
    synch_frameless = !synch_frameless;  //1-synch_frameless;
    //doing_frameless = 1-doing_frameless; // just toggle...
    cerr << synch_frameless << " Synch?\n";
    break;
#endif
  case 'W':
    cerr << "Saving raw image file\n";
    activeGGT->rtrt_dpy->scene->get_image(showing_scene)->save("images/image.raw");
    break;
  case 'w':
    cerr << "Saving ppm image file\n";
    activeGGT->rtrt_dpy->priv->dumpFrame = 1;
    break;
  case 'M':
    cerr << "Saving every frame to ppm image\n";
    switch (activeGGT->rtrt_dpy->priv->dumpFrame)
      {
      case 0:
      case 1: // Start
        activeGGT->rtrt_dpy->priv->dumpFrame = -1;
        break;
      case -1: // Stop
      case -2:
      case -3:
        activeGGT->rtrt_dpy->priv->dumpFrame = -3;
        break;
      }
    break;
  case 'd':
    {
      bool dd = activeGGT->scene()->display_depth;
      bool ds = activeGGT->scene()->display_sils;
      activeGGT->scene()->display_depth = !dd && !ds;
      activeGGT->scene()->display_sils = dd && !ds;
      activeGGT->scene()->store_depth = !ds;
    }
    break;
  case 'D':
    {
      bool ds = activeGGT->scene()->display_sils;
      activeGGT->scene()->display_depth = false;
      activeGGT->scene()->display_sils = !ds;
      activeGGT->scene()->store_depth = !ds;
    }
    break;
  default:
    printf("unknown regular key %d\n", key);
    break;
  }
} // end handleKeyPress();

// WARNING: THESE ARE NOT THE KEYPAD KEYS!
void
GGT::handleSpecialKeyCB( int key, int /*mouse_x*/, int /*mouse_y*/ )
{
  switch( key ) {
  case GLUT_KEY_LEFT:
    printf("glut_key_left\n");
    break;
  case GLUT_KEY_RIGHT:
    printf("glut_key_right\n");
    break;
  case GLUT_KEY_UP:
    printf("glut_key_up\n");
    break;
  case GLUT_KEY_DOWN:
    printf("glut_key_down\n");
    break;
  case GLUT_KEY_PAGE_DOWN:
    printf("glut_key_page_down\n");
    break;
  case GLUT_KEY_PAGE_UP:
    printf("glut_key_page_up\n");
    break;
  case GLUT_KEY_HOME:
    printf("glut_key_home\n");
    break;
  case GLUT_KEY_END:
    printf("glut_key_END\n");
    break;
  case GLUT_KEY_INSERT:
    printf("glut_key_insert\n");
    break;
  default:
    printf("unknown special key %d\n", key);
    break;
  }
}

static double    eye_dist = 0;

void
GGT::handleMousePress(int button, int mouse_x, int mouse_y)
{
  // Figure out if the shift is down at this point because you can't
  // do it in the mouse motion handler.
  int mods   = glutGetModifiers();
  shiftDown_ = mods & GLUT_ACTIVE_SHIFT;
  altDown_   = mods & GLUT_ACTIVE_ALT;
  ctrlDown_  = mods & GLUT_ACTIVE_CTRL;

  static Stats *gui_stats = new Stats(1);
  static DepthStats *ds = new DepthStats(gui_stats->ds[0]);
  Object *current_obj;
  if( shiftDown_ ) {
    Camera *C = activeGGT->rtrt_dpy->scene->get_camera( 0 );
    Ray ray;
    C->makeRay( ray, mouse_x, activeGGT->rtrt_dpy->priv->yres-mouse_y, 
		1.0/activeGGT->rtrt_dpy->priv->xres,
		1.0/activeGGT->rtrt_dpy->priv->yres );
    HitInfo hit;
    activeGGT->rtrt_dpy->scene->get_object()->intersect( ray, hit, ds,
  						     activeGGT->rtrt_dpy->ppc );
    if( hit.was_hit ) {
      current_obj = hit.hit_obj;
//        cout << "Mouse down on object "<<current_obj->get_name()<<endl;
      cbFunc mouseCB = MouseCallBack::getCB_MD( current_obj );
      if(mouseCB)
	mouseCB( current_obj, ray, hit );
    }
    return;
  }

  double     & last_time = priv->last_time;
  BallData  *& ball = priv->ball;

  // Record the mouse button so that the mouse motion handler will
  // know what operation to perform.
  mouseDown_ = button;

  activeGGT->last_x_ = mouse_x;
  activeGGT->last_y_ = mouse_y;

  switch(button){
  case GLUT_MIDDLE_BUTTON:
			
    // Find the center of rotation...
    double rad = 0.8;
    HVect center(0,0,0,1.0);
			
    // we also want to keep the old transform information
    // around (so stuff correlates correctly)
			
    Vector y_axis,x_axis;
    activeGGT->camera_->get_viewplane(y_axis, x_axis);
    Vector z_axis(activeGGT->camera_->eye - activeGGT->camera_->lookat);

    x_axis.normalize();
    y_axis.normalize();

    eye_dist = z_axis.normalize();

    prev_trans.load_frame(Point(0.0,0.0,0.0),x_axis,y_axis,z_axis);

    ball->Init();
    ball->Place(center,rad);
    HVect mouse((2.0*mouse_x)/priv->xres - 1.0,
		 2.0*(priv->yres-mouse_y*1.0)/priv->yres - 1.0,0.0,1.0);
    ball->Mouse(mouse);
    ball->BeginDrag();

    prev_time[0] = SCIRun::Time::currentSeconds();
    prev_quat[0] = mouse;
    prev_time[1] = prev_time[2] = -100;

    ball->Update();
    last_time=SCIRun::Time::currentSeconds();
    break;
  }
} // end handleMousePress()

void
GGT::handleMouseRelease(int button, int mouse_x, int mouse_y)
{
  DpyPrivate * priv      = activeGGT->priv;
  double     & last_time = priv->last_time;

  static Stats *gui_stats = new Stats(1);
  static DepthStats *ds = new DepthStats(gui_stats->ds[0]);
  Object *current_obj;
  if( activeGGT->shiftDown_ ) {
    Camera *C = activeGGT->rtrt_dpy->scene->get_camera( 0 );
    Ray ray;
    C->makeRay( ray, mouse_x, activeGGT->rtrt_dpy->priv->yres-mouse_y, 
		1.0/activeGGT->rtrt_dpy->priv->xres,
		1.0/activeGGT->rtrt_dpy->priv->yres );
    HitInfo hit;
    activeGGT->rtrt_dpy->scene->get_object()->intersect( ray, hit, ds,
  						     activeGGT->rtrt_dpy->ppc );
    if( hit.was_hit ) {
      current_obj = hit.hit_obj;
//        cout << "Mouse up on object "<<current_obj->get_name()<<endl;
      cbFunc mouseCB = MouseCallBack::getCB_MU( current_obj );
      if(mouseCB)
	mouseCB( current_obj, ray, hit );
    }
    return;
  }
  mouseDown_ = 0;

  switch(button){
  case GLUT_RIGHT_BUTTON:
    activeGGT->fovSpinner_->set_float_val( activeGGT->camera_->get_fov() );
    break;
  case GLUT_MIDDLE_BUTTON:
    {
#if 1
      if(SCIRun::Time::currentSeconds()-last_time < .1){
	// now setup the normalized quaternion
#endif
	Transform tmp_trans;
	HMatrix mNow;
	priv->ball->Value(mNow);
	tmp_trans.set(&mNow[0][0]);
			    
	Transform prv = prev_trans;
	prv.post_trans(tmp_trans);
	    
	HMatrix vmat;
	prv.get(&vmat[0][0]);
			    
	Vector y_a(vmat[0][1],vmat[1][1],vmat[2][1]);
	Vector z_a(vmat[0][2],vmat[1][2],vmat[2][2]);
	z_a.normalize();
	    
	activeGGT->camera_->up  = y_a;
	activeGGT->camera_->eye = activeGGT->camera_->lookat+z_a*eye_dist;
	activeGGT->camera_->setup();
	prev_trans = prv;

	Point origin;
	Vector lookdir;
	Vector up;
	Vector side;
	double fov;
	activeGGT->camera_->getParams(origin, lookdir, up, side, fov);
	lookdir.normalize();
	up.normalize();
	side.normalize();
	// Move the lights that are fixed to the eye
	for(int i = 0; i < activeGGT->rtrt_dpy->scene->nlights(); i++) {
	  Light *light = activeGGT->rtrt_dpy->scene->light(i);
	  if (light->fixed_to_eye) {
	    //	    light->updatePosition(light->get_pos() + dir*scl);
	    light->updatePosition(origin, 
				  Vector(side*light->eye_offset_basis.x()+
					 up*light->eye_offset_basis.y()+
					 lookdir*light->eye_offset_basis.z()),
				  lookdir);
	  }
	}
	// now you need to use the history to 
	// set up the arc you want to use...

	priv->ball->Init();
	double rad = 0.8;
	HVect center(0,0,0,1.0);
			    
	priv->ball->Place(center,rad);

	int index=2;

	if (prev_time[index] == -100)
	  index = 1;
			    
	priv->ball->vDown = prev_quat[index];
	priv->ball->vNow  = prev_quat[0];
	priv->ball->dragging = 1;

	priv->ball->Update();
			    
	priv->ball->qNorm = priv->ball->qNow.Conj();
	//double mag = ball->qNow.VecMag();
		    
      }
      priv->ball->EndDrag();
    }
    break;
  }
} // end handleMouseRelease()

void
GGT::handleWindowResizeCB( int width, int height )
{
  //  printf("window resized\n");
  static bool first=true;
  if(first){
    first=false;
  }
}

void
GGT::handleMouseCB(int button, int state, int x, int y)
{
  if( state == GLUT_DOWN ) {
    activeGGT->handleMousePress( button, x, y );
  } else {
    activeGGT->handleMouseRelease( button, x, y );
  }
}

void
GGT::handleMouseMotionCB( int mouse_x, int mouse_y )
{
  int          last_x = activeGGT->last_x_;
  int          last_y = activeGGT->last_y_;
  DpyPrivate * priv   = activeGGT->priv;

  double     & last_time = priv->last_time;
  BallData  *& ball      = priv->ball;

  static Stats *gui_stats = new Stats(1);
  static DepthStats *ds = new DepthStats(gui_stats->ds[0]);
  Object *current_obj;
  if( activeGGT->shiftDown_ ) {
    Camera *C = activeGGT->rtrt_dpy->scene->get_camera( 0 );
    Ray ray;
    C->makeRay( ray, mouse_x, activeGGT->rtrt_dpy->priv->yres-mouse_y, 
		1.0/activeGGT->rtrt_dpy->priv->xres,
		1.0/activeGGT->rtrt_dpy->priv->yres );
    HitInfo hit;
    activeGGT->rtrt_dpy->scene->get_object()->intersect( ray, hit, ds,
  						     activeGGT->rtrt_dpy->ppc );
    if( hit.was_hit ) {
      current_obj = hit.hit_obj;
//        cout << "Mouse moving on object "<<current_obj->get_name()<<endl;
      cbFunc mouseCB = MouseCallBack::getCB_MM( current_obj );
      if (mouseCB)
	mouseCB( current_obj, ray, hit );
    }
    return;
  }

  switch( activeGGT->mouseDown_ ) {
  case GLUT_RIGHT_BUTTON:
    {
      if( activeGGT->rightButtonMenuActive_ )
	{ // Note: This should actually never be the case as the right 
	  // button menu gets the click if it is active.
	  return;
	}
      if( activeGGT->shiftDown_ )
	{
	  // Move towards/away from the lookat point.
	  double scl;
	  double xmtn=-(last_x-mouse_x);
	  double ymtn=-(last_y-mouse_y);
	  xmtn/=300;
	  ymtn/=300;
	  last_x = mouse_x;
	  last_y = mouse_y;
	  if (Abs(xmtn)>Abs(ymtn)) scl=xmtn; else scl=ymtn;
	  Vector dir = activeGGT->camera_->lookat - activeGGT->camera_->eye;
	  activeGGT->camera_->eye += dir*scl;

	  Point origin;
	  Vector lookdir;
	  Vector up;
	  Vector side;
	  double fov;
	  activeGGT->camera_->getParams(origin, lookdir, up, side, fov);
	  lookdir.normalize();
	  up.normalize();
	  side.normalize();
	  // Move the lights that are fixed to the eye
	  for(int i = 0; i < activeGGT->rtrt_dpy->scene->nlights(); i++) {
	    Light *light = activeGGT->rtrt_dpy->scene->light(i);
	    if (light->fixed_to_eye) {
	      //light->updatePosition(light->get_pos() + dir*scl);
	    light->updatePosition(origin, 
				  Vector(side*light->eye_offset_basis.x()+
					 up*light->eye_offset_basis.y()+
					 lookdir*light->eye_offset_basis.z()),
				  lookdir);
	    }
	  }
	} else {
	  // Zoom in/out.
	  double scl;
	  double xmtn= last_x - mouse_x;
	  double ymtn= last_y - mouse_y;
	  xmtn/=30;
	  ymtn/=30;
	  last_x = mouse_x;
	  last_y = mouse_y;
	  if (Abs(xmtn)>Abs(ymtn)) scl=xmtn; else scl=ymtn;
	  if (scl<0) scl=1/(1-scl); else scl+=1;

	  double fov = RtoD(2*atan(scl*tan(DtoR(activeGGT->camera_->fov/2.))));
	  if( fov < MIN_FOV )
	    fov = MIN_FOV;
	  else if( fov > MAX_FOV )
	    fov = MAX_FOV;
	  activeGGT->camera_->set_fov( fov );
	}
      activeGGT->camera_->setup();
    }
    break;
  case GLUT_LEFT_BUTTON:
    {
      double xmtn =  double(last_x-mouse_x)/double(priv->xres);
      double ymtn = -double(last_y-mouse_y)/double(priv->yres);

      Vector u,v;
      activeGGT->camera_->get_viewplane(u, v);
      Vector trans(u*ymtn+v*xmtn);

      // Translate the view...
      activeGGT->camera_->eye+=trans;
      activeGGT->camera_->lookat+=trans;
      activeGGT->camera_->setup();

      Point origin;
      Vector lookdir;
      Vector up;
      Vector side;
      double fov;
      activeGGT->camera_->getParams(origin, lookdir, up, side, fov);
      lookdir.normalize();
      up.normalize();
      side.normalize();
      // Move the lights that are fixed to the eye
      for(int i = 0; i < activeGGT->rtrt_dpy->scene->nlights(); i++) {
	Light *light = activeGGT->rtrt_dpy->scene->light(i);
	if (light->fixed_to_eye) {
	  //	  light->updatePosition(light->get_pos() + trans);
	  light->updatePosition(origin, 
				Vector(side*light->eye_offset_basis.x()+
				       up*light->eye_offset_basis.y()+
				       lookdir*light->eye_offset_basis.z()),
				lookdir);
	}
      }
    }
    break;
  case GLUT_MIDDLE_BUTTON:
    {
      HVect mouse((2.0*mouse_x)/priv->xres - 1.0,
		   2.0*(priv->yres-mouse_y*1.0)/priv->yres - 1.0,0.0,1.0);

      prev_time[2] = prev_time[1];
      prev_time[1] = prev_time[0];
      prev_time[0] = SCIRun::Time::currentSeconds();

      ball->Mouse(mouse);
      ball->Update();

      prev_quat[2] = prev_quat[1];
      prev_quat[1] = prev_quat[0];
      prev_quat[0] = mouse;

      // now we should just sendthe view points through
      // the rotation (after centerd around the ball)
      // eyep lookat and up
			
      Transform tmp_trans;
      HMatrix mNow;
      ball->Value(mNow);
      tmp_trans.set(&mNow[0][0]);

      Transform prv = prev_trans;
      prv.post_trans(tmp_trans);

      HMatrix vmat;
      prv.get(&vmat[0][0]);

      Vector y_a(vmat[0][1],vmat[1][1],vmat[2][1]);
      Vector z_a(vmat[0][2],vmat[1][2],vmat[2][2]);
      z_a.normalize();

      activeGGT->camera_->up  = y_a;
      activeGGT->camera_->eye = activeGGT->camera_->lookat+z_a*eye_dist;
      activeGGT->camera_->setup();
			
      Point origin;
      Vector lookdir;
      Vector up;
      Vector side;
      double fov;
      activeGGT->camera_->getParams(origin, lookdir, up, side, fov);
      lookdir.normalize();
      up.normalize();
      side.normalize();
      // Move the lights that are fixed to the eye
      for(int i = 0; i < activeGGT->rtrt_dpy->scene->nlights(); i++) {
	Light *light = activeGGT->rtrt_dpy->scene->light(i);
	if (light->fixed_to_eye) {
	  //	    light->updatePosition(light->get_pos() + dir*scl);
	  light->updatePosition(origin, 
				Vector(side*light->eye_offset_basis.x()+
				       up*light->eye_offset_basis.y()+
				       lookdir*light->eye_offset_basis.z()),
				lookdir);
	}
      }

      last_time=SCIRun::Time::currentSeconds();
      //inertia_mode=0;
    }
    break;
  }

  activeGGT->last_x_ = mouse_x;
  activeGGT->last_y_ = mouse_y;

} // end handleMouseMotion()

void
GGT::handleSpaceballMotionCB( int sbm_x, int sbm_y, int sbm_z )
{
  double sensitivity = 100.0;

  if( abs(sbm_x) > 2 )
    activeGGT->camera_->moveLaterally( sbm_x / sensitivity );
  if( abs(sbm_y) > 2 )
    activeGGT->camera_->moveVertically( sbm_y / sensitivity );
  if( abs(sbm_z) > 2 )
    activeGGT->camera_->moveForwardOrBack( sbm_z / sensitivity );
}

void
GGT::handleSpaceballRotateCB( int sbr_x, int sbr_y, int /*sbr_z*/ )
{
  double sensitivity = 1000.0;

  if( abs(sbr_x) > 2 )
    activeGGT->camera_->changePitch( sbr_x / (sensitivity*2) );
  if( abs(sbr_y) > 2 )
    activeGGT->camera_->changeFacing( sbr_y / sensitivity );
  // Don't allow roll (at least for now)
}

void
GGT::handleSpaceballButtonCB( int button, int /*state*/ )
{
  cout << "spaceball button: " << button << "\n"; 
}


void
GGT::toggleRoutesWindowCB( int /*id*/ )
{
  if( activeGGT->routeWindowVisible )
    activeGGT->routeWindow->hide();
  else
    activeGGT->routeWindow->show();
  activeGGT->routeWindowVisible = !activeGGT->routeWindowVisible;
}

void
GGT::toggleLightsWindowCB( int /*id*/ )
{
  if( activeGGT->lightsWindowVisible )
    activeGGT->lightsWindow->hide();
  else
    activeGGT->lightsWindow->show();
  activeGGT->lightsWindowVisible = !activeGGT->lightsWindowVisible;
}

void
GGT::toggleObjectsWindowCB( int /*id*/ )
{
  if( activeGGT->objectsWindowVisible )
    activeGGT->objectsWindow->hide();
  else
    activeGGT->objectsWindow->show();
  activeGGT->objectsWindowVisible = !activeGGT->objectsWindowVisible;
}

void
GGT::toggleMaterialsWindowCB( int /*id*/ )
{
  if( activeGGT->materialsWindowVisible )
    activeGGT->materialsWindow->hide();
  else
    activeGGT->materialsWindow->show();
  activeGGT->materialsWindowVisible = !activeGGT->materialsWindowVisible;
}

void
GGT::toggleTriggersWindowCB( int /*id*/ )
{
  if( activeGGT->triggersWindowVisible )
    activeGGT->triggersWindow_->hide();
  else
    activeGGT->triggersWindow_->show();
  activeGGT->triggersWindowVisible = !activeGGT->triggersWindowVisible;
}

void
GGT::toggleSoundWindowCB( int /*id*/ )
{
  if( activeGGT->soundsWindowVisible )
    activeGGT->soundsWindow->hide();
  else
    {
      activeGGT->soundsWindow->show();
      updateSoundCB( -1 );
    }
  activeGGT->soundsWindowVisible = !activeGGT->soundsWindowVisible;
}

void
GGT::updateLightPanelCB( int /*id*/ )
{
  if( activeGGT->lights_.size() == 0 ) return;

  Light * light = activeGGT->lights_[ activeGGT->selectedLightId_ ];
  const Color & color = light->getOrigColor();
  const Point & pos   = light->get_pos();

  activeGGT->r_color_spin->set_float_val( color.red() );
  activeGGT->g_color_spin->set_float_val( color.green() );
  activeGGT->b_color_spin->set_float_val( color.blue() );

  activeGGT->lightIntensity_->set_float_val( light->get_intensity() );

  activeGGT->light_radius_spinner->set_float_val( light->radius );

  activeGGT->lightPosX_->set_float_val( pos.x() );
  activeGGT->lightPosY_->set_float_val( pos.y() );
  activeGGT->lightPosZ_->set_float_val( pos.z() );

  if( light->isOn() )
    {
      activeGGT->lightOnOffBtn_->set_name( "Turn Off" );
      activeGGT->lightsColorPanel_->enable();
      activeGGT->lightsPositionPanel_->enable();
    }
  else
    {
      activeGGT->lightOnOffBtn_->set_name( "Turn On" );
      activeGGT->lightsColorPanel_->disable();
      activeGGT->lightsPositionPanel_->disable();
    }
}

void
GGT::updateRouteCB( int /*id*/ )
{
  activeGGT->stealth_->selectPath( activeGGT->selectedRouteId_ );
  goToRouteBeginningCB( -1 );
}


void
GGT::updateSoundCB( int /*id*/ )
{
#if !defined(linux) && !defined(__APPLE__)
  activeGGT->currentSound_ = activeGGT->sounds_[ activeGGT->selectedSoundId_ ];

  Point & location = activeGGT->currentSound_->locations_[0];

  activeGGT->soundOriginX_->set_float_val( location.x() );
  activeGGT->soundOriginY_->set_float_val( location.y() );
  activeGGT->soundOriginZ_->set_float_val( location.z() );
#endif
}

void
GGT::toggleShowLightsCB( int /*id*/ )
{
  if( activeGGT->lightsBeingRendered_ ) {
    activeGGT->toggleShowLightsBtn_->set_name( "Show Lights" );
    activeGGT->rtrt_dpy->showLights_ = false;
    activeGGT->lightsBeingRendered_ = false;
  } else {
    activeGGT->toggleShowLightsBtn_->set_name( "Hide Lights" );
    activeGGT->rtrt_dpy->showLights_ = true;
    activeGGT->lightsBeingRendered_ = true;
  }
}

void
GGT::toggleLightOnOffCB( int /*id*/ )
{
  Light * light = activeGGT->lights_[ activeGGT->selectedLightId_ ];
  if( light->isOn() )
    {
      // turn it off
      activeGGT->lightOnOffBtn_->set_name( "Turn On" );
      light->turnOff();
      activeGGT->lightsColorPanel_->disable();
      activeGGT->lightsPositionPanel_->disable();
    }
  else
    {
      // turn it on
      activeGGT->lightOnOffBtn_->set_name( "Turn Off" );
      light->turnOn();
      activeGGT->lightsColorPanel_->enable();
      activeGGT->lightsPositionPanel_->enable();
    }
}

void
GGT::toggleLightSwitchesCB( int /*id*/ )
{
  if( activeGGT->lightsOn_ ) {
    activeGGT->toggleLightsOnOffBtn_->set_name( "Turn On Lights" );
    activeGGT->rtrt_dpy->turnOffAllLights_ = true;
    activeGGT->rtrt_dpy->turnOnAllLights_ = false;
    activeGGT->lightsOn_ = false;
    activeGGT->lightsColorPanel_->disable();
    activeGGT->lightsPositionPanel_->disable();
  } else {
    activeGGT->toggleLightsOnOffBtn_->set_name( "Turn Off Lights" );
    activeGGT->rtrt_dpy->turnOnAllLights_ = true;
    activeGGT->rtrt_dpy->turnOffAllLights_ = false;
    activeGGT->lightsOn_ = true;
    activeGGT->lightsColorPanel_->enable();
    activeGGT->lightsPositionPanel_->enable();
  }
}

void
GGT::updateAmbientCB( int /*id*/ )
{
  activeGGT->rtrt_dpy->scene->setAmbientLevel( activeGGT->ambientBrightness_ );
}

void
GGT::createRouteWindow( GLUI * window )
{
  GLUI_Panel * panel = window->add_panel( "Routes" );

  routeList = window->add_listbox_to_panel( panel, "Selected Route",
					    &selectedRouteId_, 
					    ROUTE_LIST_ID, updateRouteCB );
  routePositionPanel = window->add_panel_to_panel(panel, "Current Position");
  routePositionET =
    window->add_edittext_to_panel( routePositionPanel, "Marker:" );
  routePositionET->set_text( "# of #" );

  traverseRouteBtn = 
    window->add_button_to_panel( panel, "Follow Route",
				 TRAVERSE_ROUTE_BTN_ID, traverseRouteCB );
  goToRteBegBtn =
    window->add_button_to_panel( panel, "Go To Beginning",
				 GO_TO_RTE_BEGIN_BTN_ID, goToRouteBeginningCB);
  gravityBtn =
    window->add_button_to_panel( panel, "Turn Gravity On" );
  GLUI_EditText * constantZ_ET = 
    window->add_edittext_to_panel( panel, "Constant Z:" );
  constantZ_ET->disable();

  editorRO = window->add_rollout_to_panel( panel, "Edit Route", false );

  GLUI_Panel * trans = window->add_panel_to_panel( editorRO, "Transition to" );
  window->add_button_to_panel( trans, "Next Marker", 
			       GO_TO_NEXT_MARK_BTN_ID, goToNextMarkerCB );
  window->add_button_to_panel( trans, "Prev Marker",
			       GO_TO_PREV_MARK_BTN_ID, goToPrevMarkerCB );
  window->add_button_to_panel( editorRO, "Add New Marker", 
			       ADD_TO_ROUTE_BTN_ID, addToRouteCB );
  window->add_button_to_panel( editorRO, "Delete Current Marker",
			       DELETE_MARKER_BTN_ID, deleteCurrentMarkerCB );
  window->add_separator_to_panel( editorRO );
  window->add_button_to_panel( editorRO, "Delete Entire Route",
			       CLEAR_ROUTE_BTN_ID, clearRouteCB );

  routeList->disable();
  routePositionPanel->disable();
  traverseRouteBtn->disable();
  gravityBtn->disable();
  editorRO->disable();
  goToRteBegBtn->disable();

  // File menu for route file manipulation
  GLUI_Rollout * route_file = window->add_rollout( "File", true );
  window->add_button_to_panel( route_file, "New",
			       NEW_ROUTE_BUTTON_ID, getStringCB );
  window->add_button_to_panel( route_file, "Load",
			       LOAD_ROUTE_BUTTON_ID, getStringCB );
  window->add_button_to_panel( route_file, "Save",
			       SAVE_ROUTE_BUTTON_ID, getStringCB );
  window->add_button_to_panel( panel, "Close",
			       -1 , toggleRoutesWindowCB );

}

void
GGT::createGetStringWindow( GLUI * window )
{
  getStringPanel = window->add_panel( "" );

  getStringText_ = 
    window->add_edittext_to_panel( getStringPanel, "", GLUI_EDITTEXT_TEXT,
				 &(activeGGT->inputString_) );

  GLUI_Panel * buttonsPanel = window->add_panel_to_panel( getStringPanel, "" );

  getStringButton = 
    window->add_button_to_panel( buttonsPanel, "Set", CLOSE_GETSTRING_BTN );

  window->add_column_to_panel( buttonsPanel );

  window->add_button_to_panel( buttonsPanel, "Close", 
			       -1, hideGetStringWindowCB );
}

void
GGT::createObjectWindow( GLUI * window )
{
  GLUI_Panel * panel = window->add_panel( "Objects" );
					    
  Array1<Object*> & objects = rtrt_dpy->scene->objectsOfInterest_;
  for( int num = 0; num < objects.size(); num++ )
    {
      char name[ 1024 ];
      sprintf( name, "%s", Names::getName(objects[num]).c_str() );

      SelectableGroup *sg = dynamic_cast<SelectableGroup*>(objects[num]);
      if (sg) {

	window->add_separator_to_panel( panel );

	window->add_statictext_to_panel( panel, name );

	SGCallbackInfo *cbi1 = new SGCallbackInfo();
	callback_info_list.push_back(cbi1);
	cbi1->sg = sg;
	int callback_info_id = static_cast<int>(callback_info_list.size() - 1);
	
	(window->add_checkbox_to_panel( panel, "Cycle Objects", NULL,
					callback_info_id,
					SGAutoCycleCB ))->
	  set_int_val(sg->GetAutoswitch());

	(window->add_checkbox_to_panel( panel, "Display Every Frame", NULL,
				       callback_info_id,
				       SGNoSkipCB ))->
	  set_int_val(sg->GetNoSkip());

	GLUI_Spinner *sg_frame_rate =
	  window->add_spinner_to_panel( panel, "Seconds Per Frame",
					GLUI_SPINNER_FLOAT,
					&(sg->autoswitch_secs));
	sg_frame_rate->set_float_limits(0, 100);
	sg_frame_rate->set_speed(0.1);

	cbi1->current_frame_spinner = 
	  window->add_spinner_to_panel( panel, "Current Frame",
					GLUI_SPINNER_INT,
					&(cbi1->current_frame),
					callback_info_id, SGCurrentFrameCB);
	cbi1->current_frame_spinner->set_int_limits(0, sg->numObjects(),
						    GLUI_LIMIT_WRAP);
	cbi1->current_frame_spinner->set_speed(0.1);
	cbi1->current_frame_spinner->set_int_val(sg->GetChild());

	window->add_button_to_panel( panel, "Next Item",
				     callback_info_id,
				     SGNextItemCB );	
      }

      SpinningInstance *si = dynamic_cast<SpinningInstance*>(objects[num]);
      if (si) {    
	window->add_separator_to_panel( panel );

	window->add_statictext_to_panel( panel, name );

	window->add_checkbox_to_panel( panel, "Spin/Freeze", NULL,
				       num,
				       SISpinCB );
	window->add_button_to_panel( panel, "Inc Magnify",
				     num,
				     SIIncMagCB );
	window->add_button_to_panel( panel, "Dec Magnify",
				     num,
				     SIDecMagCB );
	
	window->add_button_to_panel( panel, "Slide Up",
				     num,
				     SISlideUpCB );
	window->add_button_to_panel( panel, "Slide Down",
				     num,
				     SISlideDownCB );		
      }

      CutGroup * cut = dynamic_cast<CutGroup*>( objects[num] );
      if (cut) {
	window->add_separator_to_panel( panel );

	window->add_statictext_to_panel( panel, name );

	window->add_button_to_panel( panel, "On/Off",
				     num,
				     CGOnCB );
	window->add_button_to_panel( panel, "Spin/Freeze",
				     num,
				     CGSpinCB );

      }      

      if (num&&(!(num%6))) window->add_column_to_panel( panel, true );
    }
  
  window->add_separator_to_panel( panel );
  window->add_button_to_panel( panel, "Close",
			       -1, toggleObjectsWindowCB );
} // end createObjectWindow()

void
GGT::createMaterialsWindow( GLUI * window )
{
  GLUI_Panel * panel = window->add_panel( "Materials" );
					    
  Array1<Material*> & materials = rtrt_dpy->scene->guiMaterials_;
  for( int num = 0; num < materials.size(); num++ ) {
    char name[ 1024 ];
    sprintf( name, "%s", Names::getName(materials[num]).c_str() );
    
    SketchMaterialBase *sm = dynamic_cast<SketchMaterialBase*>(materials[num]);
    if (sm) {
      
      window->add_separator_to_panel( panel );
      
      window->add_statictext_to_panel( panel, name );
      
      // Cool 2 warm
      window->add_checkbox_to_panel( panel, "Use cool2warm coloring",
				     &(sm->gui_use_cool2warm));

      // Show Silhouette
      window->add_checkbox_to_panel( panel, "Show Silhouettes",
				     &(sm->gui_show_silhouettes));

      // Silhouette thickness
      GLUI_Spinner *sil_thickness =
	window->add_spinner_to_panel( panel, "Silhouette Thickness",
				      GLUI_SPINNER_FLOAT,
				      &(sm->gui_sil_thickness));
      sil_thickness->set_float_limits(0, 10);
      sil_thickness->set_speed(0.1);

      // Silhouette Color
      GLUI_Rollout *sil_color =
	window->add_rollout_to_panel(panel, "Silhouette Color", false);
      
      GLUI_Spinner *sil_color_r = 
	window->add_spinner_to_panel( sil_color, "Red",
				      GLUI_SPINNER_FLOAT,
				      &(sm->gui_sil_color_r));
      sil_color_r->set_float_limits(0, 1);
      sil_color_r->set_speed(0.1);

      GLUI_Spinner *sil_color_g = 
	window->add_spinner_to_panel( sil_color, "Green",
				      GLUI_SPINNER_FLOAT,
				      &(sm->gui_sil_color_g));
      sil_color_g->set_float_limits(0, 1);
      sil_color_g->set_speed(0.1);

      GLUI_Spinner *sil_color_b = 
	window->add_spinner_to_panel( sil_color, "Blue",
				      GLUI_SPINNER_FLOAT,
				      &(sm->gui_sil_color_b));
      sil_color_b->set_float_limits(0, 1);
      sil_color_b->set_speed(0.1);

      // Normal type
      GLUI_RadioGroup *normal_method =
	window->add_radiogroup_to_panel( panel,
					 &(sm->gui_normal_method));
      window->add_radiobutton_to_group(normal_method, "Trilinear");
      window->add_radiobutton_to_group(normal_method, "Fancy");
    }
  }
  window->add_separator_to_panel( panel );
  window->add_button_to_panel( panel, "Close",
			       -1, toggleMaterialsWindowCB );
} // end createMaterialsWindow()

void
GGT::createSoundsWindow( GLUI * window )
{
  GLUI_Panel * panel = window->add_panel( "Sounds" );

  soundList_ = window->add_listbox_to_panel( panel, "Selected Sound",
					     &selectedSoundId_,
					     SOUND_LIST_ID, updateSoundCB );

  GLUI_Panel * soundOriginPanel = window->
    add_panel_to_panel( panel, "Location" );

  activeGGT->soundOriginX_ = window->add_edittext_to_panel
    ( soundOriginPanel, "X position:", GLUI_EDITTEXT_FLOAT );
  activeGGT->soundOriginY_ = window->add_edittext_to_panel
    ( soundOriginPanel, "Y position:", GLUI_EDITTEXT_FLOAT );
  activeGGT->soundOriginZ_ = window->add_edittext_to_panel
    ( soundOriginPanel, "Z position:", GLUI_EDITTEXT_FLOAT );


  GLUI_Panel * volumePanel = window->add_panel_to_panel( panel, "Volume" );

  currentSound_ = sounds_[0];

#if !defined(linux) && !defined(__APPLE__)
  sounds_ = rtrt_dpy->scene->getSounds();
  for( int num = 0; num < sounds_.size(); num++ )
    {
      char name[ 1024 ];
      sprintf( name, "%s", sounds_[ num ]->getName().c_str() );
      soundList_->add_item( num, name );
    }
#endif

  activeGGT->leftVolume_ = window->
    add_edittext_to_panel( volumePanel, "Left:", GLUI_EDITTEXT_FLOAT );

  activeGGT->rightVolume_ = window->
    add_edittext_to_panel( volumePanel, "Right:", GLUI_EDITTEXT_FLOAT );

  window->add_button_to_panel( panel, "Close",
			       -1, toggleSoundWindowCB );
}

void
GGT::createTriggersWindow( GLUI * window )
{
  vector<Trigger*> triggers = rtrt_dpy->scene->getTriggers();

  GLUI_Panel * panel = window->add_panel( "Triggers" );

  triggerList_ = window->add_listbox_to_panel( panel, "Selected Trigger",
					       &selectedTriggerId_ );
					     
  /*GLUI_Button * doit = */window->add_button_to_panel( panel, "Activate",
						    -1, activateTriggerCB );

  window->add_button( "Close", -1, toggleTriggersWindowCB );

  if( triggers.size() > 0 ) selectedTriggerId_ = 0;

  for( unsigned int num = 0; num < triggers.size(); num++ )
    {
      char name[ 1024 ];
      sprintf( name, "%s", triggers[ num ]->getName().c_str() );
      cout << "name: " << name << "\n";

      triggerList_->add_item( num, name );
    }
}

void
GGT::activateTriggerCB( int /* id */ )
{
  if( activeGGT->selectedTriggerId_ == -1 ) return;
  vector<Trigger*> triggers = activeGGT->rtrt_dpy->scene->getTriggers();
  Trigger * trig = triggers[ activeGGT->selectedTriggerId_ ];

  if( trig->isSoundTrigger() )
    {
      trig->activate();
    }
  else // image trigger
    {
      // Turn the Priority up so the trigger will be sure to run.
      trig->setPriority( Trigger::HighTriggerPriority );
      Trigger * next = trig->getNext();
      while( next && next != trig )
	{ // Set priorities of all triggers in sequence.
	  next->setPriority( Trigger::HighTriggerPriority );
	  next = next->getNext();
	}

      if( activeGGT->activeMTT_ ) // If a trigger is already running...
	{
	  cout << "QUEUING trigger: " << trig->getName() << "\n";
	  activeGGT->activeMTT_->deactivate(); // then tell it to stop.
	  activeGGT->queuedMTT_ = trig;        // and queue up the new trigger.
	}
      else
	{
	  cout << "activating TRIGGER: " << trig->getName() << "\n";
	  activeGGT->activeMTT_ = trig;
	  activeGGT->activeMTT_->activate();
	}
    }
}

void
GGT::startSoundThreadCB( int /*id*/ )
{
#if !defined(linux) && !defined(__APPLE__)
  activeGGT->startSoundThreadBtn_->disable();
  activeGGT->startSoundThreadBtn_->set_name( "Starting Sounds" );

  SoundThread * soundthread = NULL;

  cout << "Starting Sound Thread!\n";
  soundthread = new SoundThread( activeGGT->rtrt_dpy->getGuiCam(), 
				 activeGGT->rtrt_dpy->scene,
				 activeGGT );
  Thread * t = new Thread( soundthread, "Sound thread");
  t->detach();
#endif
}

void
GGT::soundThreadNowActive()
{
  activeGGT->enableSounds_ = true;
}

Scene*
GGT::scene()
{
  return rtrt_dpy->scene;
}

void GGT::SGAutoCycleCB( int id ) {
  SGCallbackInfo* sgcbi = (SGCallbackInfo*)callback_info_list[id];
  sgcbi->sg->toggleAutoswitch();
  // Now get the current timestep and then display it
  sgcbi->current_frame_spinner->set_int_val(sgcbi->sg->GetChild());
}

void GGT::SGNoSkipCB( int id ) {
  SelectableGroup *sg = ((SGCallbackInfo*)callback_info_list[id])->sg;
  sg->toggleNoSkip();
}

void GGT::SGNextItemCB( int id )
{
  SGCallbackInfo* sgcbi = (SGCallbackInfo*)callback_info_list[id];
  sgcbi->sg->nextChild();
  // Now get the current timestep and then display it
  sgcbi->current_frame_spinner->set_int_val(sgcbi->sg->GetChild());

  Object * newObj = sgcbi->sg->getCurrentChild();
  if( Names::hasName( newObj ) ) {
    Trigger * trig = NULL;

    if( Names::getName(newObj) == "Visible Female Volume" ) {
      trig = activeGGT->visWomanTrig_;
    } else if( Names::getName(newObj) == "Brain Volume" ) {
      trig = NULL;
    } else if( Names::getName(newObj) == "CSAFE Fire Volume" ) {
      trig = activeGGT->csafeTrig_;
    } else if( Names::getName(newObj) == "Geological Volume" ) {
      trig = activeGGT->geophysicsTrig_;
    } else if( Names::getName(newObj) == "Sheep Heart Volume" ) {
      trig = NULL;
    }

    if( trig ) {
      trig->setPriority( Trigger::HighTriggerPriority );
      if( activeGGT->activeMTT_ ) // If a trigger is already running...
	{
	  activeGGT->activeMTT_->deactivate(); // then tell it to stop.
	  activeGGT->queuedMTT_ = trig;        // and queue up the new trigger.
	}
      else // just start this trigger
	{
	  activeGGT->activeMTT_ = trig;
	  activeGGT->activeMTT_->activate();
	}
    }
  }
}

void GGT::SGCurrentFrameCB( int id ) {
  SGCallbackInfo* sgcbi = (SGCallbackInfo*)callback_info_list[id];
  sgcbi->sg->SetChild(sgcbi->current_frame_spinner->get_int_val());
}

void GGT::SISpinCB( int id ) {
  Array1<Object*> & objects = activeGGT->rtrt_dpy->scene->objectsOfInterest_;
  SpinningInstance *obj = dynamic_cast<SpinningInstance*>(objects[id]);  
  obj->toggleDoSpin();      
}
void GGT::SIIncMagCB( int id ) {
  Array1<Object*> & objects = activeGGT->rtrt_dpy->scene->objectsOfInterest_;
  SpinningInstance *obj = dynamic_cast<SpinningInstance*>(objects[id]);  
  obj->incMagnification();      
}
void GGT::SIDecMagCB( int id ) {
  Array1<Object*> & objects = activeGGT->rtrt_dpy->scene->objectsOfInterest_;
  SpinningInstance *obj = dynamic_cast<SpinningInstance*>(objects[id]);  
  obj->decMagnification();      
}
void GGT::SISlideUpCB( int id ) {
  Array1<Object*> & objects = activeGGT->rtrt_dpy->scene->objectsOfInterest_;
  SpinningInstance *obj = dynamic_cast<SpinningInstance*>(objects[id]);  
  obj->upPole();      
}
void GGT::SISlideDownCB( int id ) {
  Array1<Object*> & objects = activeGGT->rtrt_dpy->scene->objectsOfInterest_;
  SpinningInstance *obj = dynamic_cast<SpinningInstance*>(objects[id]);  
  obj->downPole();      
}

void GGT::CGOnCB( int id ) {
  Array1<Object*> & objects = activeGGT->rtrt_dpy->scene->objectsOfInterest_;
  CutGroup *obj = dynamic_cast<CutGroup*>(objects[id]);  
  obj->toggleOn();
}

void GGT::CGSpinCB( int id ) {
  Array1<Object*> & objects = activeGGT->rtrt_dpy->scene->objectsOfInterest_;
  CutGroup *obj = dynamic_cast<CutGroup*>(objects[id]);  
  obj->toggleAnimate();
}

void
GGT::addSceneLights() {
  int cnt;
  for( cnt = 0; cnt < scene()->nlights(); cnt++ ) {
    addLight( scene()->light( cnt ) );
  }
  for( ; cnt < scene()->nlights()+scene()->nPerMatlLights(); cnt++ ) {
    Light *light = scene()->per_matl_light( cnt - scene()->nlights() );
    if( light->name_ != "" )
      light->name_ = light->name_ + " (pm)";
    addLight( light );
  }
}

void
GGT::addLight( Light * light )
{
  string & name = light->name_;
  if( name != "" ) {
    int numLights = static_cast<int>(lights_.size());
    char namec[1024];
    sprintf( namec, "%s", name.c_str() );

    lightList->add_item( numLights, namec );
    lights_.push_back( light );

    updateLightPanelCB( -1 );
  }
}

void
GGT::createLightWindow( GLUI * window )
{
  GLUI_Panel * panel = window->add_panel( "Lights" );

  activeGGT->ambientBrightness_ = activeGGT->rtrt_dpy->scene->getAmbientLevel();

  ambientIntensity_ = 
    window->add_spinner_to_panel( panel, "Ambient Level:", GLUI_SPINNER_FLOAT,
				  &ambientBrightness_, -1, updateAmbientCB );
  ambientIntensity_->set_float_limits( 0.0, 1.0 );
  ambientIntensity_->set_speed( 0.05 );

  window->add_separator_to_panel( panel );

  lightList = window->add_listbox_to_panel( panel, "Selected Light:",
					    &selectedLightId_, 
					    LIGHT_LIST_ID, updateLightPanelCB);
  lightIntensity_ = 
    window->add_spinner_to_panel( panel, "Intensity:", GLUI_SPINNER_FLOAT,
				  &lightBrightness_, -1, updateIntensityCB );
  lightIntensity_->set_float_limits( 0.0, 1.0 );
  lightIntensity_->set_speed( 0.1 );

  lightOnOffBtn_ = window->add_button_to_panel( panel, "Turn Off",
						-1, toggleLightOnOffCB );

  lightsColorPanel_ = window->add_panel_to_panel( panel, "Color" );

  r_color_spin = 
    window->add_spinner_to_panel( lightsColorPanel_,"R:", GLUI_SPINNER_FLOAT );
  r_color_spin->set_float_limits( 0.0, 1.0 );
  r_color_spin->set_speed( 0.01 );

  g_color_spin = 
    window->add_spinner_to_panel( lightsColorPanel_,"G:", GLUI_SPINNER_FLOAT );
  g_color_spin->set_float_limits( 0.0, 1.0 );
  g_color_spin->set_speed( 0.01 );

  b_color_spin = 
    window->add_spinner_to_panel( lightsColorPanel_,"B:", GLUI_SPINNER_FLOAT );
  b_color_spin->set_float_limits( 0.0, 1.0 );
  b_color_spin->set_speed( 0.01 );

  lightsPositionPanel_ = window->add_panel_to_panel( panel, "Position" );
  lightPosX_ = 
    window->add_spinner_to_panel(lightsPositionPanel_,"X:",GLUI_SPINNER_FLOAT,
			  &(activeGGT->lightX_),
			  LIGHT_X_POS_ID, updateLightPositionCB );
  lightPosY_ =
    window->add_spinner_to_panel(lightsPositionPanel_,"Y:",GLUI_SPINNER_FLOAT,
			  &(activeGGT->lightY_),
			  LIGHT_Y_POS_ID, updateLightPositionCB );
  lightPosZ_ =
    window->add_spinner_to_panel(lightsPositionPanel_,"Z:",GLUI_SPINNER_FLOAT,
			  &(activeGGT->lightZ_),
			  LIGHT_Z_POS_ID, updateLightPositionCB );

  GLUI_Rollout * moreControls = 
    window->add_rollout_to_panel( panel, "More Controls", false );  

  window->add_separator_to_panel( moreControls );

  toggleLightsOnOffBtn_ = 
    window->add_button_to_panel(moreControls, "Turn Off Lights",
				TOGGLE_LIGHT_SWITCHES_ID,
				toggleLightSwitchesCB );
  toggleShowLightsBtn_ = 
    window->add_button_to_panel(moreControls, "Show Lights",
				TOGGLE_SHOW_LIGHTS_ID, toggleShowLightsCB );
  light_radius_spinner =
    window->add_spinner_to_panel(moreControls, "Radius", GLUI_SPINNER_FLOAT,
				 &light_radius, -1, updateLightRadiusCB );
  
  GLUI_Button * gotoLightBtn =
    window->add_button_to_panel(moreControls, "Goto Light" );
  gotoLightBtn->disable();

  window->add_button_to_panel( panel, "Close",
			       1, toggleLightsWindowCB );
}


void
GGT::createMenus( int winId, bool soundOn /* = false */,
		  bool showGui /* = true */ )
{
  printf("createmenus\n");

  // Register callbacks
  //  GLUI_Master.set_glutKeyboardFunc( GGT::handleKeyPress );
  GLUI_Master.set_glutReshapeFunc( GGT::handleWindowResizeCB );
  GLUI_Master.set_glutIdleFunc( GGT::idleFunc );
  
  // Build GLUI Windows
  mainWindow =
    GLUI_Master.create_glui_subwindow( mainWindowID, GLUI_SUBWINDOW_RIGHT);

  mainWindow->set_main_gfx_window( mainWindowID );
  
  if( !showGui ){
    activeGGT->mainWindow->hide();
    activeGGT->mainWindowVisible = false;
  }

  routeWindow     = GLUI_Master.create_glui( "Route",   0,900,400 );
  lightsWindow    = GLUI_Master.create_glui( "Lights",  0,900,500 );
  objectsWindow   = GLUI_Master.create_glui( "Objects", 0,900,600 );
  materialsWindow = GLUI_Master.create_glui("Materials",0,900,700 );
  soundsWindow    = GLUI_Master.create_glui( "Sounds",  0,900,800 );
  triggersWindow_ = GLUI_Master.create_glui( "Triggers",0,900,900 );

  getStringWindow = GLUI_Master.create_glui( "Input Request", 0, 900, 600 );

  //  routeWindow->set_main_gfx_window( winId );
  //  lightsWindow->set_main_gfx_window( winId );
  //  objectsWindow->set_main_gfx_window( winId );
  //  mainWindow->set_main_gfx_window( winId );

  routeWindow->hide();
  lightsWindow->hide();
  objectsWindow->hide();
  materialsWindow->hide();
  soundsWindow->hide();
  triggersWindow_->hide();

  getStringWindow->hide();

  createRouteWindow( routeWindow );
  createLightWindow( lightsWindow );
  createObjectWindow( objectsWindow );
  createMaterialsWindow( materialsWindow );
  createTriggersWindow( triggersWindow_ );
  createGetStringWindow( getStringWindow );

  /////////////////////////////////////////////////////////
  // Main Panel
  //
  GLUI_Panel * main_panel   = mainWindow->add_panel( "" );
  //  GLUI_Panel * button_panel = mainWindow->add_panel( "" );

  /////////////////////////////////////////////////////////
  // Display Parameters Panel
  //
  GLUI_Panel *display_panel = mainWindow->
    add_panel_to_panel( main_panel, "Display Parameters" );

  // Shadows
  GLUI_Panel * shadows = 
    mainWindow->add_panel_to_panel( display_panel, "Shadows" );
  shadowModeLB_ = mainWindow->
    add_listbox_to_panel( shadows, "Mode:", &rtrt_dpy->shadowMode_ );
  shadowModeLB_->add_item( No_Shadows, 
				      ShadowBase::shadowTypeNames[0] );
  shadowModeLB_->add_item( Single_Soft_Shadow,
				      ShadowBase::shadowTypeNames[1] );
  shadowModeLB_->add_item( Hard_Shadows,
				      ShadowBase::shadowTypeNames[2] );
  shadowModeLB_->add_item( Glass_Shadows,
				      ShadowBase::shadowTypeNames[3] );
  shadowModeLB_->add_item( Soft_Shadows,
				      ShadowBase::shadowTypeNames[4] );
  shadowModeLB_->add_item( Uncached_Shadows,
				      ShadowBase::shadowTypeNames[5] );
  shadowModeLB_->set_int_val( rtrt_dpy->scene->shadow_mode );

  // Ambient
  GLUI_Panel * ambient = 
    mainWindow->add_panel_to_panel(display_panel, "Ambient Light");
  ambientModeLB_ = mainWindow->
    add_listbox_to_panel( ambient, "Mode:", &rtrt_dpy->ambientMode_ );

  ambientModeLB_->add_item( Constant_Ambient, "Constant" );
  ambientModeLB_->add_item( Arc_Ambient, "Arc" );
  ambientModeLB_->add_item( Sphere_Ambient, "Sphere" );
  ambientModeLB_->set_int_val( rtrt_dpy->ambientMode_ );

  // Background Color controls
  GLUI_Panel * bgcolor = 
    mainWindow->add_panel_to_panel( display_panel, "Background Color" );
  WhiteBGButton_ = mainWindow->
    add_button_to_panel( bgcolor, "White Background",
			 WHITE_BG_BUTTON, bgColorCB);
  BlackBGButton_ = mainWindow->
    add_button_to_panel( bgcolor, "Black Background",
			 BLACK_BG_BUTTON, bgColorCB);
  OrigBGButton_ = mainWindow->
    add_button_to_panel( bgcolor, "Original Background",
			 ORIG_BG_BUTTON, bgColorCB);
  // Jitter
  GLUI_Panel * jitter = 
    mainWindow->add_panel_to_panel( display_panel, "Jitter" );
  if (rtrt_dpy->rtrt_engine->do_jitter)
    jitterButton_ = mainWindow->
      add_button_to_panel( jitter, "Turn Jitter OFF",
			   TURN_ON_JITTER_BTN, toggleJitterCB );
  else
    jitterButton_ = mainWindow->
      add_button_to_panel( jitter, "Turn Jitter ON",
			   TURN_ON_JITTER_BTN, toggleJitterCB );

  // FOV
  fovValue_ = camera_->get_fov();
  fovSpinner_ = mainWindow->
    add_spinner_to_panel( display_panel, "FOV:", GLUI_SPINNER_FLOAT,
			   &(fovValue_), FOV_SPINNER_ID,
			  updateFovCB );
  fovSpinner_->set_float_limits( MIN_FOV, MAX_FOV );
  fovSpinner_->set_speed( 0.01 );

  // Ray offset
  ray_offset_spinner = mainWindow->
    add_spinner_to_panel( display_panel, "Ray Offset", GLUI_SPINNER_FLOAT,
			   &(ray_offset), 0,
			  updateRayOffsetCB );
  ray_offset_spinner->set_float_val(rtrt_dpy->scene->get_camera(0)->get_ray_offset());
  ray_offset_spinner->set_float_limits( 0, 5000 );
  ray_offset_spinner->set_speed( 0.01 );

  // Other Controls
  GLUI_Panel * otherControls = mainWindow->
    add_panel_to_panel( display_panel, "Other Controls" );

  mainWindow->add_button_to_panel( otherControls,
	 "Toggle Hot Spot Display", TOGGLE_HOTSPOTS_ID, toggleHotspotsCB );

  mainWindow->add_button_to_panel( otherControls,
	 "Toggle Transmission Mode", TOGGLE_TRANSMISSION_MODE_ID,
					      toggleTransmissionModeCB );

  numThreadsSpinner_ = mainWindow->
    add_spinner_to_panel( otherControls, "Number of Threads",
			  GLUI_SPINNER_INT,
			  &(rtrt_dpy->numThreadsRequested_new),
			  NUM_THREADS_SPINNER_ID );
  numThreadsSpinner_->set_speed( 0.0001 );
  numThreadsSpinner_->set_int_limits( 1, MAX_NUM_THREADS );

  // ...This probably goes to the objects window...
  GLUI_Button * toggleMaterials = mainWindow->
    add_button_to_panel(otherControls,"Toggle Materials");
  toggleMaterials->disable();

  // 
  soundVolumeSpinner_ = mainWindow->
    add_spinner_to_panel( otherControls, "Sound Volume", GLUI_SPINNER_INT, 
			  &(rtrt_dpy->scene->soundVolume_),
			  SOUND_VOLUME_SPINNER_ID );
  soundVolumeSpinner_->set_speed( 0.01 );
  soundVolumeSpinner_->set_int_limits( 0, 100 );
  soundVolumeSpinner_->disable();
#if !defined(linux) && !defined(__APPLE__)
  //adding in the start sounds button after volume spinner
  //disabled if no sounds or sounds selected in beginning
  startSoundThreadBtn_ = mainWindow->
    add_button_to_panel(otherControls, "Start Sounds",-1, startSoundThreadCB);
  if( rtrt_dpy->scene->getSounds().size() == 0 )
    {
      startSoundThreadBtn_->disable();
      startSoundThreadBtn_->set_name( "No Sounds" );
    }
  else
    {
      createSoundsWindow( soundsWindow );
      if( soundOn )
	{
	  startSoundThreadBtn_->disable();
	  startSoundThreadCB( -1 );
	}
    }
#endif
  
  glyphThresholdSpinner_ = mainWindow->
    add_spinner_to_panel( otherControls, "Glyph Threshold",
			  GLUI_SPINNER_FLOAT, 
			  &glyph_threshold,
			  -1);
  glyphThresholdSpinner_->set_speed( 0.1 );
  glyphThresholdSpinner_->set_float_limits( 0, 1 );
  
  sceneDepthSpinner_ = mainWindow->
    add_spinner_to_panel( otherControls, "Sil Value",
			  GLUI_SPINNER_FLOAT, 
			  &(rtrt_dpy->scene->max_depth),
			  0, updateSceneDepthCB);
  sceneDepthSpinner_->set_speed( 0.1 );
  //  sceneDepthSpinner_->set_float_limits( 0, 500 );
  
  // 
  depthValue_ = priv->maxdepth;
  GLUI_Spinner * depthSpinner = mainWindow->
    add_spinner_to_panel( display_panel, "Ray Depth", GLUI_SPINNER_INT, 
			  &(depthValue_), DEPTH_SPINNER_ID, 
			  updateDepthCB );
  if (rtrt_dpy->scene->maxdepth > 12)
    depthSpinner->set_int_limits( 0, rtrt_dpy->scene->maxdepth * 3/2);
  else
    depthSpinner->set_int_limits( 0, 12 );
  depthSpinner->set_speed( 0.1 );

  /////////////////////////////////////////////////////////
  // Eye Position Panel
  //
  mainWindow->add_column_to_panel( main_panel, false );

#if 0
  GLUI_Panel *eye_panel = 
    mainWindow->add_panel_to_panel( main_panel, "Eye Position" );
  GLUI_Panel *pos_panel = 
    mainWindow->add_panel_to_panel( eye_panel, "Position" );

  x_pos = mainWindow->
    add_edittext_to_panel( pos_panel, "X:", GLUI_EDITTEXT_FLOAT );
  y_pos = mainWindow->
    add_edittext_to_panel( pos_panel, "Y:", GLUI_EDITTEXT_FLOAT );
  z_pos = mainWindow->
    add_edittext_to_panel( pos_panel, "Z:", GLUI_EDITTEXT_FLOAT );

  mainWindow->add_separator_to_panel( pos_panel );
  direct = 
    mainWindow->add_edittext_to_panel( pos_panel, "Facing" );

  GLUI_Panel *speed_panel = 
    mainWindow->add_panel_to_panel( eye_panel, "Speed" );

  forward_speed = mainWindow->
    add_edittext_to_panel( speed_panel, "Forward:", GLUI_EDITTEXT_FLOAT );
  upward_speed = mainWindow->
    add_edittext_to_panel( speed_panel, "Up:", GLUI_EDITTEXT_FLOAT );
  leftward_speed = mainWindow->
    add_edittext_to_panel( speed_panel, "Right:", GLUI_EDITTEXT_FLOAT );
#endif
  
  GLUI_Rollout *control_panel = mainWindow->
    add_rollout_to_panel( main_panel, "Control Sensitivity", false );
  rotateSensitivity_ = 1.0;
  GLUI_Spinner * rot = mainWindow->
    add_spinner_to_panel( control_panel, "Rotation:", GLUI_SPINNER_FLOAT,
			  &(rotateSensitivity_),
			  SENSITIVITY_SPINNER_ID, updateRotateSensitivityCB );
  rot->set_float_limits( MIN_SENSITIVITY, MAX_SENSITIVITY );
  rot->set_speed( 0.1 );

  translateSensitivity_ = 1.0;
  GLUI_Spinner * trans = mainWindow->
    add_spinner_to_panel(control_panel, "Translation:", GLUI_SPINNER_FLOAT,
			 &(translateSensitivity_),
			 SENSITIVITY_SPINNER_ID, updateTranslateSensitivityCB);
  trans->set_float_limits( MIN_SENSITIVITY, MAX_SENSITIVITY );
  trans->set_speed( 0.01 );

  // Benchmarking
  GLUI_Panel * bench_panel = 
    mainWindow->add_panel_to_panel( main_panel, "Benchmarking" );
  mainWindow->add_button_to_panel( bench_panel, "Start",
                                              START_BENCH_BTN, BenchCB );
  //  mainWindow->add_column_to_panel( bench_panel, false );
  mainWindow->add_button_to_panel( bench_panel, "Stop",
                                              STOP_BENCH_BTN, BenchCB );

  /////////////////////////////////////////////////////////
  // Route/Light/Objects/Sounds Window Buttons
  //
  GLUI_Panel * button_panel =
    mainWindow->add_panel_to_panel( main_panel, "" );

  mainWindow->
    add_button_to_panel( button_panel, "Routes",
			 ROUTE_BUTTON_ID, toggleRoutesWindowCB );

  mainWindow->
    add_button_to_panel( button_panel, "Lights",
			 LIGHTS_BUTTON_ID, toggleLightsWindowCB );

  mainWindow->
    add_button_to_panel( button_panel, "Objects",
			 OBJECTS_BUTTON_ID, toggleObjectsWindowCB );

  mainWindow->
    add_button_to_panel( button_panel, "Materials",
			 MATERIALS_BUTTON_ID, toggleMaterialsWindowCB );
  
  openSoundPanelBtn_ = mainWindow->
    add_button_to_panel( button_panel, "Sounds",
			 SOUNDS_BUTTON_ID, toggleSoundWindowCB );
  openSoundPanelBtn_->disable();
  
  mainWindow->
    add_button_to_panel( button_panel, "Triggers",
			 OBJECTS_BUTTON_ID, toggleTriggersWindowCB );

  printf("done createmenus\n");
} // end createMenus()

const string
GGT::getFacingString() const
{
  Vector lookAtVectHorizontal = 
    activeGGT->camera_->get_lookat() - activeGGT->camera_->get_eye();

  lookAtVectHorizontal.z( 0.0 );

  Vector north( 0, 1, 0 );

  double angle = RtoD( Dot( north, lookAtVectHorizontal ) );

  if( angle > 360.0 ) {
    angle = fmod( angle, 360.0 );
  }

  string dir;

  if( angle > 337.5 || angle < 22.5 ) {
    dir = "N";
  } else if( angle <  67.5 ) {
    dir = "NE";
  } else if( angle < 112.5 ) {
    dir = "E";
  } else if( angle < 157.5 ) {
    dir = "SE";
  } else if( angle < 202.5 ) {
    dir = "S";
  } else if( angle < 247.5 ) {
    dir = "SW";
  } else if( angle < 292.5 ) {
    dir = "W";
  } else if( angle < 337.5 ) {
    dir = "NW";
  } else {
    dir = "Confused";
  }
  
  return dir;
}

void
GGT::updateFovCB( int /*id*/ )
{
  activeGGT->camera_->set_fov( activeGGT->fovValue_ );
  activeGGT->camera_->setup();
}

void
GGT::updateRayOffsetCB( int /*id*/ )
{
  activeGGT->camera_->set_ray_offset( activeGGT->ray_offset );
}

void
GGT::updateSceneDepthCB( int /*id*/ )
{
  if (activeGGT->scene()->max_depth < 0) {
    activeGGT->scene()->max_depth = 0;
    activeGGT->sceneDepthSpinner_->set_float_val(0);
  }
}

void
GGT::updateDepthCB( int /*id*/ )
{
  activeGGT->priv->maxdepth = activeGGT->depthValue_;
}

void
GGT::updateRotateSensitivityCB( int /*id*/ )
{
  activeGGT->stealth_->updateRotateSensitivity(activeGGT->rotateSensitivity_);
}

void
GGT::updateTranslateSensitivityCB( int /*id*/ )
{
  activeGGT->stealth_->
    updateTranslateSensitivity(activeGGT->translateSensitivity_);
}

void
GGT::update()
{
  // Update Main Window
  x_pos->set_float_val( camera_->eye.x() );
  y_pos->set_float_val( camera_->eye.y() );
  z_pos->set_float_val( camera_->eye.z() );

  // getSpeed(0,1,2)
  forward_speed->set_float_val( stealth_->getSpeed(0) );
  leftward_speed->set_float_val( stealth_->getSpeed(1) );
  upward_speed->set_float_val( stealth_->getSpeed(2) );

  char facing[8];
  sprintf( facing, "%s", getFacingString().c_str() );
  direct->set_text( facing );

  if( priv->FrameRate < 4 && !beQuiet_ ){
    cerr << "dt=" << 1.0 / priv->FrameRate << '\n';
  }

  // Update Route Panel
  int pos, numPts;
  char status[1024];
  stealth_->getRouteStatus( pos, numPts );
  sprintf( status, "%d of %d", pos+1, numPts );
  routePositionET->set_text( status );

  updateSoundPanel();

} // end update()

bool
GGT::checkBackgroundWindow()
{
  // See if we have moved nearer to another room... return true if so.

  if( recheckBackgroundCnt_ == 10 )
    {
      recheckBackgroundCnt_ = 0;
      static vector<Point> positions;
      if( positions.size() == 0 )
	{
	  positions.push_back( Point(  -8,  8, 2) ); //  science
	  positions.push_back( Point(  10, -6, 2) ); //  living
	  positions.push_back( Point(   8, 10, 2) ); //  galaxy
	  positions.push_back( Point(  -7, -5, 2) ); //  museum

	  positions.push_back( Point(  0,  -6, 2) ); //  southTube
	  positions.push_back( Point(  0,  10, 2) ); //  northTube
	  positions.push_back( Point( -10, -2, 2) ); //  westTube
	  positions.push_back( Point(  10,  2, 2) ); //  eastTube
	}

      double minDist = 99999;
      int    index   = -1;

      const Point & eye = camera_->get_eye();

      for( int cnt = 0; cnt < 8; cnt++ )
	{
	  double dist = (positions[cnt] - eye).length2();
	  if( dist < minDist )
	    {
	      minDist = dist;
	      index = cnt;
	    }
	}
      cout << "closest to " << index << "\n";
      if( index != -1 )
	return setBackgroundImage( index );
    }
  else
    {
      recheckBackgroundCnt_++;
    }
  return false;
}

void
GGT::updateSoundPanel()
{
#if !defined(linux) && !defined(__APPLE__)
  if( soundsWindowVisible )
    {
      double right, left;
      currentSound_->currentVolumes( right, left );

      rightVolume_->set_float_val(right);
      leftVolume_->set_float_val(left);
    }
#endif
}

// Display image as a "transmission".  Ie: turn off every other scan line.
void
GGT::toggleTransmissionModeCB( int /* id */ )
{
  activeGGT->rtrt_dpy->turnOnTransmissionMode_ = 
    !activeGGT->rtrt_dpy->turnOnTransmissionMode_;
}

void
GGT::toggleHotspotsCB( int /*id*/ )
{
  switch (activeGGT->rtrt_dpy->rtrt_engine->hotSpotsMode) {
  case RTRT::HotSpotsOff:
    activeGGT->rtrt_dpy->rtrt_engine->hotSpotsMode = RTRT::HotSpotsOn;
    break;
  case RTRT::HotSpotsOn:
    activeGGT->rtrt_dpy->rtrt_engine->hotSpotsMode = RTRT::HotSpotsHalfScreen;
    break;
  case RTRT::HotSpotsHalfScreen:
    activeGGT->rtrt_dpy->rtrt_engine->hotSpotsMode = RTRT::HotSpotsOff;
    break;
  }
}


void
GGT::toggleGui()
{
  if( mainWindowVisible ) {
    routeWindow->hide();
    objectsWindow->hide();
    soundsWindow->hide();
    lightsWindow->hide();
    triggersWindow_->hide();
    mainWindow->hide();
    
    lightsWindowVisible = false;
    triggersWindowVisible = false;
    routeWindowVisible = false;
    objectsWindowVisible = false;
    soundsWindowVisible = false;

    activeGGT->rtrt_dpy->scene->hide_auxiliary_displays();
  } else {
    mainWindow->show();
    activeGGT->rtrt_dpy->scene->show_auxiliary_displays();
  }
  mainWindowVisible = !mainWindowVisible;
}


void
GGT::updateIntensityCB( int /*id*/ )
{
//  cout << "set light intensity to " << activeGGT->lightBrightness_ << "\n";

  if( activeGGT->lights_.size() == 0 ) return;

  Light * light = activeGGT->lights_[ activeGGT->selectedLightId_ ];

  light->updateIntensity( activeGGT->lightBrightness_ );

  if( activeGGT->lightBrightness_ == 0.0 )
    {
      activeGGT->lightsPositionPanel_->disable();
      activeGGT->lightsColorPanel_->disable();
      activeGGT->rtrt_dpy->turnOffLight_ = light;
    }
  else if( !light->isOn() )
    {
      activeGGT->lightsPositionPanel_->enable();
      activeGGT->lightsColorPanel_->enable();
      activeGGT->rtrt_dpy->turnOnLight_ = light;
    }
}

void
GGT::getStringCB( int id )
{
  if( id == LOAD_ROUTE_BUTTON_ID ) {
    activeGGT->getStringPanel->set_name( "Load File Name" );
    activeGGT->getStringButton->callback = loadRouteCB;
    activeGGT->getStringText_->callback = loadRouteCB;
  } else if( id == NEW_ROUTE_BUTTON_ID ) {
    activeGGT->getStringPanel->set_name( "Enter Route Name" );
    activeGGT->getStringButton->callback = newRouteCB;
    activeGGT->getStringText_->callback = newRouteCB;
  } else if( id == SAVE_ROUTE_BUTTON_ID ) {
    activeGGT->getStringPanel->set_name( "Save File Name" );
    activeGGT->getStringButton->callback = saveRouteCB;
    activeGGT->getStringText_->callback = saveRouteCB;
  } else {
    cout << "don't know what string to get\n";
    return;
  }
  
  activeGGT->getStringWindow->show();
}

void 
GGT::hideGetStringWindowCB( int /*id*/ )
{
  activeGGT->getStringWindow->hide();
}


void
GGT::loadAllRoutes()
{
  const vector<string> & routes = activeGGT->rtrt_dpy->scene->getRoutes();
  string routeName;
  char name[1024];

  for( unsigned int i = 0; i < routes.size(); i++ )
    {
      routeName = routes[i];

      sprintf( name, "%s", routeName.c_str() );
      activeGGT->routeList->add_item( routeNumber, name );
      activeGGT->routeList->set_int_val( routeNumber );
      
      routeNumber++;
    }
}

void
GGT::loadRouteCB( int /*id*/ )
{
  // glui is screwy when you have this type of window where you want
  // either the "ok" button or "return" to do the same thing.  By
  // removing the callbacks like this, you avoid an infinite loop.
  activeGGT->getStringButton->callback = NULL;
  activeGGT->getStringText_->callback = NULL;

  string routeName = activeGGT->stealth_->loadPath( activeGGT->inputString_ );

  if( routeName == "" )
    {
      cout << "loading of route failed\n";
      return;
    }

  cout << "loaded route: " << routeName << "\n";

  //  activeGGT->getStringWindow->hide();

  char name[1024];
  sprintf( name, "%s", routeName.c_str() );
  activeGGT->routeList->add_item( routeNumber, name );
  activeGGT->routeList->set_int_val( routeNumber );

  routeNumber++;

  activeGGT->routeList->enable();
  activeGGT->routePositionPanel->enable();
  activeGGT->traverseRouteBtn->enable();
  activeGGT->editorRO->enable();
  activeGGT->goToRteBegBtn->enable();

  goToRouteBeginningCB( -1 );
}

void
GGT::newRouteCB( int /*id*/ )
{
  // glui is screwy when you have this type of window where you want
  // either the "ok" button or "return" to do the same thing.  By
  // removing the callbacks like this, you avoid an infinite loop.
  activeGGT->getStringButton->callback = NULL;
  activeGGT->getStringText_->callback = NULL;

  //  activeGGT->getStringWindow->hide();

  string routeName = activeGGT->inputString_;

  if( routeName == "" )
    {
      cout << "invalid route name, not saving\n";
      return;
    }

  activeGGT->stealth_->newPath( routeName );

  char name[1024];
  sprintf( name, "%s", routeName.c_str() );
  activeGGT->routeList->add_item( routeNumber, name );
  activeGGT->routeList->set_int_val( routeNumber );
  routeNumber++;

  activeGGT->routeList->enable();
  activeGGT->routePositionPanel->enable();
  activeGGT->traverseRouteBtn->enable();
  activeGGT->editorRO->enable();
  activeGGT->goToRteBegBtn->enable();
}

void
GGT::saveRouteCB( int /*id*/ )
{
  // glui is screwy when you have this type of window where you want
  // either the "ok" button or "return" to do the same thing.  By
  // removing the callbacks like this, you avoid an infinite loop.
  activeGGT->getStringButton->callback = NULL;
  activeGGT->getStringText_->callback = NULL;

  //  activeGGT->getStringWindow->hide();
  
  if( strcmp( activeGGT->inputString_, "" ) != 0)
    {
      activeGGT->stealth_->savePath( activeGGT->inputString_ );
    }
}

void
GGT::clearRouteCB( int /*id*/ )
{
  activeGGT->stealth_->clearPath();
}

void
GGT::deleteCurrentMarkerCB( int /*id*/ )
{
  activeGGT->stealth_->deleteCurrentMarker();
}

void
GGT::addToRouteCB( int /*id*/ )
{
  activeGGT->stealth_->addToMiddleOfPath( activeGGT->camera_ );
}

void
GGT::traverseRouteCB( int /*id*/ )
{
  activeGGT->priv->followPath = !activeGGT->priv->followPath;

  // If starting/stopping following a path, stop all other movement.
  // if starting to following, increase movement once to make us move.
  activeGGT->stealth_->stopAllMovement();
  if( activeGGT->priv->followPath ) {
    activeGGT->stealth_->accelerate();
    activeGGT->traverseRouteBtn->set_name("Stop");
  } else {
    activeGGT->traverseRouteBtn->set_name("Follow Route");
  }
}

void
GGT::goToNextMarkerCB( int /*id*/ )
{
  int index = activeGGT->stealth_->getNextMarker( activeGGT->camera_ );  

  if( index == -1 ) return;

  Point origin;
  Vector lookdir;
  Vector up;
  Vector side;
  double fov;
  activeGGT->camera_->getParams(origin, lookdir, up, side, fov);
  lookdir.normalize();
  up.normalize();
  side.normalize();
  // Move the lights that are fixed to the eye
  for(int i = 0; i < activeGGT->rtrt_dpy->scene->nlights(); i++) {
    Light *light = activeGGT->rtrt_dpy->scene->light(i);
    if (light->fixed_to_eye) {
      //	    light->updatePosition(light->get_pos() + dir*scl);
      light->updatePosition(origin, 
			    Vector(side*light->eye_offset_basis.x()+
				   up*light->eye_offset_basis.y()+
				   lookdir*light->eye_offset_basis.z()),
			    lookdir);
    }
  }
}

void
GGT::goToPrevMarkerCB( int /*id*/ )
{
  int index = activeGGT->stealth_->getPrevMarker( activeGGT->camera_ );  

  if( index == -1 ) return;

  Point origin;
  Vector lookdir;
  Vector up;
  Vector side;
  double fov;
  activeGGT->camera_->getParams(origin, lookdir, up, side, fov);
  lookdir.normalize();
  up.normalize();
  side.normalize();
  // Move the lights that are fixed to the eye
  for(int i = 0; i < activeGGT->rtrt_dpy->scene->nlights(); i++) {
    Light *light = activeGGT->rtrt_dpy->scene->light(i);
    if (light->fixed_to_eye) {
      light->updatePosition(origin, 
			    Vector(side*light->eye_offset_basis.x()+
				   up*light->eye_offset_basis.y()+
				   lookdir*light->eye_offset_basis.z()),
			    lookdir);
    }
  }
}

void
GGT::goToRouteBeginningCB( int /*id*/ )
{
  int index = activeGGT->stealth_->goToBeginning( activeGGT->camera_ );  

  if( index == -1 ) return;

  Point origin;
  Vector lookdir;
  Vector up;
  Vector side;
  double fov;
  activeGGT->camera_->getParams(origin, lookdir, up, side, fov);
  lookdir.normalize();
  up.normalize();
  side.normalize();
  // Move the lights that are fixed to the eye
  for(int i = 0; i < activeGGT->rtrt_dpy->scene->nlights(); i++) {
    Light *light = activeGGT->rtrt_dpy->scene->light(i);
    if (light->fixed_to_eye) {
      light->updatePosition(origin, 
			    Vector(side*light->eye_offset_basis.x()+
				   up*light->eye_offset_basis.y()+
				   lookdir*light->eye_offset_basis.z()),
			    lookdir);
    }
  }
}

void
GGT::bgColorCB( int id )
{
  switch ( id ) {
  case WHITE_BG_BUTTON:
    activeGGT->rtrt_dpy->scene->set_bgcolor(Color(1,1,1));
    break;
  case BLACK_BG_BUTTON:
    activeGGT->rtrt_dpy->scene->set_bgcolor(Color(0,0,0));
    break;
  case ORIG_BG_BUTTON:
    activeGGT->rtrt_dpy->scene->set_original_bg();
    break;
  }
}

void
GGT::toggleAutoJitterCB( int /*id*/ )
{
  activeGGT->rtrt_dpy->doAutoJitter_ = !activeGGT->rtrt_dpy->doAutoJitter_;
}

void
GGT::toggleJitterCB( int /*id*/ )
{
  int *do_jitter = &(activeGGT->rtrt_dpy->rtrt_engine->do_jitter);
  *do_jitter = !(*do_jitter);
  if( !(*do_jitter))
    activeGGT->jitterButton_->set_name("Turn Jitter ON");
  else
    activeGGT->jitterButton_->set_name("Turn Jitter OFF");
}

void
GGT::BenchCB( int id )
{
  if (id == START_BENCH_BTN) {
    activeGGT->rtrt_dpy->start_bench();
  } else if (id == STOP_BENCH_BTN) {
    activeGGT->rtrt_dpy->stop_bench();
  }
}

void
GGT::updateLightPositionCB( int id )
{
  Light * light = activeGGT->lights_[ activeGGT->selectedLightId_ ];
  Point pos = light->get_pos();

  cout << "updating light position: " << id << "\n";
  cout << "pos was " << pos << "\n";

  cout << activeGGT->lightX_ << ", "
       << activeGGT->lightY_ << ", "
       << activeGGT->lightZ_ << "\n";

  switch( id ) {
  case LIGHT_X_POS_ID:
    pos.x( activeGGT->lightX_ );
    light->updatePosition( pos );
    break;
  case LIGHT_Y_POS_ID:
    pos.y( activeGGT->lightY_ );
    light->updatePosition( pos );
    break;
  case LIGHT_Z_POS_ID:
    pos.z( activeGGT->lightZ_ );
    light->updatePosition( pos );
    break;
  }
  cout << "pos is " << pos << "\n";
}

void
GGT::updateLightRadiusCB( int /*id*/ )
{
  if (activeGGT->light_radius < 0) {
    // This is a no no
    cerr << "light_radius cannot be less than zero...Fixing.\n";
    activeGGT->light_radius_spinner->set_float_val( 0 );
  }

  Light * light = activeGGT->lights_[ activeGGT->selectedLightId_ ];
  double radius = light->radius;

  cout << "updating light radius for light : " << light->name_ << "\n";

  light->updateRadius(activeGGT->light_radius);

  cout << "radius was " << radius << " now is "<<activeGGT->light_radius<<"\n";
}

void
GGT::cycleAmbientMode()
{
  if( rtrt_dpy->ambientMode_ == Sphere_Ambient )
    {
      rtrt_dpy->ambientMode_ = Constant_Ambient;
    }
  else
    {
      rtrt_dpy->ambientMode_++;
    }

  ambientModeLB_->set_int_val( rtrt_dpy->ambientMode_ );
}

void
GGT::cycleShadowMode()
{
  rtrt_dpy->shadowMode_ = ShadowBase::increment_shadow_type(rtrt_dpy->shadowMode_);
  shadowModeLB_->set_int_val( rtrt_dpy->shadowMode_ );
}


