
#include <sci_defs.h> // For HAVE_OOGL
#if defined(HAVE_OOGL)
#undef HAVE_OOGL
#endif
#include <Packages/rtrt/Core/Gui.h>
#include <Packages/rtrt/Core/Dpy.h>
#include <Packages/rtrt/Core/DpyPrivate.h>
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
#if !defined(linux)
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

#include <vector>
#include <Packages/rtrt/Core/MouseCallBack.h>

// From Glyph.cc
namespace rtrt {
  extern float glyph_threshold;
}
  
//oogl
ShadedPrim   * backgroundTexQuad; // from rtrt.cc
BasicTexture * backgroundTex;     // from rtrt.cc

////////////////////////////////////////////

extern "C" Display *__glutDisplay;

double ORBIT_SPEED  = 0;
double ROTATE_SPEED = 1;

using namespace rtrt;
using namespace SCIRun;
using namespace std;

static Gui * activeGui;

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
#define SOUNDS_BUTTON_ID         142
#define ATTACH_KEYPAD_BTN_ID     143
#define SICYCLE_BTN_ID           144

#define SOUND_LIST_ID            150

#define TOGGLE_HOTSPOTS_ID          190
#define TOGGLE_TRANSMISSION_MODE_ID 191
#define SOUND_VOLUME_SPINNER_ID     192
#define NUM_THREADS_SPINNER_ID      193

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

Gui::Gui() :
  activeMTT_(NULL), queuedMTT_(NULL),
  bottomGraphicTrig_(NULL), leftGraphicTrig_(NULL),
  visWomanTrig_(NULL), csafeTrig_(NULL), geophysicsTrig_(NULL),
  recheckBackgroundCnt_(10),
  selectedLightId_(0), selectedRouteId_(0), selectedObjectId_(0),
  selectedSoundId_(0), selectedTriggerId_(-1),
  routeWindowVisible(false), lightsWindowVisible(false), 
  objectsWindowVisible(false),
  soundsWindowVisible(false), triggersWindowVisible(false),
  mainWindowVisible(true),
  enableSounds_(false),
  mouseDown_(0), rightButtonMenuActive_(true), beQuiet_(true),
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

}

Gui::~Gui()
{
}

void
Gui::quit()
{
  cerr << "Quitting rtrt.\n";
  // Stop threads...
  activeGui->dpy_->scene->rtrt_engine->exit_clean(1);
  // Stop Glut mainloop.
  usleep(1000);
  Thread::exitAll( 0 );
}

void
Gui::handleMenuCB( int item )
{
  switch( item ) {
  case TOGGLE_HOT_SPOTS:
    activeGui->toggleHotspotsCB( -1 );
    break;
  case TOGGLE_GUI:
    activeGui->toggleGui();
    break;
  case TOGGLE_RIGHT_BUTTON_MENU:
    if( activeGui->rightButtonMenuActive_ )
      {
	activeGui->rightButtonMenuActive_ = false;
	glutDetachMenu(GLUT_RIGHT_BUTTON);
      }
    else
      {
	activeGui->rightButtonMenuActive_ = true;
	glutAttachMenu(GLUT_RIGHT_BUTTON);
      }
    break;
  case QUIT_MENU_ID:
    activeGui->quit();
    break;
  }
}

void
Gui::setActiveGui( Gui * gui )
{
  activeGui = gui;
}

void
Gui::setStealth( Stealth * stealth )
{
  stealth_ = stealth;
}

void
Gui::setDpy( Dpy * dpy )
{
  dpy_     = dpy;
  priv     = dpy->priv;
  stealth_ = dpy->stealth_;
  camera_  = dpy->guiCam_;
}

GLuint        fontbase, fontbase2;
XFontStruct * fontInfo;
XFontStruct * fontInfo2;

void
Gui::setupFonts()
{
  fontInfo = XLoadQueryFont(__glutDisplay, __FONTSTRING__);

  if (fontInfo == NULL) {
    cerr << "no font found" << __FILE__ << "," << __LINE__ << std::endl;
    Thread::exitAll(1);
  }

  Font id = fontInfo->fid;
  unsigned int first = fontInfo->min_char_or_byte2;
  unsigned int last = fontInfo->max_char_or_byte2;

  fontbase = glGenLists((GLuint) 2);/* last-first+1);*/
  if (fontbase == 0) {
    cout << "Out of display lists: errno: " << errno << "\n";
    Thread::exitAll(0);
  }
  glXUseXFont(id, first, last-first+1, fontbase+first);

  fontInfo2 = XLoadQueryFont(__glutDisplay, __FONTSTRING__);

  if (fontInfo2 == NULL) {
    cerr << "no font found" << __FILE__ << "," << __LINE__ << std::endl;
    Thread::exitAll(1);
  }

  id = fontInfo2->fid;
  first = fontInfo2->min_char_or_byte2;
  last = fontInfo2->max_char_or_byte2;

  fontbase2 = glGenLists((GLuint) last+1);
  if (fontbase2 == 0) {
    cout << "Out of display lists (fontbase2) : errno: " << errno << "\n";
    Thread::exitAll(0);
  }
  glXUseXFont(id, first, last-first+1, fontbase2+first);
}

void
Gui::handleTriggers()
{
  // Handle Active Trigger
  if( activeGui->activeMTT_ )
    {
      Trigger * next = NULL;
      // next is NULL if no next trigger associated with this trigger.
      bool result = activeGui->activeMTT_->advance( next );
      if( result == false ) // done, remove from active list.
	{
	  if( activeGui->queuedMTT_ )
	    {
	      if( next )
		{
		  double quedPriority = activeGui->queuedMTT_->getPriority();
		  double nextPriority = next->getPriority();
		  if( quedPriority < nextPriority )
		    {
		      cout << "using 'next' trigger: " <<next->getName()<<"\n";
		      cout << " priorities: " << nextPriority << ", "
			   << quedPriority << "\n";
		      activeGui->activeMTT_ = next;
		    }
		  else
		    {
		      activeGui->activeMTT_ = activeGui->queuedMTT_;
		      activeGui->queuedMTT_ = NULL;
		    }
		}
	      else
		{
		  cout << "moving in queued trigger: " << 
		    activeGui->queuedMTT_->getName() << "\n";
		  activeGui->activeMTT_ = activeGui->queuedMTT_;
		  activeGui->queuedMTT_ = NULL;
		}
	    }
	  else
	    {
	      activeGui->activeMTT_ = next;
	    }
	  if( activeGui->activeMTT_ ) 
	    {
	      cout << "using next trigger: " << next->getName() << "\n";
	      activeGui->activeMTT_->activate();
	    }
	}
    }

  // Check all triggers.
  vector<Trigger*> & triggers = dpy_->scene->getTriggers();
  for( unsigned int cnt = 0; cnt < triggers.size(); cnt++ )
    {
      Trigger * trigger = triggers[cnt];
      bool result = trigger->check( activeGui->camera_->eye );
      if( result == true && trigger != activeGui->activeMTT_ )
	{
	  // The trigger is in range, so determine what to do with it
	  // based on its priority and the active triggers priority.
	  if( activeGui->activeMTT_ )
	    {
	      if( trigger == activeGui->queuedMTT_ ) // already queued.
		continue;
	      double trigPriority = trigger->getPriority();
	      double currPriority = activeGui->activeMTT_->getPriority();
	      if( currPriority <= trigPriority )
		{ // Tell current to stop and queue up new trigger.
		  cout << "deactivating current trigger: " <<
		    activeGui->activeMTT_->getName() << " to start " <<
		    trigger->getName() << "\n";
		  activeGui->activeMTT_->deactivate();
		  activeGui->queuedMTT_ = trigger;
		}
	      else if( activeGui->queuedMTT_ )
		{
		  double quedPriority = activeGui->queuedMTT_->getPriority();
		  if( trigPriority > quedPriority )
		    {
		      activeGui->queuedMTT_ = trigger;
		    }
		  
		}
	    }
	  else
	    {
	      cout << "starting " << trigger->getName() << "\n";
	      activeGui->activeMTT_ = trigger;
	      activeGui->activeMTT_->activate();
	    }
	}
    }

  // Deal with bottom graphic trigger
  if( activeGui->bottomGraphicTrig_ )
    {
      Trigger * next = NULL;
      activeGui->bottomGraphicTrig_->advance( next );
      if( next )
	{
	  activeGui->bottomGraphicTrig_ = next;
	  next->activate();
	}
      else
	{
	  // Calling check() just to advance the time of the trigger.
	  activeGui->bottomGraphicTrig_->check( Point(0,0,0) );
	}
    }

  // Deal with left graphic trigger
  if( activeGui->leftGraphicTrig_ )
    {
      Trigger * next = NULL;
      activeGui->leftGraphicTrig_->advance( next );
      if( next )
	{
	  activeGui->leftGraphicTrig_ = next;
	  next->activate();
	}
      else
	{
	  // Calling check() just to advance the time of the trigger.
	  activeGui->leftGraphicTrig_->check( Point(0,0,0) );
	}
    }
} // end handleTriggers()

void
Gui::drawBackground()
{
#if defined(HAVE_OOGL)
  glutSetWindow( glutDisplayWindowId );

  glViewport(0, 0, 1280, 1024);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, 1280, 0, 1024);
  glDisable( GL_DEPTH_TEST );
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0.375, 0.375, 0.0);

  backgroundTex->reset( GL_FLOAT, &((*(activeGui->backgroundImage_))(0,0)) );
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
Gui::redrawBackgroundCB()
{
  if( activeGui->dpy_->fullScreenMode_ ) {
    if( !activeGui->backgroundImage_ ) return;
    // update loop will check for a new background, see it is different,
    // and update correctly.
    activeGui->recheckBackgroundCnt_ = 10; // forces a redraw
    activeGui->backgroundImage_ = NULL;
  }
}

bool
Gui::setBackgroundImage( int room )
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
Gui::idleFunc()
{
  Dpy               * dpy = activeGui->dpy_;
  struct DpyPrivate * priv = activeGui->priv;

  // Hacking these vars for now:
  static double lasttime  = SCIRun::Time::currentSeconds();
  static double cum_ttime = 0;
  static double cum_dt    = 0;

  if( activeGui->enableSounds_ )
    {
      // Sound Thread has finished loading sounds and is now active...
      // ...so turn on the sound GUIs.
      activeGui->enableSounds_ = false; 
      activeGui->openSoundPanelBtn_->enable();
      activeGui->soundVolumeSpinner_->enable();
      activeGui->startSoundThreadBtn_->set_name( "Sounds Started" );
    }

  glutSetWindow( activeGui->glutDisplayWindowId );
  activeGui->handleTriggers();

  // I know this is a hack... 
  if( dpy->showImage_ ){

    // Display textual information on the screen:
    char buf[100];
    sprintf( buf, "%3.1lf fps", (activeGui->priv->FrameRate) );

    bool redrawBG = false;
    if( dpy->fullScreenMode_ )
      {
	redrawBG = activeGui->checkBackgroundWindow();
	if( redrawBG ) activeGui->drawBackground();
      }

    dpy->showImage_->draw( dpy->renderWindowSize_, dpy->fullScreenMode_ );
    if( dpy->fullScreenMode_ )
      activeGui->displayText(fontbase, 133, 333, buf, Color(1,1,1));
    else
      activeGui->displayText(fontbase, 5, 5, buf, Color(1,1,1));

    if( dpy->fullScreenMode_ ) {
      glutSwapBuffers();
      // Need to draw into other buffer so that as we "continuously"
      // flip them, it doesn't look bad.
      if( redrawBG ) activeGui->drawBackground();
      dpy->showImage_->draw( dpy->renderWindowSize_, dpy->fullScreenMode_ );
      activeGui->displayText(fontbase, 133, 333, buf, Color(1,1,1));
    }

    if( activeGui->displayRStats_ )
      {
	activeGui->drawrstats(dpy->nworkers, dpy->workers_,
			      priv->showing_scene, fontbase2, 
			      priv->xres, priv->yres,
			      fontInfo2, priv->left, priv->up,
			      0.0 /* dt */);
      }
    if( activeGui->displayPStats_ )
      {
	Stats * mystats = dpy->drawstats[!priv->showing_scene];
	activeGui->drawpstats(mystats, dpy->nworkers, dpy->workers_, 
			      /*draw_framerate*/true, priv->showing_scene,
			      fontbase, lasttime, cum_ttime,
			      cum_dt);
      }

    dpy->showImage_ = NULL;

    // Let the Dpy thread start drawing the next image.
    //activeGui->priv->waitDisplay->unlock();
    if( activeGui->mainWindowVisible ) {
      activeGui->update(); // update the gui each time a frame is finished.
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
Gui::handleKeyPressCB( unsigned char key, int /*mouse_x*/, int /*mouse_y*/ )
{
  // static double FPS = 15;

  DpyPrivate * priv = activeGui->priv;

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
  activeGui->shiftDown_ = mods & GLUT_ACTIVE_SHIFT;
  activeGui->altDown_   = mods & GLUT_ACTIVE_ALT;
  activeGui->ctrlDown_  = mods & GLUT_ACTIVE_CTRL;

  switch( key ){

  // KEYPAD KEYS USED FOR MOVEMENT

  case '+':
    if (activeGui->shiftDown_) {
      // increase planet orbit speed
      if (ORBIT_SPEED<.02) ORBIT_SPEED=1;
      else ORBIT_SPEED*=1.9;
      cerr << "orbit speed: " << ORBIT_SPEED << endl;
    } else if (activeGui->ctrlDown_) {
      // increase planet rotate speed
      ROTATE_SPEED*=1.1;
    } else {
      // SPEED up or slow down
      activeGui->stealth_->accelerate();
    }
    break;
  case '-':
    if (activeGui->shiftDown_) {
      // decrease planet orbit speed
      if (ORBIT_SPEED<.1) ORBIT_SPEED=0;
      else ORBIT_SPEED*=.6;
      cerr << "orbit speed: " << ORBIT_SPEED << endl;
    } else if (activeGui->ctrlDown_) {
      // decrease planet rotate speed
      ROTATE_SPEED*=.6;
    } else {
      activeGui->stealth_->decelerate();
    }
    break;
  // PITCH up and down
  case '8':
    cout << "pitchdown\n";
    activeGui->stealth_->pitchDown();
    break;
  case '2':
    cout << "pitchup\n";
    activeGui->stealth_->pitchUp();
    break;
  // SLIDE left and right
  case '9':
    activeGui->stealth_->slideRight();
    break;
  case '7':
    activeGui->stealth_->slideLeft();
    break;
  // TURN left and right
  case '4':
    activeGui->stealth_->turnLeft();
    break;
  case '5':    // STOP rotations (pitch/turn)
    activeGui->stealth_->stopPitchAndRotate();
    break;
  case '6':
    activeGui->stealth_->turnRight();
    break;
  // SLOW down and STOP
  case '.':
    activeGui->stealth_->slowDown();
    break;
  case ' ':
  case '0':
    activeGui->stealth_->stopAllMovement();
    break;
  // ACCELERATE UPWARDS or DOWNWARDS
  case '*': 
    activeGui->stealth_->goUp();   // Accelerate UP
    break;
  case '/': 
    activeGui->stealth_->goDown(); // Accelerate DOWN
    break;
  case 'q':
    activeGui->quit();
    break;
  case 'G': // Toggle Display of Gui
    activeGui->toggleGui();
    break;
  case 'g':
    cout << "Toggling Gravity.  If you want to increase framerate, use 'F'\n";
    activeGui->stealth_->toggleGravity();
    break;
  case 't':
    if( activeGui->dpy_->scene->hotSpotMode_ )
      activeGui->dpy_->scene->hotSpotMode_ = 0;
    else
      activeGui->dpy_->scene->hotSpotMode_ = 1;
    break;
  case 'T':
    if( activeGui->dpy_->scene->hotSpotMode_ )
      activeGui->dpy_->scene->hotSpotMode_ = 0;
    else
      activeGui->dpy_->scene->hotSpotMode_ = 2;
    break;
  case 'Q':
    activeGui->beQuiet_ = !activeGui->beQuiet_;
    break;
  case 's':
    activeGui->cycleShadowMode();
    break;
  case 'h':
    activeGui->cycleAmbientMode();
    break;
  case 'z':
    handleMenuCB( TOGGLE_RIGHT_BUTTON_MENU );
    break;
  case 'v':
    {
      if(activeGui->priv->followPath) { activeGui->priv->followPath = false; }
      activeGui->stealth_->stopAllMovement();

      // Animate lookat point to center of BBox...
      Object* obj= activeGui->dpy_->scene->get_object();
      BBox bbox;
      obj->compute_bounds(bbox, 0);
      if(bbox.valid()){
	activeGui->camera_->set_lookat(bbox.center());
        
	// Move forward/backwards until entire view is in scene...
	// change this a little, make it so that the FOV must
	// be 60 deg...
	// 60 degrees sucks - try 40...
        // Let user specify using gui.

	const double FOVtry = activeGui->fovValue_;

	Vector diag(bbox.diagonal());
	double w=diag.length();
	Vector lookdir(activeGui->camera_->get_lookat() -
		       activeGui->camera_->get_eye()); 
	lookdir.normalize();
	const double scale = 1.0/(2*tan(DtoR(FOVtry/2.0)));
	double length = w*scale;
	activeGui->camera_->set_fov(FOVtry);
	activeGui->camera_->set_eye( activeGui->camera_->get_lookat() -
				    lookdir*length );
	activeGui->camera_->setup();
	activeGui->fovSpinner_->set_float_val( FOVtry );

	Point origin;
	Vector lookdir2;
	Vector up;
	Vector side;
	double fov;
	activeGui->camera_->getParams(origin, lookdir2, up, side, fov);
	lookdir2.normalize();
	up.normalize();
	side.normalize();
	// Move the lights that are fixed to the eye
	for(int i = 0; i < activeGui->dpy_->scene->nlights(); i++) {
	  Light *light = activeGui->dpy_->scene->light(i);
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
    if (activeGui->shiftDown_) {
      // toggle holo room on/off
      activeGui->dpy_->holoToggle_ = !activeGui->dpy_->holoToggle_;
      cout << "holo room is now " << activeGui->dpy_->holoToggle_ << endl;
    } else {
      activeGui->camera_->flatten(); // Right yourself (0 pitch, 0 roll)
    }
    break;
  case 'x':
    traverseRouteCB(-1);
    break;
  case 'a':
    activeGui->priv->animate =! activeGui->priv->animate;
    cout << "animate is now " << activeGui->priv->animate << "\n";
    break;
  case 'c':
    activeGui->camera_->print();
    break;
  case 'n':
    activeGui->camera_->scale_eyesep(0.9);
    cerr << "camera->eyesep="<<activeGui->camera_->get_eyesep()<<"\n";
    break;
  case 'm':
    activeGui->camera_->scale_eyesep(1.1);
    cerr << "camera->eyesep="<<activeGui->camera_->get_eyesep()<<"\n";
    break;
  case 'o':
    printf("Number materials: %d\n",activeGui->dpy_->scene->nmaterials());
    for (int m=0; m<activeGui->dpy_->scene->nmaterials(); m++) {
      CycleMaterial * cm =
	dynamic_cast<CycleMaterial*>(activeGui->dpy_->scene->get_material(m));
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
    activeGui->dpy_->nstreams++;
    break;
  case 'E':
    if(activeGui->dpy_->nstreams > 1)
      activeGui->dpy_->nstreams--;
    break;

  case 'r':
    activeGui->displayRStats_ = !activeGui->displayRStats_;
    break;
  case 'p':
    activeGui->displayPStats_ = !activeGui->displayPStats_;
    break;

  case 'f':
    if( activeGui->dpy_->fullScreenMode_ )
      activeGui->dpy_->toggleRenderWindowSize_ = true;
    else
      cout << "Can't toggle to full res on non-full screen mode.\n";
    break;

  case 27: // Escape key... need to find a symbolic name for this...
    activeGui->quit();
    break;
#if 0
    // below is for blending "pixels" in
    // frameless rendering...

  case 'y': // sychronizing mode for frameless...
    synch_frameless = !synch_frameless;  //1-synch_frameless;
    //doing_frameless = 1-doing_frameless; // just toggle...
    cerr << synch_frameless << " Synch?\n";
    break;
  case '1':
    cout << "NOTICE: Use 2 key to toggle Stereo\n";
    cout << "      : 1 key is deprecated and may go away\n";
    break;
  case '2':
    stereo=!stereo;
    break;

    FPS -= 1;
    if (FPS <= 0.0) FPS = 1.0;
    FrameRate = 1.0/FPS;
    break;

    FPS += 1.0;
    FrameRate = 1.0/FPS;
    cerr << FPS << endl;
    break;
#endif
  case 'W':
    cerr << "Saving raw image file\n";
    activeGui->dpy_->scene->get_image(showing_scene)->save("images/image.raw");
    break;
  case 'w':
    cerr << "Saving ppm image file\n";
    activeGui->dpy_->scene->get_image(showing_scene)->save_ppm("images/image.ppm");
    break;
  default:
    printf("unknown regular key %d\n", key);
    break;
  }
} // end handleKeyPress();

// WARNING: THESE ARE NOT THE KEYPAD KEYS!
void
Gui::handleSpecialKeyCB( int key, int /*mouse_x*/, int /*mouse_y*/ )
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
Gui::handleMousePress(int button, int mouse_x, int mouse_y)
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
    Camera *C = activeGui->dpy_->scene->get_camera( 0 );
    Ray ray;
    C->makeRay( ray, mouse_x, activeGui->dpy_->priv->yres-mouse_y, 
		1.0/activeGui->dpy_->priv->xres,
		1.0/activeGui->dpy_->priv->yres );
    HitInfo hit;
    activeGui->dpy_->scene->get_object()->intersect( ray, hit, ds,
  						     activeGui->dpy_->ppc );
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

  activeGui->last_x_ = mouse_x;
  activeGui->last_y_ = mouse_y;

  switch(button){
  case GLUT_MIDDLE_BUTTON:
			
    // Find the center of rotation...
    double rad = 0.8;
    HVect center(0,0,0,1.0);
			
    // we also want to keep the old transform information
    // around (so stuff correlates correctly)
			
    Vector y_axis,x_axis;
    activeGui->camera_->get_viewplane(y_axis, x_axis);
    Vector z_axis(activeGui->camera_->eye - activeGui->camera_->lookat);

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
Gui::handleMouseRelease(int button, int mouse_x, int mouse_y)
{
  DpyPrivate * priv      = activeGui->priv;
  double     & last_time = priv->last_time;

  static Stats *gui_stats = new Stats(1);
  static DepthStats *ds = new DepthStats(gui_stats->ds[0]);
  Object *current_obj;
  if( activeGui->shiftDown_ ) {
    Camera *C = activeGui->dpy_->scene->get_camera( 0 );
    Ray ray;
    C->makeRay( ray, mouse_x, activeGui->dpy_->priv->yres-mouse_y, 
		1.0/activeGui->dpy_->priv->xres,
		1.0/activeGui->dpy_->priv->yres );
    HitInfo hit;
    activeGui->dpy_->scene->get_object()->intersect( ray, hit, ds,
  						     activeGui->dpy_->ppc );
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
    activeGui->fovSpinner_->set_float_val( activeGui->camera_->get_fov() );
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
	    
	activeGui->camera_->up  = y_a;
	activeGui->camera_->eye = activeGui->camera_->lookat+z_a*eye_dist;
	activeGui->camera_->setup();
	prev_trans = prv;

	Point origin;
	Vector lookdir;
	Vector up;
	Vector side;
	double fov;
	activeGui->camera_->getParams(origin, lookdir, up, side, fov);
	lookdir.normalize();
	up.normalize();
	side.normalize();
	// Move the lights that are fixed to the eye
	for(int i = 0; i < activeGui->dpy_->scene->nlights(); i++) {
	  Light *light = activeGui->dpy_->scene->light(i);
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
Gui::handleWindowResizeCB( int width, int height )
{
  //  printf("window resized\n");
  // This is an ugly, cheaty way of getting the window id out of glut...
  static bool first=true;
  if(first){
    extern Window** __glutWindowList;
    Window win = __glutWindowList[activeGui->glutDisplayWindowId-1][1];
    cerr << "winid=" << (void*)win << '\n';
    activeGui->dpy_->release(win);
    first=false;
  }
  if (activeGui->dpy_->display_frames) {
    activeGui->dpy_->priv->xres=width;
    activeGui->dpy_->priv->yres=height;
    activeGui->camera_->setWindowAspectRatio((double)height/width);
  }
  
#if 0
  glutSetWindow( activeGui->glutDisplayWindowId );

  glViewport(0, 0, activeGui->dpy_->priv->xres, activeGui->dpy_->priv->yres);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, activeGui->dpy_->priv->xres, 0, activeGui->dpy_->priv->yres);
  glDisable( GL_DEPTH_TEST );
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0.375, 0.375, 0.0);
#endif

  return;

#if 0
  DpyPrivate * priv = activeGui->priv;

  // Resize the image...
  priv->xres = width;
  priv->yres = height;

  glViewport(0, 0, priv->xres, priv->yres);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, priv->xres, 0, priv->yres);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0.375, 0.375, 0.0);

  printf("done window resized\n");
#endif
}

void
Gui::handleMouseCB(int button, int state, int x, int y)
{
  if( state == GLUT_DOWN ) {
    activeGui->handleMousePress( button, x, y );
  } else {
    activeGui->handleMouseRelease( button, x, y );
  }
}

void
Gui::handleMouseMotionCB( int mouse_x, int mouse_y )
{
  int          last_x = activeGui->last_x_;
  int          last_y = activeGui->last_y_;
  DpyPrivate * priv   = activeGui->priv;

  double     & last_time = priv->last_time;
  BallData  *& ball      = priv->ball;

  static Stats *gui_stats = new Stats(1);
  static DepthStats *ds = new DepthStats(gui_stats->ds[0]);
  Object *current_obj;
  if( activeGui->shiftDown_ ) {
    Camera *C = activeGui->dpy_->scene->get_camera( 0 );
    Ray ray;
    C->makeRay( ray, mouse_x, activeGui->dpy_->priv->yres-mouse_y, 
		1.0/activeGui->dpy_->priv->xres,
		1.0/activeGui->dpy_->priv->yres );
    HitInfo hit;
    activeGui->dpy_->scene->get_object()->intersect( ray, hit, ds,
  						     activeGui->dpy_->ppc );
    if( hit.was_hit ) {
      current_obj = hit.hit_obj;
//        cout << "Mouse moving on object "<<current_obj->get_name()<<endl;
      cbFunc mouseCB = MouseCallBack::getCB_MM( current_obj );
      if (mouseCB)
	mouseCB( current_obj, ray, hit );
    }
    return;
  }

  switch( activeGui->mouseDown_ ) {
  case GLUT_RIGHT_BUTTON:
    {
      if( activeGui->rightButtonMenuActive_ )
	{ // Note: This should actually never be the case as the right 
	  // button menu gets the click if it is active.
	  return;
	}
      if( activeGui->shiftDown_ )
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
	  Vector dir = activeGui->camera_->lookat - activeGui->camera_->eye;
	  activeGui->camera_->eye += dir*scl;

	  Point origin;
	  Vector lookdir;
	  Vector up;
	  Vector side;
	  double fov;
	  activeGui->camera_->getParams(origin, lookdir, up, side, fov);
	  lookdir.normalize();
	  up.normalize();
	  side.normalize();
	  // Move the lights that are fixed to the eye
	  for(int i = 0; i < activeGui->dpy_->scene->nlights(); i++) {
	    Light *light = activeGui->dpy_->scene->light(i);
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

	  double fov = RtoD(2*atan(scl*tan(DtoR(activeGui->camera_->fov/2.))));
	  if( fov < MIN_FOV )
	    fov = MIN_FOV;
	  else if( fov > MAX_FOV )
	    fov = MAX_FOV;
	  activeGui->camera_->set_fov( fov );
	}
      activeGui->camera_->setup();
    }
    break;
  case GLUT_LEFT_BUTTON:
    {
      double xmtn =  double(last_x-mouse_x)/double(priv->xres);
      double ymtn = -double(last_y-mouse_y)/double(priv->yres);

      Vector u,v;
      activeGui->camera_->get_viewplane(u, v);
      Vector trans(u*ymtn+v*xmtn);

      // Translate the view...
      activeGui->camera_->eye+=trans;
      activeGui->camera_->lookat+=trans;
      activeGui->camera_->setup();

      Point origin;
      Vector lookdir;
      Vector up;
      Vector side;
      double fov;
      activeGui->camera_->getParams(origin, lookdir, up, side, fov);
      lookdir.normalize();
      up.normalize();
      side.normalize();
      // Move the lights that are fixed to the eye
      for(int i = 0; i < activeGui->dpy_->scene->nlights(); i++) {
	Light *light = activeGui->dpy_->scene->light(i);
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

      activeGui->camera_->up  = y_a;
      activeGui->camera_->eye = activeGui->camera_->lookat+z_a*eye_dist;
      activeGui->camera_->setup();
			
      Point origin;
      Vector lookdir;
      Vector up;
      Vector side;
      double fov;
      activeGui->camera_->getParams(origin, lookdir, up, side, fov);
      lookdir.normalize();
      up.normalize();
      side.normalize();
      // Move the lights that are fixed to the eye
      for(int i = 0; i < activeGui->dpy_->scene->nlights(); i++) {
	Light *light = activeGui->dpy_->scene->light(i);
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

  activeGui->last_x_ = mouse_x;
  activeGui->last_y_ = mouse_y;

} // end handleMouseMotion()

void
Gui::handleSpaceballMotionCB( int sbm_x, int sbm_y, int sbm_z )
{
  double sensitivity = 100.0;

  if( abs(sbm_x) > 2 )
    activeGui->camera_->moveLaterally( sbm_x / sensitivity );
  if( abs(sbm_y) > 2 )
    activeGui->camera_->moveVertically( sbm_y / sensitivity );
  if( abs(sbm_z) > 2 )
    activeGui->camera_->moveForwardOrBack( sbm_z / sensitivity );
}

void
Gui::handleSpaceballRotateCB( int sbr_x, int sbr_y, int sbr_z )
{
  double sensitivity = 1000.0;

  if( abs(sbr_x) > 2 )
    activeGui->camera_->changePitch( sbr_x / (sensitivity*2) );
  if( abs(sbr_y) > 2 )
    activeGui->camera_->changeFacing( sbr_y / sensitivity );
  // Don't allow roll (at least for now)
}

void
Gui::handleSpaceballButtonCB( int button, int /*state*/ )
{
  cout << "spaceball button: " << button << "\n"; 
}


void
Gui::toggleRoutesWindowCB( int /*id*/ )
{
  if( activeGui->routeWindowVisible )
    activeGui->routeWindow->hide();
  else
    activeGui->routeWindow->show();
  activeGui->routeWindowVisible = !activeGui->routeWindowVisible;
}

void
Gui::toggleLightsWindowCB( int /*id*/ )
{
  if( activeGui->lightsWindowVisible )
    activeGui->lightsWindow->hide();
  else
    activeGui->lightsWindow->show();
  activeGui->lightsWindowVisible = !activeGui->lightsWindowVisible;
}

void
Gui::toggleObjectsWindowCB( int /*id*/ )
{
  if( activeGui->objectsWindowVisible )
    activeGui->objectsWindow->hide();
  else
    activeGui->objectsWindow->show();
  activeGui->objectsWindowVisible = !activeGui->objectsWindowVisible;
}

void
Gui::toggleTriggersWindowCB( int /*id*/ )
{
  if( activeGui->triggersWindowVisible )
    activeGui->triggersWindow_->hide();
  else
    activeGui->triggersWindow_->show();
  activeGui->triggersWindowVisible = !activeGui->triggersWindowVisible;
}

void
Gui::toggleSoundWindowCB( int /*id*/ )
{
  if( activeGui->soundsWindowVisible )
    activeGui->soundsWindow->hide();
  else
    {
      activeGui->soundsWindow->show();
      updateSoundCB( -1 );
    }
  activeGui->soundsWindowVisible = !activeGui->soundsWindowVisible;
}

void
Gui::updateLightPanelCB( int /*id*/ )
{
  if( activeGui->lights_.size() == 0 ) return;

  Light * light = activeGui->lights_[ activeGui->selectedLightId_ ];
  const Color & color = light->getOrigColor();
  const Point & pos   = light->get_pos();

  activeGui->r_color_spin->set_float_val( color.red() );
  activeGui->g_color_spin->set_float_val( color.green() );
  activeGui->b_color_spin->set_float_val( color.blue() );

  activeGui->lightIntensity_->set_float_val( light->get_intensity() );

  activeGui->lightPosX_->set_float_val( pos.x() );
  activeGui->lightPosY_->set_float_val( pos.y() );
  activeGui->lightPosZ_->set_float_val( pos.z() );

  if( light->isOn() )
    {
      activeGui->lightOnOffBtn_->set_name( "Turn Off" );
      activeGui->lightsColorPanel_->enable();
      activeGui->lightsPositionPanel_->enable();
    }
  else
    {
      activeGui->lightOnOffBtn_->set_name( "Turn On" );
      activeGui->lightsColorPanel_->disable();
      activeGui->lightsPositionPanel_->disable();
    }
}

void
Gui::updateRouteCB( int /*id*/ )
{
  activeGui->stealth_->selectPath( activeGui->selectedRouteId_ );
  goToRouteBeginningCB( -1 );
}


void
Gui::updateSoundCB( int /*id*/ )
{
#if !defined(linux)
  activeGui->currentSound_ = activeGui->sounds_[ activeGui->selectedSoundId_ ];

  Point & location = activeGui->currentSound_->locations_[0];

  activeGui->soundOriginX_->set_float_val( location.x() );
  activeGui->soundOriginY_->set_float_val( location.y() );
  activeGui->soundOriginZ_->set_float_val( location.z() );
#endif
}

void
Gui::toggleShowLightsCB( int /*id*/ )
{
  if( activeGui->lightsBeingRendered_ ) {
    activeGui->toggleShowLightsBtn_->set_name( "Show Lights" );
    activeGui->dpy_->showLights_ = false;
    activeGui->lightsBeingRendered_ = false;
  } else {
    activeGui->toggleShowLightsBtn_->set_name( "Hide Lights" );
    activeGui->dpy_->showLights_ = true;
    activeGui->lightsBeingRendered_ = true;
  }
}

void
Gui::toggleLightOnOffCB( int /*id*/ )
{
  Light * light = activeGui->lights_[ activeGui->selectedLightId_ ];
  if( light->isOn() )
    {
      // turn it off
      activeGui->lightOnOffBtn_->set_name( "Turn On" );
      light->turnOff();
      activeGui->lightsColorPanel_->disable();
      activeGui->lightsPositionPanel_->disable();
    }
  else
    {
      // turn it on
      activeGui->lightOnOffBtn_->set_name( "Turn Off" );
      light->turnOn();
      activeGui->lightsColorPanel_->enable();
      activeGui->lightsPositionPanel_->enable();
    }
}

void
Gui::toggleLightSwitchesCB( int /*id*/ )
{
  if( activeGui->lightsOn_ ) {
    activeGui->toggleLightsOnOffBtn_->set_name( "Turn On Lights" );
    activeGui->dpy_->turnOffAllLights_ = true;
    activeGui->dpy_->turnOnAllLights_ = false;
    activeGui->lightsOn_ = false;
    activeGui->lightsColorPanel_->disable();
    activeGui->lightsPositionPanel_->disable();
  } else {
    activeGui->toggleLightsOnOffBtn_->set_name( "Turn Off Lights" );
    activeGui->dpy_->turnOnAllLights_ = true;
    activeGui->dpy_->turnOffAllLights_ = false;
    activeGui->lightsOn_ = true;
    activeGui->lightsColorPanel_->enable();
    activeGui->lightsPositionPanel_->enable();
  }
}

void
Gui::updateAmbientCB( int /*id*/ )
{
  activeGui->dpy_->scene->setAmbientLevel( activeGui->ambientBrightness_ );
}

void
Gui::createRouteWindow( GLUI * window )
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
Gui::createGetStringWindow( GLUI * window )
{
  getStringPanel = window->add_panel( "" );

  getStringText_ = 
    window->add_edittext_to_panel( getStringPanel, "", GLUI_EDITTEXT_TEXT,
				 &(activeGui->inputString_) );

  GLUI_Panel * buttonsPanel = window->add_panel_to_panel( getStringPanel, "" );

  getStringButton = 
    window->add_button_to_panel( buttonsPanel, "OK", CLOSE_GETSTRING_BTN );

  window->add_column_to_panel( buttonsPanel );

  window->add_button_to_panel( buttonsPanel, "Cancel", 
			       -1, hideGetStringWindowCB );
}

void
Gui::createObjectWindow( GLUI * window )
{
  GLUI_Panel * panel = window->add_panel( "Objects" );
					    
  Array1<Object*> & objects = dpy_->scene->objectsOfInterest_;
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
	int callback_info_id = callback_info_list.size() - 1;
	
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
Gui::createSoundsWindow( GLUI * window )
{
  GLUI_Panel * panel = window->add_panel( "Sounds" );

  soundList_ = window->add_listbox_to_panel( panel, "Selected Sound",
					     &selectedSoundId_,
					     SOUND_LIST_ID, updateSoundCB );

  GLUI_Panel * soundOriginPanel = window->
    add_panel_to_panel( panel, "Location" );

  activeGui->soundOriginX_ = window->add_edittext_to_panel
    ( soundOriginPanel, "X position:", GLUI_EDITTEXT_FLOAT );
  activeGui->soundOriginY_ = window->add_edittext_to_panel
    ( soundOriginPanel, "Y position:", GLUI_EDITTEXT_FLOAT );
  activeGui->soundOriginZ_ = window->add_edittext_to_panel
    ( soundOriginPanel, "Z position:", GLUI_EDITTEXT_FLOAT );


  GLUI_Panel * volumePanel = window->add_panel_to_panel( panel, "Volume" );

  currentSound_ = sounds_[0];

#ifndef linux
  sounds_ = dpy_->scene->getSounds();
  for( int num = 0; num < sounds_.size(); num++ )
    {
      char name[ 1024 ];
      sprintf( name, "%s", sounds_[ num ]->getName().c_str() );
      soundList_->add_item( num, name );
    }
#endif

  activeGui->leftVolume_ = window->
    add_edittext_to_panel( volumePanel, "Left:", GLUI_EDITTEXT_FLOAT );

  activeGui->rightVolume_ = window->
    add_edittext_to_panel( volumePanel, "Right:", GLUI_EDITTEXT_FLOAT );

  window->add_button_to_panel( panel, "Close",
			       -1, toggleSoundWindowCB );
}

void
Gui::createTriggersWindow( GLUI * window )
{
  vector<Trigger*> triggers = dpy_->scene->getTriggers();

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
Gui::activateTriggerCB( int /* id */ )
{
  if( activeGui->selectedTriggerId_ == -1 ) return;
  vector<Trigger*> triggers = activeGui->dpy_->scene->getTriggers();
  Trigger * trig = triggers[ activeGui->selectedTriggerId_ ];

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

      if( activeGui->activeMTT_ ) // If a trigger is already running...
	{
	  cout << "QUEUING trigger: " << trig->getName() << "\n";
	  activeGui->activeMTT_->deactivate(); // then tell it to stop.
	  activeGui->queuedMTT_ = trig;        // and queue up the new trigger.
	}
      else
	{
	  cout << "activating TRIGGER: " << trig->getName() << "\n";
	  activeGui->activeMTT_ = trig;
	  activeGui->activeMTT_->activate();
	}
    }
}

void
Gui::startSoundThreadCB( int /*id*/ )
{
#if !defined(linux)
  activeGui->startSoundThreadBtn_->disable();
  activeGui->startSoundThreadBtn_->set_name( "Starting Sounds" );

  SoundThread * soundthread = NULL;

  cout << "Starting Sound Thread!\n";
  soundthread = new SoundThread( activeGui->dpy_->getGuiCam(), 
				 activeGui->dpy_->scene,
				 activeGui );
  Thread * t = new Thread( soundthread, "Sound thread");
  t->detach();
#endif
}

void
Gui::soundThreadNowActive()
{
  activeGui->enableSounds_ = true;
}

void Gui::SGAutoCycleCB( int id ) {
  SGCallbackInfo* sgcbi = (SGCallbackInfo*)callback_info_list[id];
  sgcbi->sg->toggleAutoswitch();
  // Now get the current timestep and then display it
  sgcbi->current_frame_spinner->set_int_val(sgcbi->sg->GetChild());
}

void Gui::SGNoSkipCB( int id ) {
  SelectableGroup *sg = ((SGCallbackInfo*)callback_info_list[id])->sg;
  sg->toggleNoSkip();
}

void Gui::SGNextItemCB( int id )
{
  SGCallbackInfo* sgcbi = (SGCallbackInfo*)callback_info_list[id];
  sgcbi->sg->nextChild();
  // Now get the current timestep and then display it
  sgcbi->current_frame_spinner->set_int_val(sgcbi->sg->GetChild());

  Object * newObj = sgcbi->sg->getCurrentChild();
  if( Names::hasName( newObj ) ) {
    Trigger * trig = NULL;

    if( Names::getName(newObj) == "Visible Female Volume" ) {
      trig = activeGui->visWomanTrig_;
    } else if( Names::getName(newObj) == "Brain Volume" ) {
      trig = NULL;
    } else if( Names::getName(newObj) == "CSAFE Fire Volume" ) {
      trig = activeGui->csafeTrig_;
    } else if( Names::getName(newObj) == "Geological Volume" ) {
      trig = activeGui->geophysicsTrig_;
    } else if( Names::getName(newObj) == "Sheep Heart Volume" ) {
      trig = NULL;
    }

    if( trig ) {
      trig->setPriority( Trigger::HighTriggerPriority );
      if( activeGui->activeMTT_ ) // If a trigger is already running...
	{
	  activeGui->activeMTT_->deactivate(); // then tell it to stop.
	  activeGui->queuedMTT_ = trig;        // and queue up the new trigger.
	}
      else // just start this trigger
	{
	  activeGui->activeMTT_ = trig;
	  activeGui->activeMTT_->activate();
	}
    }
  }
}

void Gui::SGCurrentFrameCB( int id ) {
  SGCallbackInfo* sgcbi = (SGCallbackInfo*)callback_info_list[id];
  sgcbi->sg->SetChild(sgcbi->current_frame_spinner->get_int_val());
}

void Gui::SISpinCB( int id ) {
  Array1<Object*> & objects = activeGui->dpy_->scene->objectsOfInterest_;
  SpinningInstance *obj = dynamic_cast<SpinningInstance*>(objects[id]);  
  obj->toggleDoSpin();      
}
void Gui::SIIncMagCB( int id ) {
  Array1<Object*> & objects = activeGui->dpy_->scene->objectsOfInterest_;
  SpinningInstance *obj = dynamic_cast<SpinningInstance*>(objects[id]);  
  obj->incMagnification();      
}
void Gui::SIDecMagCB( int id ) {
  Array1<Object*> & objects = activeGui->dpy_->scene->objectsOfInterest_;
  SpinningInstance *obj = dynamic_cast<SpinningInstance*>(objects[id]);  
  obj->decMagnification();      
}
void Gui::SISlideUpCB( int id ) {
  Array1<Object*> & objects = activeGui->dpy_->scene->objectsOfInterest_;
  SpinningInstance *obj = dynamic_cast<SpinningInstance*>(objects[id]);  
  obj->upPole();      
}
void Gui::SISlideDownCB( int id ) {
  Array1<Object*> & objects = activeGui->dpy_->scene->objectsOfInterest_;
  SpinningInstance *obj = dynamic_cast<SpinningInstance*>(objects[id]);  
  obj->downPole();      
}

void Gui::CGOnCB( int id ) {
  Array1<Object*> & objects = activeGui->dpy_->scene->objectsOfInterest_;
  CutGroup *obj = dynamic_cast<CutGroup*>(objects[id]);  
  obj->toggleOn();
}

void Gui::CGSpinCB( int id ) {
  Array1<Object*> & objects = activeGui->dpy_->scene->objectsOfInterest_;
  CutGroup *obj = dynamic_cast<CutGroup*>(objects[id]);  
  obj->toggleAnimate();
}

void
Gui::addLight( Light * light )
{
  string & name = light->name_;
  if( name != "" ) {
    int numLights = lights_.size();
    char namec[1024];
    sprintf( namec, "%s", name.c_str() );

    lightList->add_item( numLights, namec );
    lights_.push_back( light );

    updateLightPanelCB( -1 );
  }
}

void
Gui::createLightWindow( GLUI * window )
{
  GLUI_Panel * panel = window->add_panel( "Lights" );

  activeGui->ambientBrightness_ = activeGui->dpy_->scene->getAmbientLevel();

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
			  &(activeGui->lightX_),
			  LIGHT_X_POS_ID, updateLightPositionCB );
  lightPosY_ =
    window->add_spinner_to_panel(lightsPositionPanel_,"Y:",GLUI_SPINNER_FLOAT,
			  &(activeGui->lightY_),
			  LIGHT_Y_POS_ID, updateLightPositionCB );
  lightPosZ_ =
    window->add_spinner_to_panel(lightsPositionPanel_,"Z:",GLUI_SPINNER_FLOAT,
			  &(activeGui->lightZ_),
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
  GLUI_Button * gotoLightBtn =
    window->add_button_to_panel(moreControls, "Goto Light" );
  gotoLightBtn->disable();

  window->add_button_to_panel( panel, "Close",
			       1, toggleLightsWindowCB );
}


void
Gui::createMenus( int winId, bool soundOn /* = false */,
		  bool showGui /* = true */ )
{
  printf("createmenus\n");

  // Need to do this at this point as glut has now been initialized.
  activeGui->setupFonts();

  activeGui->glutDisplayWindowId = winId;

  /*int modemenu = */glutCreateMenu( Gui::handleMenuCB );

  glutAddMenuEntry( "Toggle Gui [G]", TOGGLE_GUI );
  glutAddMenuEntry( "Toggle Hot Spots [t]", TOGGLE_HOT_SPOTS );
  glutAddMenuEntry( "----------------", -1);
  glutAddMenuEntry( "Toggle On/Off This Menu [z]", TOGGLE_RIGHT_BUTTON_MENU );
  glutAddMenuEntry( "----------------", -1);
  glutAddMenuEntry( "Quit [q]", QUIT_MENU_ID );
  //glutAddSubMenu("Texture mode", modemenu);
  glutAttachMenu(GLUT_RIGHT_BUTTON);

  // Build GLUI Windows
  activeGui->mainWindow = GLUI_Master.create_glui( "RTRT Controls",
						   0, 804, 0 );
  if( !showGui ){
    activeGui->mainWindow->hide();
    activeGui->mainWindowVisible = false;
  }

  activeGui->routeWindow     = GLUI_Master.create_glui( "Route",   0,900,400 );
  activeGui->lightsWindow    = GLUI_Master.create_glui( "Lights",  0,900,500 );
  activeGui->objectsWindow   = GLUI_Master.create_glui( "Objects", 0,900,600 );
  activeGui->soundsWindow    = GLUI_Master.create_glui( "Sounds",  0,900,700 );
  activeGui->triggersWindow_ = GLUI_Master.create_glui( "Triggers",0,900,800 );

  activeGui->getStringWindow = 
                    GLUI_Master.create_glui( "Input Request", 0, 900, 600 );

  //  activeGui->routeWindow->set_main_gfx_window( winId );
  //  activeGui->lightsWindow->set_main_gfx_window( winId );
  //  activeGui->objectsWindow->set_main_gfx_window( winId );
  //  activeGui->mainWindow->set_main_gfx_window( winId );

  activeGui->routeWindow->hide();
  activeGui->lightsWindow->hide();
  activeGui->objectsWindow->hide();
  activeGui->soundsWindow->hide();
  activeGui->triggersWindow_->hide();

  activeGui->getStringWindow->hide();

  activeGui->createRouteWindow( activeGui->routeWindow );
  activeGui->createLightWindow( activeGui->lightsWindow );
  activeGui->createObjectWindow( activeGui->objectsWindow );
  activeGui->createTriggersWindow( activeGui->triggersWindow_ );
  activeGui->createGetStringWindow( activeGui->getStringWindow );

  /////////////////////////////////////////////////////////
  // Main Panel
  //
  GLUI_Panel * main_panel   = activeGui->mainWindow->add_panel( "" );
  //  GLUI_Panel * button_panel = activeGui->mainWindow->add_panel( "" );

  /////////////////////////////////////////////////////////
  // Display Parameters Panel
  //
  GLUI_Panel *display_panel = activeGui->mainWindow->
    add_panel_to_panel( main_panel, "Display Parameters" );

  // FPS
  activeGui->framesPerSecondTxt = activeGui->mainWindow->
    add_edittext_to_panel( display_panel, "FPS:", GLUI_EDITTEXT_FLOAT );

  // Shadows
  GLUI_Panel * shadows = 
    activeGui->mainWindow->add_panel_to_panel( display_panel, "Shadows" );
  activeGui->shadowModeLB_ = activeGui->mainWindow->
    add_listbox_to_panel( shadows, "Mode:", &activeGui->dpy_->shadowMode_ );
  activeGui->shadowModeLB_->add_item( No_Shadows, 
				      ShadowBase::shadowTypeNames[0] );
  activeGui->shadowModeLB_->add_item( Single_Soft_Shadow,
				      ShadowBase::shadowTypeNames[1] );
  activeGui->shadowModeLB_->add_item( Hard_Shadows,
				      ShadowBase::shadowTypeNames[2] );
  activeGui->shadowModeLB_->add_item( Glass_Shadows,
				      ShadowBase::shadowTypeNames[3] );
  activeGui->shadowModeLB_->add_item( Soft_Shadows,
				      ShadowBase::shadowTypeNames[4] );
  activeGui->shadowModeLB_->add_item( Uncached_Shadows,
				      ShadowBase::shadowTypeNames[5] );
  activeGui->shadowModeLB_->set_int_val( activeGui->dpy_->scene->shadow_mode );

  // Ambient
  GLUI_Panel * ambient = 
    activeGui->mainWindow->add_panel_to_panel(display_panel, "Ambient Light");
  activeGui->ambientModeLB_ = activeGui->mainWindow->
    add_listbox_to_panel( ambient, "Mode:", &activeGui->dpy_->ambientMode_ );

  activeGui->ambientModeLB_->add_item( Constant_Ambient, "Constant" );
  activeGui->ambientModeLB_->add_item( Arc_Ambient, "Arc" );
  activeGui->ambientModeLB_->add_item( Sphere_Ambient, "Sphere" );
  activeGui->ambientModeLB_->set_int_val( activeGui->dpy_->ambientMode_ );

  // Background Color controls
  GLUI_Panel * bgcolor = 
    activeGui->mainWindow->add_panel_to_panel( display_panel, "Background Color" );
  activeGui->WhiteBGButton_ = activeGui->mainWindow->
    add_button_to_panel( bgcolor, "White Background",
			 WHITE_BG_BUTTON, bgColorCB);
  activeGui->BlackBGButton_ = activeGui->mainWindow->
    add_button_to_panel( bgcolor, "Black Background",
			 BLACK_BG_BUTTON, bgColorCB);
  activeGui->OrigBGButton_ = activeGui->mainWindow->
    add_button_to_panel( bgcolor, "Original Background",
			 ORIG_BG_BUTTON, bgColorCB);
  // Jitter
  GLUI_Panel * jitter = 
    activeGui->mainWindow->add_panel_to_panel( display_panel, "Jitter" );
  if (activeGui->dpy_->scene->rtrt_engine->do_jitter)
    activeGui->jitterButton_ = activeGui->mainWindow->
      add_button_to_panel( jitter, "Turn Jitter OFF",
			   TURN_ON_JITTER_BTN, toggleJitterCB );
  else
    activeGui->jitterButton_ = activeGui->mainWindow->
      add_button_to_panel( jitter, "Turn Jitter ON",
			   TURN_ON_JITTER_BTN, toggleJitterCB );

  // FOV
  activeGui->fovValue_ = activeGui->camera_->get_fov();
  activeGui->fovSpinner_ = activeGui->mainWindow->
    add_spinner_to_panel( display_panel, "FOV:", GLUI_SPINNER_FLOAT,
			   &(activeGui->fovValue_), FOV_SPINNER_ID,
			  updateFovCB );
  activeGui->fovSpinner_->set_float_limits( MIN_FOV, MAX_FOV );
  activeGui->fovSpinner_->set_speed( 0.01 );

  // Ray offset
  activeGui->ray_offset_spinner = activeGui->mainWindow->
    add_spinner_to_panel( display_panel, "Ray Offset", GLUI_SPINNER_FLOAT,
			   &(activeGui->ray_offset), 0,
			  updateRayOffsetCB );
  activeGui->ray_offset_spinner->set_float_val(activeGui->dpy_->scene->get_camera(0)->get_ray_offset());
  activeGui->ray_offset_spinner->set_float_limits( 0, 5000 );
  activeGui->ray_offset_spinner->set_speed( 0.01 );

  // Other Controls
  GLUI_Panel * otherControls = activeGui->mainWindow->
    add_panel_to_panel( display_panel, "Other Controls" );

  activeGui->mainWindow->add_button_to_panel( otherControls,
	 "Toggle Hot Spot Display", TOGGLE_HOTSPOTS_ID, toggleHotspotsCB );

  activeGui->mainWindow->add_button_to_panel( otherControls,
	 "Toggle Transmission Mode", TOGGLE_TRANSMISSION_MODE_ID,
					      toggleTransmissionModeCB );

  activeGui->numThreadsSpinner_ = activeGui->mainWindow->
    add_spinner_to_panel( otherControls, "Number of Threads",
			  GLUI_SPINNER_INT,
			  &(activeGui->dpy_->numThreadsRequested_),
			  NUM_THREADS_SPINNER_ID );
  activeGui->numThreadsSpinner_->set_speed( 0.0001 );
  activeGui->numThreadsSpinner_->set_int_limits( 1, MAX_NUM_THREADS );

  // ...This probably goes to the objects window...
  GLUI_Button * toggleMaterials = activeGui->mainWindow->
    add_button_to_panel(otherControls,"Toggle Materials");
  toggleMaterials->disable();

  // 
  activeGui->soundVolumeSpinner_ = activeGui->mainWindow->
    add_spinner_to_panel( otherControls, "Sound Volume", GLUI_SPINNER_INT, 
			  &(activeGui->dpy_->scene->soundVolume_),
			  SOUND_VOLUME_SPINNER_ID );
  activeGui->soundVolumeSpinner_->set_speed( 0.01 );
  activeGui->soundVolumeSpinner_->set_int_limits( 0, 100 );
  activeGui->soundVolumeSpinner_->disable();
#if !defined(linux)
  //adding in the start sounds button after volume spinner
  //disabled if no sounds or sounds selected in beginning
  activeGui->startSoundThreadBtn_ = activeGui->mainWindow->
    add_button_to_panel(otherControls, "Start Sounds",-1, startSoundThreadCB);
  if( activeGui->dpy_->scene->getSounds().size() == 0 )
    {
      activeGui->startSoundThreadBtn_->disable();
      activeGui->startSoundThreadBtn_->set_name( "No Sounds" );
    }
  else
    {
      activeGui->createSoundsWindow( activeGui->soundsWindow );
      if( soundOn )
	{
	  activeGui->startSoundThreadBtn_->disable();
	  startSoundThreadCB( -1 );
	}
    }
#endif
  
  activeGui->glyphThresholdSpinner_ = activeGui->mainWindow->
    add_spinner_to_panel( otherControls, "Glyph Threshold",
			  GLUI_SPINNER_FLOAT, 
			  &glyph_threshold,
			  -1);
  activeGui->glyphThresholdSpinner_->set_speed( 0.1 );
  activeGui->glyphThresholdSpinner_->set_float_limits( 0, 1 );
  
  // 
  activeGui->depthValue_ = activeGui->priv->maxdepth;
  GLUI_Spinner * depthSpinner = activeGui->mainWindow->
    add_spinner_to_panel( display_panel, "Ray Depth", GLUI_SPINNER_INT, 
			  &(activeGui->depthValue_), DEPTH_SPINNER_ID, 
			  updateDepthCB );
  if (activeGui->dpy_->scene->maxdepth > 12)
    depthSpinner->set_int_limits( 0, activeGui->dpy_->scene->maxdepth * 3/2);
  else
    depthSpinner->set_int_limits( 0, 12 );
  depthSpinner->set_speed( 0.1 );

  /////////////////////////////////////////////////////////
  // Eye Position Panel
  //
  activeGui->mainWindow->add_column_to_panel( main_panel, false );

  GLUI_Panel *eye_panel = 
    activeGui->mainWindow->add_panel_to_panel( main_panel, "Eye Position" );
  GLUI_Panel *pos_panel = 
    activeGui->mainWindow->add_panel_to_panel( eye_panel, "Position" );

  activeGui->x_pos = activeGui->mainWindow->
    add_edittext_to_panel( pos_panel, "X:", GLUI_EDITTEXT_FLOAT );
  activeGui->y_pos = activeGui->mainWindow->
    add_edittext_to_panel( pos_panel, "Y:", GLUI_EDITTEXT_FLOAT );
  activeGui->z_pos = activeGui->mainWindow->
    add_edittext_to_panel( pos_panel, "Z:", GLUI_EDITTEXT_FLOAT );

  activeGui->mainWindow->add_separator_to_panel( pos_panel );
  activeGui->direct = 
    activeGui->mainWindow->add_edittext_to_panel( pos_panel, "Facing" );

  GLUI_Panel *speed_panel = 
    activeGui->mainWindow->add_panel_to_panel( eye_panel, "Speed" );

  activeGui->forward_speed = activeGui->mainWindow->
    add_edittext_to_panel( speed_panel, "Forward:", GLUI_EDITTEXT_FLOAT );
  activeGui->upward_speed = activeGui->mainWindow->
    add_edittext_to_panel( speed_panel, "Up:", GLUI_EDITTEXT_FLOAT );
  activeGui->leftward_speed = activeGui->mainWindow->
    add_edittext_to_panel( speed_panel, "Right:", GLUI_EDITTEXT_FLOAT );

  GLUI_Rollout *control_panel = activeGui->mainWindow->
    add_rollout_to_panel( eye_panel, "Control Sensitivity", false );
  activeGui->rotateSensitivity_ = 1.0;
  GLUI_Spinner * rot = activeGui->mainWindow->
    add_spinner_to_panel( control_panel, "Rotation:", GLUI_SPINNER_FLOAT,
			  &(activeGui->rotateSensitivity_),
			  SENSITIVITY_SPINNER_ID, updateRotateSensitivityCB );
  rot->set_float_limits( MIN_SENSITIVITY, MAX_SENSITIVITY );
  rot->set_speed( 0.1 );

  activeGui->translateSensitivity_ = 1.0;
  GLUI_Spinner * trans = activeGui->mainWindow->
    add_spinner_to_panel(control_panel, "Translation:", GLUI_SPINNER_FLOAT,
			 &(activeGui->translateSensitivity_),
			 SENSITIVITY_SPINNER_ID, updateTranslateSensitivityCB);
  trans->set_float_limits( MIN_SENSITIVITY, MAX_SENSITIVITY );
  trans->set_speed( 0.01 );

  /////////////////////////////////////////////////////////
  // Route/Light/Objects/Sounds Window Buttons
  //
  GLUI_Panel * button_panel =
    activeGui->mainWindow->add_panel_to_panel( main_panel, "" );

  activeGui->mainWindow->
    add_button_to_panel( button_panel, "Routes",
			 ROUTE_BUTTON_ID, toggleRoutesWindowCB );
  //  activeGui->mainWindow->add_column_to_panel( button_panel );

  activeGui->mainWindow->
    add_button_to_panel( button_panel, "Lights",
			 LIGHTS_BUTTON_ID, toggleLightsWindowCB );
  //  activeGui->mainWindow->add_column_to_panel( button_panel );

  activeGui->mainWindow->
    add_button_to_panel( button_panel, "Objects",
			 OBJECTS_BUTTON_ID, toggleObjectsWindowCB );
  //  activeGui->mainWindow->add_column_to_panel( button_panel );

  activeGui->openSoundPanelBtn_ = activeGui->mainWindow->
    add_button_to_panel( button_panel, "Sounds",
			 OBJECTS_BUTTON_ID, toggleSoundWindowCB );
  activeGui->openSoundPanelBtn_->disable();
  //  activeGui->mainWindow->add_column_to_panel( button_panel );

  //  activeGui->mainWindow->add_separator_to_panel( button_panel);
  
  activeGui->mainWindow->
    add_button_to_panel( button_panel, "Triggers",
			 OBJECTS_BUTTON_ID, toggleTriggersWindowCB );


  ///////////////////////////////////////////////////////////

  if( activeGui->dpy_->fullScreenMode_ )
    {
      livingRoomImage = new PPMImage( 
	"/usr/sci/data/Geometry/interface/backgrounds/bkgrnd_livingroom.ppm",
	true );
      scienceRoomImage = new PPMImage( 
	"/usr/sci/data/Geometry/interface/backgrounds/bkgrnd_science.ppm",
	true );
      museumRoomImage = new PPMImage( 
	"/usr/sci/data/Geometry/interface/backgrounds/bkgrnd_museum.ppm",
	true );
      underwaterRoomImage = new PPMImage( 
	"/usr/sci/data/Geometry/interface/backgrounds/bkgrnd_atlantis.ppm",
	true );
      galaxyRoomImage = new PPMImage( 
	"/usr/sci/data/Geometry/interface/backgrounds/bkgrnd_galaxy.ppm",
	true );
      activeGui->backgroundImage_ = scienceRoomImage;

      ///////// CREATE SOME TRIGGERS
      ///// VIS WOMEN
      string ifpath = "/usr/sci/data/Geometry/interface/";      
      vector<Point> loc;
      PPMImage * ppm = 
	new PPMImage(ifpath+"scienceroom/science_vis-woman.ppm", true);
      loc.clear(); loc.push_back(Point(-30,-30,1.9));
      activeGui->visWomanTrig_ = new Trigger( "Visible Woman", loc, 1,
					      60,ppm,true );
      ///// GEOPHYSICS
      ppm = new PPMImage(ifpath+"scienceroom/science_geophysics.ppm", true);
      loc.clear(); loc.push_back(Point(-40,-40,1.9));
      activeGui->geophysicsTrig_ = new Trigger( "Geophysics", loc, 1,
						30,ppm,true );
      ///// C-SAFE
      ppm = new PPMImage(ifpath+"scienceroom/science_firespread.ppm", true);
      loc.clear(); loc.push_back(Point(-50,-50,1.9));
      activeGui->csafeTrig_ = new Trigger( "C-SAFE Fire", loc, 1,30,ppm,true );
    }
  printf("done createmenus\n");
} // end createMenus()

const string
Gui::getFacingString() const
{
  Vector lookAtVectHorizontal = 
    activeGui->camera_->get_lookat() - activeGui->camera_->get_eye();

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
Gui::updateFovCB( int /*id*/ )
{
  activeGui->camera_->set_fov( activeGui->fovValue_ );
  activeGui->camera_->setup();
}

void
Gui::updateRayOffsetCB( int /*id*/ )
{
  activeGui->camera_->set_ray_offset( activeGui->ray_offset );
}

void
Gui::updateDepthCB( int /*id*/ )
{
  activeGui->priv->maxdepth = activeGui->depthValue_;
}

void
Gui::updateRotateSensitivityCB( int /*id*/ )
{
  activeGui->stealth_->updateRotateSensitivity(activeGui->rotateSensitivity_);
}

void
Gui::updateTranslateSensitivityCB( int /*id*/ )
{
  activeGui->stealth_->
    updateTranslateSensitivity(activeGui->translateSensitivity_);
}

void
Gui::update()
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

  // Round to first decimal spot (for display purposes.)
  framesPerSecondTxt->set_float_val( ((int)(priv->FrameRate*10)) / 10.0 );

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
Gui::checkBackgroundWindow()
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
Gui::updateSoundPanel()
{
#if !defined(linux)
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
Gui::toggleTransmissionModeCB( int /* id */ )
{
  activeGui->dpy_->turnOnTransmissionMode_ = 
    !activeGui->dpy_->turnOnTransmissionMode_;
}

void
Gui::toggleHotspotsCB( int /*id*/ )
{
  if( activeGui->dpy_->scene->hotSpotMode_ == 2 )
    activeGui->dpy_->scene->hotSpotMode_ = 0;
  else
    activeGui->dpy_->scene->hotSpotMode_++;
}


void
Gui::toggleGui()
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

    activeGui->dpy_->scene->hide_auxiliary_displays();
  } else {
    mainWindow->show();
    activeGui->dpy_->scene->show_auxiliary_displays();
  }
  mainWindowVisible = !mainWindowVisible;
}


void
Gui::updateIntensityCB( int /*id*/ )
{
//  cout << "set light intensity to " << activeGui->lightBrightness_ << "\n";

  if( activeGui->lights_.size() == 0 ) return;

  Light * light = activeGui->lights_[ activeGui->selectedLightId_ ];

  light->updateIntensity( activeGui->lightBrightness_ );

  if( activeGui->lightBrightness_ == 0.0 )
    {
      activeGui->lightsPositionPanel_->disable();
      activeGui->lightsColorPanel_->disable();
      activeGui->dpy_->turnOffLight_ = light;
    }
  else if( !light->isOn() )
    {
      activeGui->lightsPositionPanel_->enable();
      activeGui->lightsColorPanel_->enable();
      activeGui->dpy_->turnOnLight_ = light;
    }
}

void
Gui::getStringCB( int id )
{
  if( id == LOAD_ROUTE_BUTTON_ID ) {
    activeGui->getStringPanel->set_name( "Load File Name" );
    activeGui->getStringButton->callback = loadRouteCB;
    activeGui->getStringText_->callback = loadRouteCB;
  } else if( id == NEW_ROUTE_BUTTON_ID ) {
    activeGui->getStringPanel->set_name( "Enter Route Name" );
    activeGui->getStringButton->callback = newRouteCB;
    activeGui->getStringText_->callback = newRouteCB;
  } else if( id == SAVE_ROUTE_BUTTON_ID ) {
    activeGui->getStringPanel->set_name( "Save File Name" );
    activeGui->getStringButton->callback = saveRouteCB;
    activeGui->getStringText_->callback = saveRouteCB;
  } else {
    cout << "don't know what string to get\n";
    return;
  }
  
  activeGui->getStringWindow->show();
}

void 
Gui::hideGetStringWindowCB( int /*id*/ )
{
  activeGui->getStringWindow->hide();
}


void
Gui::loadAllRoutes()
{
  const vector<string> & routes = activeGui->dpy_->scene->getRoutes();
  string routeName;
  char name[1024];

  for( unsigned int i = 0; i < routes.size(); i++ )
    {
      routeName = routes[i];

      sprintf( name, "%s", routeName.c_str() );
      activeGui->routeList->add_item( routeNumber, name );
      activeGui->routeList->set_int_val( routeNumber );
      
      routeNumber++;
    }
}

void
Gui::loadRouteCB( int /*id*/ )
{
  // glui is screwy when you have this type of window where you want
  // either the "ok" button or "return" to do the same thing.  By
  // removing the callbacks like this, you avoid an infinite loop.
  activeGui->getStringButton->callback = NULL;
  activeGui->getStringText_->callback = NULL;

  string routeName = activeGui->stealth_->loadPath( activeGui->inputString_ );

  if( routeName == "" )
    {
      cout << "loading of route failed\n";
      return;
    }

  cout << "loaded route: " << routeName << "\n";

  activeGui->getStringWindow->hide();

  char name[1024];
  sprintf( name, "%s", routeName.c_str() );
  activeGui->routeList->add_item( routeNumber, name );
  activeGui->routeList->set_int_val( routeNumber );

  routeNumber++;

  activeGui->routeList->enable();
  activeGui->routePositionPanel->enable();
  activeGui->traverseRouteBtn->enable();
  activeGui->editorRO->enable();
  activeGui->goToRteBegBtn->enable();

  goToRouteBeginningCB( -1 );
}

void
Gui::newRouteCB( int /*id*/ )
{
  // glui is screwy when you have this type of window where you want
  // either the "ok" button or "return" to do the same thing.  By
  // removing the callbacks like this, you avoid an infinite loop.
  activeGui->getStringButton->callback = NULL;
  activeGui->getStringText_->callback = NULL;

  activeGui->getStringWindow->hide();

  string routeName = activeGui->inputString_;

  if( routeName == "" )
    {
      cout << "invalid route name, not saving\n";
      return;
    }

  activeGui->stealth_->newPath( routeName );

  char name[1024];
  sprintf( name, "%s", routeName.c_str() );
  activeGui->routeList->add_item( routeNumber, name );
  activeGui->routeList->set_int_val( routeNumber );
  routeNumber++;

  activeGui->routeList->enable();
  activeGui->routePositionPanel->enable();
  activeGui->traverseRouteBtn->enable();
  activeGui->editorRO->enable();
  activeGui->goToRteBegBtn->enable();
}

void
Gui::saveRouteCB( int /*id*/ )
{
  // glui is screwy when you have this type of window where you want
  // either the "ok" button or "return" to do the same thing.  By
  // removing the callbacks like this, you avoid an infinite loop.
  activeGui->getStringButton->callback = NULL;
  activeGui->getStringText_->callback = NULL;

  activeGui->getStringWindow->hide();
  
  if( strcmp( activeGui->inputString_, "" ) )
    {
      activeGui->stealth_->savePath( activeGui->inputString_ );
    }
}

void
Gui::clearRouteCB( int /*id*/ )
{
  activeGui->stealth_->clearPath();
}

void
Gui::deleteCurrentMarkerCB( int /*id*/ )
{
  activeGui->stealth_->deleteCurrentMarker();
}

void
Gui::addToRouteCB( int /*id*/ )
{
  activeGui->stealth_->addToMiddleOfPath( activeGui->camera_->get_eye(),
					  activeGui->camera_->get_lookat() );
}

void
Gui::traverseRouteCB( int /*id*/ )
{
  activeGui->priv->followPath = !activeGui->priv->followPath;

  // If starting/stopping following a path, stop all other movement.
  // if starting to following, increase movement once to make us move.
  activeGui->stealth_->stopAllMovement();
  if( activeGui->priv->followPath ) {
    activeGui->stealth_->accelerate();
    activeGui->traverseRouteBtn->set_name("Stop");
  } else {
    activeGui->traverseRouteBtn->set_name("Follow Route");
  }
}

void
Gui::goToNextMarkerCB( int /*id*/ )
{
  Point pos, look_at;
  int index = activeGui->stealth_->getNextMarker( pos, look_at );  

  if( index == -1 ) return;

  activeGui->camera_->set_eye( pos );
  activeGui->camera_->set_lookat( look_at );
  activeGui->camera_->setup();

  Point origin;
  Vector lookdir;
  Vector up;
  Vector side;
  double fov;
  activeGui->camera_->getParams(origin, lookdir, up, side, fov);
  lookdir.normalize();
  up.normalize();
  side.normalize();
  // Move the lights that are fixed to the eye
  for(int i = 0; i < activeGui->dpy_->scene->nlights(); i++) {
    Light *light = activeGui->dpy_->scene->light(i);
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
Gui::goToPrevMarkerCB( int /*id*/ )
{
  Point pos, look_at;
  int index = activeGui->stealth_->getPrevMarker( pos, look_at );  

  if( index == -1 ) return;

  activeGui->camera_->set_eye( pos );
  activeGui->camera_->set_lookat( look_at );
  activeGui->camera_->setup();

  Point origin;
  Vector lookdir;
  Vector up;
  Vector side;
  double fov;
  activeGui->camera_->getParams(origin, lookdir, up, side, fov);
  lookdir.normalize();
  up.normalize();
  side.normalize();
  // Move the lights that are fixed to the eye
  for(int i = 0; i < activeGui->dpy_->scene->nlights(); i++) {
    Light *light = activeGui->dpy_->scene->light(i);
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
Gui::goToRouteBeginningCB( int /*id*/ )
{
  Point pos, look_at;
  int   index = activeGui->stealth_->goToBeginning( pos, look_at );  

  if( index == -1 ) return;

  activeGui->camera_->set_eye( pos );
  activeGui->camera_->set_lookat( look_at );
  activeGui->camera_->setup();

  Point origin;
  Vector lookdir;
  Vector up;
  Vector side;
  double fov;
  activeGui->camera_->getParams(origin, lookdir, up, side, fov);
  lookdir.normalize();
  up.normalize();
  side.normalize();
  // Move the lights that are fixed to the eye
  for(int i = 0; i < activeGui->dpy_->scene->nlights(); i++) {
    Light *light = activeGui->dpy_->scene->light(i);
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
Gui::bgColorCB( int id )
{
  switch ( id ) {
  case WHITE_BG_BUTTON:
    activeGui->dpy_->scene->set_bgcolor(Color(1,1,1));
    break;
  case BLACK_BG_BUTTON:
    activeGui->dpy_->scene->set_bgcolor(Color(0,0,0));
    break;
  case ORIG_BG_BUTTON:
    activeGui->dpy_->scene->set_original_bg();
    break;
  }
}

void
Gui::toggleAutoJitterCB( int /*id*/ )
{
  activeGui->dpy_->doAutoJitter_ = !activeGui->dpy_->doAutoJitter_;
}

void
Gui::toggleJitterCB( int /*id*/ )
{
  int *do_jitter = &(activeGui->dpy_->scene->rtrt_engine->do_jitter);
  *do_jitter = !(*do_jitter);
  if( !(*do_jitter))
    activeGui->jitterButton_->set_name("Turn Jitter ON");
  else
    activeGui->jitterButton_->set_name("Turn Jitter OFF");
}

void
Gui::updateLightPositionCB( int id )
{
  Light * light = activeGui->lights_[ activeGui->selectedLightId_ ];
  Point pos = light->get_pos();

  cout << "updating light position: " << id << "\n";
  cout << "pos was " << pos << "\n";

  cout << activeGui->lightX_ << ", "
       << activeGui->lightY_ << ", "
       << activeGui->lightZ_ << "\n";

  switch( id ) {
  case LIGHT_X_POS_ID:
    pos.x( activeGui->lightX_ );
    light->updatePosition( pos );
    break;
  case LIGHT_Y_POS_ID:
    pos.y( activeGui->lightY_ );
    light->updatePosition( pos );
    break;
  case LIGHT_Z_POS_ID:
    pos.z( activeGui->lightZ_ );
    light->updatePosition( pos );
    break;
  }
  cout << "pos is " << pos << "\n";
}

void
Gui::cycleAmbientMode()
{
  if( dpy_->ambientMode_ == Sphere_Ambient )
    {
      dpy_->ambientMode_ = Constant_Ambient;
    }
  else
    {
      dpy_->ambientMode_++;
    }

  ambientModeLB_->set_int_val( dpy_->ambientMode_ );
}

void
Gui::cycleShadowMode()
{
  if( dpy_->shadowMode_ == Uncached_Shadows )
    {
      dpy_->shadowMode_ = No_Shadows;
    }
  else
    {
      dpy_->shadowMode_++;
    }

  shadowModeLB_->set_int_val( dpy_->shadowMode_ );
}

/////////////////////////////////////////////////////////////////////
// Draws the string "s" to the GL window at x,y.
//
void
Gui::displayText(GLuint fontbase, double x, double y, char *s, const Color& c)
{
  glColor3f(c.red(), c.green(), c.blue());
  
  glRasterPos2d(x,y);
  /*glBitmap(0, 0, x, y, 1, 1, 0);*/
  glPushAttrib (GL_LIST_BIT);
  glListBase(fontbase);
  glCallLists((int)strlen(s), GL_UNSIGNED_BYTE, (GLubyte *)s);
  glPopAttrib ();
}

// Looks like this displays the string with a shadow on it...
void
Gui::displayShadowText(GLuint fontbase,
		       double x, double y, char *s, const Color& c)
{
  Color b(0,0,0);
  displayText(fontbase, x-1, y-1, s, b);
  displayText(fontbase, x, y, s, c);
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
Gui::draw_labels(XFontStruct* font_struct, GLuint fontbase,
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
Gui::draw_column(XFontStruct* font_struct,
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
Gui::drawpstats(Stats* mystats, int nworkers, vector<Worker*> & workers,
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
    displayText(fontbase, 80, 3, buf, Color(0,1,0));
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
Gui::drawrstats(int nworkers, vector<Worker*> & workers,
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


