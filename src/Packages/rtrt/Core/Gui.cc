
#include <Packages/rtrt/Core/Gui.h>
#include <Packages/rtrt/Core/Dpy.h>
#include <Packages/rtrt/Core/DpyPrivate.h>
#include <Packages/rtrt/Core/Stealth.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Ball.h>
#include <Packages/rtrt/Core/BallMath.h>
#include <Packages/rtrt/Core/Stealth.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Image.h>
#include <Packages/rtrt/Core/CycleMaterial.h>
#include <Packages/rtrt/Core/FontString.h>

#include <Core/Math/Trig.h>
#include <Core/Thread/Time.h>
#include <Core/Geometry/Transform.h>

#include <GL/glut.h>
#include <glui.h>

#include <unistd.h>  // for sleep

using namespace rtrt;
using namespace SCIRun;
using namespace std;

static Gui * activeGui;

static double    prev_time[3]; // history for quaternions and time
static HVect     prev_quat[3];
static Transform prev_trans;

// GLUI Component IDs
#define TURN_ON_JITTER_BTN        50

#define SENSITIVITY_SPINNER_ID    60
#define FOV_SPINNER_ID            61
#define DEPTH_SPINNER_ID          62

#define LIGHT_LIST_ID            100
#define LIGHTS_BUTTON_ID         101
#define TOGGLE_LIGHT_SWITCHES_ID 102
#define TOGGLE_SHOW_LIGHTS_ID    103

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

#define TOGGLE_HOTSPOTS_ID       190

// GLUT MENU ITEM IDS

#define TOGGLE_GUI                 1
#define TOGGLE_HOT_SPOTS           2
#define QUIT_MENU_ID               3

// Defines how much you can manually increase/decrease movement
// controls sensitivity.
#define MIN_SENSITIVITY      0.1
#define MAX_SENSITIVITY    200.0

// Used for loading in routes.  (Is monotonicly increasing)
static int routeNumber = 0;

Gui::Gui() :
  selectedLightId(0), selectedRouteId(0), selectedObjectId(0),
  routeWindowVisible(false), lightsWindowVisible(false),
  objectsWindowVisible(false), mainWindowVisible(true),
  lightList(NULL), routeList(NULL), objectList(NULL),
  r_color_spin(NULL), g_color_spin(NULL), b_color_spin(NULL), 
  lightIntensity_(NULL),
  lightBrightness_(1.0), ambientBrightness_(1.0),
  mouseDown_(0), shiftDown_(false), beQuiet_(true),
  lightsOn_(true), lightsBeingRendered_(false)
{
}

Gui::~Gui()
{
}

void
Gui::quit()
{
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
    activeGui->dpy_->scene->hotspots = !activeGui->dpy_->scene->hotspots;
    break;
  case TOGGLE_GUI:
    activeGui->toggleGui();
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
  camera_  = dpy_->guiCam_;
}

void
Gui::idleFunc()
{
  // I know this is a hack... 
  if( activeGui->dpy_->showImage_ ){

    glutSetWindow( activeGui->glutDisplayWindowId );

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, activeGui->priv->xres, 0, activeGui->priv->yres);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.375, 0.375, 0.0);

    activeGui->dpy_->showImage_->draw();
    activeGui->dpy_->showImage_ = NULL;

    // Display textual information on the screen:
    char buf[100];
    sprintf( buf, "%3.1 fps", activeGui->priv->FrameRate );
    activeGui->displayText(GLUT_BITMAP_HELVETICA_10, 10, 3, buf, Color(1,1,1));

    glutSwapBuffers(); 

    // Let the Dpy thread start drawing the next image.
    activeGui->priv->waitDisplay->unlock();
    if( activeGui->mainWindowVisible ) {
      activeGui->update(); // update the gui each time a frame is finished.
    }
  }
}

void
Gui::handleKeyPressCB( unsigned char key, int /*mouse_x*/, int /*mouse_y*/ )
{
  // static double FPS = 15;

  // DpyPrivate * priv = activeGui->priv;

  // int     & maxdepth       = priv->maxdepth;
  // bool    & stereo         = priv->stereo;  
  // bool    & animate        = priv->animate;
  // double  & FrameRate      = priv->FrameRate;
  // bool    & draw_pstats    = priv->draw_pstats;
  // bool    & draw_rstats    = priv->draw_rstats;
  // int     & showing_scene  = priv->showing_scene;

  // int     & left           = priv->left;
  // int     & up             = priv->up;

  switch( key ){

  // KEYPAD KEYS USED FOR MOVEMENT

  // SPEED up or slow down
  case '+':
    activeGui->stealth_->accelerate();
    break;
  case '-':
    activeGui->stealth_->decelerate();
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
    activeGui->dpy_->scene->hotspots = !activeGui->dpy_->scene->hotspots;
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
      }
    }
    break;
  case 13: // Enter
    activeGui->camera_->flatten(); // Right yourself (0 pitch, 0 roll)
    break;
  case 'x':
    traverseRouteCB(-1);
    break;
  case 'a':
   activeGui->priv->animate =! activeGui->priv->animate;
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
    for (int m=0; m<activeGui->dpy_->scene->nmaterials(); m++) {
      CycleMaterial * cm =
	dynamic_cast<CycleMaterial*>(activeGui->dpy_->scene->get_material(m));
      if (cm) cm->next();
    }
    break;

  case 'J': // toggle on/off "Jitter On Stop" moded...
    toggleAutoJitterCB( -1 );
    break;
  case 'j': // toggle on/off continuous jittered sampling...
    toggleJitterCB( -1 );
    break;


#if 0
  case 'p':
    draw_pstats=!draw_pstats;
    break;
  case 'r':
    draw_rstats=!draw_rstats;
    break;
    // below is for blending "pixels" in
    // frameless rendering...

  case 'y': // sychronizing mode for frameless...
    synch_frameless = !synch_frameless;  //1-synch_frameless;
    //doing_frameless = 1-doing_frameless; // just toggle...
    cerr << synch_frameless << " Synch?\n";
    break;
  case '2':
    stereo=!stereo;
    break;
  case '1':
    cout << "NOTICE: Use 2 key to toggle Stereo\n";
    cout << "      : 1 key is deprecated and may go away\n";
    break;
  case 'f':
    FPS -= 1;
    if (FPS <= 0.0) FPS = 1.0;
    FrameRate = 1.0/FPS;
    break;
  case 'F':
    FPS += 1.0;
    FrameRate = 1.0/FPS;
    cerr << FPS << endl;
    break;
  case 'w':
    cerr << "Saving file\n";
    scene->get_image(showing_scene)->save("images/image.raw");
    break;
#endif
  case 27: // Escape key... need to find a symbolic name for this...
    activeGui->quit();
    break;
  default:
    printf("unknown regular key %d\n", key);
    break;
  }
} // end handleKeyPress();

// WARNING: THERE ARE NOT THE KEYPAD KEYS!
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
  double     & last_time = priv->last_time;
  BallData  *& ball = priv->ball;

  mouseDown_ = button;
  shiftDown_ = glutGetModifiers() & GLUT_ACTIVE_SHIFT;

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

  printf("mouse was pressed\n");
} // end handleMousePress()

void
Gui::handleMouseRelease(int button, int /*mouse_x*/, int /*mouse_y*/)
{
  DpyPrivate * priv      = activeGui->priv;
  double     & last_time = priv->last_time;

  mouseDown_ = 0;

  switch(button){
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
	activeGui->camera_->eye = activeGui->camera_->lookat+z_a;
	activeGui->camera_->setup();
	prev_trans = prv;

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
  printf("window resized\n");

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

  cout << "mouse is at: " << mouse_x << ", " << mouse_y << "\n";

  switch( activeGui->mouseDown_ ) {
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
      activeGui->camera_->up=y_a;

      activeGui->camera_->eye = activeGui->camera_->lookat+z_a*eye_dist;
      activeGui->camera_->setup();
			
      last_time=SCIRun::Time::currentSeconds();
      //inertia_mode=0;
    }
    break;
  }

  activeGui->last_x_ = mouse_x;
  activeGui->last_y_ = mouse_y;


} // end handleMouseMotion()

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
Gui::updateLightPanelCB( int /*id*/ )
{
  if( activeGui->lights_.size() == 0 ) return;

  Light * light = activeGui->lights_[ activeGui->selectedLightId ];
  const Color & color = light->getOrigColor();

  activeGui->r_color_spin->set_float_val( color.red() );
  activeGui->g_color_spin->set_float_val( color.green() );
  activeGui->b_color_spin->set_float_val( color.blue() );

  activeGui->lightIntensity_->set_float_val( light->get_intensity() );

  if( light->isOn() )
    activeGui->lightsColorPanel_->enable();
  else
    activeGui->lightsColorPanel_->disable();

}

void
Gui::updateRouteCB( int /*id*/ )
{
  activeGui->stealth_->selectPath( activeGui->selectedRouteId );
  goToRouteBeginningCB( -1 );
}

void
Gui::updateObjectCB( int /*id*/ )
{
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
Gui::toggleLightSwitchesCB( int /*id*/ )
{
  if( activeGui->lightsOn_ ) {
    activeGui->toggleLightsOnOffBtn_->set_name( "Turn On Lights" );
    activeGui->dpy_->turnOffAllLights_ = true;
    activeGui->lightsOn_ = false;
    activeGui->lightsColorPanel_->disable();
  } else {
    activeGui->toggleLightsOnOffBtn_->set_name( "Turn Off Lights" );
    activeGui->dpy_->turnOnAllLights_ = true;
    activeGui->lightsOn_ = true;
    activeGui->lightsColorPanel_->enable();
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
					    &selectedRouteId, 
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

}

void
Gui::createGetStringWindow( GLUI * window )
{
  getStringPanel = window->add_panel( "" );

  window->add_edittext_to_panel( getStringPanel, "", GLUI_EDITTEXT_TEXT,
				 &(activeGui->inputString_) );
  getStringButton = 
    window->add_button_to_panel( getStringPanel, "OK", CLOSE_GETSTRING_BTN );
}

void
Gui::createObjectWindow( GLUI * window )
{
  GLUI_Panel * panel = window->add_panel( "Objects" );

  objectList = window->add_listbox_to_panel( panel, "Selected Object",
					    &selectedObjectId, 
					    OBJECT_LIST_ID, updateObjectCB );
  objectList->add_item( 1, "Not Implemented Yet" );
  objectList->add_item( 2, "Object 2" );

  window->add_statictext_to_panel( panel, "Position:" );
  window->add_statictext_to_panel( panel, "Type:" );
  window->add_statictext_to_panel( panel, "Material:" );

}

void
Gui::addLight( Light * light )
{
  int numLights = lights_.size();

  string name = light->name_;
  if( name == "" ) {
    name = "Unnamed";
  }

  char namec[1024];
  sprintf( namec, "%s", name.c_str() );

  lightList->add_item( numLights, namec );
  lights_.push_back( light );

  updateLightPanelCB( -1 );
}

void
Gui::createLightWindow( GLUI * window )
{
  GLUI_Panel * panel = window->add_panel( "Lights" );
  lightList = window->add_listbox_to_panel( panel, "Selected Light:",
					    &selectedLightId, 
					    LIGHT_LIST_ID, updateLightPanelCB);
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

  lightIntensity_ = 
    window->add_spinner_to_panel( panel, "Intensity:", GLUI_SPINNER_FLOAT,
				  &lightBrightness_, -1, updateIntensityCB );
  lightIntensity_->set_float_limits( 0.0, 1.0 );
  lightIntensity_->set_speed( 0.01 );

  window->add_separator_to_panel( panel );

  ambientIntensity_ = 
    window->add_spinner_to_panel( panel, "Ambient Level:", GLUI_SPINNER_FLOAT,
				  &ambientBrightness_, -1, updateAmbientCB );
  ambientIntensity_->set_float_limits( 0.0, 1.0 );
  ambientIntensity_->set_speed( 0.01 );

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

}


void
Gui::createMenus( int winId )
{
  printf("createmenus\n");

  activeGui->glutDisplayWindowId = winId;

  int modemenu = glutCreateMenu( Gui::handleMenuCB );

  glutAddMenuEntry( "Toggle Gui", TOGGLE_GUI );
  glutAddMenuEntry( "Toggle Hot Spots", TOGGLE_HOT_SPOTS );
  glutAddMenuEntry( "----------------", -1);
  glutAddMenuEntry( "Quit", QUIT_MENU_ID );
  //glutAddSubMenu("Texture mode", modemenu);
  glutAttachMenu(GLUT_RIGHT_BUTTON);

  // Build GLUI Windows
  activeGui->mainWindow = GLUI_Master.create_glui( "SIGGRAPH", 0, 400, 20 );

  activeGui->routeWindow = GLUI_Master.create_glui( "Route", 0, 400, 400 );
  activeGui->lightsWindow = GLUI_Master.create_glui( "Lights", 0, 500, 400 );
  activeGui->objectsWindow = GLUI_Master.create_glui( "Objects", 0, 600, 400 );

  activeGui->getStringWindow = 
                    GLUI_Master.create_glui( "Input Request", 0, 400, 400 );

  //  activeGui->routeWindow->set_main_gfx_window( winId );
  //  activeGui->lightsWindow->set_main_gfx_window( winId );
  //  activeGui->objectsWindow->set_main_gfx_window( winId );
  //  activeGui->mainWindow->set_main_gfx_window( winId );

  activeGui->routeWindow->hide();
  activeGui->lightsWindow->hide();
  activeGui->objectsWindow->hide();

  activeGui->getStringWindow->hide();

  activeGui->createRouteWindow( activeGui->routeWindow );
  activeGui->createLightWindow( activeGui->lightsWindow );
  activeGui->createObjectWindow( activeGui->objectsWindow );
  activeGui->createGetStringWindow( activeGui->getStringWindow );

  /////////////////////////////////////////////////////////
  // Main Panel
  //
  GLUI_Panel * main_panel   = activeGui->mainWindow->add_panel( "" );
  GLUI_Panel * button_panel = activeGui->mainWindow->add_panel( "" );

  /////////////////////////////////////////////////////////
  // Eye Position Panel
  //
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
  trans->set_speed( 0.1 );

  /////////////////////////////////////////////////////////
  // Display Parameters Panel
  //
  activeGui->mainWindow->add_column_to_panel( main_panel, false );

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
  activeGui->shadowModeLB_->set_int_val( activeGui->dpy_->ambientMode_ );

  // Jitter
  GLUI_Panel * jitter = 
    activeGui->mainWindow->add_panel_to_panel( display_panel, "Jitter" );
  activeGui->jitterButton_ = activeGui->mainWindow->
    add_button_to_panel( jitter, "Turn Jitter ON",
			 TURN_ON_JITTER_BTN, toggleJitterCB );

  // FOV
  activeGui->fovValue_ = activeGui->camera_->get_fov();
  activeGui->fovSpinner_ = activeGui->mainWindow->
    add_spinner_to_panel( display_panel, "FOV:", GLUI_SPINNER_INT,
			   &(activeGui->fovValue_), FOV_SPINNER_ID,
			  updateFovCB );
  activeGui->fovSpinner_->set_int_limits( 20, 150 );
  activeGui->fovSpinner_->set_speed( 0.1 );

  // Other Controls
  GLUI_Panel * otherControls = activeGui->mainWindow->
    add_panel_to_panel( display_panel, "Other Controls" );

  activeGui->mainWindow->add_button_to_panel( otherControls,
	 "Toggle Hot Spot Display", TOGGLE_HOTSPOTS_ID, toggleHotspotsCB );

  // ...This probably goes to the objects window...
  activeGui->mainWindow->add_button_to_panel(otherControls,"Toggle Materials");

  // 
  activeGui->depthValue_ = 2;
  GLUI_Spinner * depthSpinner = activeGui->mainWindow->
    add_spinner_to_panel( display_panel, "Ray Depth", GLUI_SPINNER_INT, 
			  &(activeGui->depthValue_), DEPTH_SPINNER_ID, 
			  updateDepthCB );
  depthSpinner->set_int_limits( 0, 12 );
  depthSpinner->set_speed( 0.1 );

  /////////////////////////////////////////////////////////
  // Route/Light/Objects Window Buttons
  //

  activeGui->mainWindow->
    add_button_to_panel( button_panel, "Routes",
			 ROUTE_BUTTON_ID, toggleRoutesWindowCB );
  activeGui->mainWindow->add_column_to_panel( button_panel );
  activeGui->mainWindow->
    add_button_to_panel( button_panel, "Lights",
			 LIGHTS_BUTTON_ID, toggleLightsWindowCB );
  activeGui->mainWindow->add_column_to_panel( button_panel );
  activeGui->mainWindow->
    add_button_to_panel( button_panel, "Objects",
			 OBJECTS_BUTTON_ID, toggleObjectsWindowCB );

  printf("done createmenus\n");
}

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
Gui::updateDepthCB( int /*id*/ )
{
  activeGui->priv->maxdepth = activeGui->depthValue_;
}

void
Gui::updateRotateSensitivityCB( int /*id*/ )
{
  activeGui->stealth_->updateRotateSensitivity(activeGui->rotateSensitivity_);
  printf("update control sensitivity cb called: %f\n", 
	 activeGui->rotateSensitivity_ );
}

void
Gui::updateTranslateSensitivityCB( int /*id*/ )
{
  activeGui->stealth_->
    updateTranslateSensitivity(activeGui->translateSensitivity_);

  printf("update translate sensitivity cb called: %f\n", 
	 activeGui->translateSensitivity_ );
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

} // end update()

void
Gui::toggleHotspotsCB( int /*id*/ )
{
  activeGui->dpy_->scene->hotspots = !activeGui->dpy_->scene->hotspots;
}


void
Gui::toggleGui()
{
  if( mainWindowVisible ) {
    routeWindow->hide();
    objectsWindow->hide();
    lightsWindow->hide();
    mainWindow->hide();
    
    lightsWindowVisible = false;
    routeWindowVisible = false;
    objectsWindowVisible = false;
  } else {
    mainWindow->show();
  }
  mainWindowVisible = !mainWindowVisible;
}


void
Gui::updateIntensityCB( int /*id*/ )
{
//  cout << "set light intensity to " << activeGui->lightBrightness_ << "\n";

  if( activeGui->lights_.size() == 0 ) return;

  Light * light = activeGui->lights_[ activeGui->selectedLightId ];

  light->updateIntensity( activeGui->lightBrightness_ );

  if( activeGui->lightBrightness_ == 0.0 )
    {
      activeGui->lightsColorPanel_->disable();
      activeGui->dpy_->turnOffLight_ = light;
    }
  else if( !light->isOn() )
    {
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
  } else if( id == NEW_ROUTE_BUTTON_ID ) {
    activeGui->getStringPanel->set_name( "Enter Route Name" );
    activeGui->getStringButton->callback = newRouteCB;
  } else if( id == SAVE_ROUTE_BUTTON_ID ) {
    activeGui->getStringPanel->set_name( "Save File Name" );
    activeGui->getStringButton->callback = saveRouteCB;
  } else {
    cout << "don't know what string to get\n";
    return;
  }
  
  activeGui->getStringWindow->show();
}

void
Gui::loadRouteCB( int /*id*/ )
{
  activeGui->getStringWindow->hide();
  string routeName = activeGui->stealth_->loadPath( activeGui->inputString_ );

  if( routeName == "" )
    {
      cout << "loading of route failed\n";
      return;
    }

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
}

void
Gui::toggleAutoJitterCB( int /*id*/ )
{
  activeGui->dpy_->doAutoJitter_ = !activeGui->dpy_->doAutoJitter_;
}

void
Gui::toggleJitterCB( int /*id*/ )
{
  activeGui->dpy_->doJitter_ = !activeGui->dpy_->doJitter_;
  if( !activeGui->dpy_->doJitter_ )
    activeGui->jitterButton_->set_name("Turn Jitter ON");
  else
    activeGui->jitterButton_->set_name("Turn Jitter OFF");
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
Gui::displayText(void * font, double x, double y, char *s, const Color& c)
{
  glColor3f(c.red(), c.green(), c.blue());
  glRasterPos2d(x, y);

  int len = (int) strlen(s);
  for( int i = 0; i < len; i++ ) {
    glutBitmapCharacter(font, s[i]);
  }
}

// Looks like this displays the string with a shadow on it...
void
Gui::displayShadowText(void * font,
		       double x, double y, char *s, const Color& c)
{
  Color b(0,0,0);
  displayText(font, x-1, y-1, s, b);
  displayText(font, x, y, s, c);
}

