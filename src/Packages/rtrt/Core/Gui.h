
#ifndef RTRT_GUI_H
#define RTRT_GUI_H

#include <GL/glx.h>
#include <GL/glu.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <string>
#include <vector>

// Foward declare GLUI objects
class GLUI;
class GLUI_Listbox;
class GLUI_Rollout;
class GLUI_Panel;
class GLUI_EditText;
class GLUI_StaticText;
class GLUI_Spinner;
class GLUI_Button;

namespace rtrt {

// Because we have to use static functions, only one Gui object may be
// active at a time.  (My guess is that there will only be one Gui
// object at a time anyway...)

class  Dpy;
struct DpyPrivate;
class  Stealth;
class  Camera;
class  Color;
class  Worker;
class  DepthStats;
class  Stats;
class  Light;

class Gui {

public:
  Gui();
  ~Gui();

  static void setActiveGui( Gui * gui );

  void setDpy( Dpy * dpy );
  void setStealth( Stealth * stealth );


  void addLight( Light * light );

  // Tell the GUI to update all the information in its windows.
  // (eg: eye location)
  void update();

  // Used by GLUT
  static void handleWindowResizeCB( int width, int height );
  static void handleKeyPressCB( unsigned char key,
			      int /*mouse_x*/, int /*mouse_y*/ );
  static void handleSpecialKeyCB( int key,
				int /*mouse_x*/, int /*mouse_y*/ );
  static void handleMouseCB(int button, int state, int x, int y);
  static void handleMouseMotionCB( int mouse_x, int mouse_y );
  static void createMenus( int winId ); // Call after glutInit(...)!

  static void idleFunc();

private:

  friend class Dpy;

  Dpy               * dpy_;
  struct DpyPrivate * priv;
  Camera            * camera_;
  Stealth           * stealth_;

  int                 glutDisplayWindowId;

  // Gui Component Variables

  int selectedLightId_;
  int selectedRouteId_;
  int selectedObjectId_;

  char inputString_[ 1024 ];

  GLUI_Listbox * objectList;

  GLUI         * mainWindow;

  GLUI         * routeWindow;
  GLUI         * lightsWindow;
  GLUI         * objectsWindow;

  GLUI         * getStringWindow;

  bool routeWindowVisible;
  bool lightsWindowVisible;
  bool objectsWindowVisible;
  bool mainWindowVisible;

  // Last x/y of a mouse down/move:
  int   last_x_;
  int   last_y_;

  // 0 == no button down, 1 == button 1, 2 == button 2, etc;
  int   mouseDown_;
  bool  shiftDown_; // during mouse press

  bool  rightButtonMenuActive_;

  bool  beQuiet_;

  bool  displayRStats_;
  bool  displayPStats_;

  std::vector<Light*> lights_;
  bool                lightsOn_;
  bool                lightsBeingRendered_;

  ////////////////////////////////////////////////////////////////
  //
  // mainWindow GLUI elemets:
  //

  GLUI_EditText * x_pos; // Eye position
  GLUI_EditText * y_pos;
  GLUI_EditText * z_pos;

  GLUI_EditText * forward_speed;
  GLUI_EditText * upward_speed;
  GLUI_EditText * leftward_speed;

  GLUI_EditText * direct; // Facing of camera (N, S, E, W, etc.)

  float           rotateSensitivity_;
  float           translateSensitivity_;

  GLUI_Listbox  * shadowModeLB_;
  GLUI_Listbox  * ambientModeLB_;
  GLUI_Button   * jitterButton_;

  GLUI_EditText * framesPerSecondTxt;
  GLUI_Spinner  * fovSpinner_;

  float           fovValue_;
  int             depthValue_;

  ////////////////////////////////////////////////////////////////
  //
  // lightWindow GLUI elemets:
  //

  GLUI_Listbox * lightList;

  GLUI_Panel   * lightsColorPanel_;
  GLUI_Panel   * lightsPositionPanel_;

  GLUI_Spinner * r_color_spin;
  GLUI_Spinner * g_color_spin;
  GLUI_Spinner * b_color_spin;

  GLUI_Spinner * lightPosX_;
  GLUI_Spinner * lightPosY_;
  GLUI_Spinner * lightPosZ_;

  GLUI_Spinner * lightIntensity_;
  GLUI_Spinner * ambientIntensity_;

  GLUI_Button  * toggleLightsOnOffBtn_;
  GLUI_Button  * toggleShowLightsBtn_;

  float lightBrightness_;
  float ambientBrightness_;

  // Currently selected lights position (x,y,z)
  float lightX_;
  float lightY_;
  float lightZ_;

  ////////////////////////////////////////////////////////////////
  //
  // routeWindow GLUI elemets:
  //
  GLUI_Listbox  * routeList;
  GLUI_EditText * routePositionET;
  GLUI_Panel    * routePositionPanel;
  GLUI_Rollout  * editorRO;
  GLUI_Button   * traverseRouteBtn;
  GLUI_Button   * gravityBtn;
  GLUI_Button   * goToRteBegBtn;

  ////////////////////////////////////////////////////////////////
  //
  // objectWindow GLUI elemets:
  //

  GLUI_Button * attachKeypadBtn_;
  bool          keypadAttached_;

  ////////////////////////////////////////////////////////////////

  // Returns "N", "NE", "E", etc, depending on facing.
  const std::string getFacingString() const;

  ////////////////////////////////////////////////////////////////

  static void handleMenuCB( int item );

  void handleMousePress( int button, int mouse_x, int mouse_y );
  void handleMouseRelease( int button, int mouse_x, int mouse_y );

  // Get String Window Callbacks
  void createGetStringWindow( GLUI * window );
  GLUI_Panel  * getStringPanel;
  GLUI_Button * getStringButton;
  static void getStringCB( int id );

  // Route Window Callbacks
  void createRouteWindow( GLUI * window );
  static void toggleRoutesWindowCB( int id );
  static void updateRouteCB( int id );
  static void loadRouteCB( int id );
  static void newRouteCB( int id );
  static void saveRouteCB( int id );
  static void addToRouteCB( int id );
  static void deleteCurrentMarkerCB( int id );
  static void traverseRouteCB( int id );
  static void clearRouteCB( int id );
  static void goToNextMarkerCB( int id );
  static void goToPrevMarkerCB( int id );
  static void goToRouteBeginningCB( int id );

  // Light Window Callbacks
  void createLightWindow( GLUI * window );
  static void toggleLightsWindowCB( int id );
  static void toggleLightSwitchesCB( int id ); // turn all lights on/off
  static void toggleShowLightsCB( int id );    // display light positions
  //// Update the intensity of the currently selected light.
  static void updateIntensityCB( int id );
  //// Update the intensity of the ambient light.
  static void updateAmbientCB( int id );
  static void updateLightPanelCB( int id );
  static void updateLightPositionCB( int id );

  // Object Window Callbacks
  void createObjectWindow( GLUI * window );
  static void toggleObjectsWindowCB( int id );
  static void updateObjectCB( int id );
  static void attachKeypadCB( int id );

  ////////////////////////////////////////////////////////////////

  static void toggleAutoJitterCB( int id );
  static void toggleJitterCB( int id );

  ////////////////////////////////////////////////////////////////

  static void toggleHotspotsCB( int id );

  ////////////////////////////////////////////////////////////////

  static void updateRotateSensitivityCB( int id );
  static void updateTranslateSensitivityCB( int id );
  static void updateFovCB( int id );

  static void updateDepthCB( int id );

  ////////////////////////////////////////////////////////////////
  // Helper Functions:

  void toggleGui();
  void cycleShadowMode();
  void cycleAmbientMode();
  void quit();
  void setupFonts();

  // Functions to draw text, etc on GL window.
  void displayText(GLuint fontbase, double x, double y,
		   char *s, const Color& c);
  void displayShadowText(GLuint fontbase,
			 double x, double y, char *s, const Color& c);
  void drawrstats(int nworkers, Worker** workers, int showing_scene,
		  GLuint fontbase, int xres, int yres,
		  XFontStruct* font_struct, int left, int up, double dt);
  void draw_labels(XFontStruct* font_struct, GLuint fontbase,
		   int& column, int dy, int top);
  void draw_column(XFontStruct* font_struct,
		   GLuint fontbase, char* heading, DepthStats& sum,
		   int x, int w2, int dy, int top,
		   bool first=false, double dt=1, int nworkers=0,
		   int npixels=0);
  void drawpstats(Stats* mystats, int nworkers, Worker** workers,
		  bool draw_framerate, int showing_scene,
		  GLuint fontbase, double& lasttime,
		  double& cum_ttime, double& cum_dt);

  ////////////////////////////////////////////////////////////////

};


} // end namespace rtrt

#endif
