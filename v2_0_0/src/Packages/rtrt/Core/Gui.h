
#ifndef RTRT_GUI_H
#define RTRT_GUI_H

#include <GL/glx.h>
#include <GL/glu.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <string>
#include <vector>

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/MouseCallBack.h>

// Foward declare GLUI objects
class GLUI;
class GLUI_Listbox;
class GLUI_Rollout;
class GLUI_Panel;
class GLUI_EditText;
class GLUI_StaticText;
class GLUI_Spinner;
class GLUI_Button;
class GLUI_Checkbox;

namespace rtrt {

class  Dpy;
struct DpyPrivate;
class  Stealth;
class  Camera;
class  Color;
class  Worker;
class  DepthStats;
class  Stats;
class  Light;
class  PPMImage;
class  SelectableGroup;
class  SpinningInstance;
class  CutGroup;
class  Sound;
class  Trigger;

// Because we have to use static functions, only one Gui object may be
// active at a time.  (My guess is that there will only be one Gui
// object at a time anyway...)

class Gui {

public:
  Gui();
  ~Gui();

  static void setActiveGui( Gui * gui );

  void setDpy( Dpy * dpy );
  void setStealth( Stealth * stealth );

  void setBottomGraphic( Trigger * trig ) { bottomGraphicTrig_ = trig; }
  void setLeftGraphic( Trigger * trig ) { leftGraphicTrig_ = trig; }

  void addLight( Light * light );

  // Tell the GUI to update all the information in its windows.
  // (eg: eye location)
  void update();

  // Used by GLUT
  static void redrawBackgroundCB();
  static void handleWindowResizeCB( int width, int height );
  static void handleKeyPressCB( unsigned char key,
			      int /*mouse_x*/, int /*mouse_y*/ );
  static void handleSpecialKeyCB( int key,
				int /*mouse_x*/, int /*mouse_y*/ );
  static void handleMouseCB(int button, int state, int x, int y);
  static void handleMouseMotionCB( int mouse_x, int mouse_y );
  static void handleSpaceballMotionCB( int sbm_x, int sbm_y, int sbm_z );
  static void handleSpaceballRotateCB( int sbr_x, int sbr_y, int sbr_z );
  static void handleSpaceballButtonCB( int button, int state );

  // This must be called after glutInit(...)!
  //   If showGui is false, the gui window will not be displayed.  Use
  //   'G' or right mouse menu to bring it up.
  static void createMenus( int winId, bool soundOn = false,
			   bool showGui = true );

  ///////////////////////////////////////////////////////////////////
  // 
  // Functions that allow the sound thread to interact with the gui.
  // 
  // - Tells the Gui that the sound thread has finished loading
  //   the sounds and is now active.  Gui will allow user interaction
  //   with the sound thread now.
  void    soundThreadNowActive();

  // - The SoundThread will get the current sound from us so it can
  //   later tell us things about the sound.
  //Sound * getCurrentSound() { return currentSound_; }
  // - To avoid a circular dependency (I wish I didn't have to) the
  //   sound thread will fill in this information (instead of us being
  //   able to query it) 
  //void    setSoundInformation();

  // object to keep track of while a volume is selected
  Object* selected_obj;

private:

  friend class Dpy;

  Dpy               * dpy_;
  struct DpyPrivate * priv;
  Camera            * camera_;
  Stealth           * stealth_;
  Sound             * currentSound_;

  int                 glutDisplayWindowId;

  // Main Text Trigger (MTT)
  Trigger * activeMTT_; // If queued, the activeMMT is told to deactivate
  Trigger * queuedMTT_; // and the queuedMTT will kick in as soon as it's done.

  Trigger * bottomGraphicTrig_;
  Trigger * leftGraphicTrig_;

  // Specific Triggers
  Trigger * visWomanTrig_;
  Trigger * csafeTrig_;
  Trigger * geophysicsTrig_;

  PPMImage * backgroundImage_;
  int        recheckBackgroundCnt_; // Check to see if we have
                                    // moved to another room every X cycles.

  // Gui Component Variables

  int selectedLightId_;
  int selectedRouteId_;
  int selectedObjectId_;
  int selectedSoundId_;
  int selectedTriggerId_;

  char inputString_[ 1024 ];

  GLUI         * mainWindow;

  GLUI         * routeWindow;
  GLUI         * lightsWindow;
  GLUI         * objectsWindow;
  GLUI         * soundsWindow;
  GLUI         * triggersWindow_;

  GLUI_Button  * openSoundPanelBtn_;

  GLUI         * getStringWindow;

  bool routeWindowVisible;
  bool lightsWindowVisible;
  bool objectsWindowVisible;
  bool soundsWindowVisible;
  bool triggersWindowVisible;
  bool mainWindowVisible;

  bool enableSounds_;

  // Last x/y of a mouse down/move:
  int   last_x_;
  int   last_y_;

  // 0 == no button down, 1 == button 1, 2 == button 2, etc;
  int   mouseDown_;
  bool  shiftDown_; // during mouse press
  bool  altDown_;
  bool  ctrlDown_;

  bool  rightButtonMenuActive_;

  bool  beQuiet_;

  bool  displayRStats_;
  bool  displayPStats_;

  std::vector<Light*> lights_;
  bool                lightsOn_;
  bool                lightsBeingRendered_;

  std::vector<Sound*> sounds_;

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

  GLUI_Button   * WhiteBGButton_;
  GLUI_Button   * BlackBGButton_;
  GLUI_Button   * OrigBGButton_;
  
  GLUI_Listbox  * shadowModeLB_;
  GLUI_Listbox  * ambientModeLB_;
  GLUI_Button   * jitterButton_;

  GLUI_Spinner  * soundVolumeSpinner_;
  GLUI_Spinner  * glyphThresholdSpinner_;

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

  GLUI_Button  * lightOnOffBtn_;

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

  GLUI_Spinner * numThreadsSpinner_;

  GLUI_Spinner * ray_offset_spinner;
  float ray_offset;

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
  // objectWindow GLUI elements:
  //


  ////////////////////////////////////////////////////////////////
  //
  // soundsWindow GLUI elements:
  //

  GLUI_Listbox  * soundList_;
  GLUI_EditText * leftVolume_;
  GLUI_EditText * rightVolume_;
  GLUI_Button   * startSoundThreadBtn_;
  GLUI_EditText * soundOriginX_;
  GLUI_EditText * soundOriginY_;
  GLUI_EditText * soundOriginZ_;

  ////////////////////////////////////////////////////////////////
  //
  // triggersWindow GLUI elements:
  //

  GLUI_Listbox  * triggerList_;

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
  GLUI_EditText * getStringText_;
  static void hideGetStringWindowCB( int id );
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
  static void bgColorCB( int id );

  // Light Window Callbacks
  void createLightWindow( GLUI * window );
  static void toggleLightsWindowCB( int id );
  static void toggleLightSwitchesCB( int id ); // turn all lights on/off
  static void toggleLightOnOffCB( int id );    // turn off/on current light.
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
  static void SGAutoCycleCB( int id );
  static void SGNoSkipCB( int id );
  static void SGNextItemCB( int id );
  static void SGCurrentFrameCB( int id );
  static void SISpinCB( int id );
  static void SIIncMagCB( int id );
  static void SIDecMagCB( int id );
  static void SISlideUpCB( int id );
  static void SISlideDownCB( int id );
  static void CGOnCB( int id );
  static void CGSpinCB( int id );

  // Sounds Window Callbacks
  void createSoundsWindow( GLUI * window );
  static void updateSoundCB( int id );
  static void startSoundThreadCB( int id );
  static void toggleSoundWindowCB( int id );

  // Triggers Window Callbacks
  void createTriggersWindow( GLUI * window );
  static void activateTriggerCB( int id );
  static void toggleTriggersWindowCB( int id );

  ////////////////////////////////////////////////////////////////

  static void toggleAutoJitterCB( int id );
  static void toggleJitterCB( int id );

  ////////////////////////////////////////////////////////////////

  static void toggleHotspotsCB( int id );
  static void toggleTransmissionModeCB( int id );

  ////////////////////////////////////////////////////////////////

  static void updateRotateSensitivityCB( int id );
  static void updateTranslateSensitivityCB( int id );
  static void updateFovCB( int id );

  static void updateRayOffsetCB( int id );

  static void updateDepthCB( int id );

  ////////////////////////////////////////////////////////////////
  // Helper Functions:

  void toggleGui();
  void cycleShadowMode();
  void cycleAmbientMode();
  void quit();
  void setupFonts();
  void updateSoundPanel();
  void loadAllRoutes();
  void handleTriggers();
  bool checkBackgroundWindow();
  bool setBackgroundImage( int room );
  void drawBackground();


  // Functions to draw text, etc on GL window.
  void displayText(GLuint fontbase, double x, double y,
		   char *s, const Color& c);
  void displayShadowText(GLuint fontbase,
			 double x, double y, char *s, const Color& c);
  void drawrstats(int nworkers, std::vector<Worker*> & workers,
		  int showing_scene, GLuint fontbase, int xres, int yres,
		  XFontStruct* font_struct, int left, int up, double dt);
  void draw_labels(XFontStruct* font_struct, GLuint fontbase,
		   int& column, int dy, int top);
  void draw_column(XFontStruct* font_struct,
		   GLuint fontbase, char* heading, DepthStats& sum,
		   int x, int w2, int dy, int top,
		   bool first=false, double dt=1, int nworkers=0,
		   int npixels=0);
  void drawpstats(Stats* mystats, int nworkers, std::vector<Worker*> & workers,
		  bool draw_framerate, int showing_scene,
		  GLuint fontbase, double& lasttime,
		  double& cum_ttime, double& cum_dt);

  ////////////////////////////////////////////////////////////////

};

} // end namespace rtrt

#endif
