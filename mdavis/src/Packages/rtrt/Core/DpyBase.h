
#ifndef __DPYBASE__H__
#define __DPYBASE__H__

#include <Core/Thread/Runnable.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

#include <GL/glx.h>
#include <GL/glu.h>

// Needed for the decoding of keyboard macros
#include <X11/keysym.h>

namespace rtrt {

using SCIRun::Runnable;
using std::string;

class Scene;
class Color;
  
enum MouseButton {
  MouseButton1,
  MouseButton2,
  MouseButton3
};

enum {
  SingleBuffered = 0x00000000,
  DoubleBuffered = 0x00000001
};

enum {
  BufferModeMask = 0x00000001
};

class DpyBase : public Runnable {
protected:
  
  // The width and height of the window.
  int xres, yres;

  // These functions need to be called by the derived class.  They return 0
  // if it was successful.

  // This is set to true once open_display is called.  Set back to
  // false once it is closed.
  bool opened;
  // This tells if we have an OpenGL context.  Dependent on if you
  // called open_display or open_events_display.
  bool have_ogl_context;
  
  // This opens the window.
  int open_display(Window parent = 0, bool needevents = true);
  // This opens a window that is set up for only events
  int open_events_display(Window parent = 0);
  
  // Closes the display
  int close_display();
  // Calling this will make sure the window doesn't close.  This is
  // useful for when you have a parent.  Let the parent window close.
  // Don't close the child.
  void dont_close();
  // Tells if the display should be closed.  Defaults to true.
  bool close_display_flag;

  // This function does what ever you need to do before going away.
  // It should also call close_display().
  virtual void cleanup();
  bool cleaned;
  
  // This is for global cleanups
  static void cleanup_aux(void* ptr) {
    ((DpyBase*)ptr)->cleanup();
  }

  // Flag letting the control loop know to stop executing and
  // close the display.  Setting this to true will kill the thread.
  bool on_death_row;
  // This will test on_death_row as well as if the exit_scene is true
  // with the rtrt_engine.
  virtual bool should_close();
  
  ////////////////////////////////////////////////////////////////////////
  // These event functions do a whole lot of nothing.  If you want them to
  // something else, then redefine them.

  // Called at the start of run.
  virtual void init();
  // Called whenever the window needs to be redrawn
  virtual void display();
  // Called when the window is resized.  Note: xres and yres will not be
  // updated by the event handler.  That's what this function is for.
  virtual void resize(const int width, const int height);
  // Key is pressed/released.  Use the XK_xxx constants to determine
  // which key was pressed/released
  virtual void key_pressed(unsigned long key);
  virtual void key_released(unsigned long key);
  // These handle mouse button events.  button indicates which button.
  // x and y are the location measured from the upper left corner of the
  // window.
  virtual void button_pressed(MouseButton button, const int x, const int y);
  virtual void button_released(MouseButton button, const int x, const int y);
  virtual void button_motion(MouseButton button, const int x, const int y);

  // If this flag is true, then the display() will be called at the
  // next available opportunity.  Note: more than one event may be
  // handled before display is called.
  bool redraw;

  // If there are no events this sleeps the thread for a short time to
  // prevent busy spinning.  If there are events pending it does not
  // sleep but handles as many events as are queued.
  void wait_and_handle_events();
  
  // State variables indicating if the control or shift buttons are
  // currently being pressed.
  bool control_pressed;
  bool shift_pressed;


  string window_name;
  char *cwindow_name;
  int window_mode;
  Display *dpy;
  Window win;
  XFontStruct* fontInfo;
  GLuint fontbase;
  
  Scene *scene;
public:
  // name is the name of the window.
  DpyBase(const char *name, const int window_mode = DoubleBuffered,
          bool delete_on_exit = true);
  virtual ~DpyBase();

  void Hide();
  void Show();

  void setName( const string & name );

  // Sets the resolution of the window.  Currently this only has an effect
  // before you create the window.
  virtual void set_resolution(const int width, const int height);
  
  // This run funtion will handle the basic events defined above.  If you
  // need something more sophisticated, then you can redefine this function.
  // But be carefull, you have to handle all the events yourself.  Use the
  // code for this function as a base.
  virtual void run();

  // I don't think this function is needed right now.
  //  virtual void animate(double t, bool& changed);

  // This causes the thread to end at the first opportunity closing
  // the window.  It also generates an X event so that a thread
  // waiting for an event will actually stop.
  void stop();
  
  void set_scene(Scene *new_scene) { scene = new_scene; }

public:
  // These functions try to minimize the number of things going on
  // with the xserver.
  static void xlock();
  static void xunlock();

  // This says whether to use XLockDisplay and XUnlockDisplay.
  static void initUseXThreads();
  static bool useXThreads;
};

  // helper funtions
  void printString(GLuint fontbase, double x, double y,
		   const char *s, const Color& c);
  int calc_width(XFontStruct* font_struct, const char* str);

} // end namespace rtrt

#endif // __DPYBASE__H__
