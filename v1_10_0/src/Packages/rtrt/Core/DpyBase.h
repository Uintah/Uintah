
#ifndef __DPYBASE__H__
#define __DPYBASE__H__

#include <Core/Thread/Runnable.h>

#include <string>

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

  // This opens the window.
  int open_display(Window parent = 0, bool needevents = true);
  // Closes the display
  int close_display();

  // Flag letting the control loop know to stop executing and
  // close the display.  Setting this to true will kill the thread.
  bool on_death_row;
  
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
  // next available opportunity.  Note: more than one event may be handled
  // before display is called.
  bool redraw;

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
  DpyBase(const char *name, const int window_mode = DoubleBuffered);
  virtual ~DpyBase();

  void Hide();
  void Show();

  void setName( const string & name );

  // Sets the resolution of the window.  Currently this only has an effect
  // before you create the window.
  void set_resolution(const int xres_in, const int yres_in);
  
  // This run funtion will handle the basic events defined above.  If you
  // need something more sophisticated, then you can redefine this function.
  // But be carefull, you have to handle all the events yourself.  Use the
  // code for this function as a base.
  virtual void run();

  // I don't think this function is needed right now.
  //  virtual void animate(double t, bool& changed);

  // This causes the thread to end at the first opportunity closing the window.
  void stop();
  
  void set_scene(Scene *new_scene) { scene = new_scene; }
};

  // helper funtions
  void printString(GLuint fontbase, double x, double y,
		   const char *s, const Color& c);
  int calc_width(XFontStruct* font_struct, const char* str);
  
} // end namespace rtrt

#endif // __DPYBASE__H__
