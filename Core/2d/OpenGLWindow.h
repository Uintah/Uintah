/*
 *  OpenGLWindow.h:
 *
 *  Written by:
 *   Yarden Livnat
 *   July 20001
 *
 */


#ifndef OpenGLWindow_h
#define OpenGLWindow_h

#include <string>
#include <map>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>

#include <Core/2d/Drawable.h>
#include <Core/Containers/Array1.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/TCL.h>
#include <Core/Thread/Mutex.h>
#include <Core/Geom/Color.h>


namespace SCIRun {

class OpenGLWindow {
private:
  Mutex *lock_;

  // OpenGL
  Window win;
  Display* dpy;
  GLXContext cx;
  GLuint base;  

  int xres_, yres_;
  Color bg_, fg_;
  
public:
  OpenGLWindow();
  virtual ~OpenGLWindow() {}

  void lock() { lock_->lock(); }
  void unlock() { lock_->unlock(); }

  bool init( const string &);
  bool tcl_command(TCLArgs&, void*);

protected:
  virtual void report_init() {}
  void clear();
  void pre();
  void post();
  void make_raster_font();
  void print_string( char *);

  bool initialized_;
};


} // End namespace SCIRun

#endif OpenGLWindow_h


