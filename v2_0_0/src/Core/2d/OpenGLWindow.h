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

#include <sgi_stl_warnings_off.h>
#include <string>
#include <map>
#include <sgi_stl_warnings_on.h>

#include <GL/gl.h>
#include <sci_glu.h>
#include <GL/glx.h>

#include <Core/2d/Drawable.h>
#include <Core/Containers/Array1.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/TclObj.h>
#include <Core/Thread/Mutex.h>
#include <Core/Datatypes/Color.h>


namespace SCIRun {

class OpenGLWindow : public TclObj {
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
  OpenGLWindow(GuiInterface* gui);
  virtual ~OpenGLWindow() {}

  void lock();
  void unlock();

  bool init( const string &);
  bool initialized() { return initialized_; }
  virtual void tcl_command(GuiArgs&, void*);
  
  virtual void clear();
  virtual void pre();
  virtual void post( bool = true );

  void set_cursor( const string &);
  void set_cursor_file( const string &);
  void set_binds( const string &);
  void print_string( char *);

  OpenGLWindow *sub_window( int l, int r, int t, int b );
  void resize( int, int );
  int xres() { return xres_; }
  int yres() { return yres_; }

protected:
  virtual void report_init();
  void make_raster_font();

  bool initialized_;
};


/* class OpenGLSubWindow : public OpenGLWindow { */
/*  private: */
/*   int left_, right_, top_, bottom_; */

/*  public: */
/*   OpenGLSubWindow( OpenGLWindow &ogl, int l, int r, int t, int b )  */
/*     : OpenGLWindow(ogl), left_(l), right_(r), top_(t), bottom_(b) */
/*     { */
/*     } */

/* }; */

} // End namespace SCIRun

#endif /* OpenGLWindow_h */


