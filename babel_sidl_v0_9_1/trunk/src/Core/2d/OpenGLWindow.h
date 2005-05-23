/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

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

#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>

#include <Core/2d/Drawable.h>
#include <Core/Containers/Array1.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/TclObj.h>
#include <Core/Thread/Mutex.h>
#include <Core/Datatypes/Color.h>


#ifdef _WIN32
#include <windows.h>
#endif

namespace SCIRun {

class OpenGLWindow : public TclObj {
private:
  Mutex *lock_;

  // OpenGL
#ifndef _WIN32
  Window win;
  Display* dpy;
  GLXContext cx;
#else
  HDC hdc;
  HGLRC hglrc;
#endif
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


