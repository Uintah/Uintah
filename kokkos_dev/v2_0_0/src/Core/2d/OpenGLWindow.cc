/*
 *  OpenGLWindow.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   July 20001
 *
 */


#include <GL/gl.h>
#include <sci_glu.h>
#include <GL/glx.h>

#include <tcl.h>
#include <tk.h>

#include <Core/2d/OpenGLWindow.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiInterface.h>

#include <iostream>
using namespace SCIRun;
using namespace std;
  
extern "C" Tcl_Interp* the_interp;
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);

OpenGLWindow::OpenGLWindow(GuiInterface* gui)
  : TclObj(gui, "OpenGLWindow")
{
  lock_ = scinew Mutex("OpenGLWindow");
  initialized_ = false;
  bg_ = Color( 0.8, 0.8, 0.8 );
  fg_ = Color( 0.0, 0.8, 0.0 );
}

void
OpenGLWindow::clear()
{
  glClearColor(bg_.r(), bg_.g(), bg_.b(), 0);
  glClear(GL_COLOR_BUFFER_BIT);
}

void
OpenGLWindow::tcl_command(GuiArgs& args, void*)
{
  if ( args[1] == "map" ) {
    if ( !initialized_ ) 
      init(args[2]);
  }
}

bool
OpenGLWindow::init( const string &window_name)
{
  gui->lock();

  Tk_Window tkwin=Tk_NameToWindow(the_interp,
				  const_cast<char *>(window_name.c_str()),
				  Tk_MainWindow(the_interp));
  if(!tkwin){
    cerr << "Unable to locate window!\n";
    gui->unlock();
    return false;
  }
  dpy=Tk_Display(tkwin);
  win=Tk_WindowId(tkwin);
  cx=OpenGLGetContext(the_interp, const_cast<char *>(window_name.c_str()));
  if(!cx){
    cerr << "Unable to create OpenGL Context!\n";
    gui->unlock();
    return false;
  }
  //    }
  
  // Get the window size
  xres_=Tk_Width(tkwin);
  yres_=Tk_Height(tkwin);
  //make_raster_font();

  initialized_ = true;
  
  gui->unlock();

  report_init();
  return true;
}


void 
OpenGLWindow::pre()
{
  gui->lock();

  if (!glXMakeCurrent(dpy, win, cx))
    cerr << "*glXMakeCurrent failed.\n";
  

  glViewport(0, 0, xres_, yres_);
  
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}


void
OpenGLWindow::post( bool swap)
{
  if ( swap )
    glXSwapBuffers( dpy, win);

  glXMakeCurrent(dpy, None, NULL);

  gui->unlock();
}


void
OpenGLWindow::set_cursor( const string& c )
{
  tcl_ << "set-cursor " << c;
  tcl_exec();
}

void
OpenGLWindow::set_cursor_file( const string& c )
{
  tcl_ << "set-cursor \"@../pixmaps/" << c << ".xbm black\"";
  tcl_exec();
}

void
OpenGLWindow::set_binds( const string& obj )
{
  tcl_ << "setobj " << obj;
  tcl_exec();
}


void 
OpenGLWindow::make_raster_font()
{
  if (!cx)
    return;

  XFontStruct *fontInfo;
  Font id;
  unsigned int first, last;
  Display *xdisplay;
  
  xdisplay = dpy;
  fontInfo = XLoadQueryFont(xdisplay, 
			    "-*-helvetica-medium-r-normal--12-*-*-*-p-67-iso8859-1");
  if (fontInfo == NULL) {
    printf ("no font found\n");
    exit (0);
  }
  
  id = fontInfo->fid;
  first = fontInfo->min_char_or_byte2;
  last = fontInfo->max_char_or_byte2;
  
  base = glGenLists((GLuint) last+1);
  if (base == 0) {
    printf ("out of display lists\n");
    exit (0);
  }
  glXUseXFont(id, first, last-first+1, base+first);
  /*    *height = fontInfo->ascent + fontInfo->descent;
   *width = fontInfo->max_bounds.width;  */
}

void 
OpenGLWindow::print_string(char *s)
{
  glPushAttrib (GL_LIST_BIT);
  glListBase(base);
  glCallLists(strlen(s), GL_UNSIGNED_BYTE, (GLubyte *)s);
  glPopAttrib ();
}

void
OpenGLWindow::resize( int h, int w )
{
  xres_ = h;
  yres_ = w;
  tcl_ << " resize " << h << " " << w;
  tcl_exec();
}

OpenGLWindow *
OpenGLWindow::sub_window( int , int , int , int  )
{
  return 0; // scinew OpenGLSubWindow( *this, l, r, t, b );
}

void OpenGLWindow::report_init()
{
}

void OpenGLWindow::lock()
{
  gui->lock();
}

void OpenGLWindow::unlock()
{
  gui->unlock();
}
