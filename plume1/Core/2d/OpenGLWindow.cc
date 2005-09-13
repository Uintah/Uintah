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
 *  OpenGLWindow.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   July 20001
 *
 */


#include <tcl.h>
#include <tk.h>

#include <Core/2d/OpenGLWindow.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiInterface.h>

#ifdef _WIN32
#include <tkWinInt.h>
#endif


#include <iostream>
using namespace SCIRun;
using namespace std;
  
extern "C" Tcl_Interp* the_interp;

#ifndef _WIN32
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);
#else
extern "C" HGLRC OpenGLGetContext(Tcl_Interp*, char*);
#endif

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
#ifndef _WIN32
  dpy=Tk_Display(tkwin);
  win=Tk_WindowId(tkwin);
  cx=OpenGLGetContext(the_interp, const_cast<char *>(window_name.c_str()));
  if(!cx){
    cerr << "Unable to create OpenGL Context!\n";
    gui->unlock();
    return false;
  }
#else
  hdc = TkWinGetHDC(tkwin);
  hglrc=OpenGLGetContext(the_interp, const_cast<char *>(window_name.c_str()));
  if(!hglrc){
    cerr << "Unable to create OpenGL Context!\n";
    gui->unlock();
    return false;
  }
#endif
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

#ifndef _WIN32
  if (!glXMakeCurrent(dpy, win, cx))
    cerr << "*glXMakeCurrent failed.\n";
#else
    if (!wglMakeCurrent(hdc,hglrc))
    cerr << "*wglMakeCurrent failed.\n";

#endif

  glViewport(0, 0, xres_, yres_);
  
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}


void
OpenGLWindow::post( bool swap)
{
#ifndef _WIN32
  if ( swap )
    glXSwapBuffers( dpy, win);

  glXMakeCurrent(dpy, None, NULL);
#else
  if ( swap )
    SwapBuffers( hdc );

  wglMakeCurrent(hdc, NULL);
#endif

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
#ifndef _WIN32
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
#else // WIN32
  if (!hdc)
    return;
  DWORD first, count;

  // for now, just use the system font
  SelectObject(hdc,GetStockObject(SYSTEM_FONT));

  // rasterize the standard character set.
  first = 0;
  count = 256;
  base =  glGenLists((GLuint) 1);

  if (base == 0) {
    printf("out of display lists\n");
    exit(0);
  }
  wglUseFontBitmaps( hdc, first, count, base );
#endif
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
