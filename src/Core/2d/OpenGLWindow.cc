/*
 *  OpenGLWindow.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   July 20001
 *
 */


#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>

#include <tcl.h>
#include <tk.h>

#include <Core/2d/OpenGLWindow.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/TCL.h>

namespace SCIRun {

extern "C" Tcl_Interp* the_interp;
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);



OpenGLWindow::OpenGLWindow()
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

bool
OpenGLWindow::tcl_command(TCLArgs& args, void* userdata)
{
  bool mine = true;
  if ( args[1] == "map" ) {
    if ( !initialized_ ) 
      init(args[2]);
  }
  else
    mine = false;

  return mine;
}

bool
OpenGLWindow::init( const string &window_name)
{
  TCLTask::lock();

  Tk_Window tkwin=Tk_NameToWindow(the_interp,
				  const_cast<char *>(window_name.c_str()),
				  Tk_MainWindow(the_interp));
  if(!tkwin){
    cerr << "Unable to locate window!\n";
    TCLTask::unlock();
    return false;
  }
  dpy=Tk_Display(tkwin);
  win=Tk_WindowId(tkwin);
  cx=OpenGLGetContext(the_interp, const_cast<char *>(window_name.c_str()));
  if(!cx){
    cerr << "Unable to create OpenGL Context!\n";
    TCLTask::unlock();
    return false;
  }
  //    }
  
  // Get the window size
  xres_=Tk_Width(tkwin);
  yres_=Tk_Height(tkwin);
  //make_raster_font();

  initialized_ = true;
  
  TCLTask::unlock();

  report_init();
  return true;
}


void 
OpenGLWindow::pre()
{
  TCLTask::lock();

  if (!glXMakeCurrent(dpy, win, cx))
    cerr << "*glXMakeCurrent failed.\n";
  
  // Clear the screen...
  glViewport(0, 0, xres_, yres_);
  
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

void
OpenGLWindow::post()
{
  glXSwapBuffers( dpy, win);

  GLenum errcode;
  while((errcode=glGetError()) != GL_NO_ERROR){
    cerr << "Plot got an error from GL: " 
	 << (char*)gluErrorString(errcode) << endl;
  }
  glXMakeCurrent(dpy, None, NULL);
  TCLTask::unlock();
}


void 
OpenGLWindow::make_raster_font()
{
  if (!cx)
    return;

  cerr << "Generating new raster font list!\n";
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

} // End namespace SCIRun


