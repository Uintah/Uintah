#include <Packages/rtrt/Core/DpyBase.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Time.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <Packages/rtrt/visinfo/visinfo.h>
#include <stdio.h>
#include <X11/keysym.h>
#include <algorithm>
#ifdef __GNUG__
#include <values.h>
#endif
#include <Packages/rtrt/Core/FontString.h>

using namespace rtrt;
using SCIRun::Mutex;
using SCIRun::Thread;
using namespace std;

namespace rtrt {
  extern Mutex xlock;
} // end namespace rtrt

DpyBase::DpyBase(const char *name, const int window_mode):
  xres(300), yres(300), on_death_row(false),
  redraw(true), control_pressed(false), shift_pressed(false),
  window_mode(window_mode)
{
  window_name = name;
  cwindow_name = (char*)malloc(256*sizeof(char));
  sprintf(cwindow_name, "%.255s", window_name.c_str());
}

DpyBase::~DpyBase() {
}

static int DPY_NX=0;
static int DPY_NY=0;

int DpyBase::open_display(Window parent, bool needevents) {
  xlock.lock();
  // Open an OpenGL window
  dpy = XOpenDisplay(NULL);
  if(!dpy){
    cerr << "Cannot open display\n";
    return 1;
  }
  int error, event;
  if ( !glXQueryExtension( dpy, &error, &event) ) {
    cerr << "GL extension NOT available!\n";
    XCloseDisplay(dpy);
    dpy=0;
    return 1;
  }
  int screen=DefaultScreen(dpy);
  
  // criteria is a string constant that represents the type of window
  //   ex: "db, max rgb" - double buffered, max color depth for rgb
  //     : "sb, max rgb" - single buffered, max color depth for rgb
  char* criteria;
  if (window_mode & BufferModeMask == DoubleBuffered)
    criteria = strdup("db, max rgba");
  else
    criteria = strdup("sb, max rgba");
    
  if(!visPixelFormat(criteria)){
    cerr << "Error setting pixel format for visinfo\n";
    cerr << "Syntax error in criteria: " << criteria << '\n';
    return 1;
  }
  int nvinfo;
  XVisualInfo* vi=visGetGLXVisualInfo(dpy, screen, &nvinfo);
  if(!vi || nvinfo == 0){
    cerr << "Error matching OpenGL Visual: " << criteria << '\n';
    return 1;
  }
  // free criteria
  if (criteria) free(criteria);

  Colormap cmap = XCreateColormap(dpy, RootWindow(dpy, screen),
				  vi->visual, AllocNone);
  XSetWindowAttributes atts;
  int flags=CWColormap|CWEventMask|CWBackPixmap|CWBorderPixel;
  atts.background_pixmap = None;
  atts.border_pixmap = None;
  atts.border_pixel = 0;
  atts.colormap=cmap;
  if(needevents)
    atts.event_mask=StructureNotifyMask|ExposureMask|ButtonPressMask|ButtonReleaseMask|ButtonMotionMask|KeyPressMask|KeyReleaseMask;
  else
    atts.event_mask=StructureNotifyMask;
  //DDM Added for SIGGRAPH demo to make it pop up on 2nd display
  int xlow = DPY_NX;
  int ylow = DPY_NY;
  if(!parent){
    parent = RootWindow(dpy, screen);
    xlow=ylow=0;
  } else {
    DPY_NX+=20;
    DPY_NY+=yres;
  }
  win=XCreateWindow(dpy, parent,
		    xlow, ylow, xres, yres, 0, vi->depth,
		    InputOutput, vi->visual, flags, &atts);
  cerr << "CreateWindow, parent=" << parent << ", xlow=" << xlow << ", ylow=" << ylow << ", xres=" << xres << ", yres=" << yres << '\n';
  XTextProperty tp;
  XStringListToTextProperty(&cwindow_name, 1, &tp);
  XSizeHints sh;
  sh.flags = USPosition|USSize;
  
  XSetWMProperties(dpy, win, &tp, &tp, 0, 0, &sh, 0, 0);
  
  XMapWindow(dpy, win);
  
  GLXContext cx=glXCreateContext(dpy, vi, NULL, True);
  if(!glXMakeCurrent(dpy, win, cx)){
    cerr << "glXMakeCurrent failed!\n";
  }

  // set up the fonts
  fontInfo = XLoadQueryFont(dpy, __FONTSTRING__);
  if (fontInfo == NULL) {
    cerr << "no font found\n";
    return 1;
  }
  Font id = fontInfo->fid;
  unsigned int first = fontInfo->min_char_or_byte2;
  unsigned int last = fontInfo->max_char_or_byte2;
  fontbase = glGenLists((GLuint) last+1);
  if (fontbase == 0) {
    printf ("out of display lists\n");
    exit (0);
  }
  glXUseXFont(id, first, last-first+1, fontbase+first);
  xlock.unlock();

  // Assume success at this point
  return 0;
}

void
DpyBase::setName( const string & name )
{
  window_name = name;
  sprintf( cwindow_name,"%.255s", window_name.c_str() );
  if (dpy && win) {
    xlock.lock();  
    XTextProperty tp;
    XStringListToTextProperty(&cwindow_name, 1, &tp);
    XSizeHints sh;
    sh.flags = USSize;
    XSetWMProperties(dpy, win, &tp, &tp, 0, 0, &sh, 0, 0);
 
    xlock.unlock();  
  }
}

void DpyBase::Hide() {
  xlock.lock();
  XUnmapWindow(dpy, win);
  xlock.unlock();
  XFlush(dpy);
}
void DpyBase::Show() {
  xlock.lock();
  XMapRaised(dpy, win);
  xlock.unlock();
  XFlush(dpy);
}

int DpyBase::close_display() {
  xlock.lock();
  XCloseDisplay(dpy);
  xlock.unlock();
  return 0;
}

void DpyBase::stop() {
  on_death_row = true;
}

void DpyBase::init() {
  glShadeModel(GL_FLAT);
}

void DpyBase::display() {
  glFinish();
  if (window_mode & BufferModeMask == DoubleBuffered){
    glXSwapBuffers(dpy, win);
  }
  XFlush(dpy);
}

void DpyBase::resize(const int width, const int height) {
  xres = width;
  yres = height;
  redraw = true;
}

void DpyBase::key_pressed(unsigned long /*key*/) {
}

void DpyBase::key_released(unsigned long /*key*/) {
}

void DpyBase::button_pressed(MouseButton /*button*/,
			     const int /*x*/, const int /*y*/) {
}

void DpyBase::button_released(MouseButton /*button*/,
			      const int /*x*/, const int /*y*/) {
}

void DpyBase::button_motion(MouseButton /*button*/,
			    const int /*x*/, const int /*y*/) {
}

void DpyBase::set_resolution(const int xres_in, const int yres_in) {
  xres = xres_in;
  yres = yres_in;
}

extern bool pin;
void DpyBase::run() {

  if(pin)
    Thread::self()->migrate(127);

  open_display();

  init();
  
  // Create the Xevent handler
  for(;;){
    XEvent e;
    XNextEvent(dpy, &e);
    if(e.type == MapNotify)
      break;
  }
  

  for(;;){
    // Now we need to test to see if we should die
    //if (scene->get_rtrt_engine()->stop_execution() || on_death_row) {
    if (on_death_row) {
      close_display();
      return;
    }
    
    if(redraw){
      display();
      redraw=false;
    }
    // We should try to consume all the queued events before we redraw.
    // That way we don't waste time redrawing after each event
    while (XEventsQueued(dpy, QueuedAfterReading)) {
      // Now we need to test to see if we should die
      //if (scene->get_rtrt_engine()->stop_execution() || on_death_row) {
      if (on_death_row) {
	close_display();
	return;
      }
    
      XEvent e;
      XNextEvent(dpy, &e);	
      switch(e.type){
      case Expose:
	redraw=true;
	break;
      case ConfigureNotify:
	resize(e.xconfigure.width, e.xconfigure.height);
	break;
      case KeyPress:
	{
	  unsigned long key = XKeycodeToKeysym(dpy, e.xkey.keycode, 0);
	  switch(key) {
	  case XK_Control_L:
	  case XK_Control_R:
	    //cerr << "Pressed control\n";
	    control_pressed = true;
	    break;
	  case XK_Shift_L:
	  case XK_Shift_R:
	    //cerr << "Pressed shift\n";
	    shift_pressed = true;
	    break;
	  default:
	    key_pressed(key);
	  }
	}
	break;
      case KeyRelease:
	{
	  unsigned long key = XKeycodeToKeysym(dpy, e.xkey.keycode, 0);
	  switch(key) {
	  case XK_Control_L:
	  case XK_Control_R:
	    control_pressed = false;
	    //cerr << "Releassed control\n";
	    break;
	  case XK_Shift_L:
	  case XK_Shift_R:
	    //cerr << "Releassed shift\n";
	    shift_pressed = false;
	    break;
	  default:
	    key_released(key);
	  }
	}
	break;
      case ButtonRelease:
	{
	  MouseButton button = MouseButton1;
	  switch(e.xbutton.button){
	  case Button1:
	    button = MouseButton1;
	    break;
	  case Button2:
	    button = MouseButton2;
	    break;
	  case Button3:
	    button = MouseButton3;
	    break;
	  }
	  button_released(button, e.xbutton.x, e.xbutton.y);
	}
	break;
      case ButtonPress:
	{
	  MouseButton button = MouseButton1;
	  switch(e.xbutton.button){
	  case Button1:
	    button = MouseButton1;
	    break;
	  case Button2:
	    button = MouseButton2;
	    break;
	  case Button3:
	    button = MouseButton3;
	    break;
	  }
	  button_pressed(button, e.xbutton.x, e.xbutton.y);
	}
	break;
      case MotionNotify:
	{
	  MouseButton button = MouseButton1;
	  switch(e.xmotion.state&(Button1Mask|Button2Mask|Button3Mask)) {
	  case Button1Mask:
	    button = MouseButton1;
	    break;
	  case Button2Mask:
	    button = MouseButton2;
	    break;
	  case Button3Mask:
	    button = MouseButton3;
	    break;
	  }
	  button_motion(button, e.xbutton.x, e.xbutton.y);
	}
	break;
      default:
	//	cerr << "Unknown event, type=" << e.type << '\n';
	break;
      } // end switch (e.type)
    } // end of while (there is a queued event)
  } // end of for(;;)
}

namespace rtrt {
  
void printString(GLuint fontbase, double x, double y,
		 const char *s, const Color& c) {

  glColor3f(c.red(), c.green(), c.blue());
  
  glRasterPos2d(x,y);
  /*glBitmap(0, 0, x, y, 1, 1, 0);*/
  glPushAttrib (GL_LIST_BIT);
  glListBase(fontbase);
  glCallLists((int)strlen(s), GL_UNSIGNED_BYTE, (GLubyte *)s);
  glPopAttrib ();
}

int calc_width(XFontStruct* font_struct, const char* str) {
  XCharStruct overall;
  int ascent, descent;
  int dir;
  XTextExtents(font_struct, str, (int)strlen(str), &dir, &ascent, &descent, &overall);
  return overall.width;
}

} // end namespace rtrt
