#include <Packages/rtrt/Core/DpyBase.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <Packages/rtrt/visinfo/visinfo.h>
#include <Packages/rtrt/Core/FontString.h>

#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/CleanupManager.h>

#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <stdlib.h>
#include <stdio.h>
#include <X11/keysym.h>
#include <sci_values.h>
#include <unistd.h> // For sleep()

using namespace rtrt;
using SCIRun::Mutex;
using SCIRun::Thread;
using namespace std;

#if 1
// I wanted to make this mutex a static member of DpyBase, but I had
// troubles when calling Thread::exitAll().  Somehow it was getting
// destroyed before non class members had a chance to use it.
static Mutex xmutex("X windows lock");
#else
class MyMutex {
public:
  MyMutex(): mutex_("X Windows lock") {
    cerr << "MyMutex::MyMutex() called\n";
  }
  ~MyMutex() {
    cerr << "MyMutex::~MyMutex() called\n";
  }

  void lock() { mutex_.lock(); }
  void unlock() { mutex_.unlock(); }

  Mutex mutex_;
};

static MyMutex xmutex;

#endif
bool rtrt::DpyBase::useXThreads = false;

DpyBase::DpyBase(const char *name, const int window_mode,
                 bool delete_on_exit):
  Runnable(delete_on_exit),
  xres(300), yres(300), opened(false), have_ogl_context(false),
  close_display_flag(true),
  cleaned(false), on_death_row(false),
  redraw(true), control_pressed(false), shift_pressed(false),
  window_mode(window_mode), scene(0)
{
  window_name = name;
  cwindow_name = (char*)malloc(256*sizeof(char));
  sprintf(cwindow_name, "%.255s", window_name.c_str());

  // Register the call back function to close the window when
  // Thread::exitAll(0) is called.
  SCIRun::CleanupManager::add_callback(this->cleanup_aux, this);
}

DpyBase::~DpyBase() {
  // Need to remove and call callback at same time or else you get a
  // race condition and the callback could be called twice.
  SCIRun::CleanupManager::invoke_remove_callback(this->cleanup_aux,
                                                 this);
}

static int DPY_NX=0;
static int DPY_NY=0;

int DpyBase::open_events_display(Window parent) {
  xlock();

  dpy = XOpenDisplay(NULL);
  if(!dpy){
    cerr << "Cannot open display\n";
    return 1;
  }

  int screen=DefaultScreen(dpy);
    
  XSetWindowAttributes atts;
  int flags=CWEventMask;
  atts.event_mask=StructureNotifyMask|ExposureMask|ButtonPressMask|ButtonReleaseMask|ButtonMotionMask|KeyPressMask|KeyReleaseMask;
  // Set up the parent if we need to.
  int xlow = DPY_NX;
  int ylow = DPY_NY;
  if(!parent){
    parent = RootWindow(dpy, screen);
    xlow=ylow=0;
  } else {
    DPY_NX+=20;
    DPY_NY+=yres;
  }
  cerr << "Attempting to create window for "<<window_name<<" with resolution ("<<xres<<", "<<yres<<")\n";
  win = XCreateWindow(dpy, parent,
                      xlow, ylow, xres, yres, 0,
                      DefaultDepth(dpy, screen), InputOutput,
                      DefaultVisual(dpy, screen), flags, &atts);
  
  XTextProperty tp;
  XStringListToTextProperty(&cwindow_name, 1, &tp);
  XSizeHints sh;
  sh.flags = USPosition|USSize;
  
  XSetWMProperties(dpy, win, &tp, &tp, 0, 0, &sh, 0, 0);
  XMapWindow(dpy, win);
  
  xunlock();
  
  opened = true;

  // Wait for the window to appear before proceeding
  for(;;){
    XEvent e;
    XNextEvent(dpy, &e);
    if(e.type == MapNotify)
      break;
  }
  
  return 0;
}

int DpyBase::open_display(Window parent, bool needevents) {
  xlock();
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
  char criteria[50]; // leave space for additional options
  if ((window_mode & BufferModeMask) == DoubleBuffered)
    strcpy(criteria,"db");
  else
    strcpy(criteria,"sb");
    
#if !defined(__APPLE__)
  strcat(criteria, ", max rgb");
#endif

  cerr << "criteria = \""<<criteria<<"\"\n";
  
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
  cerr << "Attempting to create window for "<<window_name<<" with resolution ("<<xres<<", "<<yres<<")\n";
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
  } else {
    have_ogl_context = true;
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
  xunlock();

  opened = true;

  // Wait for the window to appear before proceeding
  for(;;){
    XEvent e;
    XNextEvent(dpy, &e);
    if(e.type == MapNotify)
      break;
  }

  // Assume success at this point
  return 0;
}

void
DpyBase::setName( const string & name )
{
  window_name = name;
  sprintf( cwindow_name,"%.255s", window_name.c_str() );
  if (dpy && win) {
    xlock();  
    XTextProperty tp;
    XStringListToTextProperty(&cwindow_name, 1, &tp);
    XSizeHints sh;
    sh.flags = USSize;
    XSetWMProperties(dpy, win, &tp, &tp, 0, 0, &sh, 0, 0);
 
    xunlock();  
  }
}

void DpyBase::Hide() {
  xlock();
  XUnmapWindow(dpy, win);
  xunlock();
  XFlush(dpy);
}

void DpyBase::Show() {
  xlock();
  XMapRaised(dpy, win);
  xunlock();
  XFlush(dpy);
}

int DpyBase::close_display() {
  if (!opened) return 1;
  else opened = false;

  xlock();

  // This will make sure that we will not be rendering anything while
  // we try to close the windows.
  if (have_ogl_context)
    if (!glXMakeCurrent(dpy,None, NULL)) {
      cerr << "DpyBase::close_display()::glXMakeCurrent failed\n";
    }
  
  if (close_display_flag) {
    
    //    cerr << "Closing dpy:"<<window_name<<"\n";
    XCloseDisplay(dpy);
    cerr << "Closed dpy:"<<window_name<<"\n";
    
  }
  xunlock();
  
  return 0;
}

void DpyBase::dont_close() {
  cerr << window_name << ": will not be closed\n";
  close_display_flag = false;
}

void DpyBase::cleanup() {
  if (cleaned) return;
  else cleaned = true;
  
  close_display();
}

void DpyBase::stop() {
  on_death_row = true;
}

void DpyBase::init() {
  glShadeModel(GL_FLAT);
}

void DpyBase::display() {
  glFinish();
  if ((window_mode & BufferModeMask) == DoubleBuffered){
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

void DpyBase::set_resolution(const int width, const int height) {
  if (width > 0 && height > 0) {
    xres = width;
    yres = height;
  }
}

bool DpyBase::should_close() {
  return on_death_row ||
    (scene
     && scene->get_rtrt_engine()
     && scene->get_rtrt_engine()->exit_scene);
}

extern bool pin;
void DpyBase::run() {

  if(pin)
    Thread::self()->migrate(127);

  open_display();

  init();
  
  for(;;){
    // Now we need to test to see if we should die
    if (should_close()) {
      cleanup();
      return;
    }
    
    if(redraw){
      display();
      redraw=false;
    }

    if (!opened)
      // Most likely going to be stopped soon
      return;

    wait_and_handle_events();
  } // end of for(;;)
}

void DpyBase::wait_and_handle_events() {

  // I've decided that I don't like this function blocking, because it
  // requires an X event to trigger out of it.  The previous way this
  // was implemented caused the displays to pound the CPU as it was in
  // an invinate loop checking for events.  This way, if there are no
  // events, the function waits for a *user indiscernible* amount of
  // time, so the function doesn't busy spin.

  if (!XPending(dpy)) {
    usleep(1000);
    // The question arises of whether or not to return here.  I'm
    // going to try this out for now in the hopes that it will work
    // fine.
  }

  // By this time, there should be some kind of an event.  We should
  // try to consume all the queued events before we redraw.  That
  // way we don't waste time redrawing after each event.
  //  while (XEventsQueued(dpy, QueuedAfterReading)) {
  while (XPending(dpy)) {
    //    cerr << window_name << ": " << XPending(dpy) << " events left to process\n";
    // Now we need to test to see if we should die
    if (should_close()) {
      // Don't call cleanup here.  It will be called from the main run
      // function.  This is just to prevent additional event
      // processing when the window will be closed.
      return;
    }

    // Get the next event.  We know there should be one, otherwise
    // we wouldn't have gotten passed the while conditional.
    XEvent e;
    XNextEvent(dpy, &e);	
    switch(e.type){
    case Expose:
      //      cerr << window_name << ": "<< "Expose event found.  " << XPending(dpy) << " events left to process\n";
      redraw=true;
      break;
    case ConfigureNotify:
      //      cerr << window_name << ": "<< "ConfigureNotify event found.  " << XPending(dpy) << " events left to process\n";
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
}

void DpyBase::xlock() {
  xmutex.lock();
}

void DpyBase::xunlock() {
  xmutex.unlock();
}

void DpyBase::initUseXThreads() {
  if (XInitThreads()) {
    cerr << "Enabling XThreads\n";
    useXThreads = true;
  }
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
