#include <Core/Thread/Time.h>
#include <Core/Thread/Mutex.h>
#include <Packages/rtrt/Core/Color.h>
#include <GL/glx.h>
#include <GL/glu.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <unistd.h>
#include <Packages/rtrt/visinfo/visinfo.h>
//#include <sys/time.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "FontString.h"

using namespace std;
using namespace rtrt;
using namespace SCIRun;

namespace rtrt {
  extern Mutex xlock;
} // end namespace rtrt

static void printString(GLuint fontbase, double x, double y,
			char *s, const Color& c)
{
  glColor3f(c.red(), c.green(), c.blue());

  glRasterPos2d(x,y);
  /*glBitmap(0, 0, x, y, 1, 1, 0);*/
  glPushAttrib (GL_LIST_BIT);
  glListBase(fontbase);
  glCallLists((int)strlen(s), GL_UNSIGNED_BYTE, (GLubyte *)s);
  glPopAttrib ();
}

void run_gl_test() {
  Display *dpy;
  int xres = 320;
  int yres = 320;
  char* criteria1="db, stereo, max rgb, max accumrgb";
  char* criteria2="db, max rgb, max accumrgb";
  // Open an OpenGL window
  xlock.lock();
  dpy=XOpenDisplay(NULL);
  if(!dpy){
    cerr << "Cannot open display\n";
    exit(1);
  }
  int error, event;
  if ( !glXQueryExtension( dpy, &error, &event) ) {
    cerr << "GL extension NOT available!\n";
    XCloseDisplay(dpy);
    dpy=0;
    exit(1);
  }
  int screen=DefaultScreen(dpy);

  if(!visPixelFormat(criteria1)){
    cerr << "Error setting pixel format for visinfo\n";
    cerr << "Syntax error in criteria: " << criteria1 << '\n';
    exit(1);
  }
  int nvinfo;
  XVisualInfo* vi=visGetGLXVisualInfo(dpy, screen, &nvinfo);
  if(!vi || nvinfo == 0){
    if(!visPixelFormat(criteria2)){
      cerr << "Error setting pixel format for visinfo\n";
      cerr << "Syntax error in criteria: " << criteria2 << '\n';
      exit(1);
    }
    vi=visGetGLXVisualInfo(dpy, screen, &nvinfo);
    if(!vi || nvinfo == 0){
      cerr << "Error matching OpenGL Visual: " << criteria1 << " and " << criteria2 << '\n';
      exit(1);
    }
  }
  Colormap cmap = XCreateColormap(dpy, RootWindow(dpy, screen),
				  vi->visual, AllocNone);
  XSetWindowAttributes atts;
  int flags=CWColormap|CWEventMask|CWBackPixmap|CWBorderPixel;
  atts.background_pixmap = None;
  atts.border_pixmap = None;
  atts.border_pixel = 0;
  atts.colormap=cmap;
  atts.event_mask=StructureNotifyMask|ExposureMask|ButtonPressMask|ButtonReleaseMask|ButtonMotionMask|KeyPressMask;
  Window win=XCreateWindow(dpy, RootWindow(dpy, screen),
			   0, 0, xres, yres, 0, vi->depth,
			   InputOutput, vi->visual, flags, &atts);
  char* p="real time ray tracer";
  XTextProperty tp;
  XStringListToTextProperty(&p, 1, &tp);
  XSizeHints sh;
  sh.flags = USSize;
  XSetWMProperties(dpy, win, &tp, &tp, 0, 0, &sh, 0, 0);

  XMapWindow(dpy, win);

  GLXContext cx=glXCreateContext(dpy, vi, NULL, True);
  if(!glXMakeCurrent(dpy, win, cx)){
    cerr << "glXMakeCurrent failed!\n";
  }

  XFontStruct* fontInfo = XLoadQueryFont(dpy, __FONTSTRING__);
  
  if (fontInfo == NULL) {
    cerr << "no font found " << __FILE__ << "," << __LINE__ << std::endl;
  }

  Font id = fontInfo->fid;
  unsigned int first = fontInfo->min_char_or_byte2;
  unsigned int last = fontInfo->max_char_or_byte2;

  GLuint fontbase = glGenLists((GLuint) last+1);
  if (fontbase == 0) {
    printf ("out of display lists\n");
    exit (0);
  }
  glXUseXFont(id, first, last-first+1, fontbase+first);


  XFontStruct* fontInfo2 = XLoadQueryFont(dpy, __FONTSTRING__);

  if (fontInfo2 == NULL) {
    cerr << "no font found(2)\n";
    exit(1);
  }


  id = fontInfo2->fid;
  first = fontInfo2->min_char_or_byte2;
  last = fontInfo2->max_char_or_byte2;

  GLuint fontbase2 = glGenLists((GLuint) last+1);
  if (fontbase2 == 0) {
    printf ("out of display lists\n");
    exit (0);
  }
  glXUseXFont(id, first, last-first+1, fontbase2+first);

  glShadeModel(GL_FLAT);
  glReadBuffer(GL_BACK);

  // Window Created
  cerr << "Created Window\n";
  xlock.unlock();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, xres, 0, yres);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0.375, 0.375, 0.0);
  
  double framerate;
  double past=SCIRun::Time::currentSeconds();
  //  for (int i = 0; i < 20000; i++) {
  for (;;) {
    glClearColor(1,0,1, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    // draw stuff
    glColor3f(1,1,0);
    glBegin(GL_POLYGON);
    glVertex2f(50,50);
    glVertex2f(200,50);
    glVertex2f(200,200);
    glVertex2f(50,200);
    glEnd();
    
    double current=SCIRun::Time::currentSeconds();
    framerate = 1./ (current - past);
    //cerr << "dt1 = " << (current - past) << ",\tcurrent = " << current << ",\tpast = " << past << endl;
    past = current;
    char buf[100];
    sprintf(buf, "%3.1ffps", framerate);
    printString(fontbase, 10, 3, buf, Color(1,1,1));
    
    glFinish();
    glXSwapBuffers(dpy, win);
    XFlush(dpy);

    glClearColor(1,0,1, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    // draw stuff
    glColor3f(0,1,1);
    glBegin(GL_POLYGON);
    glVertex2f(100,100);
    glVertex2f(300,100);
    glVertex2f(300,300);
    glVertex2f(100,300);
    glEnd();
    
    //    for (int i = 0; i < 1e8; i++);
    current=SCIRun::Time::currentSeconds();
    framerate = 1./ (current - past);
    //cerr << "dt2 = " << (current - past) << ",\tcurrent = " << current << ",\tpast = " << past << endl;
    past = current;
    char buf2[100];
    sprintf(buf2, "%3.1ffps", framerate);
    printString(fontbase, 10, 3, buf2, Color(1,1,1));
    
    glFinish();
    glXSwapBuffers(dpy, win);
    XFlush(dpy);
  }    
}
