
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Hist2DDpy.h>
#include <Packages/rtrt/Core/VolumeVGBase.h>
#include <Packages/rtrt/Core/MinMax.h>
#include <Packages/rtrt/Core/MiscMath.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <X11/keysym.h>
#include <GL/glx.h>
#include <GL/glu.h>
#include <Packages/rtrt/visinfo/visinfo.h>
#include <iostream>
#include <stdlib.h>
#include <values.h>
#include <limits.h>

using namespace rtrt;
using SCIRun::Mutex;
using SCIRun::Thread;
using namespace std;

namespace rtrt {
  extern Mutex xlock;
} // end namespace rtrt

Hist2DDpy::Hist2DDpy()
{
  xres=300;
  yres=300;
  hist=0;
  have_line=false;
  have_p=false;
  clip=new_clip=false;
}

Hist2DDpy::Hist2DDpy(float a, float b, float c)
{
  xres=500;
  yres=500;
  hist=0;
  have_line=true;
  isoline.a=a;
  isoline.b=b;
  isoline.c=c;
  have_p=false;
  clip=new_clip=false;
}

Hist2DDpy::~Hist2DDpy()
{
}

void Hist2DDpy::set_p()
{
  if(Abs(new_isoline.a) > Abs(new_isoline.b)){
    // Find y (g) intersections
    float a=new_isoline.a;
    float b=new_isoline.b;
    float c=new_isoline.c;
    px0=-(b*gdatamin+c)/a;
    py0=gdatamin;
    px1=-(b*gdatamax+c)/a;
    py1=gdatamax;
  } else {
    // Find x (v) intersections
    float a=new_isoline.a;
    float b=new_isoline.b;
    float c=new_isoline.c;
    py0=-(a*vdatamin+c)/b;
    px0=vdatamin;
    py1=-(a*vdatamax+c)/b;
    px1=vdatamax;
  }
  set_clip();
}

void Hist2DDpy::set_line()
{
  float dx=px1-px0;
  float dy=py1-py0;
  new_isoline.a=-dy;
  new_isoline.b=dx;
  new_isoline.c=-(new_isoline.a*px0+new_isoline.b*py0);
  set_clip();
}

void Hist2DDpy::set_clip()
{
  float dx=px1-px0;
  float dy=py1-py0;
  float l2=dx*dx+dy*dy;
  new_clipline.a=dx/l2;
  new_clipline.b=dy/l2;
  new_clipline.c=-(new_clipline.a*px0+new_clipline.b*py0);
}

void Hist2DDpy::attach(VolumeVGBase* vol)
{
  vols.add(vol);
}

void Hist2DDpy::run()
{
  // Compute the global minmax
  if(vols.size()==0)
    exit(0);
  vdatamax=-MAXFLOAT;
  vdatamin=MAXFLOAT;
  gdatamax=-MAXFLOAT;
  gdatamin=MAXFLOAT;
  for(int i=0;i<vols.size();i++){
    float vmin, vmax, gmin, gmax;
    vols[i]->get_minmax(vmin, vmax, gmin, gmax);
    vdatamin=Min(vmin, vdatamin);
    vdatamax=Max(vmax, vdatamax);
    gdatamin=Min(gmin, gdatamin);
    gdatamax=Max(gmax, gdatamax);
  }
  if(!have_line){
    have_line=true;
    isoline.c=-(vdatamin+vdatamax)*0.5;
    isoline.b=0;
    isoline.a=1;
  }
  new_isoline=isoline;
  if(!have_p)
    set_p();

  xlock.lock();
  // Open an OpenGL window
  Display* dpy=XOpenDisplay(NULL);
  if(!dpy){
    cerr << "Cannot open display\n";
    Thread::exitAll(1);
  }
  int error, event;
  if ( !glXQueryExtension( dpy, &error, &event) ) {
    cerr << "GL extension NOT available!\n";
    XCloseDisplay(dpy);
    dpy=0;
    Thread::exitAll(1);
  }
  int screen=DefaultScreen(dpy);

  char* criteria="sb, max rgb";
  if(!visPixelFormat(criteria)){
    cerr << "Error setting pixel format for visinfo\n";
    cerr << "Syntax error in criteria: " << criteria << '\n';
    Thread::exitAll(1);
  }
  int nvinfo;
  XVisualInfo* vi=visGetGLXVisualInfo(dpy, screen, &nvinfo);
  if(!vi || nvinfo == 0){
    cerr << "Error matching OpenGL Visual: " << criteria << '\n';
    Thread::exitAll(1);
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
  char* p="Volume histogram";
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
  glShadeModel(GL_FLAT);
  for(;;){
    XEvent e;
    XNextEvent(dpy, &e);
    if(e.type == MapNotify)
      break;
  }
  XFontStruct* fontInfo = XLoadQueryFont(dpy, 
					 "-adobe-helvetica-bold-r-normal--17-120-100-100-p-88-iso8859-1");
  if (fontInfo == NULL) {
    cerr << "no font found\n";
    Thread::exitAll(1);
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
    

  bool need_hist=true;
  bool redraw=true;
  bool redraw_isoval=false;
  xlock.unlock();
  for(;;){
    if(need_hist){
      need_hist=false;
      compute_hist(fontbase);
      redraw=true;
    }
    if(redraw || redraw_isoval){
      if(redraw)
	redraw_isoval=false;
      draw_hist(fontbase, fontInfo, redraw_isoval);
      redraw=false;
      redraw_isoval=false;
    }
    // We should try to consume all the queued events before we redraw.
    // That way we don't waste time redrawing after each event
    while (XEventsQueued(dpy, QueuedAfterReading)) {
      XEvent e;
      XNextEvent(dpy, &e);	
      switch(e.type){
      case Expose:
	// Ignore expose events, since we will be refreshing
	// constantly anyway
	redraw=true;
	break;
      case ConfigureNotify:
	yres=e.xconfigure.height;
	if(e.xconfigure.width != xres){
	  xres=e.xconfigure.width;
	  need_hist=true;
	} else {
	  redraw=true;
	}
	break;
      case ButtonPress:
	switch(e.xbutton.button){
	case Button1:
	  move(e.xbutton.x, e.xbutton.y, Press);
	  redraw_isoval=true;
	  break;
	case Button2:
	  break;
	case Button3:
	  break;
	}
	break;
      case ButtonRelease:
	switch(e.xbutton.button){
	case Button1:
	  move(e.xbutton.x, e.xbutton.y, Release);
	  redraw_isoval=true;
	  break;
	case Button2:
	  break;
	case Button3:
	  break;
	}
	break;
      case MotionNotify:
	switch(e.xmotion.state&(Button1Mask|Button2Mask|Button3Mask)){
	case Button1Mask:
	  move(e.xbutton.x, e.xbutton.y, Motion);
	  redraw_isoval=true;
	  break;
	case Button2Mask:
	  break;
	case Button3Mask:
	  break;
	}
	break;
      case KeyPress:
	switch(XKeycodeToKeysym(dpy, e.xkey.keycode, 0)){
	case XK_space:
	  new_clip=!new_clip;
	  break;
	}
	break;
      default:
	cerr << "Unknown event, type=" << e.type << '\n';
      } // event switch
    } // while there are events
  } // for(;;)
}

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

void Hist2DDpy::compute_hist(unsigned int fid)
{
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  glViewport(0, 0, xres, yres);
  glClearColor(0, 0, .2, 1);
  glClear(GL_COLOR_BUFFER_BIT);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, 1, 0, 1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
	    
  printString(fid, .1, .5, "Recomputing histogram...\n", Color(1,1,1));
  glFlush();
  if(hist){
    //delete[] hist[0];
    //delete[] hist;
  }
  int nvhist=xres;
  int nghist=yres;
  hist=new int*[nghist];
  int** tmphist=new int*[nghist];
  int* p1=new int[nghist*nvhist];
  int* p2=new int[nghist*nvhist];
  for(int g=0;g<nghist;g++){
    hist[g]=p1;
    tmphist[g]=p2;
    for(int v=0;v<nvhist;v++){
      p1[v]=0;
    }
    p1+=nghist;
    p2+=nvhist;
  }

  for(int i=0;i<vols.size();i++){
    for(int g=0;g<nghist;g++){
      for(int v=0;v<nvhist;v++){
	tmphist[g][v]=0;
      }
    }
    vols[i]->compute_hist(nvhist, nghist, tmphist, vdatamin, vdatamax,
			  gdatamin, gdatamax);
    for(int g=0;g<nghist;g++){
      for(int v=0;v<nvhist;v++){
	hist[g][v]+=tmphist[g][v];
      }
    }
  }
  //delete[] tmphist[0];
  //delete tmphist;

  int* hp=hist[0];
  int max=0;
  int thist=nvhist*nghist;
  for(int i=0;i<thist;i++){
    if(*hp>max)
      max=*hp;
    hp++;
  }
  histmax=max;
  hp=hist[0];
  double scale=(double)INT_MAX/(double)max;
  scale*=1000;
  for(int i=0;i<thist;i++){
    double s=*hp*scale;
    if(s>INT_MAX)
      s=INT_MAX;
    *hp=(int)(s);
    hp++;
  }
  cerr << "Done building histogram: max=" << max << "\n";
}
    
static int calc_width(XFontStruct* font_struct, char* str)
{
  XCharStruct overall;
  int ascent, descent;
  int dir;
  XTextExtents(font_struct, str, (int)strlen(str), &dir, &ascent, &descent, &overall);
  return overall.width;
}

void Hist2DDpy::draw_hist(unsigned int fid, XFontStruct* font_struct,
			  bool redraw_isoval)
{
  int descent=font_struct->descent;
  int textheight=font_struct->descent+font_struct->ascent;
  if(!redraw_isoval){
    glColorMask(GL_TRUE, GL_TRUE, GL_FALSE, GL_TRUE);
    glViewport(0, 0, xres, yres);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    int nvhist=xres;
    int nghist=yres;
    glViewport(0, 0, xres, yres);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(vdatamin, vdatamax, gdatamin, gdatamax);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glRasterPos2d(vdatamin, gdatamin);
    glDrawPixels(nvhist, nghist, GL_LUMINANCE, GL_INT, hist[0]);

    /*
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluOrtho2D(0, xres, 0, h);
      char buf[100];
      sprintf(buf, "%g", datamin);
      printString(fid, 2, descent+1, buf, Color(0,1,1));
      sprintf(buf, "%g", datamax);
      int w=calc_width(font_struct, buf);
      printString(fid, xres-2-w, descent+1, buf, Color(0,1,1));
    */
  }
	
  float x0, y0, x1, y1;
  if(new_clip){
    if(Abs(new_isoline.a) > Abs(new_isoline.b)){
      // Find y (g) intersections
      float a=new_isoline.a;
      float b=new_isoline.b;
      float c=new_isoline.c;
      x0=-(b*gdatamin+c)/a;
      y0=gdatamin;
      x1=-(b*gdatamax+c)/a;
      y1=gdatamax;
    } else {
      // Find x (v) intersections
      float a=new_isoline.a;
      float b=new_isoline.b;
      float c=new_isoline.c;
      y0=-(a*vdatamin+c)/b;
      x0=vdatamin;
      y1=-(a*vdatamax+c)/b;
      x1=vdatamax;
    }
  } else {
    x0=px0;
    x1=px1;
    y0=py0;
    y1=py1;
  }

    
  glViewport(0, 0, xres, yres);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(vdatamin, vdatamax, gdatamin, gdatamax);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glColorMask(GL_FALSE, GL_FALSE, GL_TRUE, GL_TRUE);
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT);

  glColor3f(0,0,.8);
  glBegin(GL_LINES);
  glVertex2f(x0, y0);
  glVertex2f(x1, y1);
  glEnd();


  glPointSize(8.0);
  glBegin(GL_POINTS);
  glVertex2f(px0, py0);
  glVertex2f(px1, py1);
  glEnd();

  glFinish();
  int errcode;
  while((errcode=glGetError()) != GL_NO_ERROR){
    cerr << "We got an error from GL: " << (char*)gluErrorString(errcode) << endl;
  }
}

void Hist2DDpy::move(int x, int y, BTN what)
{
  y=yres-y;
  if(what == Release)
    return;

  if(what == Press){
    float xx0=(px0-vdatamin)*xres/(vdatamax-vdatamin);
    float xx1=(px1-vdatamin)*xres/(vdatamax-vdatamin);
    float yy0=(py0-gdatamin)*yres/(gdatamax-gdatamin);
    float yy1=(py1-gdatamin)*yres/(gdatamax-gdatamin);

    float dist0=Abs(xx0-x)+Abs(yy0-y);
    float dist1=Abs(xx1-x)+Abs(yy1-y);

    if(dist0 < dist1)
      whichp=0;
    else
      whichp=1;
  } else {
    float xn=float(x)/xres;
    float xval=vdatamin+xn*(vdatamax-vdatamin);
    float yn=float(y)/yres;
    float yval=gdatamin+yn*(gdatamax-gdatamin);
    if(whichp == 0){
      px0=xval;
      py0=yval;
    } else {
      px1=xval;
      py1=yval;
    }
    set_line();
  }
}

void Hist2DDpy::animate(bool& changed)
{
  if(isoline != new_isoline || clip != new_clip || new_clipline != clipline){
    isoline=new_isoline;
    clip=new_clip;
    clipline=new_clipline;
    changed=true;
  }
}

bool ImplicitLine::operator != (const ImplicitLine& x) const
{
  return a != x.a || b != x.b || c != x.c;
}

float ImplicitLine::operator()(float x, float y) const
{
  return a*x+b*y+c;
}
