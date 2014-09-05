
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

Hist2DDpy::Hist2DDpy(): DpyBase("Hist2DDpy", SingleBuffered),
			need_hist(true), redraw_isoval(false)
{
  xres=300;
  yres=300;
  hist=0;
  have_line=false;
  have_p=false;
  clip=new_clip=false;
  use_perp = new_use_perp = true;
}

Hist2DDpy::Hist2DDpy(float a, float b, float c):
  DpyBase("Hist2DDpy", SingleBuffered),
  need_hist(true), redraw_isoval(false)
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
  use_perp = new_use_perp = true;
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
  set_clip(px0, py0, px1, py1);
}

void Hist2DDpy::set_lines()
{
  float dx=px1-px0;
  float dy=py1-py0;
  new_isoline.a=-dy;
  new_isoline.b=dx;
  new_isoline.c=-(new_isoline.a*px0+new_isoline.b*py0);

  if (new_use_perp) {
    float dx = (gdatamax-gdatamin)/xres;
    float dy = (vdatamax-vdatamin)/yres;
    float ratio = dx*dx/dy/dy;
    compute_perp(new_isoline, new_perp_line, px1, py1, ratio);
    // Should use the perp's endpoints
    set_clip(px0, py0, px1, py1);
  } else {
    set_clip(px0, py0, px1, py1);
  }
}

void Hist2DDpy::set_clip(float x0, float y0, float x1, float y1)
{
  float dx=x1-x0;
  float dy=y1-y0;
  float l2=dx*dx+dy*dy;
  new_clipline.a=dx/l2;
  new_clipline.b=dy/l2;
  new_clipline.c=-(new_clipline.a*x0+new_clipline.b*y0);
}

void Hist2DDpy::attach(VolumeVGBase* vol)
{
  vols.add(vol);
}

void Hist2DDpy::init() {
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

  // Ok, figure out where the center is.  These values never change.
  cx = vdatamin+0.5f*(vdatamax-vdatamin);
  cy = gdatamin+0.5f*(gdatamax-gdatamin);
  if (vdatamin <=  147 && 147 <= vdatamax )
    cx = 147;
  if (gdatamin <=  131 && 131 <= vdatamax )
    cy = 131;

  if (use_perp) {
    px0 = cx;
    py0 = cy;
  }

  set_lines();
  
  glShadeModel(GL_FLAT);
}

// Computes the line perpendicular to l1 at (x,y) and stores it in l2.
// It turns out that this is the nice general approach, but we need to have
// some kind of ration of x to y, since that ratio will not always be one
void Hist2DDpy::compute_perp(ImplicitLine &l1, ImplicitLine &l2,
			     const float x, const float y, const float ratio) {
  l2.a = -l1.b * ratio;
  l2.b =  l1.a;
  l2.c = -(l2.a * x + l2.b * y);
  //  cout << "l1 = ["<<l1.a<<", "<<l1.b<<", "<<l1.c<<"]\n";
  //  cout << "l2 = ["<<l2.a<<", "<<l2.b<<", "<<l2.c<<"]\n";
  //  cout << "x = "<<x<<", y = "<<y<<endl;
}

#if 0
void Hist2DDpy::run()
{
  open_display();
  
  init();
  
  for(;;){
    if(redraw){
      display();
      redraw=false;
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
      case KeyPress:
	key_pressed(XKeycodeToKeysym(dpy, e.xkey.keycode, 0));
	break;
      default:
	cerr << "Unknown event, type=" << e.type << '\n';
      } // event switch
    } // while there are events
  } // for(;;)
}
#endif

void Hist2DDpy::display() {
  if(need_hist){
    need_hist=false;
    compute_hist(fontbase);
    redraw_isoval = false;
  }
  draw_hist(fontbase, fontInfo, redraw_isoval);
  redraw_isoval=false;
  redraw=false;
}

void Hist2DDpy::resize(const int width, const int height) {
#if 0
  if (xres != width || yres != height) {
    xres = width;
    yres = height;
    redraw = true;
    redraw_isoval = true;
    need_hist = true;
  }
#else
  // We want to prevent resizing for now, because the histogram creation is
  // really expensive to do over again.
  XResizeWindow(dpy, win, xres, yres);  
#endif
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

  if (new_use_perp || !new_clip)
    draw_isoline(new_isoline, px0, py0, px1, py1);
  else
    draw_isoline(new_isoline, vdatamin, gdatamin, vdatamax, gdatamax);

  if (new_use_perp)
    draw_isoline(new_perp_line, vdatamin, gdatamin, vdatamax, gdatamax);
  
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

void Hist2DDpy::draw_isoline(ImplicitLine &line,
			     float xmin, float ymin,
			     float xmax, float ymax) {
  float x0, y0, x1, y1;
  float a=line.a;
  float b=line.b;
  float c=line.c;
  //  cout << "a = "<<a<<", b = "<<b<<", c = "<<c<<endl;
  if(Abs(a) > Abs(b)){
    //    cout << "Abs(a) > Abs(b)\n";
    // Find y (g) intersections
    x0=-(b*ymin+c)/a;
    y0=ymin;
    x1=-(b*ymax+c)/a;
    y1=ymax;
  } else {
    //    cout << "Abs(a) <= Abs(b)\n";
    // Find x (v) intersections
    y0=-(a*xmin+c)/b;
    x0=xmin;
    y1=-(a*xmax+c)/b;
    x1=xmax;
  }
  //  cout << "min("<<x0<<", "<<y0<<"), max("<<x1<<", "<<y1<<")\n";
  
  glBegin(GL_LINES);
  glVertex2f(x0, y0);
  glVertex2f(x1, y1);
  glEnd();
}

void Hist2DDpy::key_pressed(unsigned long key) {
  switch(key) {
  case XK_space:
    new_clip=!new_clip;
    break;
  case XK_P:
  case XK_p:
    new_use_perp = !new_use_perp;
    if (new_use_perp) {
      px0 = cx;
      py0 = cy;
      set_lines();
      redraw_isoval=true;
      redraw = true;
    }
    break;
  }
}

void Hist2DDpy::button_pressed(MouseButton button, const int x, const int y) {
  float xx0=(px0-vdatamin)*xres/(vdatamax-vdatamin);
  float xx1=(px1-vdatamin)*xres/(vdatamax-vdatamin);
  float yy0=(py0-gdatamin)*yres/(gdatamax-gdatamin);
  float yy1=(py1-gdatamin)*yres/(gdatamax-gdatamin);
  
  float dist0=Abs(xx0-x)+Abs(yy0-yres+y);
  float dist1=Abs(xx1-x)+Abs(yy1-yres+y);

  if(dist0 < dist1)
    whichp=0;
  else
    whichp=1;
  if (new_use_perp)
    whichp=1;
  redraw_isoval=true;
  redraw = true;
}

void Hist2DDpy::button_released(MouseButton button, const int x, const int y) {
  redraw_isoval=true;
  redraw = true;
}

void Hist2DDpy::button_motion(MouseButton button, const int x, const int y) {
  float xn=float(x)/xres;
  float xval=vdatamin+xn*(vdatamax-vdatamin);
  float yn=float(yres-y)/yres;
  float yval=gdatamin+yn*(gdatamax-gdatamin);
  if(whichp == 0){
    px0=xval;
    py0=yval;
  } else {
    px1=xval;
    py1=yval;
  }
  set_lines();
  redraw_isoval=true;
  redraw = true;
}

void Hist2DDpy::animate(bool& changed)
{
  if (new_use_perp) {
    // Ok, this is a hack and should be fixed once this is verified
    if(isoline != new_perp_line ||
       clip != new_clip ||
       clipline != new_clipline ||
       use_perp != new_use_perp)
      {
	isoline=new_perp_line;
	clip=new_clip;
	clipline=new_clipline;
	use_perp = new_use_perp;
	changed=true;
      }
  } else {
    if(isoline != new_isoline ||
       clip != new_clip ||
       clipline != new_clipline ||
       use_perp != new_use_perp)
      {
	isoline=new_isoline;
	clip=new_clip;
	clipline=new_clipline;
	use_perp = new_use_perp;
	changed=true;
      }
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
