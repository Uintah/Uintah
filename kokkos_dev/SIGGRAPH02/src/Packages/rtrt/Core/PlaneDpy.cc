
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/PlaneDpy.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <GL/glx.h>
#include <GL/glu.h>
#include <Packages/rtrt/visinfo/visinfo.h>
#include <iostream>
#include <stdlib.h>
#include <values.h>
#include <stdio.h>
#include "FontString.h"

using namespace rtrt;
using namespace SCIRun;

PlaneDpy::PlaneDpy(const Vector& n, const Point& pt)
    : DpyBase("PlaneDpy"), n(n)
{
    d=Dot(n, pt);
    xres=300;
    yres=300;
}
PlaneDpy::PlaneDpy(const Vector& n, const double d)
  : DpyBase("PlaneDpy"), n(n), d(d)
{
    xres=300;
    yres=300;
}

PlaneDpy::~PlaneDpy()
{
}

void PlaneDpy::init() {
  glShadeModel(GL_FLAT);
}

void PlaneDpy::display()
{
  if(redraw) {
    int textheight=fontInfo->descent+fontInfo->ascent;

    glViewport(0, 0, xres, yres);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-1, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glColor3f(0,0,1);
    glBegin(GL_LINES);
    glVertex2f(0, -1);
    glVertex2f(0, 1);
    glEnd();
    for(int i=0;i<4;i++) {
      int s=i*yres/4;
      int e=(i+1)*yres/4;
      glViewport(0, s, xres, e-s);
      double th=double(textheight+1)/(e-s);
      double v;
      double wid=2;
      char* name;
      switch(i){
      case 3: v=n.x(); name="X"; break;
      case 2: v=n.y(); name="Y"; break;
      case 1: v=n.z(); name="Z"; break;
      case 0: v=d; name="D"; wid=20; break;
      }
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      if(i==0)
	gluOrtho2D(-10, 10, 0, 1);
      else
	gluOrtho2D(-1, 1, 0, 1);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      glColor3f(1,1,1);
      glBegin(GL_LINES);
      glVertex2f(v, th);
      glVertex2f(v, 1);
      glEnd();
      char buf[100];
      sprintf(buf, "%s: %g", name, v);
      int w=calc_width(fontInfo, buf);
      printString(fontbase, v-w/wid/yres, 1./yres, buf, Color(1,1,1));
    }
    redraw=false;
  }
  glFinish();
    
  if (window_mode & BufferModeMask == DoubleBuffered)
    glXSwapBuffers(dpy, win);
  
  XFlush(dpy);
}

void PlaneDpy::resize(const int width, const int height) {
  yres=height;
  if (width != xres) {
    xres=width;
    redraw=true;
  }
}

void PlaneDpy::button_released(MouseButton button, const int x, const int y){
  if (button==Button1) {
    starty = y;
    move(x, y);
    redraw=true;
  }	
}    

void PlaneDpy::button_motion(MouseButton button, const int x, const int /*y*/) {
  if (button == Button1) {
    move(x, starty);
    redraw=true;
  }
}

void PlaneDpy::move(int x, int y)
{
    float xn=float(x)/xres;
    float yn=float(y)/yres;
    if(yn>.75){
	d=xn*20-10;
    } else if(yn>.5){
	n.z(xn*2-1);
    } else if(yn>.25){
	n.y(xn*2-1);
    } else {
	// X...
	n.x(xn*2-1);
    }
}







