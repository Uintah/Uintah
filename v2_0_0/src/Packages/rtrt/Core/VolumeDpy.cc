
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/VolumeDpy.h>
#include <Packages/rtrt/Core/VolumeBase.h>
#include <Packages/rtrt/Core/DpyBase.h>
#include <Packages/rtrt/visinfo/visinfo.h>
#include <Packages/rtrt/Core/FontString.h>
#include <Core/Math/MinMax.h>

#include <GL/glx.h>
#include <GL/glu.h>

#include <iostream>

#include <stdlib.h>
#include <values.h>
#include <stdio.h>
#include <unistd.h>

using namespace rtrt;
using namespace std;
using namespace SCIRun;

VolumeDpy::VolumeDpy(float isoval)
    : DpyBase("VolumeDpy"), need_hist(true), redraw_isoval(false),
      isoval(isoval), new_isoval(isoval)
{
  set_resolution(400,100);
  hist=0;
}

VolumeDpy::~VolumeDpy()
{
}

void VolumeDpy::attach(VolumeBase* vol)
{
    vols.add(vol);
}

void VolumeDpy::init() {
  sleep(3);

  // Compute the global minmax
  if(vols.size()==0)
    exit(0);
  datamax=-MAXFLOAT;
  datamin=MAXFLOAT;
  for(int i=0;i<vols.size();i++){
    float min, max;
    vols[i]->get_minmax(min, max);
    datamin=Min(min, datamin);
    datamax=Max(max, datamax);
  }
  if(isoval == -123456){
    isoval=(datamin+datamax)*0.5;
  }
  new_isoval=isoval;

  glShadeModel(GL_FLAT);
}

void VolumeDpy::display() {
  if(need_hist){
    cout << "Recomputing histogram\n";
    need_hist=false;
    compute_hist();
    redraw_isoval = true;
  }
  //  draw_hist(redraw_isoval);
  draw_hist(false);
  redraw_isoval=false;

  glFinish();
  if (window_mode & BufferModeMask == DoubleBuffered)
    glXSwapBuffers(dpy, win);
  XFlush(dpy);
}

void VolumeDpy::resize(const int width, const int height) {
  yres = height;
  if(width != xres){
    xres = width;
    need_hist = true;
  }
  redraw = true;
}

void VolumeDpy::button_released(MouseButton button,
			      const int x, const int /*y*/) {
  if (button == MouseButton1) {
    move_isoval(x);
    redraw = redraw_isoval=true;
  }
}

void VolumeDpy::button_motion(MouseButton button,
			      const int x, const int /*y*/) {
  if (button == MouseButton1) {
    move_isoval(x);
    redraw = redraw_isoval=true;
  }
}

void VolumeDpy::compute_hist()
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
	    
  printString(fontbase, .1, .5, "Recomputing histogram...\n", Color(1,1,1));
  glFlush();
  if(hist){
    delete[] hist;
  }
  int nhist=xres;
  hist=new int[nhist];
  for(int i=0;i<nhist;i++){
    hist[i]=0;
  }
  int* tmphist=new int[nhist];

  for(int i=0;i<vols.size();i++){
    for(int j=0;j<nhist;j++){
      tmphist[j]=0;
    }
    vols[i]->compute_hist(nhist, tmphist, datamin, datamax);
    for(int j=0;j<nhist;j++){
      hist[j]+=tmphist[j];
    }
  }
  delete[] tmphist;
  int* hp=hist;
  int max=0;
  for(int i=0;i<nhist;i++){
    if(*hp>max)
      max=*hp;
    hp++;
  }
  histmax=max;
  cerr << "Done building histogram\n";
}
    
void VolumeDpy::draw_hist(bool redraw_isoval)
{
  int descent=fontInfo->descent;
  int textheight=fontInfo->descent+fontInfo->ascent;
  if(!redraw_isoval){
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glViewport(0, 0, xres, yres);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    int nhist=xres;
    int s=2;
    int e=yres-2;
    int h=e-s;
    glViewport(0, s, xres, e-s);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, xres, -float(textheight)*histmax/(h-textheight), histmax);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glColor3f(0,0,1);
    glBegin(GL_LINES);
    for(int i=0;i<nhist;i++){
      glVertex2i(i, 0);
      glVertex2i(i, hist[i]);
    }
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, xres, 0, h);
    char buf[100];
    sprintf(buf, "%g", datamin);
    printString(fontbase, 2, descent+1, buf, Color(0,1,1));
    sprintf(buf, "%g", datamax);
    int w=calc_width(fontInfo, buf);
    printString(fontbase, xres-2-w, descent+1, buf, Color(0,1,1));
  }

  glColorMask(GL_TRUE, GL_TRUE, GL_FALSE, GL_FALSE);
  int s=2;
  int e=yres-2;
  int h=e-s;
  glViewport(0, s, xres, e-s);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(datamin, datamax, 0, h);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glColor3f(0,0,0);
  glRectf(datamin, 0, datamax, h);
  glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE);
  glColor3f(.8,0,0);
  glBegin(GL_LINES);
  glVertex2f(new_isoval, 0);
  glVertex2f(new_isoval, h);
  glEnd();

  char buf[100];
  sprintf(buf, "%g", new_isoval);
	
  int w=calc_width(fontInfo, buf);
  float wid=(datamax-datamin)*w/xres;
  float x=new_isoval-wid/2.;
  float left=datamin+(datamax-datamin)*2/xres;
  if(x<left)
    x=left;
  printString(fontbase, x, descent+1, buf, Color(1,0,0));

  glFinish();
  int errcode;
  while((errcode=glGetError()) != GL_NO_ERROR){
    cerr << "We got an error from GL: " << (char*)gluErrorString(errcode) << endl;
  }
}

void VolumeDpy::move_isoval(const int x)
{
  float xn=float(x)/xres;
  float val=datamin+xn*(datamax-datamin);
  if(val<datamin)
    val=datamin;
  if(val>datamax)
    val=datamax;
  new_isoval=val;
}

void VolumeDpy::change_isoval(float new_val) {
  if (new_val >= datamin && new_val <= datamax)
    new_isoval = new_val;
}

void VolumeDpy::set_minmax(float min, float max) {
  datamin = min;
  datamax = max;
}

void VolumeDpy::animate(bool& changed)
{
  if(isoval != new_isoval){
    isoval=new_isoval;
    changed=true;
  }
}




