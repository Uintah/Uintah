
#include <Packages/rtrt/Core/GridSpheresDpy.h>

#include <Packages/rtrt/Core/GridSpheres.h>
#include <Packages/rtrt/Core/FontString.h>
#include <Packages/rtrt/visinfo/visinfo.h>

#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Time.h>

#ifdef __GNUG__
#  include <values.h>
#endif

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

#include <stdlib.h>
#include <stdio.h>
#include <X11/keysym.h>

using namespace std;
using namespace rtrt;

using SCIRun::Mutex;
using SCIRun::Thread;

namespace rtrt {
  extern Mutex xlock;
} // end namespace rtrt


GridSpheresDpy::GridSpheresDpy(int colordata, char *in_file) :
  hist(0), xres(500), yres(500), ndata(-1),
  colordata(colordata),newcolordata(colordata), in_file(in_file)
{
}

GridSpheresDpy::~GridSpheresDpy()
{
  if (hist)
    delete(hist);
}

void GridSpheresDpy::setup_vars() {
  cerr << "GridSpheresDpy:setup_vars:start\n";
  if (grids.size() > 0) {
    histmax=new int[ndata];

    // determine min/max
    min = new float[ndata];
    max = new float[ndata];
    // inialize values
    for(int i = 0; i < ndata; i ++) {
      min[i] = MAXFLOAT;
      max[i] = -MAXFLOAT;
    }
    // go through all the grids and compute min/max
    for(int g = 0; g < grids.size(); g++) {
      for (int i = 0; i < ndata; i++) {
	min[i]=Min(min[i], grids[g]->min[i]);
	max[i]=Max(max[i], grids[g]->max[i]);
      }
    }

    // setup scales/ranges
    scales=new float[ndata];
    color_scales=new float[ndata];
    original_min=new float[ndata];
    original_max=new float[ndata];
    range_begin=new float[ndata];
    range_end=new float[ndata];
    new_range_begin=new float[ndata];
    new_range_end=new float[ndata];
    color_begin=new float[ndata];
    color_end=new float[ndata];
    new_color_begin=new float[ndata];
    new_color_end=new float[ndata];
    for(int i=0;i<ndata;i++){
      if (var_names != 0)
	cerr << var_names[i];
      cerr << "\t\t\tmin["<<i<<"] = "<<min[i]<<",\tmax["<<i<< "] = "<<max[i] << endl;
      // this takes into account of
      // min/max equaling each other
      if (min[i] == max[i]) {
	if (max[i] > 0) {
	  max[i] *= 1.1;
	} else {
	  if (max[i] < 0)
	    max[i] *= 0.9;
	  else
	    max[i] = 1;
	}
      }
      scales[i]=color_scales[i]=1./(max[i]-min[i]);
      original_min[i]=new_range_begin[i]=range_begin[i]=color_begin[i]=
	new_color_begin[i]=min[i];
      original_max[i]=new_range_end[i]=range_end[i]=color_end[i]=
	new_color_end[i]=max[i];
    }

    ////////////////////////////////////////////////////////////
    // try to load in the data file

    if (in_file != 0) {
      // have a file name
      ifstream in(in_file);
      if(!in){
	cerr << "GridSpheresDpy::setup_vars:Error opening file: " << in_file
	     << ", using defaults.\n";
	return;
      }
      int ndata_file;
      in >> ndata_file;
      in >> colordata;
      newcolordata = colordata;
      if (ndata_file != ndata)
	return;
      for(int i=0;i<ndata_file;i++){
	in >> original_min[i] >> original_max[i];
	in >> min[i] >> max[i];
	scales[i]=1./(max[i]-min[i]);
	in >> color_begin[i] >> color_end[i];
	new_color_begin[i] = color_begin[i];
	new_color_end[i] = color_end[i];
	color_scales[i]=1./(color_end[i]-color_begin[i]);
	in >> range_begin[i] >> range_end[i];
	new_range_begin[i] = range_begin[i];
	new_range_end[i] = range_end[i];
      }
      if(!in){
	cerr << "GridSpheresDpy::setup_vars:Error reading file: " << in_file
	     << "\n";
	exit(1);
      }
    }
  }
  cerr << "GridSpheresDpy:setup_vars:end\n";
}

void GridSpheresDpy::run()
{
  //cerr << "GridSpheresDpy:run\n";
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
  atts.event_mask=StructureNotifyMask|ExposureMask|ButtonPressMask|ButtonReleaseMask|ButtonMotionMask|KeyPressMask|KeyReleaseMask;
  Window win=XCreateWindow(dpy, RootWindow(dpy, screen),
			   0, 0, xres, yres, 0, vi->depth,
			   InputOutput, vi->visual, flags, &atts);
  char* p="GridSpheres histogram";
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
  XFontStruct* fontInfo = XLoadQueryFont(dpy, __FONTSTRING__);

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
  

  setup_vars();
  bool need_hist=true;
  bool redraw=true;
  bool control_pressed=false;
  bool shift_pressed=false;
  int redraw_range=-1;
  xlock.unlock();
  for(;;){
    //cerr << "GridSpheresDpy:run:eventloop\n";
    if(need_hist){
      need_hist=false;
      compute_hist(fontbase);
      redraw=true;
    }
    if(redraw){
      draw_hist(fontbase, fontInfo, redraw_range);
      redraw=false;
    }
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
    case KeyPress:
      switch(XKeycodeToKeysym(dpy, e.xkey.keycode, 0)){
      case XK_Control_L:
      case XK_Control_R:
	//	cerr << "Pressed control\n";
	control_pressed = true;
	break;
      case XK_Shift_L:
      case XK_Shift_R:
	//	cerr << "Pressed shift\n";
	shift_pressed = true;
	break;
      case XK_w:
      case XK_W:
	write_data_file("gridspheredpy.cfg");
	break;
      }
      break;
    case KeyRelease:
      switch(XKeycodeToKeysym(dpy, e.xkey.keycode, 0)){
      case XK_Control_L:
      case XK_Control_R:
	control_pressed = false;
	//	cerr << "Releassed control\n";
	break;
      case XK_Shift_L:
      case XK_Shift_R:
	//	cerr << "Releassed shift\n";
	shift_pressed = false;
      }
      break;
    case ButtonRelease:
      break;
    case ButtonPress:
      switch(e.xbutton.button){
      case Button1:
	if (shift_pressed) {
	  //cerr << "Left button pressed with shift\n";
	  move_min_max(min, e.xbutton.x, e.xbutton.y, redraw_range);
	} else if (control_pressed) {
	  //cerr << "Left button pressed with control\n";
	  move(new_color_begin, e.xbutton.x, e.xbutton.y, redraw_range);
	} else {
	  move(new_range_begin, e.xbutton.x, e.xbutton.y, redraw_range);
	}
	redraw=true;
	break;
      case Button2:
	if (shift_pressed) {
	  //cerr << "Middle button pressed with shift\n";
	  restore_min_max(e.xbutton.y, redraw_range);
	  redraw=true;
	} else if (control_pressed) {
	  //cerr << "Middle button pressed with control\n";
	  move(new_color_begin, 0, e.xbutton.y, redraw_range);
	  move(new_color_end, xres, e.xbutton.y, redraw_range);
	  redraw=true;
	} else {
	  changecolor(e.xbutton.y);
	}
	break;
      case Button3:
	if (shift_pressed) {
	  //cerr << "Right button pressed with shift\n";
	  move_min_max(max, e.xbutton.x, e.xbutton.y, redraw_range);
	} else if (control_pressed) {
	  //cerr << "Right button pressed with control\n";
	  move(new_color_end, e.xbutton.x, e.xbutton.y, redraw_range);
	} else {
	  move(new_range_end, e.xbutton.x, e.xbutton.y, redraw_range);
	}
	redraw=true;
	break;
      }
      break;
    case MotionNotify:
      if (shift_pressed || control_pressed) break;
      switch(e.xmotion.state&(Button1Mask|Button2Mask|Button3Mask)){
      case Button1Mask:
	if (shift_pressed) {
	  //cerr << "Left button pressed with shift\n";
	} else if (control_pressed) {
	  //cerr << "Left button pressed with control\n";
	  move(new_color_begin, e.xbutton.x, e.xbutton.y, redraw_range);
	} else {
	  move(new_range_begin, e.xbutton.x, e.xbutton.y, redraw_range);
	}
	redraw=true;
	break;
      case Button2Mask:
	break;
      case Button3Mask:
	if (shift_pressed) {
	  //cerr << "Right button pressed with shift\n";
	} else if (control_pressed) {
	  //cerr << "Right button pressed with control\n";
	  move(new_color_end, e.xbutton.x, e.xbutton.y, redraw_range);
	} else {
	  move(new_range_end, e.xbutton.x, e.xbutton.y, redraw_range);
	}
	redraw=true;
	break;
      }
      break;
    default:
      cerr << "Unknown event, type=" << e.type << '\n';
    }
  }
}

static void printString(GLuint fontbase, double x, double y,
			const char *s, const Color& c)
{
  glColor3f(c.red(), c.green(), c.blue());
  
  glRasterPos2d(x,y);
  /*glBitmap(0, 0, x, y, 1, 1, 0);*/
  glPushAttrib (GL_LIST_BIT);
  glListBase(fontbase);
  glCallLists((int)strlen(s), GL_UNSIGNED_BYTE, (GLubyte *)s);
  glPopAttrib ();
}

void GridSpheresDpy::compute_hist(GLuint fid)
{
  //cerr << "GridSpheresDpy:compute_hist:start\n";
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
  int n=ndata;
  if (n == -1) {
    cerr <<"GridSpheresDpy::compute_hist: GridSpheresDpy::attach not called\n";
    cerr << "GridSpheresDpy::ndata = -1\n";
    Thread::exitAll(-1);
  }
  int nhist=xres;
  int total=n*nhist;
  //cerr << "GridSpheresDpy:compute_hist:ndata = " << ndata << "\n";  
  //cerr << "GridSpheresDpy:compute_hist:xres = " << xres << "\n";  
  //cerr << "GridSpheresDpy:compute_hist:total = " << total << "\n";  
  if (hist)
    delete(hist);
  hist=new int[total];
  for(int i=0;i<total;i++){
    hist[i]=0;
  }
  // loop over all the data and compute histograms
  for (int g = 0; g < grids.size() ; g++) {
    GridSpheres* grid = grids[g];
    float* p=grid->spheres;
    int nspheres=grid->nspheres;
    for(int i=0;i<nspheres;i++){
      int offset=0;
      for(int j=0;j<n;j++){
	float normalized=(p[j]-min[j])*scales[j];
	int idx=(int)(normalized*(nhist-1));
	if (idx >= 0 && idx < nhist)
	  hist[offset+idx]++;
#if 0
	if(idx<0 || idx>=nhist){
	  cerr << "p[" << j << "]=" << p[j] << ", min[" << j << "]=" << min[j] << ", scales[" << j << "]=" << scales[j] << '\n';
	  cerr << "idx=" << idx << '\n';
	  cerr << "idx out of bounds!\n";
	  Thread::exitAll(-1);
	}
	hist[offset+idx]++;
#endif
	offset+=nhist;
      }
      p+=n;
    }
  }
  //cerr << "GridSpheresDpy:compute_hist:past compute histograms\n";
  // determine the maximum height for each histogram
  int* hp=hist;
  for(int j=0;j<n;j++){
    int max=0;
    for(int i=0;i<nhist;i++){
      if(*hp>max)
	max=*hp;
      hp++;
    }
    histmax[j]=max;
  }
  //cerr << "GridSpheresDpy:compute_hist:end\n";
}

static int calc_width(XFontStruct* font_struct, const char* str)
{
  XCharStruct overall;
  int ascent, descent;
  int dir;
  XTextExtents(font_struct, str, (int)strlen(str), &dir, &ascent, &descent, &overall);
  return overall.width;
}

void GridSpheresDpy::draw_hist(GLuint fid, XFontStruct* font_struct,
			       int& redraw_range)
{
  //cerr << "GridSpheresDpy:draw_hist:start\n";
  int n=ndata;
  int descent=font_struct->descent;
  int textheight=font_struct->descent+font_struct->ascent+2;
  if(redraw_range == -1){
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glViewport(0, 0, xres, yres);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    int offset=0;
    int nhist=xres;
    for(int j=0;j<n;j++){
      int s=j*yres/n+2;
      int e=(j+1)*yres/n-2;
      int h=e-s;
      glViewport(0, s, xres, e-s);
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluOrtho2D(0, xres, -float(textheight)*histmax[j]/(h-textheight), histmax[j]);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      
      glColor3f(0,0,1);
      glBegin(GL_LINES);
      for(int i=0;i<nhist;i++){
	glVertex2i(i, 0);
	glVertex2i(i, hist[offset+i]);
      }
      glEnd();
      
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluOrtho2D(0, xres, 0, h);
      offset+=nhist;
      char buf[100];
      // print the min on the left
      sprintf(buf, "%g", min[j]);
      printString(fid, 2, descent+1, buf, Color(0,1,1));
      // print the variable name in the middle
      if (var_names != 0) {
	int x = (int)((xres - calc_width(font_struct,var_names[j].c_str()))/2);
	printString(fid, x, descent+1, var_names[j].c_str(), Color(1,1,1));
      }
      // print the max on the right
      sprintf(buf, "%g", max[j]);
      int w=calc_width(font_struct, buf);
      printString(fid, xres-2-w, descent+1, buf, Color(0,1,1));
    }
  }
  
  glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE);
  for(int j=0;j<n;j++){
    if(redraw_range==-1 || redraw_range==j){
      int s=j*yres/n+2;
      int e=(j+1)*yres/n-2;
      int h=e-s;
      glViewport(0, s, xres, e-s);
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluOrtho2D(min[j], max[j], 0, h);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      glColor3f(0,0,0);
      glRectf(min[j], 0, max[j], h);

      
      // draw the coloring spots
      if (new_color_begin[j] != new_color_end[j])
	color_scales[j] = 1./(new_color_end[j]-new_color_begin[j]);
      if (new_color_end[j] > new_color_begin[j]) {
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
#if 0
	glColor3f(1,1,1);
	glBegin(GL_LINES);
	cerr << "new_color_begin["<<j<<"] = "<<new_color_begin[j];
	cerr << ", new_color_end["<<j<<"] = "<<new_color_end[j]<<endl;
	cerr << "textheight-2 = "<<textheight-2<<", h = "<<h<<endl;
	float factor=xres/(max[j]-min[j]);
	int x = (int)((new_color_begin[j]-min[j])*factor);
	cerr <<"x = "<<x;
	glVertex2i(new_color_begin[j], textheight-2);
	glVertex2i(new_color_begin[j], h+2);
	x = (int)((new_color_end[j]-min[j])*factor);
	cerr <<",x2 = "<<x<<endl;
	glVertex2i(new_color_end[j], textheight-2);
	glVertex2i(new_color_end[j], h+2);
	glEnd();
#endif
	glColor3f(0,0,0);
	glRectf(min[j], textheight-2, max[j], textheight);
	glColor3f(1,1,1);
	glRectf(new_color_begin[j], textheight-2, new_color_end[j],textheight);
	glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE);
      }
      
      if(new_range_end[j] > new_range_begin[j]){
	glColor3f(.5,0,0);
	glRectf(new_range_begin[j], textheight, new_range_end[j], h);
      }
      
      char buf[100];
      sprintf(buf, "%g", new_range_begin[j]);
      
      int w=calc_width(font_struct, buf);
      float wid=(max[j]-min[j])*w/xres;
      float x=new_range_begin[j]-wid/2.;
      float left=min[j]+(max[j]-min[j])*2/xres;
      if(x<left)
	x=left;
      printString(fid, x, descent+1, buf, Color(1,0,0));
      
      sprintf(buf, "%g", new_range_end[j]);
      w=calc_width(font_struct, buf);
      wid=(max[j]-min[j])*w/xres;
      x=new_range_end[j]-wid/2.;
      float right=max[j]-(max[j]-min[j])*2/xres;
      if(x>right-wid)
	x=right-wid;
      printString(fid, x, descent+1, buf, Color(1,0,0));

    }
  }
  glFinish();
  int errcode;
  while((errcode=glGetError()) != GL_NO_ERROR){
    cerr << "We got an error from GL: " << (char*)gluErrorString(errcode) << endl;
  }
  redraw_range=-1;
  //cerr << "GridSpheresDpy:draw_hist:end\n";
}

void GridSpheresDpy::move(float* range, int x, int y, int& redraw_range)
{
  // swap y=0 orientation
  y=yres-y;
  // loop over each block and see where the event was
  for(int j = 0; j < ndata; j++){
    int s=j*yres/ndata +2;
    int e=(j+1)*yres/ndata-2;
    if(y>=s && y<e){
      // found the region the event was
      float xn=float(x)/xres; // normalize the x location to [0,1]
      // find the corresponding value at the clicked location
      float val=min[j]+xn*(max[j]-min[j]);
      // bound the value
      range[j]=bound(val,min[j],max[j]);
      redraw_range=j;
      break;
    }
  }
}

void GridSpheresDpy::move_min_max(float* range, int x, int y,int &redraw_range)
{
  // swap y=0 orientation
  y=yres-y;
  // loop over each block and see where the event was
  for(int j = 0; j < ndata; j++){
    int s=j*yres/ndata +2;
    int e=(j+1)*yres/ndata-2;
    if(y>=s && y<e){
      // found the region the event was
      float xn=float(x)/xres; // normalize the x location to [0,1]
      // find the corresponding value at the clicked location
      float val=min[j]+xn*(max[j]-min[j]);
      // bound the value
      range[j]=bound(val,min[j],max[j]);
      // set redraw_range to -1 so that the histogram lines get redrawn
      redraw_range=-1;

      // make sure that min and max don't equal each other
      if (min[j] == max[j])
	if (min[j] != 0)
	  if (min[j] > 0)
	    max[j] = 1.1 * min[j];
	  else
	    max[j] = 0.9 * min[j];
	else
	  max[j] = 1;
      // update scales, color_begin/end, range_begin/end, color_scale
      scales[j] = 1./(max[j]-min[j]);
      // bound new_color_begin and new_color_end
      new_color_begin[j] = bound(new_color_begin[j], min[j], max[j]);
      new_color_end[j] = bound(new_color_end[j], min[j], max[j]);
      // bound new_range_begin and new_range_end
      new_range_begin[j] = bound(new_range_begin[j], min[j], max[j]);
      new_range_end[j] = bound(new_range_end[j], min[j], max[j]);
      
      if (new_color_begin[j] != new_color_end[j])
	color_scales[j] = 1./(new_color_end[j]-new_color_begin[j]);

      // now we need to recompute the histogram for this range
      compute_one_hist(j);
      
      break;
    }
  }
}

void GridSpheresDpy::restore_min_max(int y, int &redraw_range) {
  // swap y=0 orientation
  y=yres-y;
  // loop over each block and see where the event was
  for(int j = 0; j < ndata; j++){
    int s=j*yres/ndata +2;
    int e=(j+1)*yres/ndata-2;
    if(y>=s && y<e){
      redraw_range = -1;
      // found the region the event was
      // reset the min and max
      min[j] = original_min[j];
      max[j] = original_max[j];

      // update scales
      scales[j] = 1./(max[j]-min[j]);

      // now we need to recompute the histogram for this range
      compute_one_hist(j);
      
      break;
    }
  }
}

void GridSpheresDpy::compute_one_hist(int j) {
#if 0
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
#endif
  int n=ndata;
  int nhist=xres;
  int offset=j*nhist;
  // initialize the histogram window
  for(int i = offset; i < (offset + nhist); i++){
    hist[i]=0;
      }
  // loop over all the data and compute histograms
  for (int g = 0; g < grids.size() ; g++) {
    GridSpheres* grid = grids[g];
    float* p=grid->spheres;
    int nspheres=grid->nspheres;
    //	p+=j*nspheres;
    for(int i=0;i<nspheres;i++){
      float normalized=(p[j]-min[j])*scales[j];
      int idx=(int)(normalized*(nhist-1));
      if (idx >= 0 && idx < nhist)
	hist[offset+idx]++;
      p+=n;
    }
  }
  //cerr << "GridSpheresDpy:compute_hist:past compute histograms\n";
  // determine the maximum height for each histogram
  int* hp=hist;
  hp+=offset;
  int max=0;
  for(int i=0;i<nhist;i++){
    if(*hp>max)
      max=*hp;
    hp++;
  }
  histmax[j]=max;
}

void GridSpheresDpy::animate(bool& changed) {

  for(int j=0;j<ndata;j++){
    if(new_range_begin[j] != range_begin[j]){
      changed=true;
      range_begin[j]=new_range_begin[j];
    }
    if(new_range_end[j] != range_end[j]){
      changed=true;
      range_end[j]=new_range_end[j];
    }
    if(new_color_begin[j] != color_begin[j]) {
      changed = true;
      color_begin[j] = new_color_begin[j];
    }
    if(new_color_end[j] != color_end[j]) {
      changed = true;
      color_end[j] = new_color_end[j];
    }
  }
  if (newcolordata != colordata) {
    changed=true;
    colordata = newcolordata;
  }
}

void GridSpheresDpy::attach(GridSpheres* g) {
  // ndata only equals -1 if ndata has not been set 
  if (ndata != -1) {
    if (ndata == g->ndata+3) {
      grids.add(g);
      g->dpy = this;
    }
    else {
      cerr << "Number of data fields does not match. Not adding to display.\n";
    }
  }
  else {
    ndata = g->ndata + 3;
    grids.add(g);
    g->dpy = this;
    var_names = g->var_names;
  }
}

void GridSpheresDpy::changecolor(int y) {
  y=yres-y;
  for(int j = 0; j < ndata; j++){
    int s=j*yres/ndata +2;
    int e=(j+1)*yres/ndata-2;
    if(y>=s && y<e){
      newcolordata = j;
      break;
    }
  }
}

void GridSpheresDpy::write_data_file(char *out_file) {
  ofstream out(out_file);
  if (!out) {
    cerr << "Error writing config file to " << out_file << endl;
    return;
  }
  out << setprecision(17);
  out << ndata << endl;
  out << colordata << endl;
  for(int i=0;i<ndata;i++){
    out << original_min[i] << " " << original_max[i] << endl;
    out << min[i] << " " << max[i] << endl;
    out << color_begin[i] << " " << color_end[i] << endl;
    out << range_begin[i] << " " << range_end[i] << endl;
  }
  cout << "Wrote config file to " << out_file << endl;
}


