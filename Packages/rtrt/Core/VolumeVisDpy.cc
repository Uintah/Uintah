

#include <Packages/rtrt/Core/VolumeVisDpy.h>
#include <Packages/rtrt/Core/VolumeVis.h>
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

using namespace rtrt;
using SCIRun::Mutex;
using SCIRun::Thread;

namespace rtrt {
  extern Mutex xlock;
} // end namespace rtrt

static void printString(GLuint fontbase, double x, double y,
			const char *s, const Color& c);
static int calc_width(XFontStruct* font_struct, const char* str);

VolumeVisDpy::VolumeVisDpy(Array1<Color*> *matls, Array1<float> *alphas,
			   float t_inc, char *in_file):
  hist(0), xres(500),yres(500), color_transform(matls),
  alpha_transform(alphas), original_t_inc(0.01), current_t_inc(0),
  t_inc(t_inc), in_file(in_file)
{}
  
VolumeVisDpy::~VolumeVisDpy()
{
  if (hist)
    delete[] hist;
}

void VolumeVisDpy::attach(VolumeVis *volume) {
  volumes.add(volume);
}

void VolumeVisDpy::setup_vars() {
  cerr << "VolumeVisDpy::setup_vars:start\n";
  if (volumes.size() > 0) {
    data_min = MAXFLOAT;
    data_max = -MAXFLOAT;

    // go through all the volumes and compute the min/max of the data values
    for(unsigned int v = 0; v < volumes.size(); v++) {
      data_min = Min(data_min, volumes[v]->data_min);
      data_max = Max(data_max, volumes[v]->data_max);
    }

    // this takes into account of
    // min/max equaling each other
    if (data_min == data_max) {
      if (data_max > 0) {
	data_max *= 1.1;
      } else {
	if (data_max < 0)
	  data_max *= 0.9;
	else
	  data_max = 1;
      }
    }
    scale = 1.0/(data_max - data_min);
    color_transform.scale(data_min,data_max);
    alpha_transform.scale(data_min,data_max);
  }// end if(volumes.size() > 0)

  // rescale the alpha values
  current_t_inc = original_t_inc;
  rescale_alphas(t_inc);
}

void VolumeVisDpy::run() {
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
  xlock.unlock();
  

  setup_vars();
  bool need_hist=true;
  bool redraw=true;
  bool control_pressed=false;
  bool shift_pressed=false;

  for(;;){
    //cerr << "GridSpheresDpy:run:eventloop\n";
    if(need_hist){
      need_hist=false;
      compute_hist(fontbase);
      redraw=true;
    }
    if(redraw){
      draw_hist(fontbase, fontInfo);
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
	cerr << "Pressed control\n";
	control_pressed = true;
	break;
      case XK_Shift_L:
      case XK_Shift_R:
	cerr << "Pressed shift\n";
	shift_pressed = true;
	break;
      case XK_w:
      case XK_W:
	break;
      }
      break;
    case KeyRelease:
      switch(XKeycodeToKeysym(dpy, e.xkey.keycode, 0)){
      case XK_Control_L:
      case XK_Control_R:
	control_pressed = false;
	cerr << "Releassed control\n";
	break;
      case XK_Shift_L:
      case XK_Shift_R:
	cerr << "Releassed shift\n";
	shift_pressed = false;
      }
      break;
    case ButtonRelease:
      break;
    case ButtonPress:
      switch(e.xbutton.button){
      case Button1:
	if (shift_pressed) {
	  cerr << "Left button pressed with shift\n";
	} else if (control_pressed) {
	  cerr << "Left button pressed with control\n";
	} else {
	}
	break;
      case Button2:
	if (shift_pressed) {
	  cerr << "Middle button pressed with shift\n";
	} else if (control_pressed) {
	  cerr << "Middle button pressed with control\n";
	} else {
	}
	break;
      case Button3:
	if (shift_pressed) {
	  cerr << "Right button pressed with shift\n";
	} else if (control_pressed) {
	  cerr << "Right button pressed with control\n";
	} else {
	}
	break;
      }
      break;
    case MotionNotify:
      if (shift_pressed || control_pressed) break;
      switch(e.xmotion.state&(Button1Mask|Button2Mask|Button3Mask)){
      case Button1Mask:
	if (shift_pressed) {
	  cerr << "Left button pressed with shift\n";
	} else if (control_pressed) {
	  cerr << "Left button pressed with control\n";
	} else {
	}
	break;
      case Button2Mask:
	break;
      case Button3Mask:
	if (shift_pressed) {
	  cerr << "Right button pressed with shift\n";
	} else if (control_pressed) {
	  cerr << "Right button pressed with control\n";
	} else {
	}
	break;
      }
      break;
    default:
      cerr << "Unknown event, type=" << e.type << '\n';
    }
  }
}

void VolumeVisDpy::compute_hist(GLuint fid) {
  //cerr << "VolumeVisDpy:compute_hist:start\n";
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

  int nhist=xres-10;
  //cerr << "VolumeVisDpy:compute_hist:xres = " << xres << "\n";
  // allocate and initialize the memory for the histogram
  if (hist)
    delete[] hist;
  hist=new int[nhist];
  for(int i=0;i<nhist;i++){
    hist[i]=0;
  }
  // loop over all the data and compute histograms
  for (int v = 0; v < volumes.size() ; v++) {
    VolumeVis* volume = volumes[v];
#if 1
    for (int x = 0; x < volume->data.dim1(); x++) {
      for (int y = 0; y < volume->data.dim2(); y++) {
	for (int z = 0; z < volume->data.dim3(); z++) {
	  int idx=(int)((volume->data(x,y,z)-data_min) * scale * (nhist-1));
	  if (idx >= 0 && idx < nhist)
	    hist[idx]++;
	  //cerr << "data = " << volume->data(x,y,z) << ", data_min="<<data_min<<", data_max = "<<data_max<<", scale = "<<scale<<", idx=" << idx << '\n';
	}
      }
    }
#else
    float* p=volume->data.get_dataptr();
    int ndata=volume->data.dim1() * volume->data.dim2() * volume->data.dim3(); 
    for(int i=0;i<ndata;i++,p++){
      float normalized=(*p-data_min)*scale;
      int idx=(int)(normalized*(nhist-1));
      //int idx=color_transform.get_lookup_index(*p++);
      if (idx >= 0 && idx < nhist)
	hist[idx]++;
      cerr << "p = " << *p << ", data_min="<<data_min<<", data_max = "<<data_max<<", scale = "<<scale<<endl;
      cerr << "idx=" << idx << '\n';
#if 0
      if(idx<0 || idx>=nhist){
	cerr << "idx out of bounds!\n";
	Thread::exitAll(-1);
      }
      hist[idx]++;
#endif
    }
#endif
  } // end loop for all volumes
  
  //cerr << "VolumeVisDpy:compute_hist:past compute histograms\n";
  // determine the maximum height for each histogram
  int* hp=hist;
  int max=0;
  for(int i=0;i<nhist;i++){
    if(*hp>max)
      max=*hp;
    hp++;
  }
  histmax=max;
  cout << "histmax = " << histmax << endl;
  //cerr << "VolumeVisDpy:compute_hist:end\n";
}

void VolumeVisDpy::draw_hist(GLuint fid, XFontStruct* font_struct) {
  int descent=font_struct->descent;
  int textheight=font_struct->descent+font_struct->ascent+2;

  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  glViewport(0, 0, xres, yres);
  glClearColor(0, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT);
  int nhist=xres-10;

  int s=yres-300; // start
  int e=yres; // end
  int h=e-s;
  glViewport(5, s, xres-5, e-s);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, xres, -float(textheight)*histmax/(h-textheight), histmax);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
    
  //  glColor3f(0,0,1);
  ScalarTransform1D<int,Color*> stripes(color_transform.get_results_ptr());
  stripes.scale(0,nhist-1);
  glBegin(GL_LINES);
  for(int i=0;i<nhist;i++){
    Color *c=stripes.lookup(i);
    //    cout << "color[i="<<i<<"] = " << *c << endl;
    glColor3f(c->red(), c->green(), c->blue());
    glVertex2i(i, 0);
    glVertex2i(i, hist[i]);
  }
  glEnd();
      
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, xres, 0, h);
  char buf[100];
  // print the min on the left
  sprintf(buf, "%g", data_min);
  printString(fid, 2, descent+1, buf, Color(1,1,1));
  // print the name in the middle
  int x = (int)((xres - calc_width(font_struct,"Data Histogram"))/2);
  printString(fid, x, descent+1, "Data Histogram", Color(1,1,1));
  // print the max on the right
  sprintf(buf, "%g", data_max);
  int w=calc_width(font_struct, buf);
  printString(fid, xres-2-w, descent+1, buf, Color(1,1,1));
  
  glFinish();
  int errcode;
  while((errcode=glGetError()) != GL_NO_ERROR){
    cerr << "We got an error from GL: " << (char*)gluErrorString(errcode) << endl;
  }
  //cerr << "VolumeVisDpy:draw_hist:end\n";
}

void VolumeVisDpy::rescale_alphas(float new_t_inc) {
  // modify the alpha matrix by the t_inc
  // we are assuming the base delta_t is 1 and that the new delta_t is t_inc
  // the formula:
  //    a_1 : original opacity
  //    a_2 : resulting opacity
  //    d_1 : original sampling distance
  //    d_2 : new sampling distance
  // a_2 = 1 - (1 - a_1)^(d_2/d_1)
  float d2_div_d1 = new_t_inc/current_t_inc;
  for(unsigned int i = 0; i < alpha_transform.size(); i++) {
    alpha_transform[i] = 1 - powf(1 - alpha_transform[i], d2_div_d1);
    cout <<"alpha_transform[i="<<i<<"] = "<<alpha_transform[i]<<", ";
  }
  current_t_inc = new_t_inc;
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

static int calc_width(XFontStruct* font_struct, const char* str)
{
  XCharStruct overall;
  int ascent, descent;
  int dir;
  XTextExtents(font_struct, str, (int)strlen(str), &dir, &ascent, &descent, &overall);
  return overall.width;
}

