

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

VolumeVisDpy::VolumeVisDpy(Array1<Color> &matls, Array1<AlphaPos> &alphas,
			   int ncolors, float t_inc, char *in_file):
  hist(0), xres(500), yres(500), colors_index(matls), alpha_list(alphas),
  ncolors(ncolors), nalphas(ncolors),
  original_t_inc(0.01), current_t_inc(t_inc), t_inc(t_inc),
  in_file(in_file)
{
  // need to allocate memory for alpha_transform and color_transform
  Array1<Color*> *c = new Array1<Color*>(ncolors);
  for(unsigned int i = 0; i < ncolors; i++)
    (*c)[i] = new Color();
  color_transform.set_results_ptr(c);
  alpha_transform.set_results_ptr(new Array1<float>(nalphas));
  alpha_stripes.set_results_ptr(new Array1<float>(nalphas));
}
  
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

  // create the colors for the color transform
  create_color_transfer();
  // rescale the alpha values
  //  current_t_inc = t_inc;
  // create the alphas for the alpha transform
  create_alpha_transfer();
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
      draw_alpha_curve(fontbase, fontInfo);
      glFinish();
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
      case XK_Page_Up:
      case XK_plus:
	rescale_alphas(current_t_inc/2);
	cout << "current_t_inc = " << current_t_inc << endl;
	break;
      case XK_Page_Down:
      case XK_minus:
	rescale_alphas(current_t_inc*2);
	cout << "current_t_inc = " << current_t_inc << endl;
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
    // must index the data using three dimesions rather than as one, because
    // a BrickArray3 uses buffers on the ends to make the bricks all the same
    // size.
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

  glViewport(0, 0, xres, yres);
  glClearColor(0, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT);
  int nhist=xres-10;

  int s=yres/2; // start
  int e=yres; // end
  int h=e-s;
  glViewport(5, s, xres-10, e-s);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, xres-10, -float(textheight)*histmax/(h-textheight), histmax);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
    
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
  
  int errcode;
  while((errcode=glGetError()) != GL_NO_ERROR){
    cerr << "We got an error from GL: " << (char*)gluErrorString(errcode) << endl;
  }
  cerr << "VolumeVisDpy:draw_hist:end\n";
}

// displays the alpha transfer curve as well as a representation of the colors
// and their corresponding opacities.  No alpha blending is used, because we
// are using a black background.  This allows us the ablility to just multiply
// the corresponding colors, by the alpha value.
void VolumeVisDpy::draw_alpha_curve(GLuint /*fid*/, XFontStruct* font_struct) {
  int descent=font_struct->descent;
  int textheight=font_struct->descent+font_struct->ascent+2;

  // set up the frame for the lower half of the user interface.
  int s=5;
  int e=yres/2-5;
  int h=e-s;
  int width = xres -10;
  glViewport(5, s, width, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, width, 0, h);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // draw the background
  // stripes allows us the oportunity to use an integer index based on the
  // width of the window to grab a color.
  ScalarTransform1D<int,Color*> stripes(color_transform.get_results_ptr());
  stripes.scale(0,width-1);
  alpha_stripes.scale(0,width-1);

  glBegin(GL_LINES);
  for(int i=0;i<width;i++){
    Color *c=stripes.lookup(i); // get the color
    float alpha = alpha_stripes.lookup(i); // get the alpha
    glColor3f(c->red()*alpha, c->green()*alpha, c->blue()*alpha);
    glVertex2i(i, 0);
    glVertex2i(i, h);
  }
  glEnd();
  
  // now draw the alpha curve
  glColor3f(1.0, 1.0, 1.0);
  glBegin(GL_LINE_STRIP);
  for(unsigned int i = 0; i < alpha_list.size(); i++) {
    //    cout << "drawing a point at ("<<alpha_list[i].x<<", "<<alpha_list[i].val<<")\n";
    glVertex2i(alpha_list[i].x*width, alpha_list[i].val*h);
  }
  glEnd();
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
  cout << endl;
  current_t_inc = new_t_inc;
}

// assuming that the data in alpha_transform and alpha_stripes is already
// allocated.
// This is not too trivial as the alpha values in alpha_list are not evenly
// spaced.  
void VolumeVisDpy::create_alpha_transfer() {
  // the ratio of values as explained in rescale_alphas
  float d2_div_d1 = current_t_inc/original_t_inc;
  // i_f is the number between 0 and 1 that represent how far we are along
  float i_f = 0;
  float i_f_inc = 1.0/(alpha_transform.size()-1);
  int a_index = 0; // the index of the alpha_list
  // slope of the line made up of alpha_list[a_index] and alpha_list[a_index+1]
  // defined as alpha_list[a_index].x <= i_f < alpha_list[a_index+1].x
  float slope = 0; 
  float c = 0; // the constant for the line equasion

  // we need to get a value for every entry in alpha_transform
  for(unsigned int i = 0; i < alpha_transform.size(); i++) {
    // this will be the interpolated value
    float val; 
    //    cout <<"a_index = "<<a_index<<", i_f = "<<i_f<<", ";
    // if this (alpha_list[a_index].x <= i_f < alpha_list[a_index+1].x) no
    // longer holds true we need to increment a_index and recompute the slope
    // and constant.
    if (i_f > alpha_list[a_index+1].x) {
      a_index++;
      //      cout << "a_index = "<<a_index<<", ";
      // slope is rise of run
      slope = (alpha_list[a_index+1].val - alpha_list[a_index].val) /
	(alpha_list[a_index+1].x - alpha_list[a_index].x);
      // c = y1 - slope * x1;
      c = alpha_list[a_index].val - slope * alpha_list[a_index].x;
      //      cout <<"slope = "<<slope<<", c = "<<c<<", ";
    }
    // calculate the interpolated value
#if 0
    if ((a_index+1) < alpha_list.size()) {
      val = slope * i_f + c;
    } else {
      // the last element
      val = alpha_list[a_index].val;
    }
#else
    val = slope * i_f + c;
#endif
    
    cout << "val = "<<val<<", ";
    alpha_stripes[i] = val;
    // apply the alpha scaling
    alpha_transform[i] = 1 - powf(1 - val, d2_div_d1);
    cout <<"alpha_transform[i="<<i<<"] = "<<alpha_transform[i]<<"\n";
    i_f += i_f_inc;
  }
}

// assuming that the data in color_transform already allocated
// This is really easy to compute.  All we do is take the color_index and
// interpolate the values.  We are assuming that each color is evenly spaced.
void VolumeVisDpy::create_color_transfer() {
  ScalarTransform1D<unsigned int,Color> c_index(&colors_index);
  c_index.scale(0,color_transform.size()-1);
  for(unsigned int i = 0; i < color_transform.size(); i++) {
    Color *c = color_transform[i];
    *c = c_index.interpolate(i);
  }
}

void VolumeVisDpy::animate(bool& changed) {
  if (current_t_inc != t_inc) {
    changed = true;
    t_inc = current_t_inc;
    cout << "t_inc now equals "<< t_inc<<endl;
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

static int calc_width(XFontStruct* font_struct, const char* str)
{
  XCharStruct overall;
  int ascent, descent;
  int dir;
  XTextExtents(font_struct, str, (int)strlen(str), &dir, &ascent, &descent, &overall);
  return overall.width;
}

