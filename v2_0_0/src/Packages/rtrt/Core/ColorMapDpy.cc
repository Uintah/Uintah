

#include <Packages/rtrt/Core/ColorMapDpy.h>
#include <Packages/rtrt/Core/DpyBase.h>
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

using namespace rtrt;
using SCIRun::Mutex;
using SCIRun::Thread;
using namespace std;

namespace rtrt {
  extern Mutex xlock;
} // end namespace rtrt

static void draw_circle(int radius, int x_center, int y_center);
static void circle_points(int x, int y, int x_center, int y_center);

ColorMapDpy::ColorMapDpy(Array1<ColorPos> &matls, int num_bins):
  DpyBase("ColorMap GUI"), colors_index(matls),
  data_min(MAXFLOAT), data_max(-MAXFLOAT),
  selected_point(-1)
{
  set_resolution(350,150);
  // Allocate memory for the color and alpha transform
  color_transform.set_results_ptr(new Array1<Color>(num_bins));
  alpha_transform.set_results_ptr(new Array1<float>(num_bins));
  // Create the transfer
  create_transfer();
}

ColorMapDpy::ColorMapDpy(char *filebase, int num_bins):
  DpyBase("ColorMap GUI"), data_min(MAXFLOAT), data_max(-MAXFLOAT),
  selected_point(-1)
{
  set_resolution(400,200);

  // Read stuff in from the file
  // Open the file
  char buf[1000];
  sprintf(buf, "%s.cmp", filebase);
  this->filebase = strdup(filebase);
  ifstream in(buf);
  if(in){
    float x, val, r,g,b;
    while (!in.eof()) {
      in >> x >> val >> r >> g >> b;
      if (!in.eof()) {
	colors_index.add(ColorPos(x, val, Color(r,g,b)));
      }
    }
    colors_index[0].x = 0;
    colors_index[colors_index.size()-1].x = 1;
  } else {
    cerr << "ColorMapDpy::ERROR " << filebase << ".cmp NO SUCH FILE!" << "\n";
    cerr << "Creating default color map.\n";
    colors_index.add(ColorPos(0, 1, Color(0.0,0.0,0.0)));
    colors_index.add(ColorPos(1, 1, Color(1.0,1.0,1.0)));
  } 
  // Allocate memory for the color and alpha transform
  color_transform.set_results_ptr(new Array1<Color>(num_bins));
  alpha_transform.set_results_ptr(new Array1<float>(num_bins));
  // Create the transfer
  create_transfer();
}

ColorMapDpy::~ColorMapDpy()
{
  if (filebase)
    free(filebase);
}

void ColorMapDpy::set_viewport_params() {
  xstart = 5;
  xend = xres-5;
  width = xend - xstart;
  ystart = 5;
  yend = yres - 5;
  height = yend - ystart;
}

void ColorMapDpy::attach(float min_in, float max_in) {
  data_min = min(data_min, min_in);
  data_max = max(data_max, max_in);
  scale = 1/(data_max - data_min);

  color_transform.scale(data_min,data_max);
  alpha_transform.scale(data_min,data_max);
}

void ColorMapDpy::init() {
  glShadeModel(GL_BLEND);
  glClearColor(0, 0, 0, 1);
  //  cerr << "VolumeVisDpy::setup_vars:start\n";

  set_viewport_params();
  glViewport(xstart, ystart, width, height);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glMatrixMode(GL_PROJECTION);
}

void ColorMapDpy::display() {
  glClear(GL_COLOR_BUFFER_BIT);

  draw_transform();
  
  glFinish();
  if (window_mode & BufferModeMask == DoubleBuffered)
    glXSwapBuffers(dpy, win);
  XFlush(dpy);
}

void ColorMapDpy::resize(const int new_width, const int new_height) {
  xres = new_width;
  yres = new_height;
  set_viewport_params();
  glViewport(xstart, ystart, width, height);
}

#if 0
void ColorMapDpy::run() {
  open_display();

  init();

  for(;;){
    XEvent e;
    XNextEvent(dpy, &e);
    if(e.type == MapNotify)
      break;
  }

  setup_vars();
  bool need_hist=true;

  for(;;){
    //cerr << "VolumeVisDpy:run:eventloop\n";

    // Now we need to test to see if we should die
    if (scene->get_rtrt_engine()->stop_execution()) {
      close_display();
      return;
    }

    if(need_hist){
      need_hist=false;
      compute_hist(fontbase);
      redraw=true;
    }
    if(redraw){
      draw_hist(fontbase, fontInfo);
      draw_alpha_curve(fontbase, fontInfo);
      glFinish();
      glXSwapBuffers(dpy, win);
      XFlush(dpy);
      redraw=false;
    }
    // We should try to consume all the queued events before we redraw.
    // That way we don't waste time redrawing after each event
    while (XEventsQueued(dpy, QueuedAfterReading)) {
      XEvent e;
      XNextEvent(dpy, &e);	
      switch(e.type){
      case Expose:
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
	  //cerr << "Pressed control\n";
	  control_pressed = true;
	  break;
	case XK_Shift_L:
	case XK_Shift_R:
	  //cerr << "Pressed shift\n";
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
	  //cerr << "Releassed control\n";
	  break;
	case XK_Shift_L:
	case XK_Shift_R:
	  //cerr << "Releassed shift\n";
	  shift_pressed = false;
	}
	break;
      case ButtonRelease:
	switch(e.xbutton.button){
	case Button1:
	  {
	    create_alpha_transfer();
	    selected_point = -1;
	    redraw = true;
	  }
	  break;
	}
	break;
      case ButtonPress:
	{
	  int xpos = e.xbutton.x;
	  int ypos = yres - e.xbutton.y;
	
	  // check boundaries
	  int s=5;
	  int end=yres/2-5;
	  if (ypos < s || ypos > end)
	    break;
	  if (xpos < 5 || xpos > xres - 5)
	    break;
	  switch(e.xbutton.button){
	  case Button1:
	    if (shift_pressed) {
	      //cerr << "Left button pressed with shift\n";
	      selected_point = -1;
	      // create a point at the location of the click
	      AlphaPos new_guy((xpos-(float)5)/(xres-10),
			       (ypos-(float)5)/(yres/2-10));
	      int index = alpha_list.size();
	      alpha_list.grow(1,5);
	      while (new_guy.x < alpha_list[index-1].x) {
		alpha_list[index] = alpha_list[index-1];
		index--;
	      }
	      // now insert new_guy
	      alpha_list[index] = new_guy;
	      // make it selected for movement
	      selected_point = index;
	    } else if (control_pressed) {
	      //cerr << "Left button pressed with control\n";
	      selected_point = -1;
	      // find the point closest and delete it
	      // can't remove the end points
	      int index = select_point(xpos,ypos);
	      if (index > 0 && index < alpha_list.size()-1)
		alpha_list.remove(index);
	    } else {
	      // find the point closest and make it selected
	      int index = select_point(xpos,ypos);
	      if (index >= 0)
		selected_point = index;
	    }
	    break;
	  case Button2:
	    if (shift_pressed) {
	      //cerr << "Middle button pressed with shift\n";
	    } else if (control_pressed) {
	      //cerr << "Middle button pressed with control\n";
	    } else {
	    }
	    break;
	  case Button3:
	    if (shift_pressed) {
	      //cerr << "Right button pressed with shift\n";
	    } else if (control_pressed) {
	      //cerr << "Right button pressed with control\n";
	    } else {
	    }
	    break;
	  }
	}
	break;
      case MotionNotify:
	{
	  if (shift_pressed || control_pressed) break;
	  switch(e.xmotion.state&(Button1Mask|Button2Mask|Button3Mask)){
	  case Button1Mask:
	    {
	      if (selected_point < 0)
		// no point is selected, so don't do anything
		break;
	      int xpos = e.xbutton.x;
	      float xnorm;
	      if (xpos >= min_x && xpos < max_x)
		xnorm = (xpos - (float)5)/(xres-10);
	      else
		xnorm = alpha_list[selected_point].x;
	      int ypos = yres - e.xbutton.y;
	      float ynorm;
	      if (ypos < 5)
		ynorm = 0;
	      else if (ypos > (yres/2-5))
		ynorm = 1;
	      else
		ynorm = (ypos - (float)5)/(yres/2-10);
	    
	      alpha_list[selected_point] = AlphaPos(xnorm,ynorm);

	      redraw = true;
	    }
	    break;
	  case Button2Mask:
	  case Button3Mask:
	    break;
	  } // end switch(button mask with motion)
	}
	break;
      default:
	cerr << "Unknown event, type=" << e.type << '\n';
      } // end switch (e.type)
    } // end of while (there is a queued event)
  } // end of for(;;)
}
#endif

// displays the alpha transfer curve as well as a representation of the colors
// and their corresponding opacities.  No alpha blending is used, because we
// are using a black background.  This allows us the ablility to just multiply
// the corresponding colors, by the alpha value.
void ColorMapDpy::draw_transform() {
  // set up the frame for the lower half of the user interface.
  //  glViewport(5, s, width, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, 1, 0, 1);
  //  glMatrixMode(GL_MODELVIEW);
  //  glLoadIdentity();

  glBegin(GL_QUAD_STRIP);
  for(int i = 0; i < colors_index.size(); i++) {
    float x = colors_index[i].x;
    float y = colors_index[i].val;
    Color c = colors_index[i].c;
    glColor3f(c.red(), c.green(), c.blue());
    glVertex2f(x, y);
    glVertex2f(x, 0);
  }
  glEnd();

  // Draw the circles around the end points
  //  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, width, 0, height);
  //  glMatrixMode(GL_MODELVIEW);
  //  glLoadIdentity();
  glColor3f(1.0, 0.5, 1.0);
  int radius = (width/100)*3;
  for(unsigned int k = 0; k < colors_index.size(); k++) {
    draw_circle(radius, (int)(colors_index[k].x*width),
		(int)(colors_index[k].val*height));
  }
}

// Preconditions:
//   assuming that the data in alpha_transform and alpha_stripes is already
//     allocated.
//   alpha_list.size() >= 2
//   alpha_list[0].x == 0
//   alpha_list[alpha_list.size()-1].x == 1

void ColorMapDpy::create_transfer() {
  // loop over the colors and fill in the slices
  int start = 0;
  int end;
  int nindex = color_transform.size() - 1;
  for (int index = 0; index < (colors_index.size()-1); index++) {
    // the start has already been computed, so you need to figure out
    // the end.
    end = (int)(colors_index[index+1].x * nindex);
    Color color = colors_index[index].c;
    Color color_inc = (colors_index[index+1].c - color) * (1.0f/(end - start));
    float val = colors_index[index].val;
    float val_inc = (colors_index[index+1].val - val) * (1.0f/(end - start));
    for (int i = start; i <= end; i++) {
      //    cout << "val = "<<val<<", ";
      color_transform[i] = color;
      color += color_inc;
      alpha_transform[i] = val;
      val += val_inc;
    }
    start = end;
  }
}

// finds the closest point to x,y in the alpha_list and returns the index
int ColorMapDpy::select_point(int xpos, int ypos) {
  // need to figure out the normalized points
  float x = (xpos - (float)5)/(xres-10);
  float y = (ypos - (float)5)/(yres-10);
  //cout << "norm (x,y) = ("<<x<<", "<<y<<")\n";
  // now loop over the values and find the point closest
  float max_distance = FLT_MAX;
  int index = -1;
  for(unsigned int i = 0; i < colors_index.size(); i++) {
    // we don't really care about the actuall distance, just the relative
    // distance, so we don't have to square root this value.
    float distance = (x - colors_index[i].x) * (x - colors_index[i].x) +
      (y - colors_index[i].val) * (y - colors_index[i].val);
    if (distance < max_distance) {
      // new close point
      max_distance = distance;
      index = i;
    }
  }
  // now to set the min_x and max_x;
  // if the selected point is either the first or last point they need to be
  // made so that they can't move left or right
  float min,max;
  if (index > 0 && index < colors_index.size()-1) {
    // center point
    min = colors_index[index-1].x;
    max = colors_index[index+1].x;
  } else {
    // end point
    min = max = colors_index[index].x;
  }
  min_x =(int)(min*(xres-10)+5);
  max_x =(int)(max*(xres-10)+5);
  //cout << "Closest to point "<<index<<".\n";
  return index;
}

void ColorMapDpy::write_data_file(char *out_file) {
  char buf[1000];
  if (out_file == 0)
    // so that the one read in isn't written over
    sprintf(buf, "%s_new.cmp", filebase);
  else
    sprintf(buf, "%s.cmp", out_file);
  FILE *out = fopen(buf, "wc");
  if(!out){
    cerr << "ColorMapDpy::write_data_file:ERROR - couldn't open " << buf <<
      " for writing.\n";
  }

  for(int i = 0; i < colors_index.size(); i++) {
    ColorPos c = colors_index[i];
    fprintf(out, "%g %g %g %g %g\n",c.x, c.val, c.c.red(), c.c.green(), c.c.blue());
  }
  fclose(out);
}

/*
  Midpoint Circle Algorithm found in Introduction to Computer Graphics

 */
void draw_circle(int radius, int x_center, int y_center) {
  int x,y,d,deltaE,deltaSE;
  x = 0;
  y = radius;
  d = 1 - radius;
  deltaE = 3;
  deltaSE = 5 - radius * 2;
  glBegin(GL_POINTS);
  circle_points(x,y,x_center,y_center);
  while (y > x) {
    if (d < 0) {
      d += deltaE;
      deltaE += 2;
      deltaSE += 2;
      x++;
    } else {
      d += deltaSE;
      deltaE += 2;
      deltaSE += 4;
      x++;
      y--;
    }
    circle_points(x,y,x_center,y_center);
  }
  glEnd();
}

/*
  found in Introduction to Computer Graphics
*/
void circle_points(int x, int y, int x_center, int y_center) {
  glVertex2i(x_center+x,y_center+y);
  glVertex2i(x_center+y,y_center+x);
  glVertex2i(x_center+y,y_center-x);
  glVertex2i(x_center+x,y_center-y);
  glVertex2i(x_center-x,y_center-y);
  glVertex2i(x_center-y,y_center-x);
  glVertex2i(x_center-y,y_center+x);
  glVertex2i(x_center-x,y_center+y);
}

