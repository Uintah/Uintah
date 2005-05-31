
#include <Packages/rtrt/Core/GridSpheresDpy.h>

#include <Packages/rtrt/Core/GridSpheres.h>
#include <Packages/rtrt/Core/FontString.h>
#include <Packages/rtrt/visinfo/visinfo.h>

#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Time.h>

#if defined(__GNUG__) || defined(__ECC)
#  include <sci_values.h>
#endif

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <sgi_stl_warnings_on.h>

#include <stdlib.h>
#include <stdio.h>
#include <X11/keysym.h>

using namespace std;
using namespace rtrt;

using SCIRun::Mutex;
using SCIRun::Thread;

GridSpheresDpy::GridSpheresDpy(int colordata, char *in_file) :
  DpyBase("GridSpheresDpy"),
  hist(0), nhist(10000), scaled_hist(0), ndata(-1),
  colordata(colordata), newcolordata(colordata),
  shade_method(1), new_shade_method(1),
  radius_index(-1), new_radius_index(-1),
  in_file(in_file),
  need_hist(true)
{
  set_resolution(500,500);
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

    // Now create the original histogram
    compute_hist(fontbase);
  }
  cerr << "GridSpheresDpy:setup_vars:end\n";
}

void GridSpheresDpy::init() {
  glShadeModel(GL_FLAT);

  setup_vars();
}

void GridSpheresDpy::display() {
  if(need_hist){
    need_hist=false;
    compute_scaled_hist();
  }

  draw_hist(fontbase, fontInfo);
  glXSwapBuffers(dpy, win);
}

void GridSpheresDpy::resize(const int width, const int height) {
  if (width != xres)
    need_hist = true;
  xres = width;
  yres = height;
  redraw = true;
}

void GridSpheresDpy::key_pressed(unsigned long key) {
  switch(key) {
  case XK_w:
  case XK_W:
    write_data_file("gridspheredpy.cfg");
    break;
  case XK_s:
  case XK_S:
    // The modulous is the number of shade methods
    new_shade_method = (new_shade_method+1)%5;
    cerr<<"Shade method:  ";
    switch (new_shade_method) {
    case 0:
      cerr<<"Color mapping with lambertian shading"<<endl;
      break;
    case 1:
      cerr<<"Constant color with lambertian shading"<<endl;
      break;
    case 2:
      cerr<<"Color mapping with textured luminance"<<endl;
      break;
    case 3:
      cerr<<"Constant color with textured luminance"<<endl;
      break;
    case 4:
      cerr<<"Color mapping with textured luminance as ambient term"<<endl;
      break;
    case 5:
      cerr<<"Constant color with textured luminance as ambient term"<<endl;
      break;
    }
    break;
  case XK_0:
    new_shade_method=0;
    cerr<<"Shade method:  Color mapping with lambertian shading"<<endl;
    break;
  case XK_1:
    new_shade_method=1;
    cerr<<"Shade method:  Constant color with lambertian shading"<<endl;
    break;
  case XK_2:
    new_shade_method=2;
    cerr<<"Shade method:  Color mapping with textured luminance"<<endl;
    break;
  case XK_3:
    new_shade_method=3;
    cerr<<"Shade method:  Constant color with textured luminance"<<endl;
    break;
  case XK_4:
    new_shade_method=4;
    cerr<<"Shade method:  Color mapping with textured luminance as ambient term"<<endl;
    break;
  case XK_5:
    new_shade_method=5;
    cerr<<"Shade method:  Constant color with textured luminance as ambient term"<<endl;
    break;
  }
}

void GridSpheresDpy::button_pressed(MouseButton button,
			     const int x_mouse, const int y_mouse) {
  switch(button) {
  case MouseButton1:
    if (shift_pressed) {
      //cerr << "Left button pressed with shift\n";
      move_min_max(min, x_mouse, y_mouse);
    } else if (control_pressed) {
      //cerr << "Left button pressed with control\n";
      move(new_color_begin, x_mouse, y_mouse);
    } else {
      move(new_range_begin, x_mouse, y_mouse);
    }
    break;
  case MouseButton2:
    if (shift_pressed) {
      //cerr << "Middle button pressed with shift\n";
      restore_min_max(y_mouse);
    } else if (control_pressed) {
      //cerr << "Middle button pressed with control\n";
      move(new_color_begin, 0, y_mouse);
      move(new_color_end, xres, y_mouse);
    } else {
      changecolor(y_mouse);
    }
    break;
  case MouseButton3:
    if (shift_pressed) {
      //cerr << "Right button pressed with shift\n";
      move_min_max(max, x_mouse, y_mouse);
    } else if (control_pressed) {
      //cerr << "Right button pressed with control\n";
      move(new_color_end, x_mouse, y_mouse);
    } else {
      move(new_range_end, x_mouse, y_mouse);
    }
    break;
  }
  redraw = true;
}

void GridSpheresDpy::button_motion(MouseButton button,
			    const int x_mouse, const int y_mouse) {
  if (shift_pressed || control_pressed) return;
  switch(button) {
  case MouseButton1:
    if (control_pressed) {
      //cerr << "Left button pressed with control\n";
      move(new_color_begin, x_mouse, y_mouse);
    } else {
      move(new_range_begin, x_mouse, y_mouse);
    }
    redraw = true;
    break;
  case MouseButton2:
    break;
  case MouseButton3:
    if (control_pressed) {
      //cerr << "Right button pressed with control\n";
      move(new_color_end, x_mouse, y_mouse);
    } else {
      move(new_range_end, x_mouse, y_mouse);
    }
    redraw = true;
    break;
  }
}

void GridSpheresDpy::compute_hist(GLuint fid)
{
  //cerr << "GridSpheresDpy:compute_hist:start\n";
  //  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  glViewport(0, 0, xres, yres);
  glClearColor(0, 0, .2, 1);
  glClear(GL_COLOR_BUFFER_BIT);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, 1, 0, 1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  
  printString(fid, .1, .5, "Recomputing histogram...\n", Color(1,1,1));
  glXSwapBuffers(dpy, win);
  glFlush();
  if (ndata == -1) {
    cerr <<"GridSpheresDpy::compute_hist: GridSpheresDpy::attach not called\n";
    cerr << "GridSpheresDpy::ndata = -1\n";
    return;
  }
  int total=ndata*nhist;
  //cerr << "GridSpheresDpy:compute_hist:ndata = " << ndata << "\n";  
  //cerr << "GridSpheresDpy:compute_hist:xres = " << xres << "\n";  
  //cerr << "GridSpheresDpy:compute_hist:total = " << total << "\n";  
  if (hist)
    delete(hist);
  hist=new int[total];
  for(int i=0;i<total;i++){
    hist[i]=0;
  }
  //
  float* original_scales = new float[ndata];
  for(int j = 0; j < ndata; j++) {
    original_scales[j] = 1.0f/(original_max[j]-original_min[j]);
    //    cerr << "original_scales["<<j<<"] = "<<original_scales[j]<<"\n";
  }
  
  // loop over all the data and compute histograms
  for (int g = 0; g < grids.size() ; g++) {
    GridSpheres* grid = grids[g];
    float* p=grid->spheres;
    int nspheres=grid->nspheres;
    for(int i=0;i<nspheres;i++){
      int offset=0;
      for(int j=0;j<ndata;j++){
	float normalized=(p[j]-original_min[j])*original_scales[j];
	int idx=(int)(normalized*(nhist-1));
        //        cerr << "(i,j) = ("<<i<<", "<<j<<")\n";
        //        cerr << "p[j] = "<<p[j]<<", normalized = "<<normalized<<", nhist-1 = "<<nhist-1<<", idx = "<<idx<<"\n";
	if (idx >= 0 && idx < nhist)
	  hist[offset+idx]++;
#if 0
	if(idx<0 || idx>=nhist){
	  cerr << "p[" << j << "]=" << p[j] << ", min[" << j << "]=" << original_min[j] << ", scales[" << j << "]=" << original_scales[j] << '\n';
	  cerr << "idx=" << idx << '\n';
	  cerr << "idx out of bounds!\n";
	  Thread::exitAll(-1);
	}
	hist[offset+idx]++;
#endif
	offset+=nhist;
      }
      p+=ndata;
    }
  }

  delete[] original_scales;
  
  //cerr << "GridSpheresDpy:compute_hist:past compute histograms\n";
  //cerr << "GridSpheresDpy:compute_hist:end\n";
}

void GridSpheresDpy::draw_hist(GLuint fid, XFontStruct* font_struct)
{
//   cerr << "GridSpheresDpy:draw_hist:start\n";
  int descent=font_struct->descent;
  int textheight=font_struct->descent+font_struct->ascent+2;

  // Rescale the histogram if we need to
  if (xres != nscaled_hist)
    compute_scaled_hist();

  glViewport(0, 0, xres, yres);
  glClearColor(0, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT);

  // Draw the histogram stuff
  int offset=0;
  for(int j=0;j<ndata;j++){
    int s=j*yres/ndata+2;
    int e=(j+1)*yres/ndata-2;
    int h=e-s;
    glViewport(0, s, xres, e-s);

    // Draw the red crop lines
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(min[j], max[j], 0, h);

    if(new_range_end[j] > new_range_begin[j]){
      glColor3f(.5,0,0);
      glRectf(new_range_begin[j], textheight, new_range_end[j], h);
    }
    
    // Draw the coloring lines
    if (new_color_begin[j] != new_color_end[j])
      color_scales[j] = 1./(new_color_end[j]-new_color_begin[j]);
    if (new_color_end[j] > new_color_begin[j]) {
      // Colors the line yellow if it is the current variable being
      // colored.
      if (j == newcolordata)
        glColor3f(1,1,0);
      else
        glColor3f(1,1,1);
      glRectf(new_color_begin[j], textheight-2, new_color_end[j],textheight);
    }
      
    // Draw the histogram lines in blue
    glColorMask(GL_FALSE, GL_FALSE, GL_TRUE, GL_FALSE);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0,xres,-float(textheight)*histmax[j]/(h-textheight),histmax[j]);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
      
    glColor3f(0,0,1);
    glBegin(GL_LINES);
    for(int i=0;i<nscaled_hist;i++){
      glVertex2i(i, 0);
      glVertex2i(i, scaled_hist[offset+i]);
    }
    glEnd();
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    ///////////////////////////////////////////////
    // Print the text on the screen
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(min[j], max[j], 0, h);

    // Print the min and max for the cropped region
    char buf[100];

    // new_range_begin
    sprintf(buf, "%g", new_range_begin[j]);
    int w=calc_width(font_struct, buf);
    float wid=(max[j]-min[j])*w/xres;
    float x=new_range_begin[j]-wid/2.;
    float left=min[j]+(max[j]-min[j])*2/xres;
    if(x<left)
      x=left;
    printString(fid, x, descent+1, buf, Color(1,0,0));

    // new_range_end
    sprintf(buf, "%g", new_range_end[j]);
    w=calc_width(font_struct, buf);
    wid=(max[j]-min[j])*w/xres;
    x=new_range_end[j]-wid/2.;
    float right=max[j]-(max[j]-min[j])*2/xres;
    if(x>right-wid)
      x=right-wid;
    printString(fid, x, descent+1, buf, Color(1,0,0));


    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, xres, 0, h);

    // Print the min and max for the variable
    offset+=nscaled_hist;
    // print the min on the left
    sprintf(buf, "%g", min[j]);
    printString(fid, 2, descent+1, buf, Color(1,1,1));
    // print the variable name in the middle
    if (var_names != 0) {
      int x = (int)((xres - calc_width(font_struct,var_names[j].c_str()))/2);
      printString(fid, x, descent+1, var_names[j].c_str(), Color(1,1,1));
    }
    // print the max on the right
    sprintf(buf, "%g", max[j]);
    w=calc_width(font_struct, buf);
    printString(fid, xres-2-w, descent+1, buf, Color(1,1,1));
  }
  
  glFinish();
  int errcode;
  while((errcode=glGetError()) != GL_NO_ERROR){
    cerr << window_name << " got an error from GL: " << (char*)gluErrorString(errcode) << endl;
  }
  //cerr << "GridSpheresDpy:draw_hist:end\n";
}

// This scales the histograms to match the min and max of the range
// as well as the horizontal resolution.
void GridSpheresDpy::compute_scaled_hist() {
  if (xres < 1)
    // This isn't a case we should handle.
    return;
  
  // Figure out how many bins we need now
  if (nscaled_hist != xres || scaled_hist == 0) {
    nscaled_hist = xres;
    //cerr << "GridSpheresDpy:compute_hist:ndata = " << ndata << "\n";  
    //cerr << "GridSpheresDpy:compute_hist:xres = " << xres << "\n";  
    //cerr << "GridSpheresDpy:compute_hist:total = " << total << "\n";
    
    // Delete the old one
    if (scaled_hist)
      delete(scaled_hist);
    // Allocate and initialize the new histogram
    scaled_hist = new int[ndata*nscaled_hist];
  }
  // Initialize the histogram values to zero to zero
  int total=ndata * nscaled_hist;
  for(int i=0;i<total;i++){
    scaled_hist[i]=0;
  }
  // For each current histogram
  int* ho = hist;
  int* sh = scaled_hist;
  for(int j = 0; j < ndata; j++) {
    for(int i = 0; i < nhist; i++) {
      // Figure out which value ho[i] maps to
      float val = (float)i*(original_max[j]-original_min[j])/(nhist-1.0f)
                  + original_min[j];
      // Use val to lookup the bin of the scaled histogram
      float normalized = (val-min[j])*scales[j];
      int idx = (int)(normalized*(nscaled_hist-1));
//       if (ho[i] > 0) {
//         cerr << "val = "<<val<<", (i,j) = ("<<i<<", "<<j<<")\n";
//         cerr << "idx = "<<idx<<", ho["<<i<<"] = "<<ho[i]<<"\n";
//       }
      if (idx >= 0 && idx < nscaled_hist)
        sh[idx] += ho[i];
    }
    ho += nhist;
    sh += nscaled_hist;
  }

  compute_histmax();
}

void GridSpheresDpy::compute_histmax() {
  // determine the maximum height for each histogram
  int* hp=scaled_hist;
  for(int j=0;j<ndata;j++){
    int max=0;
    for(int i=0;i<nscaled_hist;i++){
      if(*hp>max)
	max=*hp;
      hp++;
    }
    histmax[j]=max;
    //    cerr << "histmax["<<j<<"] = "<<histmax[j]<<"\n";
  }
}

void GridSpheresDpy::move(float* range, int x, int y)
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
      break;
    }
  }
}

void GridSpheresDpy::move_min_max(float* range, int x, int y)
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
      compute_scaled_hist();
      //compute_one_hist(j);
      
      break;
    }
  }
}

void GridSpheresDpy::restore_min_max(int y) {
  // swap y=0 orientation
  y=yres-y;
  // loop over each block and see where the event was
  for(int j = 0; j < ndata; j++){
    int s=j*yres/ndata +2;
    int e=(j+1)*yres/ndata-2;
    if(y>=s && y<e){
      // found the region the event was
      // reset the min and max
      min[j] = original_min[j];
      max[j] = original_max[j];

      // update scales
      scales[j] = 1./(max[j]-min[j]);

      // now we need to recompute the histogram for this range
      compute_scaled_hist();
      //      compute_one_hist(j);
      
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
      float normalized=(p[j]-original_min[j])*scales[j];
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

// If ndata is set then we have at least one grid attached
void GridSpheresDpy::set_radius_index(int new_ri) {
  // Negative values are no no's.
  if (new_ri < 0) return;

  // Only check the max if ndata has been set
  if (ndata > 0) {
    if (new_ri < ndata)
      new_radius_index = new_ri;
  } else {
    radius_index = new_ri;
    new_radius_index = new_ri;
  }
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
  if (new_shade_method != shade_method) {
    changed=true;
    shade_method = new_shade_method;
  }
  if (new_radius_index != radius_index) {
    changed=true;
    radius_index = new_radius_index;
  }
}

void GridSpheresDpy::attach(GridSpheres* g) {
  // ndata only equals -1 if ndata has not been set 
  if (ndata != -1) {
    if (ndata == g->ndata) {
      grids.add(g);
      g->dpy = this;
    }
    else {
      cerr << "GridSpheresDpy::attach: ERROR! ndata of the new GridSpheres ("<<g->ndata<<") does not equal that of the display's ("<<ndata<<")\n";
      cerr << "Not adding this GridSpheres\n";
      g->dpy = 0;
    }
  } else {
    // This is the first GridSpheres being added.
    ndata = g->ndata;
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
      redraw = true;
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


