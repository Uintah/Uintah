/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 * This is basicaly a ColorMap, just it works...
 * Only handles diffuse colors for now (rest is constant)
 * Peter-Pike Sloan
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Core/Geom/ColorMap.h>
#include <Dataflow/Ports/GeometryPort.h>

#include <Core/Math/CatmullRomSpline.h>
#include <Core/Malloc/Allocator.h>

#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/Color.h>
#include <Core/Math/MinMax.h>
#define Colormap XColormap

#include <Core/Geom/GeomOpenGL.h>
#include <GL/gl.h>
#include <sci_glu.h>
#include <GL/glx.h>

#include <tcl.h>
#include <tk.h>

#undef Colormap

#include <iostream>
#include <algorithm> 
#include <stdio.h>

using std::sort;
// tcl interpreter corresponding to this module

extern Tcl_Interp* the_interp;

// the OpenGL context structure

extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);

namespace SCIRun {

struct GTF_HSV {
  float hue;
  float sat;
  float val;
};
struct ColorPoint {
  float  _t;  // time - 
  Color _rgb;
  HSVColor _hsv;  // you have both reps...
};
struct AlphaPoint {
  float _t;
  float _alpha;
};

class GenTransferFunc : public Module {

  ColorMapIPort         *inport;  // input a ColorMap
  ColorMapOPort         *outport;  // outputs a ColorMap
  GeometryOPort         *ogeom;  
  GuiInt		RGBorHSV; // which mode
  GuiInt		lineVSspline; // linear vs. spline interpolate

//   Array1< ColorPoint > points;   // actual line(s)
  vector< ColorPoint > rgb_points;   // actual line(s)
  vector< ColorPoint > hsv_points;
  vector< ColorPoint > hsv_outpts;
  
  vector< AlphaPoint > alphas;
  //vector< float >	alphas;   // alpha polyline
  //  vector< float >       aTimes;   // alpha times

  Array1< Point >       otherGraph; // xyz to RGB or HSV

  int			activeLine; // active RGB,HSV,alpha,-1
  int			selNode;    // selected node
  int                   graphStat;  // status of otherGraph

  int			bdown;      // wether mouse button is down
  int 			whichWin;   // which window is selected

  int			winX[3],winY[3];  // integer size
  float			winDX[2],winDY[2]; // scales (so it's square)

  GLXContext		ctxs[3];    // OpenGL Contexts
  Display		*dpy[3];
  Window		win0;
  Window		win1;
  Window		win2;

  ColorMapHandle	cmap;  // created once, first execute...
  int cmap_generation;

  void updateViewHSV();
  void loadTs( double Ax, double Ay, double ax, double ay, vector<double>& t);
public:
  GenTransferFunc( GuiContext* ctx);

  void DrawGraphs(int flush=1); // this function just blasts away...

  void DoMotion(int win, int x, int y);

  void DoDown(int win, int x, int y, int button);
  void DoRelease(int win, int x, int y, int button);

  float GetTime(int x, int) 
    { 
      float v = x/(1.0*winX[whichWin]);
      if (v > 1.0) v = 1.0;	
      if (v < 0.0) v = 0.0;
      return v;
    }
  float GetVal(int, int y)
    { 
      float v= (winY[whichWin]-y)/(1.0*winY[whichWin]); 
      if (v > 1.0) v = 1.0;	
      if (v < 0.0) v = 0.0;
      return v;
    }

  void GetClosest(float time, float val, int& cline, int& cpoint);

  void Resize(int win);

  virtual ~GenTransferFunc();
  virtual void execute();
  void tcl_command( GuiArgs&, void* );

  int makeCurrent(void);


};

DECLARE_MAKER(GenTransferFunc)
GenTransferFunc::GenTransferFunc( GuiContext* ctx)
  : Module("GenTransferFunc",ctx,Source, "Visualization", "SCIRun"),
    inport(0), outport(0), ogeom(0),
    RGBorHSV(ctx->subVar("rgbhsv")),lineVSspline(ctx->subVar("linespline")),
    activeLine(-1),selNode(-1),graphStat(-1),bdown(-1),whichWin(-1),
    cmap_generation(-1)
{

  cmap = scinew ColorMap; // start as nothing...
  
  // initialize the transfer function

  ColorPoint p0, p1;

  p0._t = 0.0;
  p0._rgb = Color(0,.05,.1);
//   p0._hsv = HSVColor(p0._rgb);

  rgb_points.push_back(p0);

  p0._t = 1.0;
  p0._rgb = Color(1,.95,.9);
//   p0._hsv = HSVColor(p0._rgb);

  rgb_points.push_back(p0);

  // we want a similar transfer function for the HSV points
  p0._t = 0.0;
  p0._rgb = Color(0,.05, .1);
//  p0._hsv = HSVColor(p0._rgb);
  p1._t = 1.0;
  p1._rgb = Color(1.0,.95,.9);
//   p1._hsv = HSVColor(p1._rgb);
  hsv_points.push_back(p0);
  hsv_points.push_back(p1);
  
  updateViewHSV();

  AlphaPoint a;
  a._t = 0; a._alpha = 0;
  alphas.push_back(a);
  a._t = .25; 
  alphas.push_back(a);
  a._t = .5; a._alpha = .5;
  alphas.push_back(a);
  a._t = .75; a._alpha = .8;
  alphas.push_back(a);
  a._t = 1; a._alpha = .8;
  alphas.push_back(a);

//   alphas.push_back(0);
//   alphas.push_back(0);
//   alphas.push_back(.5);
//   alphas.push_back(.8);
//   alphas.push_back(.8);
//   aTimes.push_back(0);
//   aTimes.push_back(.25);
//   aTimes.push_back(.5);
//   aTimes.push_back(.75);
//   aTimes.push_back(1);


  ctxs[0] = ctxs[1] = ctxs[2] = 0;
}

GenTransferFunc::~GenTransferFunc()
{

}

void 
GenTransferFunc::loadTs( double Ax, double Ay, double ax, double ay, vector<double>& t)
{
  double ys[11] = { 0.8333333333, 0.1666666666, 0.249999999,
		    0.3333333332, 0.4166666665, 0.499999998,
		    0.5833333331, 0.6666666664, 0.749999997,
		    0.8333333333, 0.9166666663 };

  int i;
  for( i = 0; i < 11; i++ ){
    double Bx = 0; 
    double By = ys[i];
    double cx = Bx - Ax;
    double cy = By - Ay;
    double bx_ = 0;
    double by_ = 1;
    if(bx_ * ax + by_ * ay > 0 ) {
      t.push_back((bx_ * cx + by_ * cy)/(bx_ * ax + by_ * ay));
    }
  }
  sort(t.begin(), t.end()); 
}
      
    
void GenTransferFunc::updateViewHSV(void)
{
//   vector< ColorPoint > hsv_points;
//   vector< ColorPoint > hsv_outpts;

  if( hsv_points.size() < 2 ) return;
  unsigned int i,j;
  hsv_outpts.clear();
  ColorPoint newpt;
  vector<double> ts;
  for( j = 0; j <  hsv_points.size() - 1 ; j++){
    double Ax = hsv_points[j]._t;
    double Ay = hsv_points[j]._rgb[0];

    double ax = hsv_points[j+1]._t - Ax;
    double ay = hsv_points[j+1]._rgb[0] - Ay;
    loadTs( Ax, Ay, ax, ay, ts );
    hsv_outpts.push_back( hsv_points[j] );
    for(i = 0; i < ts.size(); i++){
      if( ts[i] > 0.0 && ts[i] < 1.0 ){
	double time = Ax + ts[i] * ax;
	double hx   = Ay + ts[i] * ay;
	
	double sx = hsv_points[j]._rgb[1] + 
	  (time  - hsv_points[j]._t)/(hsv_points[j+1]._t - hsv_points[j]._t) *
	  (hsv_points[j+1]._rgb[1] - hsv_points[j]._rgb[1]);
	double vx = hsv_points[j]._rgb[2] + 
	  (time  - hsv_points[j]._t)/(hsv_points[j+1]._t - hsv_points[j]._t) *
	  (hsv_points[j+1]._rgb[2] - hsv_points[j]._rgb[2]);
	newpt._t = time;
	newpt._rgb = Color( hx, sx, vx);
	hsv_outpts.push_back( newpt );
      }
    }
  }
  hsv_outpts.push_back( hsv_points[hsv_points.size() - 1] );
}



// draws/updates the graphs...

void drawBox(float x, float y, float dx, float dy)
{
  glBegin(GL_LINE_LOOP);
  glVertex2f(x-dx,y-dy);
  glVertex2f(x-dx,y+dy);
  glVertex2f(x+dx,y+dy);
  glVertex2f(x+dx,y-dy);
  glEnd();
}

void GenTransferFunc::Resize(int win)
{

  // do a make current...

  if ( ! makeCurrent() )
    return;

  switch (win) {
  case 0:
    glXMakeCurrent(dpy[win],win0,ctxs[win]);
    break;
  case 1:
    glXMakeCurrent(dpy[win],win1,ctxs[win]);
    break;
  case 2:
    glXMakeCurrent(dpy[win],win2,ctxs[win]);
    break;
  }

  if (win < 2) {  // DX DY only for first 2...
    int sx,sy;
    sx = winX[win];
    sy = winY[win];

    winDX[win] = 1.0/100.0;  // X is normaly bigger...

    winDY[win] = sx/sy*winDX[win];
  }

  // make current...

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(-1,-1,0);
  glScalef(2,2,2);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

//  gluOrtho2D(0,winX[win],0,winY[win]);
  

  glEnable(GL_COLOR_MATERIAL);  // more state?
  glColorMaterial(GL_FRONT_AND_BACK,GL_DIFFUSE);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glDisable(GL_TEXTURE_2D);

  glViewport(0,0,winX[win],winY[win]);
  glClear(GL_COLOR_BUFFER_BIT);

  gui->unlock();

}

void GenTransferFunc::DrawGraphs( int flush)
{

  static double  colors[] = { 1.,0.,0.,   0.,1.,0.,   0.,0.,1. };

  if (!makeCurrent())
    return; // don't care if you can't make current...


  glXMakeCurrent(dpy[0],win0,ctxs[0]);
  // other state should be fine...

  glDrawBuffer(GL_BACK);
  glClear(GL_COLOR_BUFFER_BIT);
  unsigned int i,j;
  if(RGBorHSV.get() ){
    for(i=0;i<3;i++) {
      float mul=0.5;
      if (activeLine == (int) i*2)
	mul = 1.0;
      glBegin(GL_LINE_STRIP);
      for(j=0;j<hsv_outpts.size();j++) {
	HSVColor hsv; 
	Color rgb;
	double val;
	switch ( i ) {
	case 0:
	  val = hsv_outpts[j]._rgb[i]*720;
	  if( val > 360 ) val = val - 360;
	  hsv = HSVColor( val, 1.0, 1.0);
	  rgb = Color( hsv );
	  break;
	case 1:
	  hsv = HSVColor( 240, hsv_outpts[j]._rgb[i], 1.0);
	  rgb = Color( hsv );
	  break;
	case 2:
	  hsv = HSVColor( 0.0, 0.0, hsv_outpts[j]._rgb[i]);
	  rgb = Color( hsv );
	  break;
	}
	glColor3f(rgb.r()*mul, rgb.g()*mul, rgb.b()*mul);
	glVertex2f(hsv_outpts[j]._t,hsv_outpts[j]._rgb[i]);
      }
      glEnd();
    }
  } else {
    for(i=0;i<3;i++) {
      float mul=0.5;
      if (activeLine == (int) i*2)
	mul = 1.0;
      glColor3f(colors[i*3 + 0]*mul,colors[i*3 + 1]*mul,colors[i*3 + 2]*mul);
      glBegin(GL_LINE_STRIP);
      for(j=0;j<rgb_points.size();j++) {
	glVertex2f(rgb_points[j]._t,rgb_points[j]._rgb[i]);
      }
      glEnd();
    }
  }

  // now draw the alpha curve...

  float mul = 0.5;
  if (activeLine == 7)
    mul = 0.7;

  glColor3f(mul,mul,mul);
  glBegin(GL_LINE_STRIP);
  
  for(i=0;i<alphas.size();i++) {
//     glVertex2f(aTimes[i],alphas[i]);
    glVertex2f(alphas[i]._t,alphas[i]._alpha);
  }
  glEnd();

  for(i=0;i<alphas.size();i++) {
    mul = 0.5;
    if ((int)i == selNode)
      mul = 0.7;
    glColor3f(mul,mul,mul);
//     drawBox(aTimes[i],alphas[i],winDX[0],winDY[0]);
    drawBox(alphas[i]._t,alphas[i]._alpha,winDX[0],winDY[0]);
  }

  // now the polylines have been draw, draw the points...
  if( RGBorHSV.get() ){
    for(i=0;i<3;i++) {
      for(j=0;j<hsv_points.size();j++) {
	float mul = 0.6;
	if (selNode == (int)j*2)
	  mul = 1.0;
	HSVColor hsv; 
	Color rgb;
	double val;
	switch (i ) {
	case 0:
	  val = hsv_points[j]._rgb[i]*720;
	  if( val > 360 ) val = val - 360;
	  hsv = HSVColor( val, 1.0, 1.0);
	  rgb = Color( hsv );
	  break;
	case 1:
	  hsv = HSVColor( 240, hsv_points[j]._rgb[i], 1.0);
	  rgb = Color( hsv );
	  break;
	case 2:
	  hsv = HSVColor( 0.0, 0.0, 1.0);
	  rgb = Color( hsv );
	  break;
	}
	glColor3f(rgb.r()*mul, rgb.g()*mul, rgb.b()*mul);
	drawBox(hsv_points[j]._t,hsv_points[j]._rgb[i],winDX[0],winDY[0]);
      }
    }
  }else {
    for(i=0;i<3;i++) {
      for(j=0;j<rgb_points.size();j++) {
	float mul = 0.6;
	if (selNode == (int)j*2)
	  mul = 1.0;
	
	glColor3f(colors[i*3 + 0]*mul,colors[i*3 + 1]*mul,colors[i*3 + 2]*mul);
	drawBox(rgb_points[j]._t,rgb_points[j]._rgb[i],winDX[0],winDY[0]);
      }
    }
  }

  glXSwapBuffers(dpy[0],win0);

  // clear the middle window (why not?)-- Because it doesn't do anything KZ
// ****** Temporary comment out of HSV Window ******  
//   glXMakeCurrent(dpy[1],win1,ctxs[1]);
//   glDrawBuffer(GL_BACK);
//   glClear(GL_COLOR_BUFFER_BIT);
//   glXSwapBuffers(dpy[1],win1);

  // go to the last window...

  glXMakeCurrent(dpy[2],win2,ctxs[2]);
  glDrawBuffer(GL_BACK);
  glClear(GL_COLOR_BUFFER_BIT);

  // 1st lay down the "pattern"

  glDisable(GL_BLEND);

  glShadeModel(GL_FLAT);
  for (j=0;j<32; j++) {
    glBegin(GL_QUAD_STRIP);
    for(i=0;i<5;i++) {
      int on=0;
      if ((i&1 && j&1) || (!(i&1) && (!(j&1)))) on=1;
      glColor3f(on, on, on);
      glVertex2f(j/31.0,i/4.0);
      glVertex2f((j+1)/31.0,i/4.0);
    }
    glEnd();
  }
  glShadeModel(GL_SMOOTH);
  glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_TRUE);
  
  glBegin(GL_QUAD_STRIP);
  for(i=0;i<alphas.size();i++) {
//     glColor4f(0,0,0,alphas[i]);
//     glVertex2f(aTimes[i],0);
//     glVertex2f(aTimes[i],1);
    glColor4f(0,0,0,alphas[i]._alpha);
    glVertex2f(alphas[i]._t,0);
    glVertex2f(alphas[i]._t,1);
  }
  glEnd();

  glBlendFunc(GL_DST_ALPHA,GL_ONE_MINUS_DST_ALPHA);
  glEnable(GL_BLEND);
  glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);

  glBegin(GL_QUAD_STRIP);
  if( RGBorHSV.get() ) {
    for(j=0;j<hsv_outpts.size();j++) {
      double h, s, v;
      Color rgb;
      h = hsv_outpts[j]._rgb[0] * 720;
      s = hsv_outpts[j]._rgb[1];
      v = hsv_outpts[j]._rgb[2];
      if (h > 360) h = h-360;
      rgb = Color( HSVColor( h, s, v ) );
      glColor3f(rgb.r(), rgb.g(), rgb.b());
      glVertex2f(hsv_outpts[j]._t,0);
      glVertex2f(hsv_outpts[j]._t,1);
    }
  } else {
    for(j=0;j<rgb_points.size();j++) {
      glColor3f(rgb_points[j]._rgb[0],rgb_points[j]._rgb[1],
		rgb_points[j]._rgb[2]);
      glVertex2f(rgb_points[j]._t,0);
      glVertex2f(rgb_points[j]._t,1);
    }
  }
  glEnd();
  glXSwapBuffers(dpy[2],win2);

  
  // here you update this ColorMap thing...
  
  if (cmap.get_rep()) {
    vector<Color> ncolors((RGBorHSV.get() ?
			   hsv_outpts.size(): rgb_points.size()));
    vector<float> times((RGBorHSV.get() ?
			   hsv_outpts.size(): rgb_points.size()));
    vector<float> a(alphas.size());
    vector<float> t(alphas.size());

    if( RGBorHSV.get() ) {
      for(i=0;i<hsv_outpts.size();i++) {
	double h, s, v;
	h = hsv_outpts[i]._rgb[0] * 720;
	s = hsv_outpts[i]._rgb[1];
	v = hsv_outpts[i]._rgb[2];
	if (h > 360) h = h-360;
	ncolors[i] = Color( HSVColor( h, s, v ) );
	times[i] = hsv_outpts[i]._t;
      }
    } else {
      for(i=0;i<rgb_points.size();i++) {
	ncolors[i] = rgb_points[i]._rgb;
	times[i] = rgb_points[i]._t;
      }
    }

    for (i=0; i < alphas.size(); i++){
      a[i] = alphas[i]._alpha;
      t[i] = alphas[i]._t;
    }
    
//     cmap.get_rep()->SetRaw(ncolors,times,alphas,aTimes,256);
    cmap.get_rep()->SetRaw(ncolors,times,a,t);

    glXMakeCurrent(dpy[0],None,NULL);

    gui->unlock();
    if ( flush && ogeom ){
      ogeom->flushViews();
      want_to_execute();
    }
    return;

  }

  glXMakeCurrent(dpy[0],None,NULL);

  gui->unlock();
}

// this is how the gui talks with this thing

void GenTransferFunc::tcl_command( GuiArgs& args, void* userdata)
{
  if (args.count() < 2) {
    args.error("No command for GenTransferFunc");
    return;
  }

  // mouse commands are motion, down, release

  if (args[1] == "mouse") {
    int x,y,whichw,whichb;
    
    string_to_int(args[4], x);
    string_to_int(args[5], y);

    string_to_int(args[2], whichw); // which window it was in

    if (args[3] == "motion") {
      if (bdown == -1) // not buttons down!
	return; 
      DoMotion(whichw,x,y);
    } else {
      string_to_int(args[6], whichb); // which button it was
      if (args[3] == "down") {
	DoDown(whichw,x,y,whichb);
      } else { // must be release
	DoRelease(whichw,x,y,whichb);
      }
    }
  } else  if (args[1] == "resize") { // resize event!
    int whichwin;
    string_to_int(args[4], whichwin);
    string_to_int(args[2], winX[whichwin]);
    string_to_int(args[3], winY[whichwin]);
    Resize(whichwin);  // kick of the resize function...
  } else if (args[1] == "expose") {
    int whichwin;
    string_to_int(args[2], whichwin);
    Resize(whichwin); // just sets up OGL stuff...
    DrawGraphs();
  }else if(args[1] == "closewindow") {
     ctxs[0] = ctxs[1] = ctxs[2] = 0;
  }else {
    Module::tcl_command(args, userdata);
  }
}

// you just have to find the time interval first
// then look at the values...

void GenTransferFunc::
GetClosest(float time, float val, int& cline, int& cpoint)
{
  unsigned int i;
  
  float minP=10000;

  float d1,d2;
  if( !RGBorHSV.get() ){
    for(i=0;i<rgb_points.size() && (rgb_points[i]._t <= time); i++) {
      ; // do nothing - just getting index...
    }

    // now i is the value 1 greater...

    if (i == rgb_points.size()) {
      i = rgb_points.size() -1;
    } 

    // now find the closest point


    d1 = time - rgb_points[i-1]._t;  d1 = d1*d1;
    d2 = rgb_points[i]._t - time;  d2 = d2*d2;
    float dy1,dy2;
    for(int j=0;j<3;j++) {
      dy1 = rgb_points[i-1]._rgb[j] - val;
      dy2 = rgb_points[i]._rgb[j]-val;
    
      float min1 = d1 + dy1*dy1;
      float min2 = d2 + dy2*dy2;
    
      if (min1 < minP) {
	minP = min1;
	cpoint = i-1;
	cline = j*2;
      }
      if (min2 < minP) {
	minP = min2;
	cpoint = i;
	cline = j*2;
      }
    }
  } else { // HSV space...
    //    error("HSV window doesn't exist.");
    for(i=0;i<hsv_points.size() && (hsv_points[i]._t <= time); i++) {
      ; // do nothing - just getting index...
    }

    // now i is the value 1 greater...

    if (i == hsv_points.size()) {
      i = hsv_points.size() -1;
    } 

    // now find the closest point
    d1 = time - hsv_points[i-1]._t;  d1 = d1*d1;
    d2 = hsv_points[i]._t - time;  d2 = d2*d2;
    float dy1,dy2;
    for(int j=0;j<3;j++) {
      dy1 = hsv_points[i-1]._rgb[j] - val;
      dy2 = hsv_points[i]._rgb[j]-val;
    
      float min1 = d1 + dy1*dy1;
      float min2 = d2 + dy2*dy2;
    
      if (min1 < minP) {
	minP = min1;
	cpoint = i-1;
	cline = j*2+1;
      }
      if (min2 < minP) {
	minP = min2;
	cpoint = i;
	cline = j*2+1;
      }
    }
  }

  // now check alpha...

//   for(i=0;i<alphas.size() && (aTimes[i] <= time); i++) {
  for(i=0;i<alphas.size() && (alphas[i]._t <= time); i++) {
    ; // do nothing - just getting index...
  }

  // now i is the value 1 greater...

  if (i == alphas.size()) {
    i = alphas.size() -1;
  } 


//   d1 = time - aTimes[i-1];  d1 = d1*d1;
//   d2 = aTimes[i] - time;  d2 = d2*d2;
  d1 = time - alphas[i-1]._t;  d1 = d1*d1;
  d2 = alphas[i]._t - time;  d2 = d2*d2;

//   d1 += (alphas[i-1]-val)*(alphas[i-1]-val);
//   d2 += (alphas[i]-val)*(alphas[i]-val);
  d1 += (alphas[i-1]._alpha-val)*(alphas[i-1]._alpha-val);
  d2 += (alphas[i]._alpha-val)*(alphas[i]._alpha-val);
  
  if (d1 < d2) {
    if (d1 < minP) {
      cline = 7;
      cpoint = i-1;
    }
  } else {
    if (d2 < minP) {
      cline = 7;
      cpoint = i;
    }
  }

  // now it should be ok...

}	

// button must be down

void GenTransferFunc::DoMotion(int, int x, int y)
{
  if ((selNode == -1) || (activeLine == -1))  // this shouldn't happen!
    return;

  // this means you have to move the selected node...

  float time = GetTime(x,y); // get the valid time value...

  float val = GetVal(x,y);   // this remaps the Y coord!

  // End conditions are special cases - can't change x!

  if (activeLine == 7) {
    if (selNode && selNode < (int)alphas.size()-1) {
//       if (aTimes[selNode-1] > time)
// 	time = aTimes[selNode-1];
//       if (aTimes[selNode+1] < time)
// 	time = aTimes[selNode+1];
      if (alphas[selNode-1]._t > time)
	time = alphas[selNode-1]._t;
      if (alphas[selNode+1]._t < time)
	time = alphas[selNode+1]._t;
    } else {
//       time = aTimes[selNode];
      time = alphas[selNode]._t;
    }
//     aTimes[selNode] = time;
    alphas[selNode]._t = time;
  } else {
    if( RGBorHSV.get() ) { // HSV
      if (selNode && selNode < (int)hsv_points.size()-1) {
	if (hsv_points[selNode-1]._t > time)
	  time = hsv_points[selNode-1]._t;
	if (hsv_points[selNode+1]._t < time)
	  time = hsv_points[selNode+1]._t;
      } else {
	time = hsv_points[selNode]._t;
      }
      hsv_points[selNode]._t = time;
    } else {
      if (selNode && selNode < (int)rgb_points.size()-1) {
	if (rgb_points[selNode-1]._t > time)
	  time = rgb_points[selNode-1]._t;
	if (rgb_points[selNode+1]._t < time)
	  time = rgb_points[selNode+1]._t;
      } else {
	time = rgb_points[selNode]._t;
      }
      rgb_points[selNode]._t = time;
    }
  }
  // now update this guy

  if (activeLine < 6) {
    if (activeLine&1) { //HSV
      hsv_points[selNode]._rgb[activeLine>>1] = val;
      updateViewHSV();
    } else {
      rgb_points[selNode]._rgb[activeLine>>1] = val;
//       rgb_points[selNode]._hsv = HSVColor(rgb_points[selNode]._rgb);
    }
  } else { // it is the alpha value...
//     alphas[selNode] = val;
    alphas[selNode]._alpha = val;
  }



  // now you just have to redraw this

  DrawGraphs(0);

}

// Button 1 is modify, 2 is insert, 3 is delete

void GenTransferFunc::DoDown(int win, int x, int y, int button)
{
  // you have to find the point to select...

  float time,val;
  int cline,cpoint; // closest info...

  time = GetTime(x,y);
  val  = GetVal(x,y);

  whichWin = win;  // this has to be pre closest selection...
  GetClosest(time,val,cline,cpoint);
  
  ColorPoint p;

  if (button == 2) {
    // you have to find what side of the closest point
    // you are on...

    activeLine = cline;
    
    p._t = time;
    
    p._rgb[cline>>1] = val;
    p._rgb[((cline>>1)+1)%3] = 0;
    p._rgb[((cline>>1)+2)%3] = 1;
    
//     p._hsv = HSVColor(p._rgb);

    if (cline != 7) {
      if(!RGBorHSV.get()){  // rgb
	if (cpoint && cpoint != (int)rgb_points.size()-1) {
	  if (rgb_points[cpoint]._t <= time)
	    cpoint = cpoint+1;
	
	} else {
	  if (!cpoint) {
	    cpoint = 1;
	  }
	}

	// place the added points for the two other curves exactly on their 
	// existing segments
	double dtime1, dtime2, dsumtime;
	dtime1=p._t-rgb_points[cpoint-1]._t;
	dtime2=rgb_points[cpoint]._t-p._t;
	double w1, w2;
	w1=w2=.5;
	dsumtime=dtime1+dtime2;
	if (dsumtime) {   // if the points are vertical, just put it midway
	  w1=dtime2/dsumtime;
	  w2=dtime1/dsumtime;
	}
	int idx;
	idx=((cline>>1)+1)%3;
	p._rgb[idx]=rgb_points[cpoint-1]._rgb[idx]*w1+rgb_points[cpoint]._rgb[idx]*w2;
	idx=((cline>>1)+2)%3;
	p._rgb[idx]=rgb_points[cpoint-1]._rgb[idx]*w1+rgb_points[cpoint]._rgb[idx]*w2;
	rgb_points.insert(rgb_points.begin() + cpoint,p);
	  
      } else { // HSV
	if (cpoint && cpoint != (int)hsv_points.size()-1) {
	  if (hsv_points[cpoint]._t <= time)
	    cpoint = cpoint+1;
	
	} else {
	  if (!cpoint) {
	    cpoint = 1;
	  }
	}

	// place the added points for the two other curves exactly on their 
	// existing segments
	double dtime1, dtime2, dsumtime;
	dtime1=p._t-hsv_points[cpoint-1]._t;
	dtime2=hsv_points[cpoint]._t-p._t;
	double w1, w2;
	w1=w2=.5;
	dsumtime=dtime1+dtime2;
	if (dsumtime) {   // if the points are vertical, just put it midway
	  w1=dtime2/dsumtime;
	  w2=dtime1/dsumtime;
	}
	int idx;
	idx=((cline>>1)+1)%3;
	p._rgb[idx]=hsv_points[cpoint-1]._rgb[idx]*w1+hsv_points[cpoint]._rgb[idx]*w2;
	idx=((cline>>1)+2)%3;
	p._rgb[idx]=hsv_points[cpoint-1]._rgb[idx]*w1+hsv_points[cpoint]._rgb[idx]*w2;
	hsv_points.insert(hsv_points.begin() + cpoint,p);
	updateViewHSV();
	
      }
    } else { // this is the alpha curve...
      if (cpoint && cpoint != (int)alphas.size()-1) {
// 	if (aTimes[cpoint] <= time)
	if (alphas[cpoint]._t <= time)
	  cpoint = cpoint+1;
      }	
      else {
	if (!cpoint) {
	  cpoint = 1;
	}
      }
      AlphaPoint a;
      a._t = time;
      a._alpha = val;
      alphas.insert(alphas.begin() + cpoint, a);
//       alphas.insert(alphas.begin() + cpoint, val);
//       aTimes.insert(aTimes.begin() + cpoint, time);
    }

  }

  selNode = cpoint;
  activeLine = cline;
  
  bdown = button;

  DrawGraphs(); // just update the display (diferent box for selected)

}

void GenTransferFunc::DoRelease(int win, int x, int y, int button)
{
  // just fake a move for the final posistion...

  DoMotion(win,x,y);

  switch(button) {
  case 1:
  case 2:
    // do nothing - just move/add...
    break;
  case 3:
    if (activeLine == 7) {
      if (selNode && (selNode != (int)alphas.size()-1)) {
	alphas.erase(alphas.begin() + selNode);
// 	aTimes.erase(alphas.begin() + selNode);
      }

    }  else {
      if( RGBorHSV.get() ) {  // HSV
	if (selNode && (selNode != (int)hsv_points.size()-1))
//	rgb_points.remove(selNode);
	hsv_points.erase(hsv_points.begin() + selNode);
	updateViewHSV();
      } else {
	
	if (selNode && (selNode != (int)rgb_points.size()-1))
	  //	rgb_points.remove(selNode);
	  rgb_points.erase(rgb_points.begin() + selNode);
      }
    }
    break;
      
  }

  selNode = -1; // leave activeLine...
  bdown = -1;

  DrawGraphs();
}

void
GenTransferFunc::execute(void)
{
  inport = (ColorMapIPort *)get_iport("ColorMap");
  outport = (ColorMapOPort *)get_oport("ColorMap");
  ogeom = (GeometryOPort *)get_oport("Geometry");
  ColorMapHandle newcmap;

  if (!inport) {
    error("Unable to initialize iport 'ColorMap'.");
    return;
  }
  if (!outport) {
    error("Unable to initialize oport 'ColorMap'.");
    return;
  }
  if (!ogeom) {
    error("Unable to initialize oport 'Geometry'.");
    return;
  }

  unsigned int i;
  
  int c = inport->get(newcmap);

  if ( c && newcmap->generation != cmap_generation )
  {
    cmap_generation = newcmap->generation;

    if (!newcmap->rawRGBA_)
    {
      error("Old version of color map!");
      return;
    }

    cmap.get_rep()->SetRaw(newcmap->rawRampColor_,
			   newcmap->rawRampColorT_,
			   newcmap->rawRampAlpha_,
			   newcmap->rawRampAlphaT_);

    rgb_points.resize(newcmap->rawRampColor_.size());
    
    for(i=0;i<rgb_points.size();i++){
      rgb_points[i]._t = newcmap->rawRampColorT_[i];
      rgb_points[i]._rgb = newcmap->rawRampColor_[i];
    }

    alphas.resize(newcmap->rawRampAlpha_.size());
    for(i = 0; i < alphas.size(); i++ ){
      alphas[i]._t = newcmap->rawRampAlphaT_[i];
      alphas[i]._alpha = newcmap->rawRampAlpha_[i];
    }
//     alphas = newcmap->rawRampAlpha;
//     aTimes = newcmap->rawRampAlphaT;
  }
  else
  {
    vector<Color> ncolors((RGBorHSV.get() ?
			   hsv_outpts.size(): rgb_points.size()));
    vector<float> times((RGBorHSV.get() ?
			   hsv_outpts.size(): rgb_points.size()));
    vector<float> a(alphas.size());
    vector<float> t(alphas.size());

    
    if( RGBorHSV.get() ) {
      for(i=0;i<hsv_outpts.size();i++) {
	double h, s, v;
	h = hsv_outpts[i]._rgb[0] * 720;
	s = hsv_outpts[i]._rgb[1];
	v = hsv_outpts[i]._rgb[2];
	if (h > 360) h = h-360;
	ncolors[i] = Color( HSVColor( h, s, v ) );
	times[i] = hsv_outpts[i]._t;
      }
    } else {
      for(i=0;i<rgb_points.size();i++){
	ncolors[i] = rgb_points[i]._rgb;
	times[i] = rgb_points[i]._t;
      }
    }
    for(i=0;i<alphas.size();i++){
      a[i] = alphas[i]._alpha;
      t[i] = alphas[i]._t;
    }

//     cmap.get_rep()->SetRaw(ncolors,times,alphas,aTimes); 

//     // Bad for Volume Rendering?
//     cmap = scinew ColorMap(ncolors,times,alphas,aTimes,256);
    cmap.get_rep()->SetRaw(ncolors,times,a,t); 

    // Bad for Volume Rendering?
    cmap = scinew ColorMap(ncolors,times,a,t);
  }
  DrawGraphs(0);
#if 0  
  if (cmap.get_rep())
  {
    delete cmap.get_rep();
  }
#endif

  outport->send(cmap);
}


int GenTransferFunc::makeCurrent(void)
{
  Tk_Window tkwin;

  // lock a mutex
  gui->lock();

  if (!ctxs[0]) {
    const string myname(".ui" + id + ".f.gl1.gl");
    tkwin = Tk_NameToWindow(the_interp, ccast_unsafe(myname),
			    Tk_MainWindow(the_interp));

    if (!tkwin) {
      warning("Unable to locate window!");

      // unlock mutex
      gui->unlock();
      return 0;
    }
    winX[0] = Tk_Width(tkwin);
    winY[0] = Tk_Height(tkwin);

    dpy[0] = Tk_Display(tkwin);
    win0 = Tk_WindowId(tkwin);

    ctxs[0] = OpenGLGetContext(the_interp, ccast_unsafe(myname));

    // check if it was created
    if(!ctxs[0])
      {
	error("Unable to create OpenGL Context!");
	gui->unlock();
	return 0;
      }
  }	
// ****** Temporary comment out of HSV Window ******
//   if (!ctxs[1]) {
//     const string myname(".ui" + id + ".f.gl2.gl");
//     tkwin = Tk_NameToWindow(the_interp, ccast_unsafe(myname),
// 			    Tk_MainWindow(the_interp));

//     if (!tkwin) {
//       error("Unable to locate window!");

//       // unlock mutex
//       gui->unlock();
//       return 0;
//     }
    
//     winX[1] = Tk_Width(tkwin);
//     winY[1] = Tk_Height(tkwin);

//     dpy[1] = Tk_Display(tkwin);
//     win1 = Tk_WindowId(tkwin);

//     ctxs[1] = OpenGLGetContext(the_interp, ccast_unsafe(myname));

//     // check if it was created
//     if(!ctxs[1])
//       {
// 	error("Unable to create OpenGL Context!");
// 	gui->unlock();
// 	return 0;
//       }
//   }	

  if (!ctxs[2]) {
    const string myname(".ui" + id + ".f.gl3.gl");
    tkwin = Tk_NameToWindow(the_interp, ccast_unsafe(myname),
			    Tk_MainWindow(the_interp));

    if (!tkwin) {
      error("Unable to locate window!");

      // unlock mutex
      gui->unlock();
      return 0;
    }
    winX[2] = Tk_Width(tkwin);
    winY[2] = Tk_Height(tkwin);

    dpy[2] = Tk_Display(tkwin);
    win2 = Tk_WindowId(tkwin);

    ctxs[2] = OpenGLGetContext(the_interp, ccast_unsafe(myname));

    // check if it was created
    if(!ctxs[2])
      {
	error("Unable to create OpenGL Context!");
	gui->unlock();
	return 0;
      }
  }	

  // ok, 3 contexts are created - now you only need to
  // do the other crap...

  return 1;
}


} // End namespace SCIRun


