/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

struct ColorPoint {
  float  _t;
  Color _rgb;
};

class GenTransferFunc : public Module
{
private:

  GuiInt		RGBorHSV_; // which mode
  GuiInt		lineVSspline_; // linear vs. spline interpolate
  GuiString             rgb_points_pickle_;
  GuiString             hsv_points_pickle_;
  GuiString             alphas_pickle_;

  vector< Color > rgbs_;   // actual line(s)
  vector< float > rgbT_;   // actual line(s)
  vector< float > alphas_;
  vector< float > alphaT_;
  bool  hsv_mode_;

  int			activeLine; // active RGB,HSV,alpha,-1
  int			selNode;    // selected node

  int			bdown;      // wether mouse button is down

  int			winX[2],winY[2];  // integer size
  float			winDX, winDY; // scales (so it's square)

  GLXContext		ctxs[2];    // OpenGL Contexts
  Display		*dpy[2];
  Window		win0;
  Window		win1;

  ColorMapHandle	cmap;  // created once, first execute...
  int cmap_generation;

  void loadTs( double Ax, double Ay, double ax, double ay, vector<double>& t);

  void tcl_pickle();
  void tcl_unpickle();

public:
  GenTransferFunc( GuiContext* ctx);

  void DrawGraphs(int flush=1); // this function just blasts away...

  void DoDown(int x, int y, int button);
  void DoMotion(int x, int y);
  void DoRelease(int x, int y, int button);

  float GetTime(int x, int)
    {
      float v = x/(1.0*winX[0]);
      if (v > 1.0) v = 1.0;	
      if (v < 0.0) v = 0.0;
      return v;
    }
  float GetVal(int, int y)
    {
      float v= (winY[0]-y)/(1.0*winY[0]);
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

  virtual void presave();

  void toggle_hsv();
};


DECLARE_MAKER(GenTransferFunc)

GenTransferFunc::GenTransferFunc( GuiContext* ctx)
  : Module("GenTransferFunc",ctx,Source, "Visualization", "SCIRun"),
    RGBorHSV_(ctx->subVar("rgbhsv")),
    lineVSspline_(ctx->subVar("linespline")),
    rgb_points_pickle_(ctx->subVar("rgb_points_pickle")),
    hsv_points_pickle_(ctx->subVar("hsv_points_pickle")),
    alphas_pickle_(ctx->subVar("alphas_pickle")),
    hsv_mode_(0),
    activeLine(-1),
    selNode(-1),
    bdown(-1),
    cmap_generation(-1)
{
  cmap = 0; //scinew ColorMap; // start as nothing.

  rgbT_.push_back(0.0);
  rgbs_.push_back(Color(0,.05,.1));

  rgbT_.push_back(1.0);
  rgbs_.push_back(Color(1,.95,.9));

  // we want a similar transfer function for the HSV points
  ColorPoint p0, p1;
  p0._t = 0.0;
  p0._rgb = Color(0,.05, .1);
  p1._t = 1.0;
  p1._rgb = Color(1.0,.95,.9);

  alphaT_.push_back(0.0);
  alphas_.push_back(0.0);

  alphaT_.push_back(0.25);
  alphas_.push_back(0.0);

  alphaT_.push_back(0.5);
  alphas_.push_back(0.5);

  alphaT_.push_back(0.75);
  alphas_.push_back(0.8);

  alphaT_.push_back(1.0);
  alphas_.push_back(0.8);

  ctxs[0] = ctxs[1] = 0;
}


GenTransferFunc::~GenTransferFunc()
{
}


void
GenTransferFunc::loadTs( double Ax, double Ay, double ax, double ay,
			 vector<double>& t)
{
  double ys[11] = { 0.8333333333, 0.1666666666, 0.249999999,
		    0.3333333332, 0.4166666665, 0.499999998,
		    0.5833333331, 0.6666666664, 0.749999997,
		    0.8333333333, 0.9166666663 };

  int i;
  for( i = 0; i < 11; i++ )
  {
    double Bx = 0;
    double By = ys[i];
    double cx = Bx - Ax;
    double cy = By - Ay;
    double bx_ = 0;
    double by_ = 1;
    if(bx_ * ax + by_ * ay > 0 )
    {
      t.push_back((bx_ * cx + by_ * cy)/(bx_ * ax + by_ * ay));
    }
  }

  sort(t.begin(), t.end());
}


void
GenTransferFunc::tcl_pickle()
{
#if 0
  unsigned int i;

  ostringstream rgbstream;
  for (i = 0; i < rgb_points_.size(); i++)
  {
    rgbstream << rgb_points_[i]._t << " "
	      << rgb_points_[i]._rgb.r() << " "
	      << rgb_points_[i]._rgb.g() << " "
	      << rgb_points_[i]._rgb.b() << " ";
  }
  rgb_points_pickle_.set(rgbstream.str());

  ostringstream hsvstream;
  for (i = 0; i < hsv_points_.size(); i++)
  {
    hsvstream << hsv_points_[i]._t << " "
	      << hsv_points_[i]._rgb.r() << " "
	      << hsv_points_[i]._rgb.g() << " "
	      << hsv_points_[i]._rgb.b() << " ";
  }
  hsv_points_pickle_.set(hsvstream.str());

  ostringstream astream;
  for (i = 0; i < alphas_.size(); i++)
  {
    astream << alphas_[i]._t << " "
	    << alphas_[i]._alpha << " ";
  }
  alphas_pickle_.set(astream.str());
#endif
}


void
GenTransferFunc::tcl_unpickle()
{
#if 0
  rgb_points_.clear();
  hsv_points_.clear();
  alphas_.clear();

  float r, g, b;
  istringstream rgbstream(rgb_points_pickle_.get());
  while (!rgbstream.eof())
  {
    ColorPoint p;
    rgbstream >> p._t;
    if (rgbstream.fail()) break;
    rgbstream >> r;
    if (rgbstream.fail()) break;
    rgbstream >> g;
    if (rgbstream.fail()) break;
    rgbstream >> b;
    if (rgbstream.fail()) break;
    p._rgb = Color(r, g, b);
    rgb_points_.push_back(p);
  }

  float h, s, v;
  istringstream hsvstream(hsv_points_pickle_.get());
  while (!hsvstream.eof())
  {
    ColorPoint p;
    hsvstream >> p._t;
    if (hsvstream.fail()) break;
    hsvstream >> h;
    if (hsvstream.fail()) break;
    hsvstream >> s;
    if (hsvstream.fail()) break;
    hsvstream >> v;
    if (hsvstream.fail()) break;
    p._rgb = Color(h, s, v);
    hsv_points_.push_back(p);
  }
  istringstream astream(alphas_pickle_.get());
  while (!astream.eof())
  {
    AlphaPoint p;
    astream >> p._t;
    if (astream.fail()) break;
    astream >> p._alpha;
    if (astream.fail()) break;
    alphas_.push_back(p);
  }
#endif
}


// Draws/updates the graphs.
void
drawBox(float x, float y, float dx, float dy)
{
  //glBegin(GL_LINE_LOOP);
  //  glVertex2f(x-dx,y-dy);
  //  glVertex2f(x-dx,y+dy);
  //  glVertex2f(x+dx,y+dy);
  //  glVertex2f(x+dx,y-dy);
  //  glEnd();
  glPointSize(5.0);
  glBegin(GL_POINTS);
  glVertex2f(x, y);
  glEnd();
}


void
GenTransferFunc::Resize(int win)
{
  // Do a make current.

  if ( ! makeCurrent() )
    return;

  switch (win) {
  case 0:
    glXMakeCurrent(dpy[win],win0,ctxs[win]);
    break;
  case 1:
    glXMakeCurrent(dpy[win],win1,ctxs[win]);
    break;
  }

  if (win == 0)
  {  // DX DY only for first 1.
    int sx,sy;
    sx = winX[win];
    sy = winY[win];

    winDX = 1.0/500.0;  // X is normaly bigger.

    winDY = sx/sy*winDX;
  }

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(-1,-1,0);
  glScalef(2,2,2);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

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


void
GenTransferFunc::DrawGraphs( int flush)
{
  static float colors[] = { 1.,0.,0.,   0.,1.,0.,   0.,0.,1. };

  if (!makeCurrent())
    return; // don't care if you can't make current...

  glXMakeCurrent(dpy[0],win0,ctxs[0]);
  // other state should be fine...

  glDrawBuffer(GL_BACK);
  glClear(GL_COLOR_BUFFER_BIT);

  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE);

  unsigned int i,j;
  for (i=0;i<3;i++)
  {
    glColor3fv(colors + i * 3);
    glBegin(GL_LINE_STRIP);
    for (j=0; j<rgbs_.size(); j++)
    {
      glVertex2f(rgbT_[j], rgbs_[j][i]);
    }
    glEnd();
  }

  // Now draw the alpha curve.
  glColor3f(0.7, 0.7, 0.7);
  glBegin(GL_LINE_STRIP);
  for(i=0;i<alphas_.size();i++)
  {
    glVertex2f(alphaT_[i], alphas_[i]);
  }
  glEnd();


  // Now draw the boxes so that they are on top.
  for (i=0;i<3;i++)
  {
    for (j=0;j<rgbs_.size();j++)
    {
      glColor3fv(colors + i * 3);
      drawBox(rgbT_[j], rgbs_[j][i], winDX, winDY);
    }
  }

  for (i=0;i<alphas_.size();i++)
  {
    glColor3f(0.7, 0.7, 0.7);
    drawBox(alphaT_[i], alphas_[i], winDX, winDY);
  }

  glXSwapBuffers(dpy[0],win0);


  // Go to the last window.
  glXMakeCurrent(dpy[1],win1,ctxs[1]);
  glDrawBuffer(GL_BACK);
  glClear(GL_COLOR_BUFFER_BIT);

  // First lay down the backgroiund pattern.
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

  // Next draw the colormap overlay.
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

#if 0
  glBegin(GL_QUAD_STRIP);
  if( RGBorHSV_.get() ) {
    for(j=0;j<hsv_outpts_.size();j++) {
      double h, s, v;
      Color rgb;
      h = hsv_outpts_[j]._rgb[0] * 720;
      s = hsv_outpts_[j]._rgb[1];
      v = hsv_outpts_[j]._rgb[2];
      if (h > 360) h = h-360;
      rgb = Color( HSVColor( h, s, v ) );
      glColor3f(rgb.r(), rgb.g(), rgb.b());
      glVertex2f(hsv_outpts_[j]._t,0);
      glVertex2f(hsv_outpts_[j]._t,1);
    }
  } else {
    for(j=0;j<rgb_points_.size();j++) {
      glColor3f(rgb_points_[j]._rgb[0],rgb_points_[j]._rgb[1],
		rgb_points_[j]._rgb[2]);
      glVertex2f(rgb_points_[j]._t,0);
      glVertex2f(rgb_points_[j]._t,1);
    }
  }
#endif
  glEnd();

  glXSwapBuffers(dpy[1],win1);

  // Update the colormap.
  cmap = scinew ColorMap(rgbs_, rgbT_, alphas_, alphaT_, 256);

  glXMakeCurrent(dpy[0],None,NULL);
  gui->unlock();

  if ( flush )
  {
    GeometryOPort *ogeomport = (GeometryOPort *)get_oport("Geometry");
    ogeomport->flushViews();
    want_to_execute();
  }
}


// This is how the gui talks with this thing.
void
GenTransferFunc::tcl_command( GuiArgs& args, void* userdata)
{
  if (args.count() < 2) {
    args.error("No command for GenTransferFunc");
    return;
  }

  // Mouse commands are motion, down, release.
  if (args[1] == "mouse") {
    int x, y, whichb;

    string_to_int(args[3], x);
    string_to_int(args[4], y);

    if (args[2] == "motion") {
      if (bdown == -1) // not buttons down!
	return;
      DoMotion(x, y);
    } else {
      string_to_int(args[5], whichb); // which button it was
      if (args[2] == "down") {
	DoDown(x,y,whichb);
      } else { // must be release
	DoRelease(x,y,whichb);
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
    DrawGraphs(0);
  }else if(args[1] == "closewindow") {
     ctxs[0] = ctxs[1] = 0;
  }else if (args[1] == "unpickle") {
     tcl_unpickle();
  }else if (args[1] == "toggle-hsv") {
     toggle_hsv();
  }else {
    Module::tcl_command(args, userdata);
  }
}


// You just have to find the time interval first
// then look at the values.

void
GenTransferFunc::GetClosest(float time, float val, int& cline, int& cpoint)
{
  unsigned int i;

  float minP=10000;

  float d1,d2;
  for (i=0; i < rgbT_.size() && (rgbT_[i] <= time); i++) ; // do nothing

  // Now i is the value 1 greater.
  if (i == rgbs_.size())
  {
    i = rgbs_.size() -1;
  }

  // Now find the closest point.
  d1 = time - rgbT_[i-1];  d1 = d1*d1;
  d2 = rgbT_[i] - time;  d2 = d2*d2;
  float dy1,dy2;
  for(int j=0;j<3;j++) {
    dy1 = rgbs_[i-1][j] - val;
    dy2 = rgbs_[i][j]-val;

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

  // Now check alpha...
  for (i=0;i<alphaT_.size() && (alphaT_[i] <= time); i++) ; // do nothing


  // Now i is the value 1 greater.
  if (i == alphas_.size())
  {
    i = alphas_.size() -1;
  }

  d1 = time - alphaT_[i-1];  d1 = d1*d1;
  d2 = alphaT_[i] - time;  d2 = d2*d2;

  d1 += (alphas_[i-1] - val)*(alphas_[i-1] - val);
  d2 += (alphas_[i] - val)*(alphas_[i] - val);

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
}	


// Button must be down.
void
GenTransferFunc::DoMotion(int x, int y)
{
  if ((selNode == -1) || (activeLine == -1))  // this shouldn't happen!
    return;

  // this means you have to move the selected node...

  float time = GetTime(x,y); // get the valid time value...

  float val = GetVal(x,y);   // this remaps the Y coord!

  // End conditions are special cases - can't change x!

  if (activeLine == 7)
  {
    if (selNode && selNode < (int)alphas_.size()-1)
    {
      if (alphaT_[selNode-1] > time)
	time = alphaT_[selNode-1];
      if (alphaT_[selNode+1] < time)
	time = alphaT_[selNode+1];
    }
    else
    {
      time = alphaT_[selNode];
    }
    alphaT_[selNode] = time;
  }
  else
  {
    if (selNode && selNode < (int)rgbs_.size()-1)
    {
      if (rgbT_[selNode-1] > time)
	time = rgbT_[selNode-1];
      if (rgbT_[selNode+1] < time)
	time = rgbT_[selNode+1];
    }
    else
    {
      time = rgbT_[selNode];
    }
    rgbT_[selNode] = time;
  }

  if (activeLine < 6)
  {
    rgbs_[selNode][activeLine>>1] = val;
  }
  else
  { // it is the alpha value.
    alphas_[selNode] = val;
  }

  // Now you just have to redraw this
  DrawGraphs(0);
}



// Button 1 is modify, 2 is insert, 3 is delete

void
GenTransferFunc::DoDown(int x, int y, int button)
{
  // You have to find the point to select.

  float time,val;
  int cline,cpoint; // Closest info.

  time = GetTime(x,y);
  val  = GetVal(x,y);

  GetClosest(time,val,cline,cpoint);

  ColorPoint p;

  if (button == 2)
  {
    // You have to find what side of the closest point
    // you are on.

    activeLine = cline;

    p._t = time;

    p._rgb[cline>>1] = val;
    p._rgb[((cline>>1)+1)%3] = 0;
    p._rgb[((cline>>1)+2)%3] = 1;

    if (cline != 7)
    {
      if (cpoint && cpoint != (int)rgbs_.size()-1)
      {
	if (rgbT_[cpoint] <= time)
	  cpoint = cpoint+1;
	
      }
      else
      {
	if (!cpoint)
	{
	  cpoint = 1;
	}
      }

      // place the added points for the two other curves exactly on their
      // existing segments
      double dtime1, dtime2, dsumtime;
      dtime1=p._t - rgbT_[cpoint-1];
      dtime2=rgbT_[cpoint] - p._t;
      double w1, w2;
      w1=w2=.5;
      dsumtime=dtime1+dtime2;
      if (dsumtime)
      {   // if the points are vertical, just put it midway
	w1=dtime2/dsumtime;
	w2=dtime1/dsumtime;
      }
      int idx;
      idx=((cline>>1)+1)%3;
      p._rgb[idx]=rgbs_[cpoint-1][idx]*w1 + rgbs_[cpoint][idx]*w2;
      idx=((cline>>1)+2)%3;
      p._rgb[idx]=rgbs_[cpoint-1][idx]*w1 + rgbs_[cpoint][idx]*w2;
      rgbs_.insert(rgbs_.begin() + cpoint, p._rgb);
      rgbT_.insert(rgbT_.begin() + cpoint, p._t);
    }
    else
    { // this is the alpha curve.
      if (cpoint && cpoint != (int)alphas_.size()-1)
      {
	if (alphaT_[cpoint] <= time)
	  cpoint = cpoint+1;
      }	
      else
      {
	if (!cpoint)
	{
	  cpoint = 1;
	}
      }
      alphaT_.insert(alphaT_.begin() + cpoint, time);
      alphas_.insert(alphas_.begin() + cpoint, val);
    }
  }

  selNode = cpoint;
  activeLine = cline;

  bdown = button;

  DrawGraphs(); // just update the display (diferent box for selected)
}


void
GenTransferFunc::DoRelease(int x, int y, int button)
{
  // Just fake a move for the final position.

  DoMotion(x,y);

  switch(button) {
  case 1:
  case 2:
    // do nothing - just move/add...
    break;
  case 3:
    if (activeLine == 7)
    {
      if (selNode && (selNode != (int)alphas_.size()-1))
      {
	alphas_.erase(alphas_.begin() + selNode);
	alphaT_.erase(alphaT_.begin() + selNode);
      }
    }
    else
    {
      if (selNode && (selNode != (int)rgbs_.size()-1))
      {
	rgbs_.erase(rgbs_.begin() + selNode);
	rgbT_.erase(rgbT_.begin() + selNode);
      }
    }
    break;
  }

  selNode = -1; // leave activeLine.
  bdown = -1;

  DrawGraphs();
}


void
GenTransferFunc::execute()
{
  ColorMapIPort *inport = (ColorMapIPort *)get_iport("ColorMap");
  ColorMapOPort *outport = (ColorMapOPort *)get_oport("ColorMap");
  GeometryOPort *ogeomport = (GeometryOPort *)get_oport("Geometry");
  ColorMapHandle newcmap;

  if (!inport) {
    error("Unable to initialize iport 'ColorMap'.");
    return;
  }
  if (!outport) {
    error("Unable to initialize oport 'ColorMap'.");
    return;
  }
  if (!ogeomport) {
    error("Unable to initialize oport 'Geometry'.");
    return;
  }

  int c = inport->get(newcmap);

  if ( c && newcmap->generation != cmap_generation )
  {
    cmap_generation = newcmap->generation;
    
    cmap = newcmap;

    rgbs_ = newcmap->get_rgbs();
    rgbT_ = newcmap->get_rgbT();
    alphas_ = newcmap->get_alphas();
    alphaT_ = newcmap->get_alphaT();
  }
  else
  {
    cmap = scinew ColorMap(rgbs_, rgbT_, alphas_, alphaT_, 256);
  }
  DrawGraphs(0);

  outport->send(cmap);
}


int
GenTransferFunc::makeCurrent()
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

  if (!ctxs[1]) {
    const string myname(".ui" + id + ".f.gl3.gl");
    tkwin = Tk_NameToWindow(the_interp, ccast_unsafe(myname),
			    Tk_MainWindow(the_interp));

    if (!tkwin) {
      error("Unable to locate window!");

      // unlock mutex
      gui->unlock();
      return 0;
    }
    winX[1] = Tk_Width(tkwin);
    winY[1] = Tk_Height(tkwin);

    dpy[1] = Tk_Display(tkwin);
    win1 = Tk_WindowId(tkwin);

    ctxs[1] = OpenGLGetContext(the_interp, ccast_unsafe(myname));

    // check if it was created
    if(!ctxs[1])
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


void
GenTransferFunc::presave()
{
  tcl_pickle();
}


void
GenTransferFunc::toggle_hsv()
{
  if (RGBorHSV_.get() != hsv_mode_)
  {
    if (!hsv_mode_)
    {
      cout << "CONVERTING TO HSV\n";
      for (unsigned int i = 0; i < rgbs_.size(); i++)
      {
	HSVColor tmp(rgbs_[i]);
	rgbs_[i].r(tmp.hue() / 360.0);
	rgbs_[i].g(tmp.sat());
	rgbs_[i].b(tmp.val());
      }
    }
    else
    {
      cout << "CONVERTING TO RGB\n";
      for (unsigned int i = 0; i < rgbs_.size(); i++)
      {
	HSVColor tmp(rgbs_[i].r() * 360.0, rgbs_[i].g(), rgbs_[i].b());
	rgbs_[i] = tmp;
      }
    }
    hsv_mode_ = RGBorHSV_.get();
  }

  // Now you just have to redraw this
  DrawGraphs();
}


} // End namespace SCIRun





