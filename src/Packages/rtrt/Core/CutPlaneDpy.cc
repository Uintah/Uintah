
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/CutPlaneDpy.h>
#include <Packages/rtrt/Core/Ball.h>
#include <Packages/rtrt/Core/BallMath.h>
#include <Packages/rtrt/Core/MiscMath.h>
#include <Packages/rtrt/Core/FontString.h>
#include <Packages/rtrt/visinfo/visinfo.h>

#include <Core/Math/MinMax.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>

#include <X11/keysym.h>
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

namespace rtrt {
  extern Mutex xlock;
} // end namespace rtrt


CutPlaneDpy::CutPlaneDpy(const Vector& n, const Point& cen)
  : PlaneDpy(n, cen), cen(cen)
{
  set_resolution(150,150);
  on = true;
  dscale = 0.5;
  doanimate = true;
}
CutPlaneDpy::CutPlaneDpy(const Vector& n, const double d)
  : PlaneDpy(n, d)
{
  set_resolution(150,150);
  cen = Point(0,0,0)+n*d;
  on = true;
  dscale = 0.5;
  doanimate = true;
}

CutPlaneDpy::~CutPlaneDpy()
{
}

void CutPlaneDpy::init() {
  sleep(3);

  glShadeModel(GL_FLAT);
  //setup the rotation ball
  //copy plane eqn before rotation

  ball = new BallData();
  prev_trans = new Transform();

  double rad = 0.8;
  HVect center(0,0,0,1);
  Vector v1, v2;
  n.find_orthogonal(v1,v2);
  //n.normalize(); //has artifacts in "d" if enabled  
  prev_trans->load_frame(Point(0,0,0), v1, v2, n );
  ball->Init();
  ball->Place(center, rad);

  rotsphere = false;
}

void CutPlaneDpy::display() {
  if(redraw){ 
    int textheight=fontInfo->descent+fontInfo->ascent;
    
    glViewport(0, 0, xres, yres);
    if (on)
      glClearColor(0, 0, 0, 1);	    
    else
      glClearColor(0.3, 0.3, 0.3, 1);	    	      
    glClear(GL_COLOR_BUFFER_BIT);
    
    
    if (rotsphere) {
      glViewport(0, 0, xres, yres);
      ball->Draw();
    }

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-1, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glColor3f(0,0,1);
    
    if (!rotsphere) {
      glLineWidth(4.0);
      glBegin(GL_LINES);
      glVertex2f(0, -1);
      glVertex2f(0, 1);
      glEnd();
    }
    
    for(int i=0;i<4;i++){
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
      case 0: v=d; name="D"; wid=20/dscale; break;
      }
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      if(i==0)
	gluOrtho2D(-wid/2, wid/2, 0, 1);
      else
	gluOrtho2D(-1, 1, 0, 1);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      glColor3f(1,1,1);
      glLineWidth(4.0);
      glBegin(GL_LINES);
      glVertex2f(v, th);
      if (rotsphere)
	glVertex2f(v, 0.5);
      else
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


void CutPlaneDpy::resize(const int width, const int height) {
  yres=height;
  if(width != xres){
    xres=width;
    redraw=true;
  }
}

void CutPlaneDpy::key_pressed(unsigned long key) {
  switch(key) {
  case XK_o:
    on = !on;
    redraw=true;	
    break;
  case XK_p:
    cerr << "A " << n.x() << '\t';
    cerr << "B " << n.y() << '\t';
    cerr << "C " << n.z() << '\t';
    cerr << "D " << d << '\t';
    if (on)
      cerr << "enabled" << '\t';
    else
      cerr << "disabled" << '\t';
    cerr << endl;
    break;
  case XK_d:
    d -= 0.1;
    redraw=true;
    break;
  case XK_f:
    d += 0.1;
    redraw=true;
    break;
  case XK_z:
    dscale *= .2;
    redraw=true;
    break;
  case XK_x:
    dscale *= 5.0;
    redraw=true;
    break;
  }
}


void CutPlaneDpy::button_pressed(MouseButton button, const int x, const int y) {
  
  prev_doanimate = doanimate;
  doanimate = false;
			       
  if (button == MouseButton1) {
    //set each ABCD for plane manually
    starty = y;
    move(x, y);
    redraw = true;
  }

  if (button == MouseButton2) {
    //begin a cutplane rotation
    //Update ball with first mouse position	      
    HVect mouse(
		(2.0*x)/xres - 1.0, 
		2.0*(yres-y*1.0)/yres - 1.0,
		0.0,
		1.0
		);
    ball->Mouse(mouse);
    ball->BeginDrag();
    ball->Update();
    
    rotsphere=true; //turn on display of rot axis ball
    redraw=true;
  }	      
}

void CutPlaneDpy::button_motion(MouseButton button, const int x, const int y) {
  if (button == MouseButton1) {
    //set plane ABCD manually
    move(x, starty);
    redraw=true;
  }
  if (button == MouseButton2) {
    // Rotate the cutplane

    //use the new mouse position to rotate the ball
    HVect mouse(
		(2.0*x)/xres - 1.0,
		2.0*(yres-y*1.0)/yres - 1.0,
		0.0,
		1.0);	      	              
    ball->Mouse(mouse);
    ball->Update();

    //get the orientation of the ball and apply it to the original plane orientation
    Transform tmp_trans;
    HMatrix mNow;
    ball->Value(mNow);
    tmp_trans.set(&mNow[0][0]);    
    Transform prv = *prev_trans; //original plane eqn
    prv.post_trans(tmp_trans); //apply the ball's transform to it
    HMatrix vmat;
    prv.get(&vmat[0][0]); 
    Vector x_a(vmat[0][2],vmat[1][2],vmat[2][2]); //get x, where we stored the pln normal
    
    //store result into the plane
    n.x(x_a.x());
    n.y(x_a.y());
    n.z(x_a.z());

    d = Dot(n, cen); //maintain d to center rotation around initial point

    redraw=true;
  }
  
}

void CutPlaneDpy::button_released(MouseButton button, const int x, const int y)
{
  if (button == MouseButton1) {
    starty=y;
    move(x, y);

    //reset the ball so that it rotates starting with this plane orientation
    double rad = 0.8;
    HVect center(0,0,0,1);
    Vector v1, v2;
    n.find_orthogonal(v1,v2);
    prev_trans->load_frame(Point(0,0,0), v1, v2, n); //store pln normal to rotate
    ball->Init(); //would be better to find a way to update orientation instead of reseting it
    ball->Place(center, rad);

    redraw=true;
  }
  if (button == MouseButton2) {
    //same as for motion
    HVect mouse(
		(2.0*x)/xres - 1.0,
		2.0*(yres-y*1.0)/yres - 1.0,
		0.0,
		1.0);	      	              
    ball->Mouse(mouse);
    ball->Update();


    Transform tmp_trans;
    HMatrix mNow;
    ball->Value(mNow);
    tmp_trans.set(&mNow[0][0]);
    Transform prv = *prev_trans;
    prv.post_trans(tmp_trans);	  
    HMatrix vmat;
    prv.get(&vmat[0][0]);    
    Vector x_a(vmat[0][2],vmat[1][2],vmat[2][2]);
    n.x(x_a.x());
    n.y(x_a.y());
    n.z(x_a.z());
    
    d = Dot(n, cen);

    //finish ball motion
    ball->EndDrag();

    //stop displaying it
    rotsphere=false;
    redraw=true;
  }
  doanimate = prev_doanimate;
}

void CutPlaneDpy::move(int x, int y)
{
    float xn=float(x)/xres;
    float yn=float(y)/yres;
    if(yn>.75){
      //D
      double dt = d;
      d=xn*20/dscale-10/dscale;
      cen = cen + n*(d-dt); //change center of rotation to be on the new plane
    } else if(yn>.5){
      //Z
      n.z(xn*2-1);
      d = Dot(n, cen); //maintain center of rotation
    } else if(yn>.25){
      //Y
      n.y(xn*2-1);
      d = Dot(n, cen);
    } else {
      //X
      n.x(xn*2-1);
      d = Dot(n, cen);
    }
}


void CutPlaneDpy::animate(double t) {
  if (doanimate) {
    //constant chosen to match the spinning things in the siggraph demo
#define CPDPY_RATE 2*0.1
    
    Transform prv = *prev_trans; //original plane eqn
    prv.pre_rotate(t*CPDPY_RATE, Vector(0,0,1));
#undef CPDPY_RATE
    HMatrix vmat;
    prv.get(&vmat[0][0]); 
    Vector x_a(vmat[0][2],vmat[1][2],vmat[2][2]); //get x, where we stored the pln normal
  
    //store result into the plane
    n.x(x_a.x());
    n.y(x_a.y());
    n.z(x_a.z());
    
    d = Dot(n, cen); //maintain d to center rotation around initial point  
    
    //redraw = true;
  }
}



