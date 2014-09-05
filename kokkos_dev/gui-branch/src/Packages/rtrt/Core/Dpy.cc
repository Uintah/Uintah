#include <Packages/rtrt/Core/Dpy.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Image.h>
#include <Packages/rtrt/Core/Transform.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Packages/rtrt/Core/Ball.h>
#include <Packages/rtrt/Core/BallMath.h>
#include <Packages/rtrt/visinfo/visinfo.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/MiscMath.h>
#include <Packages/rtrt/Core/MusilRNG.h>
#include <Packages/rtrt/Core/Scene.h>
#include <iostream>
#include <GL/glx.h>
#include <GL/glu.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <unistd.h>
#include <sys/time.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <strings.h>
/* #include <SpeedShop/api.h> */

using namespace rtrt;
//using SCIRun::Mutex;
//using SCIRun::Thread;
//using SCIRun::Time;
using namespace SCIRun;
using std::endl;
using std::cerr;

extern void run_gl_test();

Mutex rtrt::xlock("X windows startup lock");

#define GL_INCLUDE 1

//Mutex rtrt::cameralock("Frameless Camera Synch lock");

Mutex io("io lock");

//extern int do_jitter;

inline double RtoD(double rad) {
  return rad*180/M_PI;
}

inline double DtoR(double deg) {
  return deg*M_PI/180.;
}

#define MIN_SELECT_TIME 20000

static void printString(GLuint fontbase, double x, double y,
			char *s, const Color& c)
{
  glColor3f(c.red(), c.green(), c.blue());

  glRasterPos2d(x,y);
  /*glBitmap(0, 0, x, y, 1, 1, 0);*/
  glPushAttrib (GL_LIST_BIT);
  glListBase(fontbase);
  glCallLists((int)strlen(s), GL_UNSIGNED_BYTE, (GLubyte *)s);
  glPopAttrib ();
}

static void printString2(GLuint fontbase, double x, double y,
			 char *s, const Color& c)
{
  Color b(0,0,0);
  printString(fontbase, x-1, y-1, s, b);
  printString(fontbase, x, y, s, c);
}

namespace rtrt {
  struct Dpy_private {
  int xres,yres;
  Display *dpy;

  bool exposed;
  int shadow_mode;
  bool animate;
  bool draw_pstats;
  bool draw_rstats;
  int showing_scene;
  int maxdepth;

  float base_threshold;
  float full_threshold;

  int left;
  int up;

  bool stereo;

  Camera *camera;

  BallData *ball;

  int last_x,last_y;

  double last_time;

  double FrameRate;

  int doing_frameless;

  //Transform prev_trans;

};

} // end namespace rtrt

Dpy::Dpy(Scene* scene, char* criteria1, char* criteria2,
	 int nworkers, bool bench, int ncounters, int c0, int c1,
	 float, float, bool display_frames, int frameless)
  : scene(scene), criteria1(criteria1), criteria2(criteria2),
    nworkers(nworkers), bench(bench), ncounters(ncounters),
    c0(c0), c1(c1), frameless(frameless),synch_frameless(0),
    display_frames(display_frames)
{
  //  barrier=new Barrier("Frame end", nworkers+1);
  barrier=new Barrier("Frame end");
  workers=new Worker*[nworkers];
  drawstats[0]=new Stats(1000);
  drawstats[1]=new Stats(1000);
  priv = new Dpy_private;

}

void Dpy::register_worker(int i, Worker* worker)
{
  workers[i]=worker;
}

Dpy::~Dpy()
{
}

static void drawpstats(Stats* mystats, int nworkers, Worker** workers,
		       bool draw_framerate, int showing_scene,
		       GLuint fontbase, double& lasttime,
		       double& cum_ttime, double& cum_dt)
{
  double thickness=.3;
  double border=.5;

  double mintime=1.e99;
  double maxtime=0;
  for(int i=0;i<nworkers;i++){
    Stats* stats=workers[i]->get_stats(showing_scene);
    int nstats=stats->nstats();
    if(stats->time(0)<mintime)
      mintime=stats->time(0);
    if(stats->time(nstats-1)>maxtime)
      maxtime=stats->time(nstats-1);
  }
  double maxworker=maxtime;
  if(mystats->time(0)<mintime)
    mintime=mystats->time(0);
  int nstats=mystats->nstats();
  if(mystats->time(nstats-1)>maxtime)
    maxtime=mystats->time(nstats-1);
    
  if(draw_framerate){
    char buf[100];
    double total_dt=0;
    for(int i=0;i<nworkers;i++){
      Stats* stats=workers[i]->get_stats(showing_scene);
      int nstats=stats->nstats();
      double dt=maxtime-stats->time(nstats-1);
      total_dt+=dt;
    }
    double ttime=(maxtime-lasttime)*nworkers;
    double imbalance=total_dt/ttime*100;
    cum_ttime+=ttime;
    cum_dt+=total_dt;
    double cum_imbalance=cum_dt/cum_ttime*100;
    sprintf(buf, "%5.1fms  %5.1f%% %5.1f%%", (maxtime-maxworker)*1000,
	    imbalance, cum_imbalance);
    printString(fontbase, 80, 3, buf, Color(0,1,0));
    lasttime=maxtime;
  }
  //cerr << mintime << " " << maxworker << " " << maxtime << " " << (maxtime-maxworker)*1000 << '\n';

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(mintime, maxtime, -border, nworkers+2+border);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
	    
  for(int i=0;i<nworkers;i++){
    Stats* stats=workers[i]->get_stats(showing_scene);
    int nstats=stats->nstats();
    double tlast=stats->time(0);
    for(int j=1;j<nstats;j++){
      double tnow=stats->time(j);
      glColor3dv(stats->color(j));
      glRectd(tlast, i+1, tnow, i+thickness+1);
      tlast=tnow;
    }
  }

  double tlast=mystats->time(0);
  int i=nworkers+1;
  for(int j=1;j<nstats;j++){
    double tnow=mystats->time(j);
    glColor3dv(mystats->color(j));
    glRectd(tlast, i, tnow, i+thickness);
    tlast=tnow;
  }	
}

int calc_width(XFontStruct* font_struct, char* str)
{
  XCharStruct overall;
  int ascent, descent;
  int dir;
  XTextExtents(font_struct, str, (int)strlen(str), &dir, &ascent, &descent, &overall);
  return overall.width;
}

#define PS(str) \
printString2(fontbase, x, y, str, c); y-=dy; \
width=calc_width(font_struct, str); \
maxwidth=width>maxwidth?width:maxwidth;

static void draw_labels(XFontStruct* font_struct, GLuint fontbase,
			int& column, int dy, int top)
{
  int x=column;
  int y=top;
  int maxwidth=0;
  int width;
  Color c(0,1,0);
  PS("");
  PS("Number of Rays");
  PS("Hit background");
  PS("Reflection rays");
  PS("Tranparency rays");
  PS("Shadow rays");
  PS("Rays in shadow");
  PS("Shadow cache tries");
  PS("Shadow cache misses");
  PS("BV intersections");
  PS("BV primitive intersections");
  PS("Light BV intersections");
  PS("Light BV prim intersections");
  PS("Sphere intersections");
  PS("Sphere hits");
  PS("Sphere light intersections");
  PS("Sphere light hits");
  PS("Sphere light hit penumbra");
  PS("Tri intersections");
  PS("Tri hits");
  PS("Tri light intersections");
  PS("Tri light hits");
  PS("Tri light hit penumbra");
  PS("Rect intersections");
  PS("Rect hits");
  PS("Rect light intersections");
  PS("Rect light hits");
  PS("Rect light hit penumbra");
  y-=dy;
  PS("Rays/second");
  PS("Rays/second/processor");
  PS("Rays/pixel");
  column+=maxwidth;
}
#undef PS

#define PN(n) \
sprintf(buf, "%d", n); \
width=calc_width(font_struct, buf); \
printString2(fontbase, x-width-w2, y, buf, c); y-=dy;

#define PD(n) \
sprintf(buf, "%g", n); \
width=calc_width(font_struct, buf); \
printString2(fontbase, x-width-w2, y, buf, c); y-=dy;

#define PP(n, den) \
if(den==0) \
percent=0; \
else \
percent=100.*n/den; \
sprintf(buf, "%d", n); \
width=calc_width(font_struct, buf); \
printString2(fontbase, x-width-w2, y, buf, c); \
sprintf(buf, " (%4.1f%%)", percent); \
printString2(fontbase, x-w2, y, buf, c); \
y-=dy;
  

static void draw_column(XFontStruct* font_struct,
			GLuint fontbase, char* heading, DepthStats& sum,
			int x, int w2, int dy, int top,
			bool first=false, double dt=1, int nworkers=0,
			int npixels=0)
{
  char buf[100];
  int y=top;
  Color c(0,1,0);
  double percent;

  int width=calc_width(font_struct, heading);
  printString2(fontbase, x-width-w2, y, heading, Color(1,0,0)); y-=dy;
  
  PN(sum.nrays);
  PP(sum.nbg, sum.nrays);
  PP(sum.nrefl, sum.nrays);
  PP(sum.ntrans, sum.nrays);
  PN(sum.nshadow);
  PP(sum.inshadow, sum.nshadow);
  PP(sum.shadow_cache_try, sum.nshadow);
  PP(sum.shadow_cache_miss, sum.shadow_cache_try);
  PN(sum.bv_total_isect);
  PP(sum.bv_prim_isect, sum.bv_total_isect);
  PN(sum.bv_total_isect_light);
  PP(sum.bv_prim_isect_light, sum.bv_total_isect_light);
  
  PN(sum.sphere_isect);
  PP(sum.sphere_hit, sum.sphere_isect);
  PN(sum.sphere_light_isect);
  PP(sum.sphere_light_hit, sum.sphere_light_isect);
  PP(sum.sphere_light_penumbra, sum.sphere_light_isect);
  
  PN(sum.tri_isect);
  PP(sum.tri_hit, sum.tri_isect);
  PN(sum.tri_light_isect);
  PP(sum.tri_light_hit, sum.tri_light_isect);
  PP(sum.tri_light_penumbra, sum.tri_light_isect);
  
  PN(sum.rect_isect);
  PP(sum.rect_hit, sum.rect_isect);
  PN(sum.rect_light_isect);
  PP(sum.rect_light_hit, sum.rect_light_isect);
  PP(sum.rect_light_penumbra, sum.rect_light_isect);
  if(first){
    y-=dy;
    double rps=sum.nrays/dt;
    PD(rps);
    double rpspp=rps/nworkers;
    PD(rpspp);
    double rpp=sum.nrays/(double)npixels;
    PD(rpp);
  }
}
#undef PS

static void drawrstats(int nworkers, Worker** workers,
		       int showing_scene,
		       GLuint fontbase, int xres, int yres,
		       XFontStruct* font_struct,
		       int left, int up,
		       double dt)
{
  DepthStats sums[MAXDEPTH];
  DepthStats sum;
  bzero(sums, sizeof(sums));
  bzero(&sum, sizeof(sum));
  int md=0;
  for(int i=0;i<nworkers;i++){
    int depth=md;
    while(depth<MAXDEPTH){
      Stats* st=workers[i]->get_stats(showing_scene);
      if(st->ds[depth].nrays==0)
	break;
      depth++;
    }
    md=depth;
  }

  for(int i=0;i<nworkers;i++){
    for(int depth=0;depth<md;depth++){
      Stats* st=workers[i]->get_stats(showing_scene);
      sums[depth].addto(st->ds[depth]);
    }
  }
  for(int depth=0;depth<md;depth++){
    sum.addto(sums[depth]);
  }

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, xres, 0, yres);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0.375, 0.375, 0.0);

  XCharStruct overall;
  int ascent, descent;
  char* str="123456789 (100%)";
  int dir;
  XTextExtents(font_struct, str, (int)strlen(str), &dir, &ascent, &descent, &overall);
  int dy=ascent+descent;
  int dx=overall.width;
  int column=3-left;
  int top=yres-3-dy+up;
  char* str2="(100%)";
  XTextExtents(font_struct, str2, (int)strlen(str2), &dir, &ascent, &descent, &overall);
  int w2=overall.width;
  draw_labels(font_struct, fontbase, column, dy, top);
  column+=dx;
  draw_column(font_struct, fontbase, "Total", sum, column, w2, dy, top,
	      true, dt, nworkers, xres*yres);
  column+=dx;
  for(int depth=0;depth<md;depth++){
    char buf[20];
    sprintf(buf, "%d", depth);
    draw_column(font_struct, fontbase, buf, sums[depth], column, w2, dy, top);
    column+=dx;
  }
}

float Galpha=1.0; // doh!  global variable...  probably should be moved...

void Dpy::get_input()
{
    //bool& exposed = priv->exposed;
  int& shadow_mode= priv->shadow_mode;
  bool& animate= priv->animate;
  bool& draw_pstats= priv->draw_pstats;
  bool& draw_rstats= priv->draw_rstats;
  int& showing_scene= priv->showing_scene;
  int& maxdepth= priv->maxdepth;

  float& base_threshold= priv->base_threshold;
  float& full_threshold= priv->full_threshold;

  int& left= priv->left;
  int& up= priv->up;

  bool& stereo= priv->stereo;  

  Camera *& camera = priv->camera;

  BallData *& ball = priv->ball;
  
  int& last_x = priv->last_x;
  int& last_y = priv->last_y;

  double& last_time = priv->last_time;

  double& FrameRate = priv->FrameRate;

  //int& doing_frameless = priv->doing_frameless;

  static double eye_dist=0;

  static Transform prev_trans;

  static double prev_time[3]; // history for quaternions and time
  static HVect prev_quat[3];

  static double FPS = 15;

  while (XEventsQueued(priv->dpy, QueuedAfterReading)){
    XEvent e;
    XNextEvent(priv->dpy, &e);	
    switch(e.type){
    case Expose:
      // Ignore expose events, since we will be refreshing
      // constantly anyway
      priv->exposed=true;
      break;
    case KeyPress:
      switch(XKeycodeToKeysym(priv->dpy, e.xkey.keycode, 0)){
      case XK_Escape:
      case XK_q:
	scene->rtrt_engine->exit_clean(1);
	//	Thread::exitAll(1);
	break;
      case XK_2:
	stereo=true;
	break;
      case XK_1:
	stereo=false;
	break;
      case XK_a:
	animate=!animate;
	break;
      case XK_plus:
	break;
      case XK_minus:
	break;
      case XK_f:
	FPS -= 1;
	if (FPS <= 0.0) FPS = 1.0;
	FrameRate = 1.0/FPS;
	break;
      case XK_g:
	FPS += 1.0;
	FrameRate = 1.0/FPS;
	cerr << FPS << endl;
	break;

	// turning on/off jittered sampling...
      case XK_j:
	scene->rtrt_engine->do_jitter=1-scene->rtrt_engine->do_jitter;
	break;

	// below is for blending "pixels" in
	// frameless rendering...

      case XK_Home:
	Galpha += 0.1;
	if (Galpha > 1.0) Galpha = 1.0;
	cerr << Galpha << endl;
	break;
      case XK_End:
	Galpha -= 0.1;
	if (Galpha < 0.0) Galpha = 0.0;
	cerr << Galpha << endl;
	break;

      case XK_y: // sychronizing mode for frameless...
	synch_frameless = 1-synch_frameless;
	//doing_frameless = 1-doing_frameless; // just toggle...
	cerr << synch_frameless << " Synch?\n";
	break;

      case XK_c:
	camera->print();
	break;
      case XK_p:
	draw_pstats=!draw_pstats;
	break;
      case XK_r:
	draw_rstats=!draw_rstats;
	break;
      case XK_t:
	  scene->hotspots=!scene->hotspots;
	  break;
      case XK_v:
	{
	  // Animate lookat point to center of BBox...
	  Object* obj=scene->get_object();
	  BBox bbox;
	  obj->compute_bounds(bbox, 0);
	  if(bbox.valid()){
	    camera->set_lookat(bbox.center());
        
	    // Move forward/backwards until entire view is in scene...
	    // change this a little, make it so that the FOV must
	    // be 60 deg...
	    // 60 degrees sucks - try 40

	    const double FOVtry=40;

	    Vector diag(bbox.diagonal());
	    double w=diag.length();
	    Vector lookdir(camera->get_lookat()-camera->get_eye()); 
	    lookdir.normalize();
	    const double scale = 1.0/(2*tan(DtoR(FOVtry/2.0)));
	    double length = w*scale;
	    camera->set_fov(FOVtry);
	    camera->set_eye(camera->get_lookat() - lookdir*length);
	    camera->setup();
	  }
	}
      break;
      case XK_w:
	cerr << "Saving file\n";
	scene->get_image(showing_scene)->save("images/image.raw");
	break;
      case XK_s:
	shadow_mode++;
	if(shadow_mode>3)
	  shadow_mode=0;
	break;
      case XK_h:
	scene->ambient_hack = !scene->ambient_hack;
	break;
      case XK_KP_Add:
	maxdepth++;
	cerr << "maxdepth=" << maxdepth << '\n';
	break;
      case XK_KP_Subtract:
	maxdepth--;
	if(maxdepth<0)
	  maxdepth=0;
	cerr << "maxdepth=" << maxdepth << '\n';
	break;
      case XK_KP_Up:
	base_threshold*=2;
	cerr << "base_threshold=" << base_threshold << '\n';
	break;
      case XK_KP_Down:
	base_threshold/=2;
	cerr << "base_threshold=" << base_threshold << '\n';
	break;
      case XK_KP_Page_Up:
	full_threshold*=2;
	cerr << "full_threshold=" << full_threshold << '\n';
	break;
      case XK_KP_Page_Down:
	full_threshold/=2;
	cerr << "full_threshold=" << full_threshold << '\n';
	break;
      case XK_Up:
	up+=4;
	break;
      case XK_Down:
	up-=4;
	break;
      case XK_Left:
	left+=4;
	break;
      case XK_Right:
	left-=4;
	break;
      }
      break;
    case ButtonPress:
      switch(e.xbutton.button){
      case Button1:
	last_x=e.xbutton.x;
	last_y=e.xbutton.y;
	break;
      case Button2:
	{
	  //inertia_mode=0;
	  last_x=e.xbutton.x;
	  last_y=e.xbutton.y;
			
	  // Find the center of rotation...
	  double rad = 0.8;
	  HVect center(0,0,0,1.0);
			
	  // we also want to keep the old transform information
	  // around (so stuff correlates correctly)
			
	  Vector y_axis,x_axis;
	  camera->get_viewplane(y_axis, x_axis);
	  Vector z_axis(camera->eye-camera->lookat);
			

	  x_axis.normalize();
	  y_axis.normalize();

	  eye_dist = z_axis.normalize();

	  prev_trans.load_frame(Point(0.0,0.0,0.0),x_axis,y_axis,z_axis);
			
	  ball->Init();
	  ball->Place(center,rad);
	  HVect mouse((2.0*e.xbutton.x)/priv->xres - 1.0,2.0*(priv->yres-e.xbutton.y*1.0)/priv->yres - 1.0,0.0,1.0);
	  ball->Mouse(mouse);
	  ball->BeginDrag();

	  prev_time[0] = Time::currentSeconds();
	  prev_quat[0] = mouse;
	  prev_time[1] = prev_time[2] = -100;

	  ball->Update();
	  last_time=Time::currentSeconds();
	  break;
	}
      case Button3:
	last_x=e.xbutton.x;
	last_y=e.xbutton.y;
	break;
      }
    case MotionNotify:
      switch(e.xmotion.state&(Button1Mask|Button2Mask|Button3Mask)){
      case Button1Mask:
	{
	  double xmtn=double(last_x-e.xmotion.x)/double(priv->xres);
	  double ymtn=-double(last_y-e.xmotion.y)/double(priv->yres);
	  last_x = e.xmotion.x;
	  last_y = e.xmotion.y;

	  Vector u,v;
	  camera->get_viewplane(u, v);
	  Vector trans(u*ymtn+v*xmtn);

	  // Translate the view...
	  camera->eye+=trans;
	  camera->lookat+=trans;
	  camera->setup();
	}
      break;
      case Button2Mask:
	{
	  HVect mouse((2.0*e.xmotion.x)/priv->xres - 1.0,2.0*(priv->yres-e.xmotion.y*1.0)/priv->yres - 1.0,0.0,1.0);

	  prev_time[2] = prev_time[1];
	  prev_time[1] = prev_time[0];
	  prev_time[0] = Time::currentSeconds();

	  ball->Mouse(mouse);
	  ball->Update();

	  prev_quat[2] = prev_quat[1];
	  prev_quat[1] = prev_quat[0];
	  prev_quat[0] = mouse;

	  // now we should just sendthe view points through
	  // the rotation (after centerd around the ball)
	  // eyep lookat and up
			
	  Transform tmp_trans;
	  HMatrix mNow;
	  ball->Value(mNow);
	  tmp_trans.set(&mNow[0][0]);

	  Transform prv = prev_trans;
	  prv.post_trans(tmp_trans);

	  HMatrix vmat;
	  prv.get(&vmat[0][0]);

	  Vector y_a(vmat[0][1],vmat[1][1],vmat[2][1]);
	  Vector z_a(vmat[0][2],vmat[1][2],vmat[2][2]);
	  z_a.normalize();
	  camera->up=y_a;
	  camera->eye=camera->lookat+z_a*eye_dist;
	  camera->setup();
			
	  last_time=Time::currentSeconds();
	  //inertia_mode=0;
	}
      break;
      case Button3Mask:
	{
	  if(e.xmotion.state & ShiftMask){
	    double scl;
	    double xmtn=last_x-e.xmotion.x;
	    double ymtn=last_y-e.xmotion.y;
	    xmtn/=300;
	    ymtn/=300;
	    last_x = e.xmotion.x;
	    last_y = e.xmotion.y;
	    if (Abs(xmtn)>Abs(ymtn)) scl=xmtn; else scl=ymtn;
	    Vector dir=camera->lookat-camera->eye;
	    camera->eye+=dir*scl;
	  } else {
	    double scl;
	    double xmtn=last_x-e.xmotion.x;
	    double ymtn=last_y-e.xmotion.y;
	    xmtn/=30;
	    ymtn/=30;
	    last_x = e.xmotion.x;
	    last_y = e.xmotion.y;
	    if (Abs(xmtn)>Abs(ymtn)) scl=xmtn; else scl=ymtn;
	    if (scl<0) scl=1/(1-scl); else scl+=1;
			
	    camera->fov=RtoD(2*atan(scl*tan(DtoR(camera->fov/2.))));
	  }
	  camera->setup();
	}
      break;
      }
      break;
    case ButtonRelease:
      switch(e.xbutton.button){
      case Button1:
	// Nothing...
	break;
      case Button2:
	{
#if 1
	  if(Time::currentSeconds()-last_time < .1){
	    // now setup the normalized quaternion
#endif
	    Transform tmp_trans;
	    HMatrix mNow;
	    ball->Value(mNow);
	    tmp_trans.set(&mNow[0][0]);
			    
	    Transform prv = prev_trans;
	    prv.post_trans(tmp_trans);
	    
	    HMatrix vmat;
	    prv.get(&vmat[0][0]);
			    
	    Vector y_a(vmat[0][1],vmat[1][1],vmat[2][1]);
	    Vector z_a(vmat[0][2],vmat[1][2],vmat[2][2]);
	    z_a.normalize();
	    
	    camera->up=y_a;
	    camera->eye=camera->lookat+z_a*eye_dist;
	    camera->setup();
	    prev_trans = prv;

	    // now you need to use the history to 
	    // set up the arc you want to use...

	    ball->Init();
	    double rad = 0.8;
	    HVect center(0,0,0,1.0);
			    
	    ball->Place(center,rad);

	    int index=2;

	    if (prev_time[index] == -100)
	      index = 1;
			    
	    ball->vDown = prev_quat[index];
	    ball->vNow  = prev_quat[0];
	    ball->dragging = 1;

	    ball->Update();
			    
	    ball->qNorm = ball->qNow.Conj();
	    //double mag = ball->qNow.VecMag();
			    
	    // Go into inertia mode...
	    //inertia_mode=1;
#if 0			    
	    if (mag < 0.00001) { // arbitrary ad-hoc threshold
	      //inertia_mode = 0;
	    } else {
	      double c = 1.0/mag;
	      double dt = prev_time[0] - prev_time[index];// time between last 2 events
	      ball->qNorm.x *= c;
	      ball->qNorm.y *= c;
	      ball->qNorm.z *= c;
	      angular_v = 2*acos(ball->qNow.w)*1000.0/dt;
	      cerr << dt << endl;
	    }
#endif
	  } 
#if 0
	  else {
	    //inertia_mode=0;
	  }
#endif
	  ball->EndDrag();
	}
      break;
      case Button3:
	// Nothing
	break;
      }
      break;
    case ConfigureNotify:
      {
	// Resize the image...
	priv->xres=e.xconfigure.width;
	priv->yres=e.xconfigure.height;
	glViewport(0, 0, priv->xres, priv->yres);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, priv->xres, 0, priv->yres);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.375, 0.375, 0.0);
      }
    break;
    default:
      cerr << "Unknown event, type=" << e.type << '\n';
    }
  }
}

int Dpy::get_num_procs() {
  return nworkers;
}

void Dpy::run()
{
  io.lock();
  cerr << "display is pid " << getpid() << '\n';
  io.unlock();
  if (scene->rtrt_engine->display_run_gl_test)
    run_gl_test();
  if(ncounters)
    counters=new Counters(ncounters, c0, c1);
  else
    counters=0;

  ////////////////////////////////////////////////////////////
  // open the display
  priv->xres=scene->get_image(0)->get_xres();
  priv->yres=scene->get_image(0)->get_yres();
  Window win;
  GLuint fontbase, fontbase2;
  XFontStruct* fontInfo2;
  xlock.lock();
  if (display_frames) {
    
  // Open an OpenGL window
  priv->dpy=XOpenDisplay(NULL);
  if(!priv->dpy){
    cerr << "Cannot open display\n";
    Thread::exitAll(1);
  }
  int error, event;
  if ( !glXQueryExtension( priv->dpy, &error, &event) ) {
    cerr << "GL extension NOT available!\n";
    XCloseDisplay(priv->dpy);
    priv->dpy=0;
    Thread::exitAll(1);
  }
  int screen=DefaultScreen(priv->dpy);

  if(!visPixelFormat(criteria1)){
    cerr << "Error setting pixel format for visinfo\n";
    cerr << "Syntax error in criteria: " << criteria1 << '\n';
    Thread::exitAll(1);
  }
  int nvinfo;
  XVisualInfo* vi=visGetGLXVisualInfo(priv->dpy, screen, &nvinfo);
  if(!vi || nvinfo == 0){
    if(!visPixelFormat(criteria2)){
      cerr << "Error setting pixel format for visinfo\n";
      cerr << "Syntax error in criteria: " << criteria2 << '\n';
      Thread::exitAll(1);
    }
    vi=visGetGLXVisualInfo(priv->dpy, screen, &nvinfo);
    if(!vi || nvinfo == 0){
      cerr << "Error matching OpenGL Visual: " << criteria1 << " and " << criteria2 << '\n';
      Thread::exitAll(1);
    }
  }
  Colormap cmap = XCreateColormap(priv->dpy, RootWindow(priv->dpy, screen),
				  vi->visual, AllocNone);
  XSetWindowAttributes atts;
  int flags=CWColormap|CWEventMask|CWBackPixmap|CWBorderPixel;
  atts.background_pixmap = None;
  atts.border_pixmap = None;
  atts.border_pixel = 0;
  atts.colormap=cmap;
  atts.event_mask=StructureNotifyMask|ExposureMask|ButtonPressMask|ButtonReleaseMask|ButtonMotionMask|KeyPressMask;
  win=XCreateWindow(priv->dpy, RootWindow(priv->dpy, screen),
			   0, 0, priv->xres, priv->yres, 0, vi->depth,
			   InputOutput, vi->visual, flags, &atts);
  char* p="real time ray tracer";
  XTextProperty tp;
  XStringListToTextProperty(&p, 1, &tp);
  XSizeHints sh;
  sh.flags = USSize;
  XSetWMProperties(priv->dpy, win, &tp, &tp, 0, 0, &sh, 0, 0);

  XMapWindow(priv->dpy, win);

  GLXContext cx=glXCreateContext(priv->dpy, vi, NULL, True);
  if(!glXMakeCurrent(priv->dpy, win, cx)){
    cerr << "glXMakeCurrent failed!\n";
  }

  XFontStruct* fontInfo = XLoadQueryFont(priv->dpy, 
					 "-adobe-helvetica-bold-r-normal--17-120-100-100-p-88-iso8859-1");
  if (fontInfo == NULL) {
    cerr << "no font found\n";
    Thread::exitAll(1);
  }
  Font id = fontInfo->fid;
  unsigned int first = fontInfo->min_char_or_byte2;
  unsigned int last = fontInfo->max_char_or_byte2;

  fontbase = glGenLists((GLuint) last+1);
  if (fontbase == 0) {
    printf ("out of display lists\n");
    exit (0);
  }
  glXUseXFont(id, first, last-first+1, fontbase+first);


  XFontStruct* fontInfo2 = XLoadQueryFont(priv->dpy, 
					  "-adobe-helvetica-bold-r-normal--12-120-*-*-p-*-iso8859-1");
  if (fontInfo2 == NULL) {
    cerr << "no font found(2)\n";
    Thread::exitAll(1);
  }


  id = fontInfo2->fid;
  first = fontInfo2->min_char_or_byte2;
  last = fontInfo2->max_char_or_byte2;

  GLuint fontbase2 = glGenLists((GLuint) last+1);
  if (fontbase2 == 0) {
    printf ("out of display lists\n");
    exit (0);
  }
  glXUseXFont(id, first, last-first+1, fontbase2+first);

  glShadeModel(GL_FLAT);
  glReadBuffer(GL_BACK);
  
  for(;;){
    XEvent e;
    XNextEvent(priv->dpy, &e);
    if(e.type == MapNotify)
      break;
  }
  }
  int rendering_scene=0;
  priv->showing_scene=1;

  //    start_rendering(rendering_scene);

  priv->last_x=0;
  priv->last_y=0;
  //int inertia_mode=0;
  //double angular_v=0; // angular velocity for inertia
  //Transform prev_trans;
  priv->ball = new BallData();
  priv->ball->Init();
  //double prev_time[3]; // history for quaternions and time
  priv->last_time=0;
  //HVect prev_quat[3];
  //double eye_dist=0;
  double last_frame=Time::currentSeconds();
  bool draw_framerate=true;
  priv->draw_pstats=false;
  priv->draw_rstats=false;
  int benchcount=0;
  int benchstartcount=0;
  priv->shadow_mode=scene->shadow_mode;
  int benchframes=100;
  if(bench){
    cerr << "Doing benchmark\n";
    benchcount=benchframes;
    benchstartcount=10;
  }
  double benchstart=-1234;
  Object* obj=scene->get_object();
  priv->animate=true;
  priv->maxdepth=scene->maxdepth;
  priv->base_threshold=0.005;
  priv->full_threshold=0.01;
  priv->left=0;
  priv->up=0;
  MusilRNG rng;
  int frame=0;
  bool last_changed=true;
  bool midchanged=true;
  int accum_count;
  priv->exposed=true;
  xlock.unlock();
  double lasttime=Time::currentSeconds();
  double cum_ttime=0;
  double cum_dt=0;
  priv->stereo=false;

  //bool& exposed = priv->exposed;
  int& shadow_mode= priv->shadow_mode;
  bool& animate= priv->animate;
  bool& draw_pstats= priv->draw_pstats;
  bool& draw_rstats= priv->draw_rstats;
  int& showing_scene= priv->showing_scene;
  int& maxdepth= priv->maxdepth;
    
  float& base_threshold= priv->base_threshold;
  float& full_threshold= priv->full_threshold;
    
  int& left= priv->left;
  int& up= priv->up;
    
  bool& stereo= priv->stereo;  
    
  //Camera *& camera = priv->camera;
    
  //BallData *& ball = priv->ball;
    
  //int& last_x = priv->last_x;
  //int& last_y = priv->last_y;

  if (!frameless) {

    // counter
    int counter = 1;
    for(;;){
      // Sync with the rendering processes...
      frame++;
      drawstats[showing_scene]->add(Time::currentSeconds(), Color(1,0,0));
      barrier->wait(nworkers+1);
      // exit if you are supposed to
      if (scene->rtrt_engine->stop_execution())
	Thread::exit();

      scene->refill_work(rendering_scene, nworkers);
      //bool changed=false;
      bool changed=true;
      {
	Camera* cam1=scene->get_camera(showing_scene);
	Camera* cam2=scene->get_camera(rendering_scene);
	if(*cam1 != *cam2){
	  *cam1=*cam2;
	  changed=true;
	}
	if(animate && scene->animate)
	  obj->animate(Time::currentSeconds(), changed);
	if(scene->shadow_mode != shadow_mode){
	  scene->shadow_mode=shadow_mode;
	  changed=true;
	}
	if(scene->maxdepth != maxdepth){
	  scene->maxdepth=maxdepth;
	  changed=true;
	}
	if(scene->base_threshold != base_threshold){
	  cerr << "Old: " << scene->base_threshold << '\n';
	  cerr << "New: " << base_threshold << '\n';
	  scene->base_threshold=base_threshold;
	  changed=true;
	}
	if(scene->full_threshold != full_threshold){
	  scene->full_threshold=full_threshold;
	  changed=true;
	}
	if(frame<5){
	  changed=true;
	}
	if(priv->exposed){
	  changed=true;
	  priv->exposed=false;
	}
	if(!changed && !scene->no_aa){
	  double x1, x2, w;
	  do {
	    x1 = 2.0 * rng() - 1.0;
	    x2 = 2.0 * rng() - 1.0;
	    w = x1 * x1 + x2 * x2;
	  } while ( w >= 1.0 );

	  w = sqrt( (-2.0 * log( w ) ) / w );
	  scene->xoffset = x1 * w * 0.5;
	  scene->yoffset = x2 * w * 0.5;
	} else {
	  scene->xoffset=0;
	  scene->yoffset=0;
	}
      }
      drawstats[showing_scene]->add(Time::currentSeconds(), Color(0,1,0));
      // sync all the workers.  scene->get_image(rendering_scene) should now
      // be completed
      barrier->wait(nworkers+1);
      // dump the frame and quit for now
      if (counter == 0) {
	if (!display_frames) {
	  scene->get_image(showing_scene)->save("displayless.raw");
	  cerr <<"Wrote frame to displayless.raw\n";
	}
      }
      counter--;
      // This is the last stat for the rendering scene (cyan)
      drawstats[showing_scene]->add(Time::currentSeconds(), Color(0,1,1));
      counters->end_frame();
	
      Stats* st=drawstats[rendering_scene];
      st->reset();
      double tnow=Time::currentSeconds();
      st->add(tnow, Color(1,0,0));
      double dt=tnow-last_frame;
      double framerate=1./dt;
#if 1
      if(framerate<4){
	cerr << "dt=" << dt << '\n';
      }
#endif
      last_frame=tnow;
      if(ncounters){
	fprintf(stderr, "%2d: %12lld", c0, counters->count0());	
	for(int i=0;i<nworkers;i++){
	  fprintf(stderr, "%12lld", workers[i]->get_counters()->count0());
	}
	fprintf(stderr, "\n");
	if(ncounters>1){
	  fprintf(stderr, "%2d: %12lld", c1, counters->count1());	
	  for(int i=0;i<nworkers;i++){
	    fprintf(stderr, "%12lld", workers[i]->get_counters()->count1());
	  }
	  fprintf(stderr, "\n\n");
	}
      }

      bool dontswap;
      if (display_frames) {
#ifdef GL_INCLUDE
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluOrtho2D(0, priv->xres, 0, priv->yres);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      glTranslatef(0.375, 0.375, 0.0);
#endif

#if 1
      dontswap=false;
      if(!bench){
	if(scene->no_aa || last_changed){
	  accum_count=0;
	  scene->get_image(showing_scene)->draw();
	} else {
	  if(accum_count==0){
	    // Load last image...
	    glReadBuffer(GL_FRONT);
	    glAccum(GL_LOAD, 1.0);
	    glReadBuffer(GL_BACK);
	    accum_count=1;
	  }
	  accum_count++;
	  glAccum(GL_MULT, 1.-1./accum_count);
	  scene->get_image(showing_scene)->draw();
	  glAccum(GL_ACCUM, 1.0/accum_count);
	  if(accum_count>=4){
	    /* The picture jumps around quite a bit when we show
	     * one that doesn't have very many samples, so we
	     * don't show the accumulated picture until we have
	     * at least 4 samples
	     */
	    glAccum(GL_RETURN, 1.0);
	  } else {
	    dontswap=true;
	  }
	}
      }
#else
      glClearColor(.3,.3,.3, 1);
      glClear(GL_COLOR_BUFFER_BIT);
#endif
      }
      Image* im=scene->get_image(showing_scene);
      midchanged=changed;
      last_changed=midchanged;
      // color blue
      st->add(Time::currentSeconds(), Color(0,0,1));
      //      st->add(Time::currentSeconds(), Color(0.4,0.2,1));
      priv->camera=scene->get_camera(showing_scene);
      if (display_frames) {
      // color 
      st->add(Time::currentSeconds(), Color(1,0.5,0));
      if(!bench && draw_framerate){
	char buf[100];
	sprintf(buf, "%3.1ffps", framerate);
	if(im->get_stereo()){
	  glDrawBuffer(GL_BACK_LEFT);
	  printString(fontbase, 10, 3, buf, Color(1,1,1));
	  glDrawBuffer(GL_BACK_RIGHT);
	}
	printString(fontbase, 10, 3, buf, Color(1,1,1));
      }
      // color light blue
      st->add(Time::currentSeconds(), Color(0.5,0.5,1));
      if(draw_pstats){
	Stats* mystats=drawstats[showing_scene];
	if(im->get_stereo()){
	  glDrawBuffer(GL_BACK_LEFT);
	  drawpstats(mystats, nworkers, workers, draw_framerate,
		     showing_scene, fontbase, lasttime, cum_ttime,
		     cum_dt);
	  glDrawBuffer(GL_BACK_RIGHT);
	}
	drawpstats(mystats, nworkers, workers, draw_framerate,
		   showing_scene, fontbase, lasttime, cum_ttime,
		   cum_dt);
      }
      // color grey
      st->add(Time::currentSeconds(), Color(0.5,0.5,0.5));
      if(draw_rstats){
	if(im->get_stereo()){
	  glDrawBuffer(GL_BACK_LEFT);
	  drawrstats(nworkers, workers,
		     showing_scene, fontbase2, priv->xres, priv->yres,
		     fontInfo2, left, up, dt);
	  glDrawBuffer(GL_BACK_RIGHT);
	}
	drawrstats(nworkers, workers,
		   showing_scene, fontbase2, priv->xres, priv->yres,
		   fontInfo2, left, up, dt);
      }
      st->add(Time::currentSeconds(), Color(1,0,1));
      glFinish();
#ifdef GL_INCLUDE
      st->add(Time::currentSeconds(), Color(1,0.5,1));
      if(!dontswap)
	glXSwapBuffers(priv->dpy, win);
      XFlush(priv->dpy);
#endif
      st->add(Time::currentSeconds(), Color(1,0.5,0.3));
      } // end if (display_frames)
      
      if(scene->logframes){
	  scene->frameno++;
	  if(!scene->frametime_fp){
	      scene->frametime_fp=fopen("framerate.out", "w");
	  }
	  double curtime=Time::currentSeconds();
	  double dt=curtime-scene->lasttime;
	  scene->lasttime=curtime;
	  Camera* camera=scene->get_camera(showing_scene);
	  Point eye(camera->get_eye());
	  Point lookat(camera->get_lookat());
	  Vector up(camera->get_up());
	  double fov(camera->get_fov());
	  fprintf(scene->frametime_fp, "%d %g -eye %g %g %g -lookat %g %g %g -up %g %g %g -fov %g\n",
		  scene->frameno, 1./dt,
		  eye.x(), eye.y(), eye.z(),
		  lookat.x(), lookat.y(), lookat.z(),
		  up.x(), up.y(), up.z(),
		  fov);
      }

      if (display_frames)
	get_input(); // this does all of the x stuff...

      if(im->get_xres() != priv->xres || im->get_yres() != priv->yres || im->get_stereo() != stereo){
	delete im;
	im=new Image(priv->xres, priv->yres, stereo);
	scene->set_image(showing_scene, im);
      }
      if(bench){
	if(benchstartcount){
	  --benchstartcount;
	  if(benchstartcount==0){
	    cerr << "Warmup done, starting bench\n";
	    benchstart=Time::currentSeconds();
	  }
	} else if(!--benchcount){
	  double dt=Time::currentSeconds()-benchstart;
	  cerr << "Benchmark completed in " <<  dt << " seconds ("
	       << benchframes/dt << " frames/second)\n";
	  Thread::exitAll(0);
	}
      }
      st->add(Time::currentSeconds(), Color(1,0,0));
      rendering_scene=1-rendering_scene;
      showing_scene=1-showing_scene;
    }
  } else { // doing frameless rendering...

    showing_scene=rendering_scene=0;  // only 1 buffer for frameless...
    double starttime = 0;

    priv->FrameRate = 1.0/15.0;
    priv->doing_frameless = 0;
    double& FrameRate = priv->FrameRate;

    for (;;) {
      // exit if you are supposed to
      if (scene->rtrt_engine->stop_execution())
	Thread::exit();

      frame++;
      //int do_synch=0;
      if (synch_frameless) { // synchronize stuff...
#if 0
	io.lock();
	cerr << "Display on first\n";
	io.unlock();
#endif
	barrier->wait(nworkers+1); // wait here for all of the procs...
#if 0
	io.lock();
	cerr << "Display out of first\n";
	io.unlock();
#endif
	//do_synch=1; // do the right thing for barriers...
      } else {
	starttime = Time::currentSeconds();
      }

      bool changed=true;
      {
	Camera* cam1=scene->get_camera(showing_scene);
	Camera* cam2=scene->get_camera(rendering_scene);
	if(*cam1 != *cam2){
	  *cam1=*cam2;
	  changed=true;
	}
	if(animate)
	  obj->animate(Time::currentSeconds(), changed);
	if(scene->shadow_mode != shadow_mode){
	  scene->shadow_mode=shadow_mode;
	  changed=true;
	}
	if(scene->maxdepth != maxdepth){
	  scene->maxdepth=maxdepth;
	  changed=true;
	}
	if(scene->base_threshold != base_threshold){
	  cerr << "Old: " << scene->base_threshold << '\n';
	  cerr << "New: " << base_threshold << '\n';
	  scene->base_threshold=base_threshold;
	  changed=true;
	}
	if(scene->full_threshold != full_threshold){
	  scene->full_threshold=full_threshold;
	  changed=true;
	}
	if(frame<5){
	  changed=true;
	}
	if(priv->exposed){
	  changed=true;
	  priv->exposed=false;
	}
	if(!changed && !scene->no_aa){
	  double x1, x2, w;
	  do {
	    x1 = 2.0 * rng() - 1.0;
	    x2 = 2.0 * rng() - 1.0;
	    w = x1 * x1 + x2 * x2;
	  } while ( w >= 1.0 );
	  
	  w = sqrt( (-2.0 * log( w ) ) / w );
	  scene->xoffset = x1 * w * 0.5;
	  scene->yoffset = x2 * w * 0.5;
	} else {
	  scene->xoffset=0;
	  scene->yoffset=0;
	}
      }

      // no stats for frameless stuff for now...

#if 0
      drawstats[showing_scene]->add(Time::currentSeconds(), Color(0,1,0));

      // This is the last stat for the rendering scene (cyan)
      drawstats[showing_scene]->add(Time::currentSeconds(), Color(0,1,1));
      counters->end_frame();
      
      Stats* st=drawstats[rendering_scene];
      st->reset();
#endif
      double tnow=Time::currentSeconds();
      //st->add(tnow, Color(1,0,0));
      double dt=tnow-last_frame;
      double framerate=1./dt;
      if(framerate<4){
	cerr << "dt=" << dt << '\n';
      }
      last_frame=tnow;
#if 0
      if(ncounters){
	fprintf(stderr, "%2d: %12ld", c0, counters->count0());	
	for(int i=0;i<nworkers;i++){
	  fprintf(stderr, "%12ld", workers[i]->get_counters()->count0());
	}
	fprintf(stderr, "\n");
	if(ncounters>1){
	  fprintf(stderr, "%2d: %12ld", c1, counters->count1());	
	  for(int i=0;i<nworkers;i++){
	    fprintf(stderr, "%12ld", workers[i]->get_counters()->count1());
	  }
	  fprintf(stderr, "\n\n");
	}
      }
#endif      
#ifdef GL_INCLUDE
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluOrtho2D(0, priv->xres, 0, priv->yres);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      glTranslatef(0.375, 0.375, 0.0);
#endif
      // zoom stuff to fit to the window...
#if 0
      if ((xScale != 1.0) || (yScale != 1.0)) {
	glPixelZoom(1.0/xScale,1.0/yScale);
      }
#endif

#if 1
      bool dontswap=false;
      if(!bench){
	if(scene->no_aa || last_changed){
	  accum_count=0;
	  scene->get_image(showing_scene)->draw();
	} else {
	  if(accum_count==0){
	    // Load last image...
	    glReadBuffer(GL_FRONT);
	    glAccum(GL_LOAD, 1.0);
	    glReadBuffer(GL_BACK);
	    accum_count=1;
	  }
	  accum_count++;
	  glAccum(GL_MULT, 1.-1./accum_count);
	  scene->get_image(showing_scene)->draw();
	  glAccum(GL_ACCUM, 1.0/accum_count);
	  if(accum_count>=4){
	    /* The picture jumps around quite a bit when we show
	     * one that doesn't have very many samples, so we
	     * don't show the accumulated picture until we have
	     * at least 4 samples
	     */
	    glAccum(GL_RETURN, 1.0);
	  } else {
	    dontswap=true;
	  }
	}
      }
#else
      glClearColor(.3,.3,.3, 1);
      glClear(GL_COLOR_BUFFER_BIT);
#endif
      Image* im=scene->get_image(showing_scene);
      midchanged=changed;
      last_changed=midchanged;
      //st->add(Time::currentSeconds(), Color(0,0,1));
      
      if(!bench && draw_framerate){
	char buf[100];
	sprintf(buf, "%3.1ffps", framerate);
	if(im->get_stereo()){
	  glDrawBuffer(GL_BACK_LEFT);
	  printString(fontbase, 10, 3, buf, Color(1,1,1));
	  glDrawBuffer(GL_BACK_RIGHT);
	}
	printString(fontbase, 10, 3, buf, Color(1,1,1));
      }
      if(draw_pstats){
	Stats* mystats=drawstats[showing_scene];
	if(im->get_stereo()){
	  glDrawBuffer(GL_BACK_LEFT);
	  drawpstats(mystats, nworkers, workers, draw_framerate,
		     showing_scene, fontbase, lasttime, cum_ttime,
		     cum_dt);
	  glDrawBuffer(GL_BACK_RIGHT);
	}
	drawpstats(mystats, nworkers, workers, draw_framerate,
		   showing_scene, fontbase, lasttime, cum_ttime,
		   cum_dt);
      }
      if(draw_rstats){
	if(im->get_stereo()){
	  glDrawBuffer(GL_BACK_LEFT);
	  drawrstats(nworkers, workers,
		     showing_scene, fontbase2, priv->xres, priv->yres,
		     fontInfo2, left, up, dt);
	  glDrawBuffer(GL_BACK_RIGHT);
	}
	drawrstats(nworkers, workers,
		   showing_scene, fontbase2, priv->xres, priv->yres,
		   fontInfo2, left, up, dt);
      }
      //st->add(Time::currentSeconds(), Color(1,0,1));
      glFinish();
#ifdef GL_INCLUDE
      if(!dontswap)
	glXSwapBuffers(priv->dpy, win);
      XFlush(priv->dpy);
#endif
      //st->add(Time::currentSeconds(), Color(1,1,0));
      
      int did_synch=synch_frameless;

      // lock this - for synchronization stuff...
      scene->rtrt_engine->cameralock.lock(); 
      priv->camera=scene->get_camera(showing_scene);
      
      get_input(); // this does all of the x stuff...
      
      scene->rtrt_engine->cameralock.unlock();

      if (did_synch) {
#if 0
	io.lock();
	cerr << "Display on second\n";
	io.unlock();
#endif
	barrier->wait(nworkers+1);
#if 0
	io.lock();
	cerr << "Display out of second\n";
	io.unlock();
#endif
      }

#if 0
      if (do_synch) {
	// just block in the begining???
	barrier->wait(nworkers+1); // block - this lets them get camera params...
	synch_frameless = priv->doing_frameless; // new ones always catch on 1st barrier...
      }
#endif

      // always have stereo - camera sychronization used to determine if
      // stereo pixels are rendered...

#if 0      
      if(im->get_xres() != priv->xres || im->get_yres() != priv->yres || im->get_stereo() != stereo){
	delete im;
	im=new Image(priv->xres, priv->yres, stereo);
	scene->set_image(showing_scene, im);
      }
#endif

      if(bench){
	if(benchstartcount){
	  --benchstartcount;
	  if(benchstartcount==0){
	    cerr << "Warmup done, starting bench\n";
	    benchstart=Time::currentSeconds();
	  }
	} else if(!--benchcount){
	  double dt=Time::currentSeconds()-benchstart;
	  cerr << "Benchmark completed in " <<  dt << " seconds ("
	       << benchframes/dt << " frames/second)\n";
	  Thread::exitAll(0);
	}
      }
      //st->add(Time::currentSeconds(), Color(1,0,0));
      //rendering_scene=1-rendering_scene;
      //showing_scene=1-showing_scene;

      double endtime = Time::currentSeconds();
      if (!synch_frameless)
	Time::waitFor(FrameRate - (endtime-starttime));
    }
  }
}

Barrier* Dpy::get_barrier()
{
    return barrier;
}

