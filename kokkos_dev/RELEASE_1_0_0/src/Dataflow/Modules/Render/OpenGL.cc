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
 *  OpenGL.cc: Render geometry using opengl
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   December 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Modules/Render/OpenGL.h>

#ifdef __sgi
#include <ifl/iflFile.h>
#endif

extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);
extern Tcl_Interp* the_interp;

namespace SCIRun {

#define DO_REDRAW 0
#define DO_PICK 1
#define DO_GETDATA 2
#define REDRAW_DONE 4
#define PICK_DONE 5
#define DO_IMAGE 6
#define IMAGE_DONE 7

static map<clString, ObjTag*>::iterator viter;

int CAPTURE_Z_DATA_HACK = 0;

static OpenGL* current_drawer=0;
static const int pick_buffer_size = 512;
static const double pick_window = 10.0;

static Renderer* make_OpenGL()
{
  return scinew OpenGL;
}

int query_OpenGL()
{
  TCLTask::lock();
  int have_opengl=glXQueryExtension
    (Tk_Display(Tk_MainWindow(the_interp)), NULL, NULL);
  TCLTask::unlock();
  if (!have_opengl)
    cerr << "glXQueryExtension() returned NULL.\n"
      "** XFree86 NOTE **  Do you have the line 'Load \"glx\"'"
      " in the Modules section of your XF86Config file?"
         << endl;
  return have_opengl;
}

RegisterRenderer OpenGL_renderer("OpenGL", &query_OpenGL, &make_OpenGL);

OpenGL::OpenGL()
  : tkwin(0),
    helper(0),
    send_mb("OpenGL renderer send mailbox",10),
    recv_mb("OpenGL renderer receive mailbox", 10),
    get_mb("OpenGL renderer request mailbox", 5),
    img_mb("OpenGL renderer image data mailbox", 5)
{
  encoding_mpeg = false;
  drawinfo=scinew DrawInfoOpenGL;
  fpstimer.start();
  
  /* Grey */
  stylusMaterial[0] = scinew Material(Color(0,0,0), Color(.3,.3,.3), Color(.5,.5,.5), 20);
  /* White */
  stylusMaterial[1] = scinew Material(Color(0,0,0), Color(1.0,1.0,1.0), Color(.5,.5,.5), 20);
  /* Yellow */
  stylusMaterial[2] = scinew Material(Color(0,0,0), Color(1.0,1.0,0.0), Color(.5,.5,.5), 20);
  /* Light red */
  stylusMaterial[3] = scinew Material(Color(0,0,0), Color(.8,0,0), Color(.5,.5,.5), 20);
  /* Dark red */
  stylusMaterial[4] = scinew Material(Color(0,0,0), Color(.2,0,0), Color(.5,.5,.5), 20);
  /* Light green */
  stylusMaterial[5] = scinew Material(Color(0,0,0), Color(0,.8,0), Color(.5,.5,.5), 20);
  /* Dark green */
  stylusMaterial[6] = scinew Material(Color(0,0,0), Color(0,.2,0), Color(.5,.5,.5), 20);
  /* Light blue */
  stylusMaterial[7] = scinew Material(Color(0,0,0), Color(0,0,.8), Color(.5,.5,.5), 20);
  /* Dark blue */
  stylusMaterial[8] = scinew Material(Color(0,0,0), Color(0,0,.2), Color(.5,.5,.5), 20);
  
  stylusCylinder[0] = scinew GeomCappedCylinder(Point(0,-3,0), Point(0,3,0), 0.3, 20, 10);
  stylusCylinder[1] = scinew GeomCappedCylinder(Point(0,3,0), Point(0,3.3,0), 0.3, 20, 10);
  
  stylusTriangle[0] = scinew GeomTri(Point(0,-1.5,0), Point(0,1.5,0), Point(1.5,0,0));
  stylusTriangle[1] = scinew GeomTri(Point(0,-1.5,0), Point(0,1.5,0), Point(-1.5,0,0));
  stylusTriangle[2] = scinew GeomTri(Point(0,-1.5,0), Point(0,1.5,0), Point(0,0,1.5));
  stylusTriangle[3] = scinew GeomTri(Point(0,-1.5,0), Point(0,1.5,0), Point(0,0,-1.5));
  
  pinchMaterial = scinew Material(Color(0,0,0), Color(0,.8,0), Color(.5,.5,.5), 20);
  pinchSphere = scinew GeomSphere(Point(0,0,0), 0.4, 20, 10);
  
  pinchText[0] = scinew GeomText(clString(""),Point(1,1,1));
  pinchText[1] = scinew GeomText(clString(""),Point(1,1,1));
  
  pinchCylinder[0] = scinew GeomCappedCylinder(Point(0,0,0), Point(1,0,0), 0.2, 20, 10);
  pinchCylinder[1] = scinew GeomCappedCylinder(Point(0,0,0), Point(-1,0,0), 0.2, 20, 10);
}

OpenGL::~OpenGL()
{
  fpstimer.stop();
				// make sure we finish up mpeg that
				// was in progress
  if(encoding_mpeg) encoding_mpeg = false;
}

clString OpenGL::create_window(ViewWindow*,
			       const clString& name,
			       const clString& width,
			       const clString& height)
{
  myname=name;
  width.get_int(xres);
  height.get_int(yres);
  static int direct=1;
  int d=direct;
  direct=0;
  return "opengl "+name+" -geometry "+width+"x"+height+" -doublebuffer true -direct "+(d?"true":"false")+" -rgba true -redsize 1 -greensize 1 -bluesize 1 -depthsize 2";
}

void OpenGL::initState(void)
{
  
}

class OpenGLHelper : public Runnable {
  OpenGL* opengl;
public:
  OpenGLHelper(OpenGL* opengl);
  virtual ~OpenGLHelper();
  virtual void run();
};

OpenGLHelper::OpenGLHelper(OpenGL* opengl)
  : opengl(opengl)
{
}

OpenGLHelper::~OpenGLHelper()
{
}

void OpenGLHelper::run()
{
  cerr << "Calling allow..." << getpid() << "\n";
  Thread::allow_sgi_OpenGL_page0_sillyness();
  opengl->redraw_loop();
}

void OpenGL::redraw(Viewer* s, ViewWindow* r, double _tbeg, double _tend,
		    int _nframes, double _framerate)
{
  viewer=s;
  viewwindow=r;
  tbeg=_tbeg;
  tend=_tend;
  nframes=_nframes;
  framerate=_framerate;
  // This is the first redraw - if there is not an OpenGL thread,
  // start one...
  if(!helper){
    my_openglname=clString("OpenGL: ")+myname;
    helper=new OpenGLHelper(this);
    Thread* t=new Thread(helper, my_openglname());
    t->detach();
  }
  
  send_mb.send(DO_REDRAW);
  int rc=recv_mb.receive();
  if(rc != REDRAW_DONE){
    cerr << "Wanted redraw_done, but got: " << r << endl;
  }
}


void OpenGL::redraw_loop()
{
  int r;
  
  // Tell the ViewWindow that we are started...
  TimeThrottle throttle;
  throttle.start();
  double newtime=0;
  while(1) {
    int nreply=0;
    if(viewwindow->inertia_mode){
      double current_time=throttle.time();
      if(framerate==0)
	framerate=30;
      double frametime=1./framerate;
      double delta=current_time-newtime;
      if(delta > 1.5*frametime){
	framerate=1./delta;
	frametime=delta;
	newtime=current_time;
      } if(delta > .85*frametime){
	framerate*=.9;
	frametime=1./framerate;
	newtime=current_time;
      } else if(delta < .5*frametime){
	framerate*=1.1;
	if(framerate>30)
	  framerate=30;
	frametime=1./framerate;
	newtime=current_time;
      }
      newtime+=frametime;
      throttle.wait_for_time(newtime);
      
      while (send_mb.tryReceive(r)) {
	if (r == DO_PICK) {
	  real_get_pick(viewer, viewwindow, send_pick_x, send_pick_y,
			ret_pick_obj, ret_pick_pick, ret_pick_index);
	  recv_mb.send(PICK_DONE);
	} else if(r== DO_GETDATA) {
	  GetReq req(get_mb.receive());
	  real_getData(req.datamask, req.result);
	} else if(r== DO_IMAGE) {
	  ImgReq req(img_mb.receive());
	  real_saveImage(req.name, req.type);
	} else {
	  // Gobble them up...
	  nreply++;
	}
      }
      
      // you want to just rotate around the current rotation
      // axis - the current quaternion is viewwindow->ball->qNow	    
      // the first 3 components of this 
      
      viewwindow->ball->SetAngle(newtime*viewwindow->angular_v);
      
      View tmpview(viewwindow->rot_view);
      
      Transform tmp_trans;
      HMatrix mNow;
      viewwindow->ball->Value(mNow);
      tmp_trans.set(&mNow[0][0]);
      
      Transform prv = viewwindow->prev_trans;
      prv.post_trans(tmp_trans);
      
      HMatrix vmat;
      prv.get(&vmat[0][0]);
      
      Point y_a(vmat[0][1],vmat[1][1],vmat[2][1]);
      Point z_a(vmat[0][2],vmat[1][2],vmat[2][2]);
      
      tmpview.up(y_a.vector());
      
      if (viewwindow->inertia_mode == 1) {
	tmpview.eyep((z_a*(viewwindow->eye_dist)) + tmpview.lookat().vector());
	viewwindow->view.set(tmpview);      
      } else if (viewwindow->inertia_mode == 2) {
	tmpview.lookat(tmpview.eyep()-(z_a*(viewwindow->eye_dist)).vector());
	viewwindow->view.set(tmpview);      
      }
      
    } else {
      for (;;) {
	int r=send_mb.receive();
	if (r == DO_PICK) {
	  real_get_pick(viewer, viewwindow, send_pick_x, send_pick_y, 
			ret_pick_obj, ret_pick_pick, ret_pick_index);
	  recv_mb.send(PICK_DONE);
	} else if(r== DO_GETDATA){
	  GetReq req(get_mb.receive());
	  real_getData(req.datamask, req.result);
	} else if(r== DO_IMAGE) {
	  ImgReq req(img_mb.receive());
	  real_saveImage(req.name, req.type);
	} else {
	  nreply++;
	  break;
	}
      }
      newtime=throttle.time();
      throttle.stop();
      throttle.clear();
      throttle.start();
    }
    redraw_frame();
    for(int i=0;i<nreply;i++)
      recv_mb.send(REDRAW_DONE);
  }
}

void OpenGL::make_image()
{
  imglist=glGenLists(1);
  glNewList(imglist, GL_COMPILE_AND_EXECUTE);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, xres-1, 0.0, yres-1, -10.0, 10.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glDisable(GL_DEPTH_TEST);
  glRasterPos2i(xres-168-5, yres-5);
  glDrawPixels(168, 95, GL_LUMINANCE, GL_UNSIGNED_BYTE, logoimg);
  glEnable(GL_DEPTH_TEST);
  glEndList();
}

void OpenGL::redraw_frame()
{
  // Get window information
  TCLTask::lock();
  Tk_Window new_tkwin=Tk_NameToWindow(the_interp,
				      const_cast<char *>(myname()),
				      Tk_MainWindow(the_interp));
  if(!new_tkwin){
    cerr << "Unable to locate window!\n";
    TCLTask::unlock();
    return;
  }
  if(tkwin != new_tkwin){
    tkwin=new_tkwin;
    dpy=Tk_Display(tkwin);
    win=Tk_WindowId(tkwin);
    cx=OpenGLGetContext(the_interp, const_cast<char *>(myname()));
    if(!cx){
      cerr << "Unable to create OpenGL Context!\n";
      TCLTask::unlock();
      return;
    }
    glXMakeCurrent(dpy, win, cx);
    glXWaitX();
    current_drawer=this;
    GLint data[1];
    glGetIntegerv(GL_MAX_LIGHTS, data);
    maxlights=data[0];
    // Look for multisample extension...
#ifdef __sgi
    if(strstr((char*)glGetString(GL_EXTENSIONS), "GL_SGIS_multisample")){
      cerr << "Enabling multisampling...\n";
      glEnable(GL_MULTISAMPLE_SGIS);
      glSamplePatternSGIS(GL_1PASS_SGIS);
    }
#endif
  }
  
  TCLTask::unlock();
  
  // Start polygon counter...
  WallClockTimer timer;
  timer.clear();
  timer.start();
  
  initState();
  
  // Get the window size
  xres=Tk_Width(tkwin);
  yres=Tk_Height(tkwin);
  
  // Make ourselves current
  if(current_drawer != this){
    current_drawer=this;
    TCLTask::lock();
    glXMakeCurrent(dpy, win, cx);
    TCLTask::unlock();
  }
  
  // Get a lock on the geometry database...
  // Do this now to prevent a hold and wait condition with TCLTask
  viewer->geomlock.readLock();
  
  TCLTask::lock();
  
  // Clear the screen...
  glViewport(0, 0, xres, yres);
  Color bg(viewwindow->bgcolor.get());
  glClearColor(bg.r(), bg.g(), bg.b(), 1);
  
  clString saveprefix(viewwindow->saveprefix.get());
  
  // Setup the view...
  View view(viewwindow->view.get());
  lastview=view;
  double aspect=double(xres)/double(yres);
  // XXX - UNICam change-- should be '1.0/aspect' not 'aspect' below
  double fovy=RtoD(2*Atan(1.0/aspect*Tan(DtoR(view.fov()/2.))));
  
  drawinfo->reset();
  int do_stereo=viewwindow->do_stereo.get();
  
#ifdef __sgi
  //  --  BAWGL  -- 
  int do_bawgl = viewwindow->do_bawgl.get();
  SCIBaWGL* bawgl = viewwindow->get_bawgl();
  
  if(!do_bawgl)
    bawgl->shutdown_ok();
  //  --  BAWGL  -- 
#endif
  // Compute znear and zfar...
  
  if(compute_depth(viewwindow, view, znear, zfar)){
    
    // Set up graphics state
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);
    glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
    
    clString globals("global");
    viewwindow->setState(drawinfo,globals);
    drawinfo->pickmode=0;
    
    GLenum errcode;
    while((errcode=glGetError()) != GL_NO_ERROR){
      cerr << "We got an error from GL: " << (char*)gluErrorString(errcode) << endl;
    }
    
    // Do the redraw loop for each time value
    double dt=(tend-tbeg)/nframes;
    double frametime=framerate==0?0:1./framerate;
    TimeThrottle throttle;
    throttle.start();
    Vector eyesep(0,0,0);
    if(do_stereo){
      //      double eye_sep_dist=0.025/2;
      double eye_sep_dist=viewwindow->sbase.get()*
	(viewwindow->sr.get()?0.048:0.0125);
      Vector u, v;
      view.get_viewplane(aspect, 1.0, u, v);
      u.normalize();
      double zmid=(znear+zfar)/2.;
      eyesep=u*eye_sep_dist*zmid;
    }
    
#ifdef __sgi
    GLfloat realStylusMatrix[16], realPinchMatrix[16];
    int stylusID, pinchID;
    int stylus, pinch;
    GLfloat scale;
    char scalestr[512];
    
    //  --  BAWGL  -- 
    if( do_bawgl )
    {
      bawgl->getAllEyePositions();
      
      stylusID = bawgl->getControllerID(BAWGL_STYLUS);
      pinchID = bawgl->getControllerID(BAWGL_PINCH);
      
      bawgl->getControllerMatrix(stylusID, BAWGL_ONE, 
				 realStylusMatrix, BAWGL_REAL_SPACE);
      bawgl->getControllerState(stylusID, &stylus);
      bawgl->getControllerMatrix(pinchID, BAWGL_LEFT, 
				 realPinchMatrix, BAWGL_REAL_SPACE);
      bawgl->getControllerState(pinchID, &pinch);
    }
#endif
    
    for(int t=0;t<nframes;t++){
      int n=1;
#ifdef __sgi
      if( do_stereo || do_bawgl ) n=2;
      for(int i=0;i<n;i++){
	if( do_stereo || do_bawgl ){
#else
	  if( do_stereo ) n=2;
	  for(int i=0;i<n;i++){
	    if( do_stereo ){
#endif
	      glDrawBuffer(i==0?GL_BACK_LEFT:GL_BACK_RIGHT);
	    } else {
	      glDrawBuffer(GL_BACK);
	    }
	    
	    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	    glViewport(0, 0, xres, yres);
	    
	    double modeltime=t*dt+tbeg;
	    viewwindow->set_current_time(modeltime);
	    
#ifdef __sgi
	    if( do_bawgl ) // render head tracked stereo
	    {
	      bawgl->setViewPort(0, 0, xres, yres);
	      
	      if( i==1 )
	      {
		bawgl->setModelViewMatrix(BAWGL_RIGHT_EYE);
		bawgl->setProjectionMatrix(BAWGL_RIGHT_EYE);
	      }
	      else
	      {
		bawgl->setModelViewMatrix(BAWGL_LEFT_EYE);
		bawgl->setProjectionMatrix(BAWGL_LEFT_EYE);
	      }
	      
	      bawgl->setSurfaceView();
	      
	      glPushMatrix();
	      
	      bawgl->setVirtualView();
	    } else 
	      //  --  BAWGL  -- 
#endif
	    {  // render normal
	      glMatrixMode(GL_PROJECTION);
	      glLoadIdentity();
	      gluPerspective(fovy, aspect, znear, zfar);
	      glMatrixMode(GL_MODELVIEW);
	      glLoadIdentity();
	      Point eyep(view.eyep());
	      Point lookat(view.lookat());
	      if(do_stereo){
		if(i==0){
		  eyep-=eyesep;
		  if (!viewwindow->sr.get())
		    lookat-=eyesep;
		} else {
		  eyep+=eyesep;
		  if (!viewwindow->sr.get())
		    lookat+=eyesep;
		}
	      }
	      Vector up(view.up());
	      gluLookAt(eyep.x(), eyep.y(), eyep.z(),
			lookat.x(), lookat.y(), lookat.z(),
			up.x(), up.y(), up.z());
	    }
	    
	    // Set up Lighting
	    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	    Lighting& l=viewer->lighting;
	    int idx=0;
	    int ii;
	    for(ii=0;ii<l.lights.size();ii++){
	      Light* light=l.lights[ii];
	      light->opengl_setup(view, drawinfo, idx);
	    }
	    for(ii=0;ii<idx && ii<maxlights;ii++)
	      glEnable((GLenum)(GL_LIGHT0+ii));
	    for(;ii<maxlights;ii++)
	      glDisable((GLenum)(GL_LIGHT0+ii));
	    
	    // now set up the fog stuff
	    
	    glFogi(GL_FOG_MODE,GL_LINEAR);
	    glFogf(GL_FOG_START,float(znear));
	    glFogf(GL_FOG_END,float(zfar));
	    GLfloat bgArray[4];
	    bgArray[0]=bg.r(); 
	    bgArray[1]=bg.g(); 
	    bgArray[2]=bg.b(); 
	    bgArray[3]=1.0;
	    glFogfv(GL_FOG_COLOR, bgArray);
	    
	    // now make the ViewWindow setup its clipping planes...
	    viewwindow->setClip(drawinfo);
	    
            // UNICAM addition
            glGetDoublev (GL_MODELVIEW_MATRIX, get_depth_model);
            glGetDoublev (GL_PROJECTION_MATRIX, get_depth_proj);
            glGetIntegerv(GL_VIEWPORT, get_depth_view);
	    
	    // Draw it all...
	    current_time=modeltime;
	    viewwindow->do_for_visible(this, (ViewWindowVisPMF)&OpenGL::redraw_obj);
	    
#ifdef __sgi
	    //  --  BAWGL  -- 
	    if( do_bawgl ) // render stylus and pinch 'metaphores'
	    {
	      glPopMatrix();
	      
	      glPushMatrix();
	      glMultMatrixf(realStylusMatrix);
	      stylusCylinder[0]->draw(drawinfo, stylusMaterial[0], 
				      current_time);
	      
	      if( stylus == BAWGL_STYLUS_ON )
	      {
		stylusCylinder[1]->draw(drawinfo, stylusMaterial[2], 
					current_time);
		stylusTriangle[0]->draw(drawinfo, stylusMaterial[3], 
					current_time);
		stylusTriangle[1]->draw(drawinfo, stylusMaterial[4], 
					current_time);
		stylusTriangle[2]->draw(drawinfo, stylusMaterial[7], 
					current_time);
		stylusTriangle[3]->draw(drawinfo, stylusMaterial[8], 
					current_time);
	      }
	      else
	      {
		stylusCylinder[1]->draw(drawinfo, stylusMaterial[1], 
					current_time);
	      }
	      glPopMatrix();
	      
	      if( !bawgl->pick )
	      {
		glPushMatrix();
		glMultMatrixf(realPinchMatrix);
		pinchSphere->draw(drawinfo, pinchMaterial, 
				  current_time);
		glPopMatrix();
	      }
	      
	      if( bawgl->scale )
	      {
		glPushMatrix();
		glMultMatrixf(realPinchMatrix);
		scale = bawgl->scaleFrom - realPinchMatrix[13];
		
		glPushMatrix();
		if( scale > 0 )
		{
		  glPushMatrix();
		  glScalef(scale, 1.0, 1.0);
		  pinchCylinder[0]->draw(drawinfo, 
					 stylusMaterial[3], 
					 current_time);
		  glPopMatrix();
		  
		  glTranslatef(scale, 0.0, 0.0);
		  pinchSphere->draw(drawinfo, 
				    pinchMaterial, current_time);
		}
		else
		{
		  glPushMatrix();
		  glScalef(-scale, 1.0, 1.0);
		  pinchCylinder[1]->draw(drawinfo, 
					 stylusMaterial[3], 
					 current_time);
		  glPopMatrix();
		  
		  glTranslatef(scale, 0.0, 0.0);
		  pinchSphere->draw(drawinfo, 
				    pinchMaterial, 
				    current_time);
		}
		glPopMatrix();
		
		delete pinchText[0];
		sprintf(scalestr, "Scale: %.2f", bawgl->virtualViewScale);
		
		pinchText[0] = scinew GeomText(clString(scalestr),Point(1,1,1));
		pinchText[0]->draw(drawinfo, pinchMaterial, current_time);
		
		glPopMatrix();
	      }
	      
	      if( bawgl->navigate )
	      {
		glPushMatrix();
		glMultMatrixf(realPinchMatrix);
		
		scale = bawgl->navigateFrom - realPinchMatrix[13];
		
		glPushMatrix();
		if( scale > 0 )
		{
		  glPushMatrix();
		  glScalef(scale, 1.0, 1.0);
		  pinchCylinder[0]->draw(drawinfo, stylusMaterial[7], 
					 current_time);
		  glPopMatrix();
		  
		  glTranslatef(scale, 0.0, 0.0);
		  pinchSphere->draw(drawinfo, pinchMaterial, 
				    current_time);
		}
		else
		{
		  glPushMatrix();
		  glScalef(-scale, 1.0, 1.0);
		  pinchCylinder[1]->draw(drawinfo, stylusMaterial[7], 
					 current_time);
		  glPopMatrix();
		  
		  glTranslatef(scale, 0.0, 0.0);
		  pinchSphere->draw(drawinfo, pinchMaterial, 
				    current_time);
		}
		glPopMatrix();      
		
		delete pinchText[1];
		sprintf(scalestr, "Velocity: %.2f", -1000*bawgl->velocity);
		
		pinchText[1] = scinew GeomText(clString(scalestr),Point(1,1,1));
		pinchText[1]->draw(drawinfo, pinchMaterial, 
				   current_time);
		
		glPopMatrix();
		
	      }
	    }
	    //  --  BAWGL  -- 
#endif
	  }
	  
#if 0
	  if(viewwindow->drawimg.get()){
	    if(!imglist)
	      make_image();
	    else
	      glCallList(imglist);
	  }
#endif
	  
          // save z-buffer data
          if (CAPTURE_Z_DATA_HACK) {
            CAPTURE_Z_DATA_HACK = 0;
            glReadPixels( 0, 0,
                          xres, yres,
                          GL_DEPTH_COMPONENT, GL_FLOAT,
                          pixel_depth_data );
	    //            cerr << "(read from (0,0) to (" << xres << "," << yres << ")" << endl;
          }
	  
	  // Wait for the right time before swapping buffers
	  //TCLTask::unlock();
	  double realtime=t*frametime;
	  throttle.wait_for_time(realtime);
	  //TCLTask::lock();
	  TCL::execute("update idletasks");
	  
	  // Show the pretty picture
	  glXSwapBuffers(dpy, win);
#ifdef __sgi
#ifdef LIBIMAGE
	  if(saveprefix != ""){
	    // Save out the image...
	    char filename[200];
	    sprintf(filename, "%s%04d.rgb", saveprefix(), t);
	    unsigned short* reddata=scinew unsigned short[xres*yres];
	    unsigned short* greendata=scinew unsigned short[xres*yres];
	    unsigned short* bluedata=scinew unsigned short[xres*yres];
	    glReadPixels(0, 0, xres, yres, GL_RED, GL_UNSIGNED_SHORT, reddata);
	    glReadPixels(0, 0, xres, yres, GL_GREEN, GL_UNSIGNED_SHORT, greendata);
	    glReadPixels(0, 0, xres, yres, GL_BLUE, GL_UNSIGNED_SHORT, bluedata);
	    IMAGE* image=iopen(filename, "w", RLE(1), 3, xres, yres, 3);
	    unsigned short* rr=reddata;
	    unsigned short* gg=greendata;
	    unsigned short* bb=bluedata;
	    for(int y=0;y<yres;y++){
	      for(int x=0;x<xres;x++){
		rr[x]>>=8;
		gg[x]>>=8;
		bb[x]>>=8;
	      }
	      putrow(image, rr, y, 0);
	      putrow(image, gg, y, 1);
	      putrow(image, bb, y, 2);
	      rr+=xres;
	      gg+=xres;
	      bb+=xres;
	    }
	    iclose(image);
	    delete[] reddata;
	    delete[] greendata;
	    delete[] bluedata;
	  }
#endif // LIBIMAGE
#endif // __sgi
	}
	throttle.stop();
	double fps=nframes/throttle.time();
	int fps_whole=(int)fps;
	int fps_hund=(int)((fps-fps_whole)*100);
	ostringstream str;
	str << viewwindow->id << " setFrameRate " << fps_whole << "." << fps_hund;
	TCL::execute(str.str().c_str());
	viewwindow->set_current_time(tend);
      } else {
	// Just show the cleared screen
	viewwindow->set_current_time(tend);
	
#ifdef __sgi
	//  --  BAWGL  -- 
	if( do_stereo || do_bawgl ) {
	  glDrawBuffer(GL_BACK_LEFT);
	  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	  glDrawBuffer(GL_BACK_RIGHT);
        } else {
	  glDrawBuffer(GL_BACK);
	}
	//  --  BAWGL  -- 
#endif
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if(viewwindow->drawimg.get()){
#if 0
	  if(!imglist)
	    make_image();
	  else
	    glCallList(imglist);
#endif
	}
	glXSwapBuffers(dpy, win);
      }
      
      viewer->geomlock.readUnlock();
      
				// Look for errors
      GLenum errcode;
      while((errcode=glGetError()) != GL_NO_ERROR){
	cerr << "We got an error from GL: " << (char*)gluErrorString(errcode) << endl;
      }
      
				// Report statistics
      timer.stop();
      fpstimer.stop();
      double fps=nframes/fpstimer.time();
      fps+=0.05;			// Round to nearest tenth
      int fps_whole=(int)fps;
      int fps_tenths=(int)((fps-fps_whole)*10);
      fpstimer.clear();
      fpstimer.start();		// Start it running for next time
      ostringstream str;
      str << viewwindow->id << " updatePerf \"";
      str << drawinfo->polycount << " polygons in " << timer.time()
	  << " seconds\" \"" << drawinfo->polycount/timer.time()
	  << " polygons/second\"" << " \"" << fps_whole << "."
	  << fps_tenths << " frames/sec\"" << '\0';
      //    cerr <<"updatePerf: <" << str.str() << ">\n";	
      /***********************************/
      /* movie makin' movie-movie makin' */
      /***********************************/
      if (viewwindow->doingMovie) {
	
	clString segname(viewwindow->curName);
	int lasthash=-1;
	for (int ii=0; ii<segname.len(); ii++) {
	  if (segname()[ii] == '/') lasthash=ii;
	}
	clString pathname;
	if (lasthash == -1) pathname = "./";
	else pathname = segname.substr(0, lasthash+1);
	clString fname = segname.substr(lasthash+1, -1);
	
	//      cerr << "Saving a movie!\n";
	if( viewwindow->makeMPEG ){
	  if(!encoding_mpeg){
	    encoding_mpeg = true;
	    fname = fname + ".mpg";
	    StartMpeg( fname );
	  }
	  AddMpegFrame();
	} else { // dump each frame
	  /* if mpeg has just been turned off, close the file. */
	  if(encoding_mpeg){
	    encoding_mpeg = false;
	    EndMpeg();
	  }
	  unsigned char movie[10];
	  int startDiv = 100;
	  int idx=0;
	  int fi = viewwindow->curFrame;
	  while (startDiv >= 1) {
	    movie[idx] = '0' + fi/startDiv;
	    fi = fi - (startDiv)*(fi/startDiv);
	    startDiv /= 10;
	    idx++;
	  }
	  movie[idx] = 0;
	  fname = fname + ".raw";
	  clString framenum((char *)movie);
	  framenum = framenum + ".";
	  clString fullpath(pathname + framenum + fname);
	  cerr << "Dumping "<<fullpath<<"....  ";
	  dump_image(fullpath);
	  cerr << " done!\n";
	  viewwindow->curFrame++;
	}
      }
      else {
	if(encoding_mpeg) {// make sure we finish up mpeg that was in progress
	  encoding_mpeg = false;
	  EndMpeg();
	}
      }
      TCL::execute(str.str().c_str());
      TCLTask::unlock();
}
    
void OpenGL::hide()
{
  tkwin=0;
  if(current_drawer==this)
    current_drawer=0;
}


void OpenGL::get_pick(Viewer*, ViewWindow*, int x, int y,
		      GeomObj*& pick_obj, GeomPick*& pick_pick,
		      int& pick_index)
{
  send_pick_x=x;
  send_pick_y=y;
  send_mb.send(DO_PICK);
  for(;;){
    int r=recv_mb.receive();
    if(r != PICK_DONE){
      cerr << "WANTED A PICK!!! (got back " << r << endl;
    } else {
      pick_obj=ret_pick_obj;
      pick_pick=ret_pick_pick;
      pick_index=ret_pick_index;
      break;
    }
  }
}

void OpenGL::real_get_pick(Viewer*, ViewWindow* ViewWindow, int x, int y,
			   GeomObj*& pick_obj, GeomPick*& pick_pick,
			   int& pick_index)
{
  pick_obj=0;
  pick_pick=0;
  pick_index = 0x12345678;
  // Make ourselves current
  if(current_drawer != this){
    current_drawer=this;
    TCLTask::lock();
    glXMakeCurrent(dpy, win, cx);
    TCLTask::unlock();
  }
  // Setup the view...
  View view(viewwindow->view.get());
  viewer->geomlock.readLock();
  
  // Compute znear and zfar...
  double znear;
  double zfar;
  if(compute_depth(ViewWindow, view, znear, zfar)){
    // Setup picking...
    TCLTask::lock();
    
    GLuint pick_buffer[pick_buffer_size];
    glSelectBuffer(pick_buffer_size, pick_buffer);
    glRenderMode(GL_SELECT);
    glInitNames();
#if (_MIPS_SZPTR == 64)
    glPushName(0);
    glPushName(0);
    glPushName(0x12345678);
    glPushName(0x12345678);
    glPushName(0x12345678);
#else
    glPushName(0); //for the pick
    glPushName(0x12345678); //for the object
    glPushName(0x12345678); //for the object's face index
#endif

    if(!viewwindow->do_bawgl.get()){ //Regular flavor picking
      double aspect=double(xres)/double(yres);
      // XXX - UNICam change-- should be '1.0/aspect' not 'aspect' below
      double fovy=RtoD(2*Atan(1.0/aspect*Tan(DtoR(view.fov()/2.))));
      glViewport(0, 0, xres, yres);
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      GLint viewport[4];
      glGetIntegerv(GL_VIEWPORT, viewport);
      gluPickMatrix(x, viewport[3]-y, pick_window, pick_window, viewport);
      gluPerspective(fovy, aspect, znear, zfar);
      
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      Point eyep(view.eyep());
      Point lookat(view.lookat());
      Vector up(view.up());
      gluLookAt(eyep.x(), eyep.y(), eyep.z(),
		lookat.x(), lookat.y(), lookat.z(),
		up.x(), up.y(), up.z());
    }
    else { //BAWGL flavored picking setup!!!
      SCIBaWGL* bawgl = viewwindow->get_bawgl();
      bawgl->setModelViewMatrix(BAWGL_MIDDLE_EYE);
      bawgl->setPickProjectionMatrix(BAWGL_MIDDLE_EYE, x, y, pick_window);
      
      bawgl->setSurfaceView();
      bawgl->setVirtualView();      
    }

    drawinfo->lighting=0;
    drawinfo->set_drawtype(DrawInfoOpenGL::Flat);
    drawinfo->pickmode=1;
    //drawinfo->pickable=0;
    
    // Draw it all...
    viewwindow->do_for_visible(this, (ViewWindowVisPMF)&OpenGL::pick_draw_obj);
    
#if (_MIPS_SZPTR == 64)
    glPopName();
    glPopName();
    glPopName();
#else
    glPopName();
    glPopName();
#endif
    
    glFlush();
    int hits=glRenderMode(GL_RENDER);
    GLenum errcode;
    while((errcode=glGetError()) != GL_NO_ERROR){
      cerr << "We got an error from GL: " << (char*)gluErrorString(errcode) << endl;
    }
    TCLTask::unlock();
    GLuint min_z;
#if (_MIPS_SZPTR == 64)
    unsigned long hit_obj=0;
    GLuint hit_obj_index = 0x12345678;
    unsigned long hit_pick=0;
    GLuint hit_pick_index = 0x12345678;  // need for object indexing
#else
    GLuint hit_obj=0;
    //GLuint hit_obj_index = 0x12345678;  // need for object indexing
    GLuint hit_pick=0;
    GLuint hit_pick_index = 0x12345678;  // need for object indexing
#endif
    cerr << "hits=" << hits << endl;
    if(hits >= 1){
      int idx=0;
      min_z=0;
      int have_one=0;
      for (int h=0; h<hits; h++) {
	int nnames=pick_buffer[idx++];
	GLuint z=pick_buffer[idx++];
	//cerr << "h=" << h << ", nnames=" << nnames << ", z=" << z << endl;
	if (nnames > 1 && (!have_one || z < min_z)) {
	  min_z=z;
	  have_one=1;
	  idx++; // Skip Max Z
#if (_MIPS_SZPTR == 64)
	  idx+=nnames-5; // Skip to the last one...
	  unsigned int ho1=pick_buffer[idx++];
	  unsigned int ho2=pick_buffer[idx++];
	  hit_pick=((long)ho1<<32)|ho2;
	  //hit_obj_index = pick_buffer[idx++];
	  unsigned int hp1=pick_buffer[idx++];
	  unsigned int hp2=pick_buffer[idx++];
	  hit_obj=((long)hp1<<32)|hp2;
	  hit_pick_index = pick_buffer[idx++];
#else
	  // hit_obj=pick_buffer[idx++];
	  // hit_obj_index=pick_buffer[idx++];
	  //for(int i=idx; i<idx+nnames; ++i) cerr << pick_buffer[i] << endl;
	  idx+=nnames-3; // Skip to the last one...
	  hit_pick=pick_buffer[idx++];
	  hit_obj=pick_buffer[idx++];
	  hit_pick_index=pick_buffer[idx++];
#endif
	  cerr << "new min... (obj=" << hit_obj
	       << ", pick="          << hit_pick
	       << ", index = "       << hit_pick_index << ")\n";
	} else {
	  idx+=nnames+1;
	}
      }
      
      pick_obj=(GeomObj*)hit_obj;
      pick_pick=(GeomPick*)hit_pick;
      pick_obj->getId(pick_index); //(int)hit_pick_index;
      cerr << "pick_pick=" << pick_pick << ", pick_index="<<pick_index<<endl;
    }
  }
  viewer->geomlock.readUnlock();
}

void OpenGL::dump_image(const clString& name, const clString& type) {
  ofstream dumpfile(name());
  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT,vp);
  int n=3*vp[2]*vp[3];
  cerr << "Dumping: " << vp[2] << "x" << vp[3] << endl;
  unsigned char* pxl=scinew unsigned char[n];
  glPixelStorei(GL_PACK_ALIGNMENT,1);
  glReadBuffer(GL_FRONT);
  glReadPixels(0,0,vp[2],vp[3],GL_RGB,GL_UNSIGNED_BYTE,pxl);
  dumpfile.write((const char *)pxl,n);
  delete[] pxl;
}

void OpenGL::put_scanline(int y, int width, Color* scanline, int repeat)
{
  float* pixels=scinew float[width*3];
  float* p=pixels;
  int i;
  for(i=0;i<width;i++){
    *p++=scanline[i].r();
    *p++=scanline[i].g();
    *p++=scanline[i].b();
  }
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glTranslated(-1, -1, 0);
  glScaled(2./xres, 2./yres, 1.0);
  glDepthFunc(GL_ALWAYS);
  glDrawBuffer(GL_FRONT);
  for(i=0;i<repeat;i++){
    glRasterPos2i(0, y+i);
    glDrawPixels(width, 1, GL_RGB, GL_FLOAT, pixels);
  }
  glDepthFunc(GL_LEQUAL);
  glDrawBuffer(GL_BACK);
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  delete[] pixels;
}

void OpenGL::pick_draw_obj(Viewer* viewer, ViewWindow*, GeomObj* obj)
{
#if (_MIPS_SZPTR == 64)
  unsigned long o=(unsigned long)obj;
  unsigned int o1=(o>>32)&0xffffffff;
  unsigned int o2=o&0xffffffff;
  glPopName();
  glPopName();
  glPopName();
  glPushName(o1);
  glPushName(o2);
  glPushName(0x12345678);
#else
  glPopName();
  glPushName((GLuint)obj);
  glPushName(0x12345678);
#endif
  obj->draw(drawinfo, viewer->default_matl.get_rep(), current_time);
}

void OpenGL::redraw_obj(Viewer* viewer, ViewWindow* viewwindow, GeomObj* obj)
{
  drawinfo->viewwindow = viewwindow;
  obj->draw(drawinfo, viewer->default_matl.get_rep(), current_time);
}

void ViewWindow::setState(DrawInfoOpenGL* drawinfo,clString tclID)
{
  clString val;
  double dval;
  clString type(tclID+"-"+"type");
  clString lighting(tclID+"-"+"light");
  clString fog(tclID+"-"+"fog");
  clString cull(tclID+"-"+"cull");
  clString dl(tclID+"-"+"dl");
  clString debug(tclID+"-"+"debug");
  clString psize(tclID+"-"+"psize");
  clString movie(tclID+"-"+"movie");
  clString movieName(tclID+"-"+"movieName");
  clString movieFrame(tclID+"-"+"movieFrame");
  clString use_clip(tclID+"-"+"clip");
  
  if (!get_gui_stringvar(id,type,val)) {
    cerr << "Error illegal name!\n";
    return;
  }
  else {
    if(val == "Wire"){
      drawinfo->set_drawtype(DrawInfoOpenGL::WireFrame);
      drawinfo->lighting=0;
    } else if(val == "Flat"){
      drawinfo->set_drawtype(DrawInfoOpenGL::Flat);
      drawinfo->lighting=0;
    } else if(val == "Gouraud"){
      drawinfo->set_drawtype(DrawInfoOpenGL::Gouraud);
      drawinfo->lighting=1;
    }
    else if (val == "Default") {
      //	    drawinfo->currently_lit=drawinfo->lighting;
      //	    drawinfo->init_lighting(drawinfo->lighting);
      clString globals("global");
      setState(drawinfo,globals);	    
      return; // if they are using the default, con't change
    } else {
      cerr << "Unknown shading(" << val << "), defaulting to phong" << endl;
      drawinfo->set_drawtype(DrawInfoOpenGL::Gouraud);
      drawinfo->lighting=1;
    }
    
    // now see if they want a bounding box...
    
    if (get_gui_stringvar(id,debug,val)) {
      if (val == "0")
	drawinfo->debug = 0;
      else
	drawinfo->debug = 1;
    }	
    else {
      cerr << "Error, no debug level set!\n";
      drawinfo->debug = 0;
    }
    
    if (get_gui_doublevar(id,psize,dval)) {
      drawinfo->point_size = dval;
    }
    
    if (get_gui_stringvar(id,use_clip,val)) {
      if (val == "0")
	drawinfo->check_clip = 0;
      else
	drawinfo->check_clip = 1;
    }	
    else {
      cerr << "Error, no clipping info\n";
      drawinfo->check_clip = 0;
    }
    // only set with globals...
    if (get_gui_stringvar(id,movie,val)) {
      get_gui_stringvar(id,movieName,curName);
      clString curFrameStr;
      get_gui_stringvar(id,movieFrame,curFrameStr);
      //	    curFrameStr.get_int(curFrame);
      //	    cerr << "curFrameStr="<<curFrameStr<<"  curFrame="<<curFrame<<"\n";
      if (val == "0") {
	doingMovie = 0;
	makeMPEG = 0;
      } else {
	if (!doingMovie) {
	  doingMovie = 1;
	  curFrame=0;
	  if( val == "1" )
	    makeMPEG = 0;
	  else
	    makeMPEG = 1;
	}
      }
    }
    
    drawinfo->init_clip(); // set clipping 
    
    if (get_gui_stringvar(id,cull,val)) {
      if (val == "0")
	drawinfo->cull = 0;
      else
	drawinfo->cull = 1;
    }	
    else {
      cerr << "Error, no culling info\n";
      drawinfo->cull = 0;
    }
    if (get_gui_stringvar(id,dl,val)) {
      if (val == "0")
	drawinfo->dl = 0;
      else
	drawinfo->dl = 1;
    }	
    else {
      cerr << "Error, no display list info\n";
      drawinfo->dl = 0;
    }
    if (!get_gui_stringvar(id,lighting,val))
      cerr << "Error, no lighting!\n";
    else {
      if (val == "0"){
	drawinfo->lighting=0;
      }
      else if (val == "1") {
	drawinfo->lighting=1;
      }
      else {
	cerr << "Unknown lighting setting(" << val << "\n";
      }
      
      if (get_gui_stringvar(id,fog,val)) {
	if (val=="0"){
	  drawinfo->fog=0;
	}
	else {
	  drawinfo->fog=1;
	}
      }
      else {
	cerr << "Fog not defined properly!\n";
	drawinfo->fog=0;
      }
      
    }
  }
  drawinfo->currently_lit=drawinfo->lighting;
  drawinfo->init_lighting(drawinfo->lighting);
  
  
}

void ViewWindow::setDI(DrawInfoOpenGL* drawinfo,clString name)
{
  ObjTag* vis;
  
  viter = visible.find(name);
  if (viter != visible.end()) { // if found
    vis = (*viter).second;
    setState(drawinfo,to_string(vis->tagid));
  }
}

// set the bits for the clipping planes that are on...

void ViewWindow::setClip(DrawInfoOpenGL* drawinfo)
{
  clString val;
  int i;
  
  drawinfo->clip_planes = 0; // set them all of for default
  clString num_clip("clip-num");
  
  if (get_gui_stringvar(id,"clip-visible",val) && 
      get_gui_intvar(id,num_clip,i)) {
    
    int cur_flag = CLIP_P5;
    if ( (i>0 && i<7) ) {
      while(i--) {
	
	clString vis("clip-visible-"+to_string(i+1));
	
	
	if (get_gui_stringvar(id,vis,val)) {
	  if (val == "1") {
	    double plane[4];
	    clString nx("clip-normal-x-"+to_string(i+1));
	    clString ny("clip-normal-y-"+to_string(i+1));
	    clString nz("clip-normal-z-"+to_string(i+1));
	    clString nd("clip-normal-d-"+to_string(i+1));
	    
	    int rval=0;
	    
	    rval = get_gui_doublevar(id,nx,plane[0]);
	    rval = get_gui_doublevar(id,ny,plane[1]);
	    rval = get_gui_doublevar(id,nz,plane[2]);
	    rval = get_gui_doublevar(id,nd,plane[3]);
	    
	    double mag = sqrt(plane[0]*plane[0] +
			      plane[1]*plane[1] +
			      plane[2]*plane[2]);
	    plane[0] /= mag;
	    plane[1] /= mag;
	    plane[2] /= mag;
	    plane[3] = -plane[3]; // so moves in planes direction...
	    glClipPlane((GLenum)(GL_CLIP_PLANE0+i),plane);	
	    
	    if (drawinfo->check_clip)
	      glEnable((GLenum)(GL_CLIP_PLANE0+i));
	    else
	      glDisable((GLenum)(GL_CLIP_PLANE0+i));
	    
	    drawinfo->clip_planes |= cur_flag;
	    
	    if (!rval ) {
	      cerr << "Error, variable is hosed!\n";
	    }
	  }
	  else {
	    glDisable((GLenum)(GL_CLIP_PLANE0+i));
	  }
	  
	}
	cur_flag >>= 1; // shift the bit we are looking at...
      }
    }
  }
}


void GeomViewerItem::draw(DrawInfoOpenGL* di, Material *m, double time)
{
  // here we need to query the ViewWindow with our name and give it our
  // di so it can change things if they need to be...
  di->viewwindow->setDI(di,name);
  
  // lets get the childs bounding box, and draw it...
  
  BBox bb;
  //    child->reset_bbox();
  
  child->get_bounds(bb);
  if(!bb.valid())
    return;
  
  // might as well try and draw the arcball also...
  
  Point min,max;
  
  min = bb.min();
  max = bb.max();
  if (!di->debug)
    child->draw(di,m,time);
  
  if (di->debug) {
    
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glDepthMask(GL_FALSE);
    
    glColor4f(1.0,0.0,1.0,0.2);
    
    glDisable(GL_LIGHTING);	
    
    glBegin(GL_QUADS);
    //front
    glVertex3d(max.x(),min.y(),max.z());
    //	glColor4f(0.0,1.0,0.0,0.8);
    glVertex3d(max.x(),max.y(),max.z());
    glColor4f(0.0,1.0,0.0,0.2);
    glVertex3d(min.x(),max.y(),max.z());
    glVertex3d(min.x(),min.y(),max.z());
    //back
    glVertex3d(max.x(),max.y(),min.z());
    glVertex3d(max.x(),min.y(),min.z());
    //	glColor4f(1.0,0.0,0.0,0.8);
    glVertex3d(min.x(),min.y(),min.z());
    glColor4f(0.0,1.0,0.0,0.2);
    glVertex3d(min.x(),max.y(),min.z());
    
    glColor4f(1.0,0.0,0.0,0.2);
    
    //left
    glVertex3d(min.x(),min.y(),max.z());
    glVertex3d(min.x(),max.y(),max.z());
    glVertex3d(min.x(),max.y(),min.z());
    //	glColor4f(1.0,0.0,0.0,0.8);
    glVertex3d(min.x(),min.y(),min.z());
    glColor4f(1.0,0.0,0.0,0.2);
    
    //right
    glVertex3d(max.x(),min.y(),min.z());
    glVertex3d(max.x(),max.y(),min.z());
    //	glColor4f(0.0,1.0,0.0,0.8);
    glVertex3d(max.x(),max.y(),max.z());
    glColor4f(1.0,0.0,0.0,0.2);
    glVertex3d(max.x(),min.y(),max.z());
    
    
    glColor4f(0.0,0.0,1.0,0.2);
    
    //top
    glVertex3d(min.x(),max.y(),max.z());
    //	glColor4f(0.0,1.0,0.0,0.8);
    glVertex3d(max.x(),max.y(),max.z());
    glColor4f(0.0,0.0,1.0,0.2);
    glVertex3d(max.x(),max.y(),min.z());
    glVertex3d(min.x(),max.y(),min.z());
    //bottom
    //	glColor4f(1.0,0.0,0.0,0.8);
    glVertex3d(min.x(),min.y(),min.z());
    glColor4f(0.0,0.0,1.0,0.2);
    glVertex3d(max.x(),min.y(),min.z());
    glVertex3d(max.x(),min.y(),max.z());
    glVertex3d(min.x(),min.y(),max.z());
    
    glEnd();
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_CULL_FACE);
  }
}

#define GETCONFIG(attrib) \
if(glXGetConfig(dpy, &vinfo[i], attrib, &value) != 0){\
  args.error("Error getting attribute: " #attrib); \
  return; \
}

void OpenGL::listvisuals(TCLArgs& args)
{
  TCLTask::lock();
  
  cerr << "Calling allow..." << getpid() << "\n";
  Thread::allow_sgi_OpenGL_page0_sillyness();
  Tk_Window topwin=Tk_NameToWindow(the_interp,
				   const_cast<char *>(args[2]()),
				   Tk_MainWindow(the_interp));
  if(!topwin){
    cerr << "Unable to locate window!\n";
    TCLTask::unlock();
    return;
  }
  dpy=Tk_Display(topwin);
  int screen=Tk_ScreenNumber(topwin);
  Array1<clString> visualtags;
  Array1<int> scores;
  visuals.remove_all();
  int nvis;
  XVisualInfo* vinfo=XGetVisualInfo(dpy, 0, NULL, &nvis);
  if(!vinfo){
    args.error("XGetVisualInfo failed");
    return;
  }
  int i;
  for(i=0;i<nvis;i++){
    int score=0;
    int value;
    GETCONFIG(GLX_USE_GL);
    if(!value)
      continue;
    GETCONFIG(GLX_RGBA);
    if(!value)
      continue;
    GETCONFIG(GLX_LEVEL);
    if(value != 0)
      continue;
    if(vinfo[i].screen != screen)
      continue;
    char buf[20];
    sprintf(buf, "id=%02x, ", (unsigned int)(vinfo[i].visualid));
    clString tag(buf);
    GETCONFIG(GLX_DOUBLEBUFFER);
    if(value){
      score+=200;
      tag+=clString("double, ");
    } else {
      tag+=clString("single, ");
    }
    GETCONFIG(GLX_STEREO);
    if(value){
      score+=1;
      tag+=clString("stereo, ");
    }
    tag+=clString("rgba=");
    GETCONFIG(GLX_RED_SIZE);
    tag+=to_string(value)+":";
    score+=value;
    GETCONFIG(GLX_GREEN_SIZE);
    tag+=to_string(value)+":";
    score+=value;
    GETCONFIG(GLX_BLUE_SIZE);
    tag+=to_string(value)+":";
    score+=value;
    GETCONFIG(GLX_ALPHA_SIZE);
    tag+=to_string(value);
    score+=value;
    GETCONFIG(GLX_DEPTH_SIZE);
    tag+=clString(", depth=")+to_string(value);
    score+=value*5;
    GETCONFIG(GLX_STENCIL_SIZE);
    tag+=clString(", stencil=")+to_string(value);
    tag+=clString(", accum=");
    GETCONFIG(GLX_ACCUM_RED_SIZE);
    tag+=to_string(value)+":";
    GETCONFIG(GLX_ACCUM_GREEN_SIZE);
    tag+=to_string(value)+":";
    GETCONFIG(GLX_ACCUM_BLUE_SIZE);
    tag+=to_string(value)+":";
    GETCONFIG(GLX_ACCUM_ALPHA_SIZE);
    tag+=to_string(value);
#ifdef __sgi
    tag+=clString(", samples=");
    GETCONFIG(GLX_SAMPLES_SGIS);
    if(value)
      score+=50;
#endif
    tag+=to_string(value);
    
    tag+=clString(", score=")+to_string(score);
    //cerr << score << ": " << tag << '\n';
    
    visualtags.add(tag);
    visuals.add(&vinfo[i]);
    scores.add(score);
  }
  for(i=0;i<scores.size()-1;i++){
    for(int j=i+1;j<scores.size();j++){
      if(scores[i] < scores[j]){
	// Swap...
	int tmp1=scores[i];
	scores[i]=scores[j];
	scores[j]=tmp1;
	clString tmp2=visualtags[i];
	visualtags[i]=visualtags[j];
	visualtags[j]=tmp2;
	XVisualInfo* tmp3=visuals[i];
	visuals[i]=visuals[j];
	visuals[j]=tmp3;
      }
    }
  }
  args.result(TCLArgs::make_list(visualtags));
  TCLTask::unlock();
}

void OpenGL::setvisual(const clString& wname, int which, int width, int height)
{
  tkwin=0;
  current_drawer=0;
  //cerr << "choosing visual " << which << '\n';
  TCL::execute(clString("opengl ")+wname+" -visual "+to_string((int)visuals[which]->visualid)+" -direct true -geometry "+to_string(width)+"x"+to_string(height));
  //cerr << clString("opengl ")+wname+" -visual "+to_string((int)visuals[which]->visualid)+" -direct true -geometry "+to_string(width)+"x"+to_string(height) << endl;
  //cerr << "done choosing visual\n";
  
  myname = wname;
}

void OpenGL::saveImage(const clString& fname,
		       const clString& type) //= "raw")
{
  send_mb.send(DO_IMAGE);
  img_mb.send(ImgReq(fname,type));
}

void OpenGL::real_saveImage(const clString& name,
			    const clString& type) //= "raw")
{
  GLint vp[4];
  
  if(current_drawer != this){
    current_drawer=this;
    TCLTask::lock();
    glXMakeCurrent(dpy, win, cx);
    TCLTask::unlock();
  }
  
  glGetIntegerv(GL_VIEWPORT,vp);
  int n=3*vp[2]*vp[3];
  unsigned char* pxl=scinew unsigned char[n];
  glPixelStorei(GL_PACK_ALIGNMENT,1);
  glReadBuffer(GL_FRONT);
  glReadPixels(0,0,vp[2],vp[3],GL_RGB,GL_UNSIGNED_BYTE,pxl);
  
  if(type == "raw"){
    cerr << "Saving raw file "<<name<<":  size = " << vp[2] << "x" << vp[3] << endl;
    ofstream dumpfile(name());
    dumpfile.write((const char *)pxl,n);
    dumpfile.close();
  }
#ifdef __sgi
  else if(type == "rgb" || type == "ppm" || type == "jpg" ){
    cerr << "Saving file "<< name <<endl;
    iflSize dims(vp[2], vp[3], 3);
    iflFileConfig fc(&dims, iflUChar);
    iflStatus sts;
    iflFile* file = iflFile::create(name(), NULL, &fc, NULL, &sts);
    sts = file->setTile(0, 0, 0, vp[2], vp[3], 1, pxl);
    file->close();
  } 
#endif
  else {
    cerr<<"Error unknown image file type\n";
  }
  delete[] pxl;
}


void OpenGL::getData(int datamask, FutureValue<GeometryData*>* result)
{
  send_mb.send(DO_GETDATA);
  get_mb.send(GetReq(datamask, result));
}

void OpenGL::real_getData(int datamask, FutureValue<GeometryData*>* result)
{
  GeometryData* res = new GeometryData;
  if(datamask&GEOM_VIEW){
    res->view=new View(lastview);
    res->xres=xres;
    res->yres=yres;
    res->znear=znear;
    res->zfar=zfar;
  }
  if(datamask&(GEOM_COLORBUFFER|GEOM_DEPTHBUFFER)){
    TCLTask::lock();
  }
  if(datamask&GEOM_COLORBUFFER){
    ColorImage* img = res->colorbuffer = new ColorImage(xres, yres);
    float* data=new float[xres*yres*3];
    cerr << "xres=" << xres << ", yres=" << yres << endl;
    WallClockTimer timer;
    timer.start();
    glReadPixels(0, 0, xres, yres, GL_RGB, GL_FLOAT, data);
    timer.stop();
    cerr << "done in " << timer.time() << " seconds\n";
    float* p=data;
    for(int y=0;y<yres;y++){
      for(int x=0;x<xres;x++){
	img->put_pixel(x, y, Color(p[0], p[1], p[2]));
	p+=3;
      }
    }
    delete[] data;
  }
  if(datamask&GEOM_DEPTHBUFFER){
    DepthImage* img=res->depthbuffer=new DepthImage(xres, yres);
    unsigned int* data=new unsigned int[xres*yres*3];
    cerr << "reading depth...\n";
    WallClockTimer timer;
    timer.start();
    glReadPixels(0, 0, xres, yres, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, data);
    timer.stop();
    cerr << "done in " << timer.time() << " seconds\n";
    unsigned int* p=data;
    for(int y=0;y<yres;y++){
      for(int x=0;x<xres;x++){
	img->put_pixel(x, y, (*p++)*(1./4294967295.));
      }
    }
    delete[] data;
  }
  if(datamask&(GEOM_COLORBUFFER|GEOM_DEPTHBUFFER)){
    GLenum errcode;
    while((errcode=glGetError()) != GL_NO_ERROR){
      cerr << "We got an error from GL: " << (char*)gluErrorString(errcode) << endl;
    }
    TCLTask::unlock();
  }
  result->send(res);
}

void OpenGL::StartMpeg(const clString& fname)
{
  // let's get a file pointer pointing to the output file
#ifdef MPEG
  output=fopen(fname(), "w");
  if (!output){
    cerr<<"Failed to open file "<< fname()<<" for writing\n";
    return;
  }
  // get the default options
  MPEGe_default_options( &options );
  // then change a couple
  strcpy(options.frame_pattern, "II");
  // was ("IIIIIIIIIIIIIII");
  options.search_range[1]=0;
  if( !MPEGe_open(output, &options ) ){
    cerr<<"MPEGe library initialisation failure!:"<<options.error<<endl;
    return;
  }
#endif // MPEG
}

void OpenGL::AddMpegFrame()
{
#ifdef MPEG
  static ImVfb *image=NULL; /* we only wnat to alloc memory for these once */
  int width, height;
  ImVfbPtr ptr;

  cerr<<"Adding Mpeg Frame\n";
  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT,vp);

  width = vp[2];
  height = vp[3];

  // int n=3*width*height;

  //  memcpy( ptr, pxl, n*sizeof(unsigned char));

  // set up the ImVfb used to store the image 
  if( !image ){
    image=MPEGe_ImVfbAlloc( width,height, IMVFBRGB, TRUE );
    if( !image ){
      cerr<<"Couldn't allocate memory for frame buffer\n";
      exit(2);
    }
  }

  // get to the first pixel in the image
  ptr=ImVfbQPtr( image,0,0 );
  glPixelStorei(GL_PACK_ALIGNMENT,1);
  glReadBuffer(GL_FRONT);
  glReadPixels(0,0,width,height,GL_RGB,GL_UNSIGNED_BYTE,ptr);

  int r=3*width;
  static unsigned char* row = 0;
  if( !row )
    row = scinew unsigned char[r];
  unsigned char* p0, *p1;

  int k, j;
  for(k = height -1, j = 0; j < height/2; k--, j++){
    p0 = ptr + r*j;
    p1 = ptr + r*k;
    memcpy( row, p0, r);
    memcpy( p0, p1, r);
    memcpy( p1, row, r);
  }
    
  if( !MPEGe_image(image, &options) ){
    cerr<<"MPEGe_image failure:"<<options.error<<endl;
  }
#endif // MPEG
}

void OpenGL::EndMpeg()
{
#ifdef MPEG
  if( !MPEGe_close(&options) ){
    cerr<<"Had a bit of difficulty closing the file:"<<options.error;
  }
  
  fclose(output);
  cerr<<"Ending Mpeg\n";
#endif // MPEG
}

// return world-space depth to point under pixel (x, y)
int OpenGL::pick_scene( int x, int y, Point *p )
{
  // y = 0 is bottom of screen (not top of screen, which is what X
  // events reports)
  y = (yres - 1) - y;
  int index = x + (y * xres);
  double z = pixel_depth_data[index];
  //    cerr << "z="<<z<<"\n";
  if (p) {
    // unproject the window point (x, y, z)
    GLdouble world_x, world_y, world_z;
    gluUnProject(x, y, z,
		 get_depth_model, get_depth_proj, get_depth_view,
		 &world_x, &world_y, &world_z);
    
    *p = Point(world_x, world_y, world_z);
  }
  
  // if z is close to 1, then assume no object was picked
  return (z < .999999);
}

GetReq::GetReq()
{
}

GetReq::GetReq(int datamask, FutureValue<GeometryData*>* result)
  : datamask(datamask), result(result)
{
}

ImgReq::ImgReq()
{
}
ImgReq::ImgReq(const clString& n, const clString& t)
  : name(n), type(t)
{
}

} // End namespace SCIRun
