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
#include <Dataflow/Modules/Render/logo.h>
#include <Core/Containers/StringUtil.h>
#ifdef __APPLE__
#include <float.h>
#define MAXDOUBLE DBL_MAX
#else
#include <values.h>
#endif

// CollabVis code begin
#ifdef HAVE_COLLAB_VIS
#include <Core/Datatypes/Image.h>
#endif
// CollabVis code end

#ifdef HAVE_MAGICK
namespace C_Magick {
#include <magick/magick.h>
}
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


static map<string, ObjTag*>::iterator viter;


int CAPTURE_Z_DATA_HACK = 0;

static OpenGL* current_drawer=0;
static const int pick_buffer_size = 512;
static const double pick_window = 10.0;



bool
OpenGL::query(GuiInterface* gui)
{
  gui->lock();
  int have_opengl=glXQueryExtension
    (Tk_Display(Tk_MainWindow(the_interp)), NULL, NULL);
  gui->unlock();
  if (!have_opengl)
    cerr << "glXQueryExtension() returned NULL.\n"
      "** XFree86 NOTE **  Do you have the line 'Load \"glx\"'"
      " in the Modules section of your XF86Config file?\n";
  return have_opengl;
}


OpenGL::OpenGL(GuiInterface* gui) :
  gui(gui),
  helper(0),
  tkwin(0),
  do_hi_res(false),
  encoding_mpeg_(false),

  send_mb("OpenGL renderer send mailbox",10),
  recv_mb("OpenGL renderer receive mailbox", 10),
  get_mb("OpenGL renderer request mailbox", 5),
  img_mb("OpenGL renderer image data mailbox", 5),
  // CollabVis Code begin
#ifdef HAVE_COLLAB_VIS
  doZTexView(false),
  doZTexTransform(false),
#endif
  helper_thread_(0),
  dead_(false)
  // CollabVis code end
{
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

  pinchText[0] = scinew GeomText("",Point(1,1,1));
  pinchText[1] = scinew GeomText("",Point(1,1,1));

  pinchCylinder[0] = scinew GeomCappedCylinder(Point(0,0,0), Point(1,0,0), 0.2, 20, 10);
  pinchCylinder[1] = scinew GeomCappedCylinder(Point(0,0,0), Point(-1,0,0), 0.2, 20, 10);
}


OpenGL::~OpenGL()
{
  // Finish up the mpeg that was in progress.
  if (encoding_mpeg_)
  {
    encoding_mpeg_ = false;
    EndMpeg();
  }

  int r;
  while (send_mb.tryReceive(r)) ;
  while (recv_mb.tryReceive(r)) ;

  fpstimer.stop();
}


class OpenGLHelper : public Runnable {
  OpenGL* opengl;
public:
  OpenGLHelper(OpenGL* opengl);
  virtual ~OpenGLHelper();
  virtual void run();
};


OpenGLHelper::OpenGLHelper(OpenGL* opengl) :
  opengl(opengl)
{
}


OpenGLHelper::~OpenGLHelper()
{
}


void
OpenGLHelper::run()
{
  Thread::allow_sgi_OpenGL_page0_sillyness();
  opengl->redraw_loop();
}


void
OpenGL::redraw(Viewer* s, ViewWindow* r, double _tbeg, double _tend,
	       int _nframes, double _framerate)
{
  if (dead_) return;
  viewer_ = s;
  viewwindow=r;
  tbeg=_tbeg;
  tend=_tend;
  nframes=_nframes;
  framerate=_framerate;
  // This is the first redraw - if there is not an OpenGL thread,
  // start one...
  if(!helper)
  {
    my_openglname= "OpenGL: " + myname_;
    helper=new OpenGLHelper(this);
    helper_thread_ = new Thread(helper, my_openglname.c_str());
  }

  send_mb.send(DO_REDRAW);
  int rc=recv_mb.receive();
  if(rc != REDRAW_DONE)
  {
    cerr << "Wanted redraw_done, but got: " << r << "\n";
  }
}


void
OpenGL::kill_helper()
{
  // kill the helper thread
  dead_ = true;
  if (helper_thread_)
  {
    send_mb.send(86);
    helper_thread_->join();
    helper_thread_ = 0;
  }
}


void
OpenGL::redraw_loop()
{
  int r,resx,resy;
  string fname, ftype;
  // Tell the ViewWindow that we are started...
  TimeThrottle throttle;
  throttle.start();
  double newtime=0;
  for(;;)
  {
    int nreply=0;
    if(viewwindow->inertia_mode)
    {
      double current_time=throttle.time();
      if(framerate==0)
	framerate=30;
      double frametime=1./framerate;
      double delta=current_time-newtime;
      if(delta > 1.5*frametime)
      {
	framerate=1./delta;
	frametime=delta;
	newtime=current_time;
      }
      if(delta > .85*frametime)
      {
	framerate*=.9;
	frametime=1./framerate;
	newtime=current_time;
      }
      else if(delta < .5*frametime)
      {
	framerate*=1.1;
	if(framerate>30)
	{
	  framerate=30;
	}
	frametime=1./framerate;
	newtime=current_time;
      }
      newtime+=frametime;
      throttle.wait_for_time(newtime);

      while (send_mb.tryReceive(r))
      {
	if (r == 86)
	{
	  throttle.stop();
	  return;
	}
	else if (r == DO_PICK)
	{
	  real_get_pick(viewer_, viewwindow, send_pick_x, send_pick_y,
			ret_pick_obj, ret_pick_pick, ret_pick_index);
	  recv_mb.send(PICK_DONE);
	}
	else if(r== DO_GETDATA)
	{
	  GetReq req(get_mb.receive());
	  real_getData(req.datamask, req.result);
	}
	else if(r== DO_IMAGE)
	{
	  ImgReq req(img_mb.receive());
	  do_hi_res = true;
	  fname = req.name;
	  ftype = req.type;
	  resx = req.resx;
	  resy = req.resy;
	}
	else
	{
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

      if (viewwindow->inertia_mode == 1)
      {
	tmpview.eyep((z_a*(viewwindow->eye_dist)) + tmpview.lookat().vector());
	viewwindow->view.set(tmpview);
      }
      else if (viewwindow->inertia_mode == 2)
      {
	tmpview.lookat(tmpview.eyep()-(z_a*(viewwindow->eye_dist)).vector());
	viewwindow->view.set(tmpview);
      }

    }
    else
    {
      for (;;)
      {
	int r=send_mb.receive();
	if (r == 86)
	{
	  throttle.stop();
	  return;
	}
	else if (r == DO_PICK)
	{
	  real_get_pick(viewer_, viewwindow, send_pick_x, send_pick_y,
			ret_pick_obj, ret_pick_pick, ret_pick_index);
	  recv_mb.send(PICK_DONE);
	}
	else if(r== DO_GETDATA)
	{
	  GetReq req(get_mb.receive());
	  real_getData(req.datamask, req.result);
	}
	else if(r== DO_IMAGE)
	{
	  ImgReq req(img_mb.receive());
	  do_hi_res = true;
	  fname = req.name;
	  ftype = req.type;
	  resx = req.resx;
	  resy = req.resy;
	}
	else
	{
	  nreply++;
	  break;
	}
      }
      newtime=throttle.time();
      throttle.stop();
      throttle.clear();
      throttle.start();
    }

    if(do_hi_res)
    {
      render_and_save_image(resx,resy,fname,ftype);
      do_hi_res=false;
    }

    redraw_frame();
    for(int i=0;i<nreply;i++) {
      recv_mb.send(REDRAW_DONE);
    }
  } // end for(;;)
}



void
OpenGL::render_and_save_image(int x, int y,
			      const string& fname, const string &ftype)
{
#ifndef HAVE_MAGICK
  if (ftype != "ppm" && ftype != "raw")
  {
    cerr << "Error - ImageMagick is not enabled, can only save .ppm or .raw files.\n";
    return;
  }
#endif

  cerr << "Saving Image: " << fname << " with width=" << x
       << " and height=" << y <<"...\n";


  // FIXME: this next line was apparently meant to raise the Viewer to the
  //        top... but it doesn't actually seem to work
  Tk_RestackWindow(tkwin,Above,NULL);


  gui->lock();
  // Make sure our GL context is current
  if(current_drawer != this)
  {
    current_drawer=this;
    glXMakeCurrent(dpy, win, cx);
  }
  deriveFrustum();
  gui->unlock();

  // Get Viewport dimensions
  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT,vp);

  hi_res.resx = x;
  hi_res.resy = y;
  // The EXACT # of screen rows and columns needed to render the image
  hi_res.ncols = (double)hi_res.resx/(double)vp[2];
  hi_res.nrows = (double)hi_res.resy/(double)vp[3];

  // The # of screen rows and columns that will be rendered to make the image
  const int nrows = (int)ceil(hi_res.nrows);
  const int ncols = (int)ceil(hi_res.ncols);

  ofstream *image_file;
#ifdef HAVE_MAGICK
  C_Magick::Image *image;
  C_Magick::ImageInfo *image_info;
#endif
  int channel_bytes, num_channels;
  bool do_magick;

  if (ftype == "ppm" || ftype == "raw")
  {
    image_file = scinew ofstream(fname.c_str());
    channel_bytes = 1;
    num_channels = 3;
    do_magick = false;
    if (ftype == "ppm")
    {
      (*image_file) << "P6" << std::endl;
      (*image_file) << hi_res.resx << " " << hi_res.resy << std::endl;
      (*image_file) << 255 << std::endl;
    }
  }
  else
  {
#ifdef HAVE_MAGICK
    C_Magick::InitializeMagick(0);
    num_channels = 4;
    channel_bytes = 2;
    do_magick = true;
    image_info=C_Magick::CloneImageInfo((C_Magick::ImageInfo *)0);
    strcpy(image_info->filename,fname.c_str());
    image_info->colorspace = C_Magick::RGBColorspace;
    image_info->quality = 90;
    image=C_Magick::AllocateImage(image_info);
    image->columns=hi_res.resx;
    image->rows=hi_res.resy;
#endif
  }

  const int pix_size = channel_bytes*num_channels;

  // Write out a screen height X image width chunk of pixels at a time
  unsigned char* pixels=
    scinew unsigned char[hi_res.resx*vp[3]*pix_size];

  // Start writing image_file
  static unsigned char* tmp_row = 0;
  if (!tmp_row )
    tmp_row = scinew unsigned char[hi_res.resx*pix_size];


  for (hi_res.row = nrows - 1; hi_res.row >= 0; --hi_res.row)
  {
    int read_height = hi_res.resy - hi_res.row * vp[3];
    read_height = (vp[3] < read_height) ? vp[3] : read_height;

    if (do_magick)
    {
#ifdef HAVE_MAGICK
      pixels = (unsigned char *)C_Magick::SetImagePixels
	(image,0,vp[3]*(nrows - 1 - hi_res.row),hi_res.resx,read_height);
#endif
    }

    if (!pixels)
    {
      cerr << "No ImageMagick Memory! Aborting...\n";
      break;
    }

    for (hi_res.col = 0; hi_res.col < ncols; hi_res.col++)
    {
      // render the col and row in the hi_res struct
      redraw_frame();
      gui->lock();
	
      // Tell OpenGL where to put the data in our pixel buffer
      glPixelStorei(GL_PACK_ALIGNMENT,1);
      glPixelStorei(GL_PACK_SKIP_PIXELS, hi_res.col * vp[2]);
      glPixelStorei(GL_PACK_SKIP_ROWS,0);
      glPixelStorei(GL_PACK_ROW_LENGTH, hi_res.resx);

      int read_width = hi_res.resx - hi_res.col * vp[2];
      read_width = (vp[2] < read_width) ? vp[2] : read_width;

      // Read the data from OpenGL into our memory
      glReadBuffer(GL_FRONT);
      glReadPixels(0,0,read_width, read_height,
		   (num_channels == 3) ? GL_RGB : GL_BGRA,
		   (channel_bytes == 1) ? GL_UNSIGNED_BYTE : GL_UNSIGNED_SHORT,
		   pixels);

      gui->unlock();
    }
    // OpenGL renders upside-down to image_file writing
    unsigned char *top_row, *bot_row;	
    int top, bot;
    for(top = read_height-1, bot = 0; bot < read_height/2; top--, bot++)
    {
      top_row = pixels + hi_res.resx*top*pix_size;
      bot_row = pixels + hi_res.resx*bot*pix_size;
      memcpy(tmp_row, top_row, hi_res.resx*pix_size);
      memcpy(top_row, bot_row, hi_res.resx*pix_size);
      memcpy(bot_row, tmp_row, hi_res.resx*pix_size);
    }
    if (do_magick)
    {
#ifdef HAVE_MAGICK
      C_Magick::SyncImagePixels(image);
#endif
    } else
      image_file->write((char *)pixels, hi_res.resx*read_height*pix_size);
  }
  gui->lock();

  // Set OpenGL back to nice PixelStore values for somebody else
  glPixelStorei(GL_PACK_SKIP_PIXELS,0);
  glPixelStorei(GL_PACK_ROW_LENGTH,0);

  if (do_magick)
  {
#ifdef HAVE_MAGICK
    if (!C_Magick::WriteImage(image_info,image))
    {
      cerr << "\nCannont Write " << fname << " because: "
	   << image->exception.reason << std::endl;
    }
	
    C_Magick::DestroyImageInfo(image_info);
    C_Magick::DestroyImage(image);
    C_Magick::DestroyMagick();
#endif
  }
  else
  {
    image_file->close();
    delete[] pixels;
  }

  gui->unlock();

  if (tmp_row)
  {
    delete[] tmp_row;
    tmp_row = 0;
  }

  extern bool regression_testing_flag;
  if (regression_testing_flag)
  {
    Thread::exitAll(0);
  }
}



void
OpenGL::make_image()
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



void
OpenGL::redraw_frame()
{
  if (dead_) return;
  gui->lock();
  Tk_Window new_tkwin=Tk_NameToWindow(the_interp, ccast_unsafe(myname_),
				      Tk_MainWindow(the_interp));
  if(!new_tkwin)
  {
    cerr << "Unable to locate window!\n";
    gui->unlock();
    return;
  }
  if(tkwin != new_tkwin)
  {
    tkwin=new_tkwin;
    dpy=Tk_Display(tkwin);
    win=Tk_WindowId(tkwin);
    // Race condition,  create context before the window is done.
    while (win==0)
    {
      gui->unlock();
      Thread::yield();
      gui->lock();
      win = Tk_WindowId(tkwin);
    }
    cx=OpenGLGetContext(the_interp, ccast_unsafe(myname_));
    if(!cx)
    {
      cerr << "Unable to create OpenGL Context!\n";
      gui->unlock();
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
    if(strstr((char*)glGetString(GL_EXTENSIONS), "GL_SGIS_multisample"))
    {
      cerr << "Enabling multisampling...\n";
      glEnable(GL_MULTISAMPLE_SGIS);
      glSamplePatternSGIS(GL_1PASS_SGIS);
    }
#endif
  }

  gui->unlock();

  // Start polygon counter...
  WallClockTimer timer;
  timer.clear();
  timer.start();

  // Get the window size
  xres=Tk_Width(tkwin);
  yres=Tk_Height(tkwin);

  // Make ourselves current
  if(current_drawer != this)
  {
    current_drawer=this;
    gui->lock();
    glXMakeCurrent(dpy, win, cx);
    gui->unlock();
  }

  // Get a lock on the geometry database...
  // Do this now to prevent a hold and wait condition with TCLTask
  viewer_->geomlock_.readLock();

  gui->lock();

  // Clear the screen...
  glViewport(0, 0, xres, yres);
  Color bg(viewwindow->bgcolor.get());
  glClearColor(bg.r(), bg.g(), bg.b(), 1);

  string saveprefix(viewwindow->saveprefix.get());

  // Setup the view...
  View view(viewwindow->view.get());
  lastview=view;
  double aspect=double(xres)/double(yres);
  // XXX - UNICam change-- should be '1.0/aspect' not 'aspect' below
  double fovy=RtoD(2*Atan(1.0/aspect*Tan(DtoR(view.fov()/2.))));

  drawinfo->reset();

  int do_stereo=viewwindow->do_stereo.get();
  drawinfo->ambient_scale_=viewwindow->ambient_scale.get();
  drawinfo->diffuse_scale_=viewwindow->diffuse_scale.get();
  drawinfo->specular_scale_=viewwindow->specular_scale.get();
  drawinfo->shininess_scale_=viewwindow->shininess_scale.get();
  drawinfo->emission_scale_=viewwindow->emission_scale.get();
  drawinfo->line_width_=viewwindow->line_width.get();
  drawinfo->point_size_=viewwindow->point_size.get();
  drawinfo->polygon_offset_factor_=viewwindow->polygon_offset_factor.get();
  drawinfo->polygon_offset_units_=viewwindow->polygon_offset_units.get();
  
#ifdef __sgi
  //  --  BAWGL  --
  int do_bawgl = viewwindow->do_bawgl.get();
  SCIBaWGL* bawgl = viewwindow->get_bawgl();

  if(!do_bawgl)
    bawgl->shutdown_ok();
  //  --  BAWGL  --
#endif
  // Compute znear and zfar...

  if(compute_depth(viewwindow, view, znear, zfar))
  {

    // Set up graphics state
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);
    glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);

    string globals("global");
    viewwindow->setState(drawinfo,globals);
    drawinfo->pickmode=0;

    GLenum errcode;
    while((errcode=glGetError()) != GL_NO_ERROR)
    {
      cerr << "We got an error from GL: " << (char*)gluErrorString(errcode)
	   << "\n";
    }

    // Do the redraw loop for each time value
    double dt=(tend-tbeg)/nframes;
    double frametime=framerate==0?0:1./framerate;
    TimeThrottle throttle;
    throttle.start();
    Vector eyesep(0,0,0);
    if(do_stereo)
    {
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


    for(int t=0;t<nframes;t++)
    {
      int n=1;
#ifdef __sgi
      bool stereo_or_bawgl = do_stereo || do_bawgl;
#else
      bool stereo_or_bawgl = do_stereo;
#endif
      if( stereo_or_bawgl ) n=2;
      for(int i=0;i<n;i++)
      {
	if( stereo_or_bawgl )
	{
	  glDrawBuffer(i==0?GL_BACK_LEFT:GL_BACK_RIGHT);
	}
	else
	{
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
	}
	else
	  //  --  BAWGL  --
#endif
	{ // render normal
	  // Setup view.
	  glMatrixMode(GL_PROJECTION);
	  glLoadIdentity();
	  if (viewwindow->ortho_view())
	  {
	    const double len = (view.lookat() - view.eyep()).length();
	    const double yval = tan(fovy * M_PI / 360.0) * len;
	    const double xval = yval * aspect;
	    glOrtho(-xval, xval, -yval, yval, znear, zfar);
	  }
	  else
	  {
	    gluPerspective(fovy, aspect, znear, zfar);
	  }
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
	  if(do_hi_res)
	    setFrustumToWindowPortion();
	}
	
	// Set up Lighting
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	const Lighting& l=viewer_->lighting_;
	int idx=0;
	int ii;
	for(ii=0;ii<l.lights.size();ii++)
	{
	  LightHandle light=l.lights[ii];
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
	
	// set up point size, line size, and polygon offset
	glPointSize(drawinfo->point_size_);
	glLineWidth(drawinfo->line_width_);
	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
	if (drawinfo->polygon_offset_factor_ || 
	    drawinfo->polygon_offset_units_) {
	  glPolygonOffset(drawinfo->polygon_offset_factor_, 
			  drawinfo->polygon_offset_units_);
	  glEnable(GL_POLYGON_OFFSET_FILL);
	} else {
	  glDisable(GL_POLYGON_OFFSET_FILL);
	}

	// Draw it all...
	current_time=modeltime;
	viewwindow->do_for_visible(this, &OpenGL::redraw_obj);

	if (viewwindow->show_rotation_axis)
	{
	  render_rotation_axis(view, do_stereo, i, eyesep);
	}

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
		
	    sprintf(scalestr, "Scale: %.2f", bawgl->virtualViewScale);
		
	    pinchText[0] = scinew GeomText(scalestr, Point(1,1,1));
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
	
	    sprintf(scalestr, "Velocity: %.2f", -1000*bawgl->velocity);
	
	    pinchText[1] = scinew GeomText(scalestr, Point(1,1,1));
	    pinchText[1]->draw(drawinfo, pinchMaterial,
			       current_time);
	
	    glPopMatrix();
	
	  }
	}
	//  --  BAWGL  --
#endif
      }
	
#if 0
      if(viewwindow->drawimg.get())
      {
	if(!imglist)
	  make_image();
	else
	  glCallList(imglist);
      }
#endif
	
      // save z-buffer data
      if (CAPTURE_Z_DATA_HACK)
      {
	CAPTURE_Z_DATA_HACK = 0;
	glReadPixels( 0, 0,
		      xres, yres,
		      GL_DEPTH_COMPONENT, GL_FLOAT,
		      pixel_depth_data );
	//            cerr << "(read from (0,0) to (" << xres << "," << yres << ")\n";
      }
	
      // Wait for the right time before swapping buffers
      //gui->unlock();
      double realtime=t*frametime;
      if(nframes>1)
	throttle.wait_for_time(realtime);
      //gui->lock();
      gui->execute("update idletasks");

      // Show the pretty picture
      glXSwapBuffers(dpy, win);

      // CollabVis code begin
#ifdef HAVE_COLLAB_VIS
      if ( viewwindow->serverNeedsImage() ) {
	/* BUG - we need to grab an image whose dimensions are a multiple
	   of 4. */
	int _x = xres - ( xres % 4 );
	int _y = yres - ( yres % 4 );
	char * image = scinew char[_x*_y*3];

        // test code
        for ( int i = 0; i < 640 * 512 * 3; i+=3 ) {
          image[i] = 0;
          image[i+1] = 0;
          image[i+2] = 0;
        }


        glReadBuffer(GL_BACK);
	glReadPixels( 0, 0, _x, _y, GL_RGB, GL_UNSIGNED_BYTE,
		      (GLubyte *)image);
	
        cerr << "Dimensions: " << _x << " x " << _y << endl;
        // DEBUG CODE
        unsigned char * testImage = (unsigned char *) image;
        int numColoredPixels = 0;
        for ( int i = 0; i < 640 * 512 * 3; i+=3 ) {
          if((unsigned int)testImage[ i ] != 0 || (unsigned int)testImage[ i+1 ] != 0 || (unsigned int)testImage[ i+2 ] != 0){
            //cerr << "<" << (unsigned int)testImage[ i ] << ", " << (unsigned int)testImage[ i+1 ] << ", " << (unsigned int)testImage[ i+2 ] << ">  ";
            numColoredPixels++;
          }
        }
        cerr << "**************************NUM COLORED PIXELS = " << numColoredPixels << endl;
 
        // test code
        glRasterPos2i( 0, 0 );
        glDrawPixels( _x, _y, GL_RGB, GL_UNSIGNED_BYTE, image);

	viewwindow->sendImageToServer( image, _x, _y );
	
      }
#endif
      // CollabVis code end
      
      //  #ifdef __sgi
      //  	  if(saveprefix != ""){
      //  	    // Save out the image...
      //  	    char filename[200];
      //  	    sprintf(filename, "%s%04d.rgb", saveprefix.c_str(), t);
      //  	    unsigned char* rgbdata=scinew unsigned char[xres*yres*3];
      //  	    glReadPixels(0, 0, xres, yres, GL_RGB, GL_UNSIGNED_BYTE, rgbdata);
      //  	    iflSize dims(xres, yres, 1);
      //  	    iflFileConfig fc(&dims, iflUChar);
      //  	    iflStatus sts;
      //  	    iflFile *file=iflFile::create(filename, NULL, &fc, NULL, &sts);
      //  	    if (sts != iflOKAY) {
      //  	      cerr << "Unable to save image file "<<filename<<"\n";
      //  	      break;
      //  	    }
      //  	    sts = file->setTile(0, 0, 0, xres, yres, 1, rgbdata);
      //  	    if (sts != iflOKAY) {
      //  	      cerr << "Unable to save image tile to "<<filename<<"\n";
      //  	      break;
      //  	    }
      //  	    sts = file->flush();
      //  	    if (sts != iflOKAY) {
      //  	      cerr << "Unable to write tile to "<<filename<<"\n";
      //  	      break;
      //  	    }
      //  	    file->close();
      //  	    delete[] rgbdata;
      //  	  }
      //  #endif // __sgi
    }
    throttle.stop();
    double fps;
    if (throttle.time()>0)
      fps=nframes/throttle.time();
    else
      fps=nframes;
    int fps_whole=(int)fps;
    int fps_hund=(int)((fps-fps_whole)*100);
    ostringstream str;
    str << viewwindow->id << " setFrameRate " << fps_whole << "." << fps_hund;
    gui->execute(str.str());
    viewwindow->set_current_time(tend);
  }
  else
  {
    // Just show the cleared screen
    viewwindow->set_current_time(tend);
	
#ifdef __sgi
    //  --  BAWGL  --
    if( do_stereo || do_bawgl )
    {
      glDrawBuffer(GL_BACK_LEFT);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glDrawBuffer(GL_BACK_RIGHT);
    }
    else
    {
      glDrawBuffer(GL_BACK);
    }
    //  --  BAWGL  --
#endif
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if(viewwindow->drawimg.get())
    {
#if 0
      if(!imglist)
	make_image();
      else
	glCallList(imglist);
#endif
    }
    glXSwapBuffers(dpy, win);
  }

  viewer_->geomlock_.readUnlock();

  // Look for errors
  GLenum errcode;
  while((errcode=glGetError()) != GL_NO_ERROR)
  {
    cerr << "We got an error from GL: " << (char*)gluErrorString(errcode)
	 << "\n";
  }

  // Report statistics
  timer.stop();
  fpstimer.stop();
  double fps;
  if (fpstimer.time()>0)
    fps=nframes/fpstimer.time();
  else
    fps=100;
  fps+=0.05;			// Round to nearest tenth
  int fps_whole=(int)fps;
  int fps_tenths=(int)((fps-fps_whole)*10);
  fpstimer.clear();
  fpstimer.start();		// Start it running for next time
  ostringstream str;
  double pps;
  if (timer.time()>0)
    pps=drawinfo->polycount/timer.time();
  else
    pps=drawinfo->polycount;
  str << viewwindow->id << " updatePerf \"";
  str << drawinfo->polycount << " polygons in " << timer.time()
      << " seconds\" \"" << pps
      << " polygons/second\"" << " \"" << fps_whole << "."
      << fps_tenths << " frames/sec\"" << '\0';
  //    cerr <<"updatePerf: <" << str.str() << ">\n";	
  /***********************************/
  /* movie makin' movie-movie makin' */
  /***********************************/
  if (viewwindow->doingMovie)
  {
	
    string segname(viewwindow->curName);
    int lasthash=-1;
    for (unsigned int ii=0; ii<segname.size(); ii++)
    {
      if (segname[ii] == '/') lasthash=ii;
    }
    string pathname;
    if (lasthash == -1) pathname = "./";
    else pathname = segname.substr(0, lasthash+1);
    string fname = segname.substr(lasthash+1, segname.size()-(lasthash+1));
	
    //      cerr << "Saving a movie!\n";
    if( viewwindow->makeMPEG )
    {
      if(!encoding_mpeg_)
      {
	encoding_mpeg_ = true;
	fname = fname + ".mpg";
	StartMpeg( fname );
      }
      AddMpegFrame();
    }
    else
    { // dump each frame
      /* if mpeg has just been turned off, close the file. */
      if(encoding_mpeg_)
      {
	encoding_mpeg_ = false;
	EndMpeg();
      }
      unsigned char movie[10];
      int startDiv = 100;
      int idx=0;
      int fi = viewwindow->curFrame;
      while (startDiv >= 1)
      {
	movie[idx] = '0' + fi/startDiv;
	fi = fi - (startDiv)*(fi/startDiv);
	startDiv /= 10;
	idx++;
      }
      movie[idx] = 0;
      fname = fname + ".raw";
      string framenum((char *)movie);
      framenum = framenum + ".";
      string fullpath(pathname + framenum + fname);
      cerr << "Dumping "<<fullpath<<"....  ";
      dump_image(fullpath);
      cerr << " done!\n";
      viewwindow->curFrame++;
    }
  }
  else
  {
    if(encoding_mpeg_)
    { // Finish up mpeg that was in progress.
      encoding_mpeg_ = false;
      EndMpeg();
    }
  }
  gui->execute(str.str());
  gui->unlock();

}



void
OpenGL::get_pick(Viewer*, ViewWindow*, int x, int y,
		 GeomHandle& pick_obj, GeomPickHandle& pick_pick,
		 int& pick_index)
{
  send_pick_x=x;
  send_pick_y=y;
  send_mb.send(DO_PICK);
  for(;;)
  {
    int r=recv_mb.receive();
    if(r != PICK_DONE)
    {
      cerr << "WANTED A PICK!!! (got back " << r << "\n";
    }
    else
    {
      pick_obj=ret_pick_obj;
      pick_pick=ret_pick_pick;
      pick_index=ret_pick_index;
      break;
    }
  }
}



void
OpenGL::real_get_pick(Viewer*, ViewWindow* ViewWindow, int x, int y,
		      GeomHandle& pick_obj, GeomPickHandle& pick_pick,
		      int& pick_index)
{
  pick_obj=0;
  pick_pick=0;
  pick_index = 0x12345678;
  // Make ourselves current
  if(current_drawer != this)
  {
    current_drawer=this;
    gui->lock();
    glXMakeCurrent(dpy, win, cx);
    gui->unlock();
  }
  // Setup the view...
  View view(viewwindow->view.get());
  viewer_->geomlock_.readLock();

  // Compute znear and zfar...
  double znear;
  double zfar;
  if(compute_depth(ViewWindow, view, znear, zfar))
  {
    // Setup picking...
    gui->lock();

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

    if(!viewwindow->do_bawgl.get())
    { //Regular flavor picking
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
    else
    { //BAWGL flavored picking setup!!!
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
    while((errcode=glGetError()) != GL_NO_ERROR)
    {
      cerr << "We got an error from GL: " << (char*)gluErrorString(errcode)
	   << "\n";
    }
    gui->unlock();
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
    //GLuint hit_pick_index = 0x12345678;  // need for object indexing
#endif
    //cerr << "hits=" << hits << "\n";
    if(hits >= 1)
    {
      int idx=0;
      min_z=0;
      int have_one=0;
      for (int h=0; h<hits; h++)
      {
	int nnames=pick_buffer[idx++];
	GLuint z=pick_buffer[idx++];
	//cerr << "h=" << h << ", nnames=" << nnames << ", z=" << z << "\n";
	if (nnames > 1 && (!have_one || z < min_z))
	{
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
	  //hit_pick_index = pick_buffer[idx++];
	  idx++;
#else
	  // hit_obj=pick_buffer[idx++];
	  // hit_obj_index=pick_buffer[idx++];
	  //for(int i=idx; i<idx+nnames; ++i) cerr << pick_buffer[i] << "\n";
	  idx+=nnames-3; // Skip to the last one...
	  hit_pick=pick_buffer[idx++];
	  hit_obj=pick_buffer[idx++];
	  idx++;
	  //hit_pick_index=pick_buffer[idx++];
#endif
	  //cerr << "new min... (obj=" << hit_obj
	  //     << ", pick="          << hit_pick
	  //     << ", index = "       << hit_pick_index << ")\n";
	}
	else
	{
	  idx+=nnames+1;
	}
      }

      pick_obj=(GeomObj*)hit_obj;
      pick_pick=(GeomPick*)hit_pick;
      pick_obj->getId(pick_index); //(int)hit_pick_index;
      //cerr << "pick_pick=" << pick_pick << ", pick_index="<<pick_index<<"\n";

      // x,y,min_z is our point... we can unproject it to get model-space pt
    }
  }
  viewer_->geomlock_.readUnlock();
}



void
OpenGL::dump_image(const string& name, const string& /* type */)
{
  ofstream dumpfile(name.c_str());
  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT,vp);
  int n=3*vp[2]*vp[3];
  cerr << "Dumping: " << vp[2] << "x" << vp[3] << "\n";
  unsigned char* pxl=scinew unsigned char[n];
  glPixelStorei(GL_PACK_ALIGNMENT,1);
  glReadBuffer(GL_FRONT);
  glReadPixels(0,0,vp[2],vp[3],GL_RGB,GL_UNSIGNED_BYTE,pxl);
  dumpfile.write((const char *)pxl,n);
  delete[] pxl;
}



void
OpenGL::put_scanline(int y, int width, Color* scanline, int repeat)
{
  float* pixels=scinew float[width*3];
  float* p=pixels;
  int i;
  for(i=0;i<width;i++)
  {
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
  for(i=0;i<repeat;i++)
  {
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



void
OpenGL::pick_draw_obj(Viewer* viewer, ViewWindow*, GeomHandle obj)
{
#if (_MIPS_SZPTR == 64)
  unsigned long o=(unsigned long)(obj.get_rep());
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
  glPushName((GLuint)(obj.get_rep()));
  glPushName(0x12345678);
#endif
  obj->draw(drawinfo, viewer->default_material_.get_rep(), current_time);
}



void
OpenGL::redraw_obj(Viewer* viewer, ViewWindow* viewwindow, GeomHandle obj, bool sticky)
{
  drawinfo->viewwindow = viewwindow;
  if( !sticky){
    obj->draw(drawinfo, viewer->default_material_.get_rep(), current_time);
  } else {
    int ii = 0;
    // Disable clipping planes for sticky objects.
    vector<bool> cliplist(6, false);
    for (ii = 0; ii < 6; ii++)
    {
      if (glIsEnabled((GLenum)(GL_CLIP_PLANE0+ii)))
      {
	glDisable((GLenum)(GL_CLIP_PLANE0+ii));
	cliplist[ii] = true;
      }
    }
    obj->draw(drawinfo, viewer->default_material_.get_rep(), current_time);
    
    // Reenable clipping planes.
    for (ii = 0; ii < 6; ii++)
    {
      if (cliplist[ii])
      {
	glEnable((GLenum)(GL_CLIP_PLANE0+ii));
      }
    }
  }
}



void
ViewWindow::setState(DrawInfoOpenGL* drawinfo, const string& tclID)
{
  string val;
  string type(tclID+"-"+"type");
  string lighting(tclID+"-"+"light");
  string fog(tclID+"-"+"fog");
  string cull(tclID+"-"+"cull");
  string dl(tclID+"-"+"dl");
  string debug(tclID+"-"+"debug");
  string movie(tclID+"-"+"movie");
  string movieName(tclID+"-"+"movieName");
  string movieFrame(tclID+"-"+"movieFrame");
  string use_clip(tclID+"-"+"clip");
  if (!ctx->getSub(type,val))
  {
    cerr << "Error illegal name!\n";
    return;
  }
  else
  {
    if(val == "Wire")
    {
      drawinfo->set_drawtype(DrawInfoOpenGL::WireFrame);
      drawinfo->lighting=0;
    }
    else if(val == "Flat")
    {
      drawinfo->set_drawtype(DrawInfoOpenGL::Flat);
      drawinfo->lighting=0;
    }
    else if(val == "Gouraud")
    {
      drawinfo->set_drawtype(DrawInfoOpenGL::Gouraud);
      drawinfo->lighting=1;
    }
    else if (val == "Default")
    {
      string globals("global");
      setState(drawinfo,globals);	
      return; // if they are using the default, con't change
    }
    else
    {
      cerr << "Unknown shading(" << val << "), defaulting to phong" << "\n";
      drawinfo->set_drawtype(DrawInfoOpenGL::Gouraud);
      drawinfo->lighting=1;
    }

    // Now see if they want a bounding box.

    if (ctx->getSub(debug,val))
    {
      if (val == "0")
	drawinfo->debug = 0;
      else
	drawinfo->debug = 1;
    }	
    else
    {
      cerr << "Error, no debug level set!\n";
      drawinfo->debug = 0;
    }

    if (ctx->getSub(use_clip,val))
    {
      if (val == "0")
	drawinfo->check_clip = 0;
      else
	drawinfo->check_clip = 1;
    }	
    else
    {
      cerr << "Error, no clipping info\n";
      drawinfo->check_clip = 0;
    }
    // only set with globals.
    if (ctx->getSub(movie,val))
    {
      ctx->getSub(movieName,curName);
      string curFrameStr;
      ctx->getSub(movieFrame,curFrameStr);
      if (val == "0")
      {
	doingMovie = 0;
	makeMPEG = 0;
      }
      else
      {
	if (!doingMovie)
	{
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

    if (ctx->getSub(cull,val))
    {
      if (val == "0")
	drawinfo->cull = 0;
      else
	drawinfo->cull = 1;
    }	
    else
    {
      cerr << "Error, no culling info\n";
      drawinfo->cull = 0;
    }
    if (ctx->getSub(dl,val))
    {
      if (val == "0")
	drawinfo->dl = 0;
      else
	drawinfo->dl = 1;
    }	
    else
    {
      cerr << "Error, no display list info\n";
      drawinfo->dl = 0;
    }
    if (!ctx->getSub(lighting,val))
      cerr << "Error, no lighting!\n";
    else
    {
      if (val == "0")
      {
	drawinfo->lighting=0;
      }
      else if (val == "1")
      {
	drawinfo->lighting=1;
      }
      else
      {
	cerr << "Unknown lighting setting(" << val << "\n";
      }

      if (ctx->getSub(fog,val))
      {
	if (val=="0")
	{
	  drawinfo->fog=0;
	}
	else
	{
	  drawinfo->fog=1;
	}
      }
      else
      {
	cerr << "Fog not defined properly!\n";
	drawinfo->fog=0;
      }

    }
  }
  drawinfo->currently_lit=drawinfo->lighting;
  drawinfo->init_lighting(drawinfo->lighting);
}



void
ViewWindow::setDI(DrawInfoOpenGL* drawinfo,string name)
{
  ObjTag* vis;

  viter = visible.find(name);
  if (viter != visible.end())
  { // if found
    vis = (*viter).second;
    setState(drawinfo,to_string(vis->tagid));
  }
}


// Set the bits for the clipping planes that are on.

void
ViewWindow::setClip(DrawInfoOpenGL* drawinfo)
{
  string val;
  int i;

  drawinfo->clip_planes = 0; // set them all of for default
  string num_clip("clip-num");

  if (ctx->getSub("clip-visible",val) &&
      ctx->getSub(num_clip,i))
  {

    int cur_flag = CLIP_P5;
    if ( (i>0 && i<7) )
    {
      while(i--)
      {
	
	string vis("clip-visible-"+to_string(i+1));
	
	
	if (ctx->getSub(vis,val))
	{
	  if (val == "1")
	  {
	    double plane[4];
	    string nx("clip-normal-x-"+to_string(i+1));
	    string ny("clip-normal-y-"+to_string(i+1));
	    string nz("clip-normal-z-"+to_string(i+1));
	    string nd("clip-normal-d-"+to_string(i+1));
	
	    int rval=0;
	
	    rval = ctx->getSub(nx,plane[0]);
	    rval = ctx->getSub(ny,plane[1]);
	    rval = ctx->getSub(nz,plane[2]);
	    rval = ctx->getSub(nd,plane[3]);
	
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
	
	    if (!rval )
	    {
	      cerr << "Error, variable is hosed!\n";
	    }
	  }
	  else
	  {
	    glDisable((GLenum)(GL_CLIP_PLANE0+i));
	  }
	
	}
	cur_flag >>= 1; // shift the bit we are looking at...
      }
    }
  }
}



void
GeomViewerItem::draw(DrawInfoOpenGL* di, Material *m, double time)
{
  // Here we need to query the ViewWindow with our name and give it our
  // di so it can change things if they need to be.
  di->viewwindow->setDI(di, name_);

  BBox bb;
  child_->get_bounds(bb);
  if (!(di->debug && bb.valid()))
  {
    child_->draw(di,m,time);
  }
  else
  {
    const Point &min(bb.min());
    const Point &max(bb.max());

    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glDepthMask(GL_FALSE);

    glDisable(GL_LIGHTING);	

    glBegin(GL_QUADS);

    //top
    glColor4f(0.0, 0.0, 1.0, 0.4);
    glVertex3d(max.x(),min.y(),max.z());
    glVertex3d(max.x(),max.y(),max.z());
    glVertex3d(min.x(),max.y(),max.z());
    glVertex3d(min.x(),min.y(),max.z());

    //bottom
    glColor4f(0.0, 0.0, 1.0, 0.2);
    glVertex3d(max.x(),max.y(),min.z());
    glVertex3d(max.x(),min.y(),min.z());
    glVertex3d(min.x(),min.y(),min.z());
    glVertex3d(min.x(),max.y(),min.z());

    //right
    glColor4f(1.0, 0.0, 0.0, 0.4);
    glVertex3d(max.x(),min.y(),min.z());
    glVertex3d(max.x(),max.y(),min.z());
    glVertex3d(max.x(),max.y(),max.z());
    glVertex3d(max.x(),min.y(),max.z());

    //left
    glColor4f(1.0, 0.0, 0.0, 0.2);
    glVertex3d(min.x(),min.y(),max.z());
    glVertex3d(min.x(),max.y(),max.z());
    glVertex3d(min.x(),max.y(),min.z());
    glVertex3d(min.x(),min.y(),min.z());

    //top
    glColor4f(0.0, 1.0, 0.0, 0.4);
    glVertex3d(min.x(),max.y(),max.z());
    glVertex3d(max.x(),max.y(),max.z());
    glVertex3d(max.x(),max.y(),min.z());
    glVertex3d(min.x(),max.y(),min.z());

    //bottom
    glColor4f(0.0, 1.0, 0.0, 0.2);
    glVertex3d(min.x(),min.y(),min.z());
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


void
OpenGL::listvisuals(GuiArgs& args)
{
  gui->lock();

  Thread::allow_sgi_OpenGL_page0_sillyness();
  Tk_Window topwin=Tk_NameToWindow(the_interp, ccast_unsafe(args[2]),
				   Tk_MainWindow(the_interp));
  if(!topwin)
  {
    cerr << "Unable to locate window!\n";
    gui->unlock();
    return;
  }
  dpy=Tk_Display(topwin);
  int screen=Tk_ScreenNumber(topwin);
  vector<string> visualtags;
  vector<int> scores;
  visuals.clear();
  int nvis;
  XVisualInfo* vinfo=XGetVisualInfo(dpy, 0, NULL, &nvis);
  if(!vinfo)
  {
    args.error("XGetVisualInfo failed");
    gui->unlock();
    return;
  }
  int i;
  for(i=0;i<nvis;i++)
  {
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
    string tag(buf);
    GETCONFIG(GLX_DOUBLEBUFFER);
    if(value)
    {
      score+=200;
      tag += "double, ";
    }
    else
    {
      tag += "single, ";
    }
    GETCONFIG(GLX_STEREO);
    if(value)
    {
      score+=1;
      tag += "stereo, ";
    }
    tag += "rgba=";
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
    tag += ", depth=" + to_string(value);
    score+=value*5;
    GETCONFIG(GLX_STENCIL_SIZE);
    tag += ", stencil="+to_string(value);
    tag += ", accum=";
    GETCONFIG(GLX_ACCUM_RED_SIZE);
    tag += to_string(value) + ":";
    GETCONFIG(GLX_ACCUM_GREEN_SIZE);
    tag += to_string(value) + ":";
    GETCONFIG(GLX_ACCUM_BLUE_SIZE);
    tag += to_string(value) + ":";
    GETCONFIG(GLX_ACCUM_ALPHA_SIZE);
    tag += to_string(value);
#ifdef __sgi
    tag += ", samples=";
    GETCONFIG(GLX_SAMPLES_SGIS);
    if(value)
      score+=50;
#endif
    tag += to_string(value);

    tag += ", score=" + to_string(score);

    visualtags.push_back(tag);
    visuals.push_back(&vinfo[i]);
    scores.push_back(score);
  }
  for(i=0;(unsigned int)i<scores.size()-1;i++)
  {
    for(unsigned int j=i+1;j<scores.size();j++)
    {
      if(scores[i] < scores[j])
      {
	// Swap.
	int tmp1=scores[i];
	scores[i]=scores[j];
	scores[j]=tmp1;
	string tmp2=visualtags[i];
	visualtags[i]=visualtags[j];
	visualtags[j]=tmp2;
	XVisualInfo* tmp3=visuals[i];
	visuals[i]=visuals[j];
	visuals[j]=tmp3;
      }
    }
  }
  args.result(GuiArgs::make_list(visualtags));
  gui->unlock();
}



void
OpenGL::setvisual(const string& wname, int which, int width, int height)
{
  tkwin=0;
  current_drawer=0;

  gui->execute("opengl " + wname +
	       " -visual " + to_string((int)visuals[which]->visualid) +
	       " -direct true" +
	       " -geometry " + to_string(width) + "x" + to_string(height));

  myname_ = wname;
}



void
OpenGL::deriveFrustum()
{
  double pmat[16];
  glGetDoublev(GL_PROJECTION_MATRIX, pmat);
  const double G = (pmat[10]-1)/(pmat[10]+1);
  frustum.znear = -(pmat[14]*(G-1))/(2*G);
  frustum.zfar = frustum.znear*G;
  frustum.left = frustum.znear*(pmat[8]-1)/pmat[0];
  frustum.right = frustum.znear*(pmat[8]+1)/pmat[0];
  frustum.bottom = frustum.znear*(pmat[9]-1)/pmat[5];
  frustum.top = frustum.znear*(pmat[9]+1)/pmat[5];
  frustum.width = frustum.right - frustum.left;
  frustum.height = frustum.top - frustum.bottom;
}



void
OpenGL::setFrustumToWindowPortion()
{
  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  glFrustum(frustum.left + frustum.width / hi_res.ncols * hi_res.col,
	    frustum.left + frustum.width / hi_res.ncols * (hi_res.col+1),
	    frustum.bottom + frustum.height / hi_res.nrows * hi_res.row,
	    frustum.bottom + frustum.height / hi_res.nrows * (hi_res.row+1),
	    frustum.znear, frustum.zfar);
}



void
OpenGL::saveImage(const string& fname,
		  const string& type,
		  int x, int y) //= "ppm")
{
  send_mb.send(DO_IMAGE);
  img_mb.send(ImgReq(fname,type,x,y));
}

// CollabVis code begin
#ifdef HAVE_COLLAB_VIS

void OpenGL::setZTexTransform( double * matrix ) {
  ZTexTransform.set( matrix );
  doZTexTransform = true;
}

void OpenGL::setZTexView( const View &v ) {
  ZTexView = v;
  doZTexView = true;
}
#endif
// CollabVis code end

void
OpenGL::getData(int datamask, FutureValue<GeometryData*>* result)
{
  send_mb.send(DO_GETDATA);
  get_mb.send(GetReq(datamask, result));
}

void
OpenGL::real_getData(int datamask, FutureValue<GeometryData*>* result)
{
  GeometryData* res = new GeometryData;
  if(datamask&GEOM_VIEW)
  {
    res->view=new View(lastview);
    res->xres=xres;
    res->yres=yres;
    res->znear=znear;
    res->zfar=zfar;
  }
  if(datamask&(GEOM_COLORBUFFER|GEOM_DEPTHBUFFER/*CollabVis*/|GEOM_MATRICES))
  {
    gui->lock();
  }
  if(datamask&GEOM_COLORBUFFER)
  {
    ColorImage* img = res->colorbuffer = new ColorImage(xres, yres);
    float* data=new float[xres*yres*3];
    cerr << "xres=" << xres << ", yres=" << yres << "\n";
    WallClockTimer timer;
    timer.start();
    // CollabVis code begin
#ifdef HAVE_COLLAB_VIS
    //cerr << "[HAVE_COLLAB_VIS] (OpenGL::real_getData) 0" << endl;
    glReadBuffer(GL_FRONT);
#endif
    // CollabVis code end
    glReadPixels(0, 0, xres, yres, GL_RGB, GL_FLOAT, data);
    timer.stop();
    cerr << "done in " << timer.time() << " seconds\n";
    float* p=data;
    for(int y=0;y<yres;y++)
    {
      for(int x=0;x<xres;x++)
      {
	img->put_pixel(x, y, Color(p[0], p[1], p[2]));
	p+=3;
      }
    }
    delete[] data;
  }
  if(datamask&GEOM_DEPTHBUFFER)
  {
    DepthImage* img=res->depthbuffer=new DepthImage(xres, yres);
    unsigned int* data=new unsigned int[xres*yres*3];
    cerr << "reading depth...\n";
    WallClockTimer timer;
    timer.start();
    glReadPixels(0, 0, xres, yres, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, data);
    timer.stop();
    cerr << "done in " << timer.time() << " seconds\n";
    // CollabVis code begin
#ifndef HAVE_COLLAB_VIS
    //cerr << "[HAVE_COLLAB_VIS] (OpenGL::real_getData) 1" << endl;
    unsigned int* p=data;
    for(int y=0;y<yres;y++)
    {
      for(int x=0;x<xres;x++)
      {
	img->put_pixel(x, y, (*p++)*(1./4294967295.));
      }
    }
    delete[] data;
#else
    res->depthbuffer = (DepthImage *)data;
#endif
    // CollabVis code end
  }
  // CollabVis code begin
#ifdef HAVE_COLLAB_VIS
  //cerr << "[HAVE_COLLAB_VIS] (OpenGL::real_getData) 2" << endl;
  if (datamask&GEOM_MATRICES) {
    /* Get the necessary matrices */
    glGetDoublev(GL_MODELVIEW_MATRIX, res->modelview);
    glGetDoublev(GL_PROJECTION_MATRIX, res->projection);
    glGetIntegerv(GL_VIEWPORT, res->viewport);
  }
  
#endif
// CollabVis code end

  if(datamask&(GEOM_COLORBUFFER|GEOM_DEPTHBUFFER/*CollabVis*/|GEOM_MATRICES))
  {
    GLenum errcode;
    while((errcode=glGetError()) != GL_NO_ERROR)
    {
      cerr << "We got an error from GL: " <<
	(char*)gluErrorString(errcode) << "\n";
    }
    gui->unlock();
  }

  // CollabVis code begin
#ifdef HAVE_COLLAB_VIS
  if ( datamask & GEOM_TRIANGLES ) {
    triangles = new Array1<float>;
    viewwindow->do_for_visible(this,
			       (ViewWindowVisPMF)&OpenGL::collect_triangles);
    res->depthbuffer = (DepthImage*)triangles;
  }
  // CollabVis code end
#endif
  
  result->send(res);
}

// CollabVis code begin
#ifdef HAVE_COLLAB_VIS
void OpenGL::collect_triangles(Viewer *viewer,
			       ViewWindow *viewwindow,
			       GeomHandle obj)
{
  cerr << "[HAVE_COLLAB_VIS] (OpenGL::collect_triangles) 0" << endl;
  obj->get_triangles(*triangles);
  cerr << "found " << (*triangles).size()/3 << "  triangles" << endl;

}
// CollabVis code end
#endif

void
OpenGL::StartMpeg(const string& fname)
{
#ifdef HAVE_MPEG
  // Get a file pointer pointing to the output file.
  mpeg_file_ = fopen(fname.c_str(), "w");
  if (!mpeg_file_)
  {
    cerr << "Failed to open file " << fname << " for writing\n";
    return;
  }
  // Get the default options.
  MPEGe_default_options( &mpeg_options_ );
  // Change a couple of the options.
  strcpy(mpeg_options_.frame_pattern, "II");  // was ("IIIIIIIIIIIIIII");
  mpeg_options_.search_range[1]=0;
  if( !MPEGe_open(mpeg_file_, &mpeg_options_ ) )
  {
    cerr << "MPEGe library initialisation failure!:" <<
      mpeg_options_.error << "\n";
    return;
  }
#endif // HAVE_MPEG
}



void
OpenGL::AddMpegFrame()
{
#ifdef HAVE_MPEG
  // Looks like we'll blow up if you try to make more than one movie
  // at a time, as the memory per frame is static.
  static ImVfb *image=NULL; /* Only alloc memory for these once. */
  int width, height;
  ImVfbPtr ptr;

  cerr<<"Adding Mpeg Frame\n";
  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT,vp);

  width = vp[2];
  height = vp[3];

  // Set up the ImVfb used to store the image.
  if( !image )
  {
    image=MPEGe_ImVfbAlloc( width, height, IMVFBRGB, true );
    if( !image )
    {
      cerr<<"Couldn't allocate memory for frame buffer\n";
      exit(2);
    }
  }

  // Get to the first pixel in the image.
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
  for(k = height -1, j = 0; j < height/2; k--, j++)
  {
    p0 = ptr + r*j;
    p1 = ptr + r*k;
    memcpy( row, p0, r);
    memcpy( p0, p1, r);
    memcpy( p1, row, r);
  }

  if( !MPEGe_image(image, &mpeg_options_) )
  {
    cerr << "MPEGe_image failure:" << mpeg_options_.error << "\n";
  }
#endif // HAVE_MPEG
}



void
OpenGL::EndMpeg()
{
#ifdef HAVE_MPEG
  if( !MPEGe_close(&mpeg_options_) )
  {
    cerr << "Had a bit of difficulty closing the file:" << mpeg_options_.error;
  }

  cerr << "Ending Mpeg\n";
#endif // HAVE_MPEG
}



// Return world-space depth to point under pixel (x, y).
bool
OpenGL::pick_scene( int x, int y, Point *p )
{
  // y = 0 is bottom of screen (not top of screen, which is what X
  // events reports)
  y = (yres - 1) - y;
  int index = x + (y * xres);
  double z = pixel_depth_data[index];
  if (p)
  {
    // Unproject the window point (x, y, z).
    GLdouble world_x, world_y, world_z;
    gluUnProject(x, y, z,
		 get_depth_model, get_depth_proj, get_depth_view,
		 &world_x, &world_y, &world_z);

    *p = Point(world_x, world_y, world_z);
  }

  // if z is close to 1, then assume no object was picked
  return (z < .999999);
}



bool
OpenGL::compute_depth(ViewWindow* viewwindow, const View& view,
		      double& znear, double& zfar)
{
  znear=MAXDOUBLE;
  zfar=-MAXDOUBLE;
  BBox bb;
  viewwindow->get_bounds(bb);
  if(bb.valid())
  {
    // We have something to draw.
    Point min(bb.min());
    Point max(bb.max());
    Point eyep(view.eyep());
    Vector dir(view.lookat()-eyep);
    const double dirlen2 = dir.length2();
    if (dirlen2 < 1.0e-6 || dirlen2 != dirlen2)
      return false;
    dir.normalize();
    double d=-Dot(eyep, dir);
    for(int ix=0;ix<2;ix++)
    {
      for(int iy=0;iy<2;iy++)
      {
	for(int iz=0;iz<2;iz++)
	{
	  Point p(ix?max.x():min.x(),
		  iy?max.y():min.y(),
		  iz?max.z():min.z());
	  double dist=Dot(p, dir)+d;
	  znear=Min(znear, dist);
	  zfar=Max(zfar, dist);
	}
      }
    }
    if(znear <= 0)
    {
      if(zfar <= 0)
      {
	// Everything is behind us - it doesn't matter what we do.
	znear=1.0;
	zfar=2.0;
      }
      else
      {
	znear=zfar*.001;
      }
    }
    return true;
  }
  else
  {
    return false;
  }
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


ImgReq::ImgReq(const string& n, const string& t, int x, int y)
  : name(n), type(t), resx(x), resy(y)
{
}


static GeomHandle
createGenAxes()
{     
  MaterialHandle dk_red = scinew Material(Color(0,0,0), Color(.2,0,0),
					  Color(.5,.5,.5), 20);
  MaterialHandle dk_green = scinew Material(Color(0,0,0), Color(0,.2,0),
					    Color(.5,.5,.5), 20);
  MaterialHandle dk_blue = scinew Material(Color(0,0,0), Color(0,0,.2),
					   Color(.5,.5,.5), 20);
  MaterialHandle lt_red = scinew Material(Color(0,0,0), Color(.8,0,0),
					  Color(.5,.5,.5), 20);
  MaterialHandle lt_green = scinew Material(Color(0,0,0), Color(0,.8,0),
					    Color(.5,.5,.5), 20);
  MaterialHandle lt_blue = scinew Material(Color(0,0,0), Color(0,0,.8),
					   Color(.5,.5,.5), 20);

  GeomGroup* xp = scinew GeomGroup; 
  GeomGroup* yp = scinew GeomGroup;
  GeomGroup* zp = scinew GeomGroup;
  GeomGroup* xn = scinew GeomGroup;
  GeomGroup* yn = scinew GeomGroup;
  GeomGroup* zn = scinew GeomGroup;

  const double sz = 1.0;
  xp->add(scinew GeomCylinder(Point(0,0,0), Point(sz, 0, 0), sz/20));
  xp->add(scinew GeomCone(Point(sz, 0, 0), Point(sz+sz/5, 0, 0), sz/10, 0));
  yp->add(scinew GeomCylinder(Point(0,0,0), Point(0, sz, 0), sz/20));
  yp->add(scinew GeomCone(Point(0, sz, 0), Point(0, sz+sz/5, 0), sz/10, 0));
  zp->add(scinew GeomCylinder(Point(0,0,0), Point(0, 0, sz), sz/20));
  zp->add(scinew GeomCone(Point(0, 0, sz), Point(0, 0, sz+sz/5), sz/10, 0));
  xn->add(scinew GeomCylinder(Point(0,0,0), Point(-sz, 0, 0), sz/20));
  xn->add(scinew GeomCone(Point(-sz, 0, 0), Point(-sz-sz/5, 0, 0), sz/10, 0));
  yn->add(scinew GeomCylinder(Point(0,0,0), Point(0, -sz, 0), sz/20));
  yn->add(scinew GeomCone(Point(0, -sz, 0), Point(0, -sz-sz/5, 0), sz/10, 0));
  zn->add(scinew GeomCylinder(Point(0,0,0), Point(0, 0, -sz), sz/20));
  zn->add(scinew GeomCone(Point(0, 0, -sz), Point(0, 0, -sz-sz/5), sz/10, 0));
  GeomGroup* all=scinew GeomGroup;
  all->add(scinew GeomMaterial(xp, lt_red));
  all->add(scinew GeomMaterial(yp, lt_green));
  all->add(scinew GeomMaterial(zp, lt_blue));
  all->add(scinew GeomMaterial(xn, dk_red));
  all->add(scinew GeomMaterial(yn, dk_green));
  all->add(scinew GeomMaterial(zn, dk_blue));
  
  return all;
}


// i is the frame number, usually refers to left or right when do_stereo
// is set.

void
OpenGL::render_rotation_axis(const View &view,
			     bool do_stereo, int i, const Vector &eyesep)
{
  static GeomHandle axis_obj = 0;
  if (axis_obj.get_rep() == 0) axis_obj = createGenAxes();

  GLint viewport[4];
  glGetIntegerv(GL_VIEWPORT, viewport);

  const int xysize = Min(viewport[2], viewport[3]) / 4;
  glViewport(viewport[2] - xysize, viewport[3] - xysize, xysize, xysize);
  const double aspect = 1.0;

  // fovy 16 eyedist 10 is approximately the default axis view.
  // fovy 32 eyedist 5 gives an exagerated perspective.
  const double fovy = 32.0;
  const double eyedist = 5.0;
  const double znear = eyedist - 2.0;
  const double zfar = eyedist + 2.0;

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  gluPerspective(fovy, aspect, znear, zfar);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  Vector oldeye(view.eyep().asVector() - view.lookat().asVector());
  oldeye.normalize();
  Point eyep((oldeye * eyedist).asPoint());
  Point lookat(0.0, 0.0, 0.0);
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
  if(do_hi_res)
  {
    // Draw in upper right hand corner of total image, not viewport image.
    const int xysize = Min(hi_res.resx, hi_res.resy) / 4;
    const int xoff = hi_res.resx - hi_res.col * viewport[2];
    const int yoff = hi_res.resy - hi_res.row * viewport[3];
    glViewport(xoff - xysize, yoff - xysize, xysize, xysize);
  }

  // Disable fog for the orientation axis.
  const bool fog = drawinfo->fog;
  drawinfo->fog = false;

  // Set up Lighting
  glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
  const Lighting& l=viewer_->lighting_;
  int idx=0;
  int ii;
  for(ii=0;ii<l.lights.size();ii++)
  {
    LightHandle light=l.lights[ii];
    light->opengl_setup(view, drawinfo, idx);
  }
  for(ii=0;ii<idx && ii<maxlights;ii++)
    glEnable((GLenum)(GL_LIGHT0+ii));
  for(;ii<maxlights;ii++)
    glDisable((GLenum)(GL_LIGHT0+ii));

  // Disable clipping planes for the orientation icon.
  vector<bool> cliplist(6, false);
  for (ii = 0; ii < 6; ii++)
  {
    if (glIsEnabled((GLenum)(GL_CLIP_PLANE0+ii)))
    {
      glDisable((GLenum)(GL_CLIP_PLANE0+ii));
      cliplist[ii] = true;
    }
  }

  drawinfo->viewwindow = viewwindow;
  
  // Use depthrange to force the icon to move forward.
  // Ideally the rest of the scene should be drawn at 0.05 1.0,
  // so there was no overlap at all, but that would require
  // mucking about in the picking code.
  glDepthRange(0.0, 0.05);
  axis_obj->draw(drawinfo, 0, current_time);
  glDepthRange(0.0, 1.0);

  drawinfo->fog = fog;  // Restore fog state.

  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);

  // Reenable clipping planes.
  for (ii = 0; ii < 6; ii++)
  {
    if (cliplist[ii])
    {
      glEnable((GLenum)(GL_CLIP_PLANE0+ii));
    }
  }
}

} // End namespace SCIRun
