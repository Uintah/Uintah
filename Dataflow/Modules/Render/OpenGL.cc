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


#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>
#include <sci_defs/bits_defs.h>
#include <Dataflow/Modules/Render/OpenGL.h>
#include <Dataflow/Modules/Render/PBuffer.h> // #defines HAVE_PBUFFER
#include <Dataflow/Modules/Render/logo.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Util/Environment.h>
#include <sci_values.h>

#ifdef HAVE_MAGICK
namespace C_Magick {
#include <magick/api.h>
}
#endif

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

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


OpenGL::OpenGL(GuiInterface* gui, Viewer *viewer, ViewWindow *vw) :
  xres_(0),
  yres_(0),
  doing_image_p_(false),
  doing_movie_p_(false),
  make_MPEG_p_(false),
  current_movie_frame_(0),
  movie_name_("./movie.%04d"),
  // private member variables
  gui_(gui),
  helper_(0),
  helper_thread_(0),
  tkwin_(0),
  x11_dpy_(0),
  myname_("INVAID"),
  viewer_(viewer),
  view_window_(vw),
  drawinfo_(scinew DrawInfoOpenGL),
  x11_visuals_(),
  have_pbuffer_(false),
  dead_(false),
  do_hi_res_(false),
  encoding_mpeg_(false),
  max_gl_lights_(0),
  animate_num_frames_(0),
  animate_time_begin_(0.0),
  animate_time_end_(0.0),
  animate_framerate_(0.0),
  znear_(0.0),
  zfar_(0.0),
  current_time_(0.0),
  frame_count_(1),
  cached_view_(),
  send_pick_x_(0),
  send_pick_y_(0),
  ret_pick_index_(0),
  ret_pick_obj_(0),
  ret_pick_pick_(0),
  send_mailbox_("OpenGL renderer send mailbox",10),
  recv_mailbox_("OpenGL renderer receive mailbox", 10),
  get_mailbox_("OpenGL renderer request mailbox", 5),
  img_mailbox_("OpenGL renderer image data mailbox", 5),
#ifdef HAVE_PBUFFER
  pbuffer_(scinew PBuffer())
#else
  pbuffer_(0)
#endif
{
  fps_timer_.start();
}


OpenGL::~OpenGL()
{
  // Finish up the mpeg that was in progress.
  if (encoding_mpeg_)
  {
    encoding_mpeg_ = false;
    EndMpeg();
  }
  kill_helper();
  int r;
  while (send_mailbox_.tryReceive(r)) ;
  while (recv_mailbox_.tryReceive(r)) ;

  fps_timer_.stop();

  delete drawinfo_;
  drawinfo_ = 0;

  gui_->lock();
  Tk_Window new_tkwin=Tk_NameToWindow(the_interp, ccast_unsafe(myname_),
				      Tk_MainWindow(the_interp));
  if(!new_tkwin)
  {
    cerr << "Unable to locate window!\n";
    gui_->unlock();
    return;
  }
  glXMakeCurrent(Tk_Display(new_tkwin), None, NULL);
  gui_->unlock();

  if (pbuffer_) { delete pbuffer_; }
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
OpenGL::redraw(double _tbeg, double _tend, int _nframes, double _framerate)
{
  if (dead_) return;
  animate_time_begin_=_tbeg;
  animate_time_end_=_tend;
  animate_num_frames_=_nframes;
  animate_framerate_=_framerate;
  // This is the first redraw - if there is not an OpenGL thread,
  // start one...
  if(!helper_)
  {
    helper_=new OpenGLHelper(this);
    helper_thread_ = new Thread(helper_, string("OpenGL: "+myname_).c_str());
  }
  send_mailbox_.send(DO_REDRAW);
  int rc=recv_mailbox_.receive();
  if(rc != REDRAW_DONE)
  {
    cerr << "Wanted redraw_done, but got: " << rc << "\n";
  }
}


void
OpenGL::kill_helper()
{
  // kill the helper thread
  dead_ = true;
  if (helper_thread_)
  {
    send_mailbox_.send(86);
    helper_thread_->join();
    helper_thread_ = 0;
  }
}


void
OpenGL::redraw_loop()
{
  int r;
  int resx = -1;
  int resy = -1;
  string fname, ftype;
  // Tell the ViewWindow that we are started...
  TimeThrottle throttle;
  throttle.start();
  double newtime=0;
  for(;;)
  {
    int nreply=0;
    if(view_window_->inertia_mode_)
    {
      current_time_=throttle.time();
      if(animate_framerate_==0)
	animate_framerate_=30;
      double frametime=1./animate_framerate_;
      double delta=current_time_-newtime;
      if(delta > 1.5*frametime)
      {
	animate_framerate_=1./delta;
	frametime=delta;
	newtime=current_time_;
      }
      if(delta > .85*frametime)
      {
	animate_framerate_*=.9;
	frametime=1./animate_framerate_;
	newtime=current_time_;
      }
      else if(delta < .5*frametime)
      {
	animate_framerate_*=1.1;
	if(animate_framerate_>30)
	{
	  animate_framerate_=30;
	}
	frametime=1./animate_framerate_;
	newtime=current_time_;
      }
      newtime+=frametime;
      throttle.wait_for_time(newtime);
      while (send_mailbox_.tryReceive(r))
      {
	if (r == 86)
	{
	  throttle.stop();
	  return;
	}
	else if (r == DO_PICK)
	{
	  real_get_pick(send_pick_x_, send_pick_y_, ret_pick_obj_,
			ret_pick_pick_, ret_pick_index_);
	  recv_mailbox_.send(PICK_DONE);
	}
	else if(r== DO_GETDATA)
	{
	  GetReq req(get_mailbox_.receive());
	  real_getData(req.datamask, req.result);
	}
	else if(r== DO_IMAGE)
	{
	  ImgReq req(img_mailbox_.receive());
	  do_hi_res_ = true;
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
      // axis - the current quaternion is viewwindow->ball_->qNow	
      // the first 3 components of this

      view_window_->ball_->SetAngle(newtime*view_window_->angular_v_);
      View tmpview(view_window_->rot_view_);
      Transform tmp_trans;
      HMatrix mNow;
      view_window_->ball_->Value(mNow);
      tmp_trans.set(&mNow[0][0]);
      Transform prv = view_window_->prev_trans_;
      prv.post_trans(tmp_trans);
      HMatrix vmat;
      prv.get(&vmat[0][0]);
      Point y_a(vmat[0][1],vmat[1][1],vmat[2][1]);
      Point z_a(vmat[0][2],vmat[1][2],vmat[2][2]);
      tmpview.up(y_a.vector());
      if (view_window_->inertia_mode_ == 1)
      {
	tmpview.eyep((z_a*(view_window_->eye_dist_))+
		     tmpview.lookat().vector());
	view_window_->gui_view_.set(tmpview);
      }
      else if (view_window_->inertia_mode_ == 2)
      {
	tmpview.lookat(tmpview.eyep()-
		       (z_a*(view_window_->eye_dist_)).vector());
	view_window_->gui_view_.set(tmpview);
      }

    }
    else
    {
      for (;;)
      {
	int r=send_mailbox_.receive();
	if (r == 86)
	{
	  throttle.stop();
	  return;
	}
	else if (r == DO_PICK)
	{
	  real_get_pick(send_pick_x_, send_pick_y_, ret_pick_obj_,
			ret_pick_pick_, ret_pick_index_);
	  recv_mailbox_.send(PICK_DONE);
	}
	else if(r== DO_GETDATA)
	{
	  GetReq req(get_mailbox_.receive());
	  real_getData(req.datamask, req.result);
	}
	else if(r== DO_IMAGE)
	{
	  ImgReq req(img_mailbox_.receive());
	  do_hi_res_ = true;
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

    if(do_hi_res_)
    {
      render_and_save_image(resx,resy,fname,ftype);
      do_hi_res_=false;
    }
    redraw_frame();
    for(int i=0;i<nreply;i++) {
      recv_mailbox_.send(REDRAW_DONE);
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
    cerr << "Error - ImageMagick is not enabled, " 
	 << "can only save .ppm or .raw files.\n";
    return;
  }
#endif

  cerr << "Saving Image: " << fname << " with width=" << x
       << " and height=" << y <<"...\n";


#ifndef HAVE_PBUFFER
  // Don't need to raise if using pbuffer.
  // FIXME: this next line was apparently meant to raise the Viewer to the
  //        top... but it doesn't actually seem to work
  Tk_RestackWindow(tkwin_,Above,NULL);
#endif

  gui_->lock();
  // Make sure our GL context is current
  if(current_drawer != this)
  {
    current_drawer=this;
    glXMakeCurrent(x11_dpy_, x11_win_, x11_gl_context_);
  }
  deriveFrustum();

  // Get Viewport dimensions
  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT,vp);
  gui_->unlock();

  hi_res_.resx = x;
  hi_res_.resy = y;
  // The EXACT # of screen rows and columns needed to render the image
  hi_res_.ncols = (double)hi_res_.resx/(double)vp[2];
  hi_res_.nrows = (double)hi_res_.resy/(double)vp[3];

  // The # of screen rows and columns that will be rendered to make the image
  const int nrows = (int)ceil(hi_res_.nrows);
  const int ncols = (int)ceil(hi_res_.ncols);

  ofstream *image_file = NULL;
#ifdef HAVE_MAGICK
  C_Magick::Image *image = NULL;
  C_Magick::ImageInfo *image_info = NULL;
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
      (*image_file) << hi_res_.resx << " " << hi_res_.resy << std::endl;
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
    image->columns=hi_res_.resx;
    image->rows=hi_res_.resy;
#endif
  }

  const int pix_size = channel_bytes*num_channels;

  // Write out a screen height X image width chunk of pixels at a time
  unsigned char* pixels=
    scinew unsigned char[hi_res_.resx*vp[3]*pix_size];

  // Start writing image_file
  static unsigned char* tmp_row = 0;
  if (!tmp_row )
    tmp_row = scinew unsigned char[hi_res_.resx*pix_size];


  for (hi_res_.row = nrows - 1; hi_res_.row >= 0; --hi_res_.row)
  {
    int read_height = hi_res_.resy - hi_res_.row * vp[3];
    read_height = (vp[3] < read_height) ? vp[3] : read_height;

    if (do_magick)
    {
#ifdef HAVE_MAGICK
      pixels = (unsigned char *)C_Magick::SetImagePixels
	(image,0,vp[3]*(nrows - 1 - hi_res_.row),hi_res_.resx,read_height);
#endif
    }

    if (!pixels)
    {
      cerr << "No ImageMagick Memory! Aborting...\n";
      break;
    }

    for (hi_res_.col = 0; hi_res_.col < ncols; hi_res_.col++)
    {
      // render the col and row in the hi_res struct
      doing_image_p_ = true; // forces pbuffer if available
      redraw_frame();
      doing_image_p_ = false;
      gui_->lock();
	
      // Tell OpenGL where to put the data in our pixel buffer
      glPixelStorei(GL_PACK_ALIGNMENT,1);
      glPixelStorei(GL_PACK_SKIP_PIXELS, hi_res_.col * vp[2]);
      glPixelStorei(GL_PACK_SKIP_ROWS,0);
      glPixelStorei(GL_PACK_ROW_LENGTH, hi_res_.resx);

      int read_width = hi_res_.resx - hi_res_.col * vp[2];
      read_width = (vp[2] < read_width) ? vp[2] : read_width;

      // Read the data from OpenGL into our memory
      glReadBuffer(GL_FRONT);
      glReadPixels(0,0,read_width, read_height,
		   (num_channels == 3) ? GL_RGB : GL_BGRA,
		   (channel_bytes == 1) ? GL_UNSIGNED_BYTE : GL_UNSIGNED_SHORT,
		   pixels);
      gui_->unlock();
    }
    // OpenGL renders upside-down to image_file writing
    unsigned char *top_row, *bot_row;	
    int top, bot;
    for(top = read_height-1, bot = 0; bot < read_height/2; top--, bot++)
    {
      top_row = pixels + hi_res_.resx*top*pix_size;
      bot_row = pixels + hi_res_.resx*bot*pix_size;
      memcpy(tmp_row, top_row, hi_res_.resx*pix_size);
      memcpy(top_row, bot_row, hi_res_.resx*pix_size);
      memcpy(bot_row, tmp_row, hi_res_.resx*pix_size);
    }
    if (do_magick)
    {
#ifdef HAVE_MAGICK
      C_Magick::SyncImagePixels(image);
#endif
    } else
      image_file->write((char *)pixels, hi_res_.resx*read_height*pix_size);
  }
  gui_->lock();

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

  gui_->unlock();

  if (tmp_row)
  {
    delete[] tmp_row;
    tmp_row = 0;
  }

  if (sci_getenv("SCI_REGRESSION_TESTING"))
  {
    Thread::exitAll(0);
  }
}


void
OpenGL::redraw_frame()
{
  if (dead_) return;
  gui_->lock();
  if (dead_) { // ViewWindow was deleted from gui_
    gui_->unlock(); 
    return;
  }
  Tk_Window new_tkwin=Tk_NameToWindow(the_interp, ccast_unsafe(myname_),
				      Tk_MainWindow(the_interp));
  if(!new_tkwin)
  {
    cerr << "Unable to locate window!\n";
    gui_->unlock();
    return;
  }
  if(tkwin_ != new_tkwin)
  {
    tkwin_=new_tkwin;
    x11_dpy_=Tk_Display(tkwin_);
    x11_win_=Tk_WindowId(tkwin_);
    // Race condition,  create context before the window is done.
    while (x11_win_==0)
    {
      gui_->unlock();
      Thread::yield();
      gui_->lock();
      x11_win_ = Tk_WindowId(tkwin_);
    }
    x11_gl_context_=OpenGLGetContext(the_interp, ccast_unsafe(myname_));
    if(!x11_gl_context_)
    {
      cerr << "Unable to create OpenGL Context!\n";
      gui_->unlock();
      return;
    }
    //cerr << "The next line will crash!: " << (unsigned int)x11_dpy_ <<  ", " 
    //	 << (unsigned int)x11_win_ << ", " << (unsigned int)x11_gl_context_ 
    //	 << std::endl;
    glXMakeCurrent(x11_dpy_, x11_win_, x11_gl_context_);
    //    cerr << "See! it crashed";
    glXWaitX();
#if defined(HAVE_GLEW)
    sci_glew_init();
#endif
    current_drawer=this;
    GLint data[1];
    glGetIntegerv(GL_MAX_LIGHTS, data);
    max_gl_lights_=data[0];
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

  gui_->unlock();

  // Start polygon counter...
  WallClockTimer timer;
  timer.clear();
  timer.start();

  // Get the window size
  xres_=Tk_Width(tkwin_);
  yres_=Tk_Height(tkwin_);

  // Make ourselves current
  if(current_drawer != this)
  {
    current_drawer=this;
    gui_->lock();
    glXMakeCurrent(x11_dpy_, x11_win_, x11_gl_context_);
    gui_->unlock();
  }
  // Set up a pbuffer associated with x11_dpy_
  // Get a lock on the geometry database...
  // Do this now to prevent a hold and wait condition with TCLTask
  viewer_->geomlock_.readLock();

  gui_->lock();
#if defined(HAVE_PBUFFER)
  int screen = Tk_ScreenNumber(tkwin_);
  if( xres_ != pbuffer_->width() || yres_ != pbuffer_->height() ){
    //cerr<<"creating new pbuffer: width = "<<xres<<", height == "<<yres<<"\n";
    pbuffer_->destroy();
    if( !pbuffer_->create( x11_dpy_, screen, xres_, yres_, 8, 8 ) ) {
      //  printf( "Pbuffer create failed.  PBuffering will not be used.\n" );
    } else {
      have_pbuffer_ = true;
    }
  }

  if((doing_movie_p_ || doing_image_p_) && 
     pbuffer_->is_valid()){
    pbuffer_->makeCurrent();
    glDrawBuffer( GL_FRONT );
  } else if( have_pbuffer_ && pbuffer_->is_current() ) {
    glXMakeCurrent(x11_dpy_, x11_win_, x11_gl_context_);
  }
#endif

  // Clear the screen...
  glViewport(0, 0, xres_, yres_);
  Color bg(view_window_->gui_bgcolor_.get());
  glClearColor(bg.r(), bg.g(), bg.b(), 1);

  // Setup the view...
  View view(view_window_->gui_view_.get());
  cached_view_=view;
  double aspect=double(xres_)/double(yres_);
  // XXX - UNICam change-- should be '1.0/aspect' not 'aspect' below
  double fovy=RtoD(2*Atan(1.0/aspect*Tan(DtoR(view.fov()/2.))));

  drawinfo_->reset();

  int do_stereo=view_window_->gui_do_stereo_.get();
  drawinfo_->ambient_scale_ = view_window_->gui_ambient_scale_.get();
  drawinfo_->diffuse_scale_ = view_window_->gui_diffuse_scale_.get();
  drawinfo_->specular_scale_ = view_window_->gui_specular_scale_.get();
  drawinfo_->shininess_scale_ = view_window_->gui_shininess_scale_.get();
  drawinfo_->emission_scale_ = view_window_->gui_emission_scale_.get();
  drawinfo_->line_width_ = view_window_->gui_line_width_.get();
  drawinfo_->point_size_ = view_window_->gui_point_size_.get();
  drawinfo_->polygon_offset_factor_ =
    view_window_->gui_polygon_offset_factor_.get();
  drawinfo_->polygon_offset_units_ =
    view_window_->gui_polygon_offset_units_.get();

  if(compute_depth(view, znear_, zfar_))
  {

    // Set up graphics state
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);
    glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
    string globals("global");
    view_window_->setState(drawinfo_,globals);
    drawinfo_->pickmode=0;

    CHECK_OPENGL_ERROR("after setting up the graphics state: ")

#if defined(HAVE_PBUFFER)
    if( pbuffer_->is_current() &&
	(!doing_movie_p_ && !doing_image_p_) ){
      glXMakeCurrent(x11_dpy_, x11_win_, x11_gl_context_);
    }
#endif
    // Do the redraw loop for each time value
    double dt=(animate_time_end_-animate_time_begin_)/animate_num_frames_;
    double frametime=animate_framerate_==0?0:1./animate_framerate_;
    TimeThrottle throttle;
    throttle.start();
    Vector eyesep(0,0,0);
    if(do_stereo)
    {
      //      double eye_sep_dist=0.025/2;
      double eye_sep_dist=view_window_->gui_sbase_.get()*
	(view_window_->gui_sr_.get()?0.048:0.0125);
      Vector u, v;
      view.get_viewplane(aspect, 1.0, u, v);
      u.normalize();
      double zmid=(znear_+zfar_)/2.;
      eyesep=u*eye_sep_dist*zmid;
    }

    for(int t=0;t<animate_num_frames_;t++)
    {
      int n=1;
      if( do_stereo ) n=2;
      for(int i=0;i<n;i++)
      {
	if( do_stereo )
	{
	  glDrawBuffer(i==0?GL_BACK_LEFT:GL_BACK_RIGHT);
	}
	else
	{
#if defined(HAVE_PBUFFER)
	  if(have_pbuffer_){
	    if(!doing_movie_p_ && !doing_image_p_){
	      if( pbuffer_->is_current() )
                cerr<<"pbuffer is current while not doing Movie\n";
#endif
	    glDrawBuffer(GL_BACK);
#if defined(HAVE_PBUFFER)
	    } else {
	    glDrawBuffer(GL_FRONT);
	    }
	  } else {
	    glDrawBuffer(GL_BACK);
	  }
#endif	
	}
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0, 0, xres_, yres_);

	double modeltime=t*dt+animate_time_begin_;
	view_window_->set_current_time(modeltime);
	
	{ // render normal
	  // Setup view.
	  glMatrixMode(GL_PROJECTION);

	  glLoadIdentity();

	  if (view_window_->gui_ortho_view_.get())
	  {
	    const double len = (view.lookat() - view.eyep()).length();
	    const double yval = tan(fovy * M_PI / 360.0) * len;
	    const double xval = yval * aspect;
	    glOrtho(-xval, xval, -yval, yval, znear_, zfar_);
	  }
	  else
	  {
	    gluPerspective(fovy, aspect, znear_, zfar_);
	  }
	  glMatrixMode(GL_MODELVIEW);

	  glLoadIdentity();
	  Point eyep(view.eyep());
	  Point lookat(view.lookat());
	  if(do_stereo){
	    if(i==0){
	      eyep-=eyesep;
	      if (!view_window_->gui_sr_.get())
		lookat-=eyesep;
	    } else {
	      eyep+=eyesep;
	      if (!view_window_->gui_sr_.get())
		lookat+=eyesep;
	    }
	  }
	  Vector up(view.up());
	  gluLookAt(eyep.x(), eyep.y(), eyep.z(),
		    lookat.x(), lookat.y(), lookat.z(),
		    up.x(), up.y(), up.z());
	  if(do_hi_res_)
	    setFrustumToWindowPortion();
	}
	
	// Set up Lighting
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	const Lighting& l = viewer_->lighting_;
	int idx=0;
	int ii;
	for(ii=0;ii<l.lights.size();ii++)
	{
	  LightHandle light=l.lights[ii];
	  light->opengl_setup(view, drawinfo_, idx);
	}
	for(ii=0;ii<idx && ii<max_gl_lights_;ii++)
	  glEnable((GLenum)(GL_LIGHT0+ii));
	for(;ii<max_gl_lights_;ii++)
	  glDisable((GLenum)(GL_LIGHT0+ii));

	// Now set up the fog stuff.
	double fognear, fogfar;
	compute_fog_depth(view, fognear, fogfar,
			  view_window_->gui_fog_visibleonly_.get());
	glFogi(GL_FOG_MODE, GL_LINEAR);
	const float fnear =
	  fognear + (fogfar - fognear) * view_window_->gui_fog_start_.get();
	glFogf(GL_FOG_START, fnear);
	const double ffar =
	  fognear + (fogfar - fognear) / 
	  Max(view_window_->gui_fog_end_.get(), 0.001);
	glFogf(GL_FOG_END, ffar);
	GLfloat bgArray[4];
	if (view_window_->gui_fogusebg_.get())
	{
	  bgArray[0] = bg.r();
	  bgArray[1] = bg.g();
	  bgArray[2] = bg.b();
	}
	else
	{
	  Color fogcolor(view_window_->gui_fogcolor_.get());
	  bgArray[0] = fogcolor.r();
	  bgArray[1] = fogcolor.g();
	  bgArray[2] = fogcolor.b();
	}	  
	bgArray[3]=1.0;
	glFogfv(GL_FOG_COLOR, bgArray);

	// now make the ViewWindow setup its clipping planes...
	view_window_->setClip(drawinfo_);
        view_window_->setMouse(drawinfo_);
        
	// UNICAM addition
	glGetDoublev (GL_MODELVIEW_MATRIX, modelview_matrix_);
	glGetDoublev (GL_PROJECTION_MATRIX, projection_matrix_);
	glGetIntegerv(GL_VIEWPORT, viewport_matrix_);

	// set up point size, line size, and polygon offset
	glPointSize(drawinfo_->point_size_);
	glLineWidth(drawinfo_->line_width_);
	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
	if (drawinfo_->polygon_offset_factor_ || 
	    drawinfo_->polygon_offset_units_) {
	  glPolygonOffset(drawinfo_->polygon_offset_factor_, 
			  drawinfo_->polygon_offset_units_);
	  glEnable(GL_POLYGON_OFFSET_FILL);
	} else {
	  glDisable(GL_POLYGON_OFFSET_FILL);
	}

	// Draw it all...
	current_time_=modeltime;
	view_window_->do_for_visible(this, &OpenGL::redraw_obj);

	if (view_window_->gui_raxes_.get())
	{
	  render_rotation_axis(view, do_stereo, i, eyesep);
	}

      }
	
      // save z-buffer data
      if (CAPTURE_Z_DATA_HACK)
      {
	CAPTURE_Z_DATA_HACK = 0;
	glReadPixels(0, 0, xres_, yres_, GL_DEPTH_COMPONENT, GL_FLOAT,
		     depth_buffer_ );
      }
	
      // Wait for the right time before swapping buffers
      //gui->unlock();
      double realtime=t*frametime;
      if(animate_num_frames_>1)
      {
	throttle.wait_for_time(realtime);
      }
      //gui->lock();
      gui_->execute("update idletasks");

      // Show the pretty picture
#if defined(HAVE_PBUFFER)
      if( !have_pbuffer_ ||
	  (!doing_movie_p_ && !doing_image_p_) )
#endif
	glXSwapBuffers(x11_dpy_, x11_win_);
    }
    throttle.stop();
    double fps;
    if (throttle.time()>0)
      fps=animate_num_frames_/throttle.time();
    else
      fps=animate_num_frames_;
    //int fps_whole=(int)fps;
    //int fps_hund=(int)((fps-fps_whole)*100);
    ostringstream str;
    // str << view_window_->id << " setFrameRate "<<fps_whole<<"."<< fps_hund;
    gui_->execute(str.str());
    view_window_->set_current_time(animate_time_end_);
  }
  else
  {
    // Just show the cleared screen
    view_window_->set_current_time(animate_time_end_);
	
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
#if defined(HAVE_PBUFFER)
      if( !have_pbuffer_ || (!doing_movie_p_ && !doing_image_p_))
#endif
    glXSwapBuffers(x11_dpy_, x11_win_);
  }

  viewer_->geomlock_.readUnlock();

  // Look for errors
  CHECK_OPENGL_ERROR("after drawing objects: ")

  // Report statistics
  timer.stop();
  fps_timer_.stop();
  ostringstream str;
  // Frame counter This is used to keep track of how many frames the
  // statistics are being used for.  When we go to update the on
  // screen statistics we will check to see if the elapsed time is
  // greater than .33 seconds.  If not we increment this counter and
  // don't display the statistics.  When the accumulated time becomes
  // greater than .33 seconds, we update our statistics.
  if (fps_timer_.time()>.33) {
    double fps;
    fps=animate_num_frames_*frame_count_/fps_timer_.time();
//     cerr << "fps = " << fps <<",\tframe_count_ = "<<frame_count_
// 	 <<",\tfps_timer_.time = "<<fps_timer_.time()
// 	 <<",\tanimate_num_frames_ = "<<animate_num_frames_<<"\n";
    cerr.flush();
    frame_count_ = 1;
    fps+=0.05;			// Round to nearest tenth
    int fps_whole=(int)fps;
    int fps_tenths=(int)((fps-fps_whole)*10);
    fps_timer_.clear();
    fps_timer_.start();		// Start it running for next time
    double pps;
    if (timer.time()>0)
      pps=drawinfo_->polycount/timer.time();
    else
      pps=drawinfo_->polycount;
    str << view_window_->id_ << " updatePerf \"";
    str << drawinfo_->polycount << " polygons in " << timer.time()
      	<< " seconds\" \"" << pps
    	<< " polygons/second\"" << " \"" << fps_whole << "."
    	<< fps_tenths << " frames/sec\"" << '\0';
  } else if (fps_timer_.time() > 0) {
    frame_count_++;
    fps_timer_.start();
  } else {
    fps_timer_.start();
  }    
  /*****************************************/
  /* movie-movie makin' movie-movie makin' */
  /*****************************************/
  if (doing_movie_p_) {	
    //      cerr << "Saving a movie!\n";
    if(make_MPEG_p_ ) {

      if(encoding_mpeg_) {
	AddMpegFrame();
      } else {

	string fname = movie_name_;

	// only add extension if not allready there
	if(!(fname.find(".mpg") != std::string::npos ||
	     fname.find(".MPG") != std::string::npos ||
	     fname.find(".mpeg") != std::string::npos ||
	     fname.find(".MPEG") != std::string::npos)) {
	  fname = fname + string(".mpg");
	}
	
	// Dump the mpeg in the local dir ... ignoring any path since mpeg
	// can not handle it.
	//std::string::size_type pos = fname.find_last_of("/");
	//if( pos != std::string::npos ) {
        //cerr << "Removing the mpeg path." << std::endl;
        //fname = fname.erase(0, pos+1);
        //}

	if( fname.find("%") != std::string::npos ) {
	  cerr << "Remove the C Style format for the frames." << std::endl;
	  cerr << "The format should be of the form: 'my_movie'" << std::endl;
	} else {
	  cerr << "Dumping mpeg " << fname << std::endl;

	  StartMpeg( fname );
	  AddMpegFrame();

	  encoding_mpeg_ = true;
	}
      }

    } else { // dump each frame
      if(encoding_mpeg_) { // Finish up mpeg that was in progress.
	encoding_mpeg_ = false;
	EndMpeg();
      }

      std::string::size_type pos = movie_name_.find_last_of("%0");

      if( pos == std::string::npos ||
	  movie_name_[pos+2] != 'd' ||
	  movie_name_.find("%") != movie_name_.find_last_of("%") ) {
	cerr << "Bad C Style format for the frames." << std::endl;
	cerr << "The format should be of the form: './my_movie.%04d'";
	cerr << std::endl;
      } else {

	char fname[256];
	sprintf(fname, movie_name_.c_str(), current_movie_frame_++);
	string fullpath = string(fname) + string(".ppm");
	cerr << "Dumping " << fullpath << "....  ";
	dump_image(fullpath);
	cerr << " done!\n";
      }
    }
  } else {
    if(encoding_mpeg_) { // Finish up mpeg that was in progress.
      encoding_mpeg_ = false;
      EndMpeg();
    }
  }
  gui_->execute(str.str());
  gui_->unlock();
}



void
OpenGL::get_pick(int x, int y,
		 GeomHandle& pick_obj, GeomPickHandle& pick_pick,
		 int& pick_index)
{
  send_pick_x_=x;
  send_pick_y_=y;
  send_mailbox_.send(DO_PICK);
  for(;;)
  {
    int r=recv_mailbox_.receive();
    if(r != PICK_DONE)
    {
      cerr << "WANTED A PICK!!! (got back " << r << "\n";
    }
    else
    {
      pick_obj=ret_pick_obj_;
      pick_pick=ret_pick_pick_;
      pick_index=ret_pick_index_;
      break;
    }
  }
}



void
OpenGL::real_get_pick(int x, int y,
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
    gui_->lock();
    glXMakeCurrent(x11_dpy_, x11_win_, x11_gl_context_);
    cerr<<"viewer current\n";
    gui_->unlock();
  }
  // Setup the view...
  View view(view_window_->gui_view_.get());
  viewer_->geomlock_.readLock();

  // Compute znear_ and zfar...
  double znear;
  double zfar;
  if(compute_depth(view, znear, zfar))
  {
    // Setup picking...
    gui_->lock();

    GLuint pick_buffer[pick_buffer_size];
    glSelectBuffer(pick_buffer_size, pick_buffer);
    glRenderMode(GL_SELECT);
    glInitNames();
#ifdef SCI_64BITS
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

    // Picking
    double aspect=double(xres_)/double(yres_);
    // XXX - UNICam change-- should be '1.0/aspect' not 'aspect' below
    double fovy=RtoD(2*Atan(1.0/aspect*Tan(DtoR(view.fov()/2.))));
    glViewport(0, 0, xres_, yres_);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    gluPickMatrix(x, viewport[3]-y, pick_window, pick_window, viewport);
    if (view_window_->gui_ortho_view_.get())
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
    Vector up(view.up());
    gluLookAt(eyep.x(), eyep.y(), eyep.z(),
	      lookat.x(), lookat.y(), lookat.z(),
	      up.x(), up.y(), up.z());

    drawinfo_->lighting=0;
    drawinfo_->set_drawtype(DrawInfoOpenGL::Flat);
    drawinfo_->pickmode=1;
    //drawinfo_->pickable=0;

    // Draw it all...
    view_window_->do_for_visible(this,
				 (ViewWindowVisPMF)&OpenGL::pick_draw_obj);

#ifdef SCI_64BITS
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
    gui_->unlock();
    GLuint min_z;
#ifdef SCI_64BITS
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
    cerr << "hits=" << hits << "\n";
    if(hits >= 1)
    {
      int idx=0;
      min_z=0;
      int have_one=0;
      for (int h=0; h<hits; h++)
      {
	int nnames=pick_buffer[idx++];
	GLuint z=pick_buffer[idx++];
	cerr << "h=" << h << ", nnames=" << nnames << ", z=" << z << "\n";
	if (nnames > 1 && (!have_one || z < min_z))
	{
	  min_z=z;
	  have_one=1;
	  idx++; // Skip Max Z
#ifdef SCI_64BITS
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
	  cerr << "new min... (obj=" << hit_obj
	       << ", pick="          << hit_pick << std::endl;
	    //   << ", index = "       << hit_pick_index << ")\n";
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


// dump a ppm image
void
OpenGL::dump_image(const string& name, const string& /* type */)
{
  ofstream dumpfile(name.c_str());
  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT,vp);
  int pix_size = 3;  // for RGB
  int n=pix_size*vp[2]*vp[3];
  unsigned char* pxl=scinew unsigned char[n];
  glPixelStorei(GL_PACK_ALIGNMENT,1);
  glReadBuffer(GL_FRONT);
  glReadPixels(0,0,vp[2],vp[3],GL_RGB,GL_UNSIGNED_BYTE,pxl);


#if defined(HAVE_PBUFFER)
  if( have_pbuffer_ && pbuffer_->is_valid() && pbuffer_->is_current() ){
    glXMakeCurrent( x11_dpy_, x11_win_, x11_gl_context_ ); 
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, vp[2], 0.0, vp[3], -10.0, 10.0);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glDrawBuffer(GL_BACK);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(vp[2],vp[3], GL_RGB, GL_UNSIGNED_BYTE,pxl);
    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glXSwapBuffers(x11_dpy_,x11_win_);
  }
#endif

  // Print out the ppm  header
  dumpfile << "P6" << std::endl;
  dumpfile << vp[2] << " " << vp[3] << std::endl;
  dumpfile << 255 << std::endl;

  // OpenGL renders upside-down to ppm_file writing
  unsigned char *top_row, *bot_row;	
  unsigned char *tmp_row = scinew unsigned char[ vp[2] * pix_size];
  int top, bot;
  for( top = vp[3] - 1, bot = 0; bot < vp[3]/2; top --, bot++){
    top_row = pxl + vp[2] *top*pix_size;
    bot_row = pxl + vp[2]*bot*pix_size;
    memcpy(tmp_row, top_row, vp[2]*pix_size);
    memcpy(top_row, bot_row, vp[2]*pix_size);
    memcpy(bot_row, tmp_row, vp[2]*pix_size);
  }
  // now dump the file
  dumpfile.write((const char *)pxl,n);

  delete [] pxl;
  delete [] tmp_row;
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
  glScaled(2./xres_, 2./yres_, 1.0);
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
#ifdef SCI_64BITS
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
  obj->draw(drawinfo_, viewer->default_material_.get_rep(), current_time_);
}



void
OpenGL::redraw_obj(Viewer* viewer, ViewWindow* viewwindow, GeomHandle obj)
{
  drawinfo_->viewwindow = viewwindow;
  obj->draw(drawinfo_, viewer->default_material_.get_rep(), current_time_);
}



void
ViewWindow::setState(DrawInfoOpenGL* drawinfo, const string& tclID)
{
  GuiString type(ctx_->subVar(tclID+"-type",false));
  if (type.valid()) {
    if (type.get() == "Default")
    {
      setState(drawinfo,"global");	
      return; // if they are using the default, con't change
    } 
    else if(type.get() == "Wire")
    {
    drawinfo->set_drawtype(DrawInfoOpenGL::WireFrame);
    drawinfo->lighting=0;
    }
    else if(type.get() == "Flat")
    {
      drawinfo->set_drawtype(DrawInfoOpenGL::Flat);
      drawinfo->lighting=0;
    }
    else if(type.get() == "Gouraud")
    {
      drawinfo->set_drawtype(DrawInfoOpenGL::Gouraud);
      drawinfo->lighting=1;
    }
    else
    {
      cerr << "Unknown shading(" << type.get() << "), defaulting to phong\n";
      drawinfo->set_drawtype(DrawInfoOpenGL::Gouraud);
      drawinfo->lighting=1;
    }
  }

  // Now see if they want a bounding box.
  GuiInt debug(ctx_->subVar(tclID+"-debug",false));
  if (debug.valid()) 
    drawinfo->debug = debug.get();


  GuiString movieName(ctx_->subVar(tclID+"-movieName",false));
  if (movieName.valid())
    renderer_->movie_name_ = movieName.get();

  GuiInt movie(ctx_->subVar(tclID+"-movie",false));
  if (movie.valid()) {
    if (!movie.get()) {
      renderer_->doing_movie_p_ = 0;
      renderer_->make_MPEG_p_ = 0;
    } else if (!renderer_->doing_movie_p_) {
      GuiInt movieFrame(ctx_->subVar(tclID+"-movieFrame",false));
      if (movieFrame.valid())
	renderer_->current_movie_frame_ = movieFrame.get();

      renderer_->doing_movie_p_ = 1;
      if (movie.get() == 1)
	renderer_->make_MPEG_p_ = 0;
      else if (movie.get() == 2)
	renderer_->make_MPEG_p_ = 1;
    }
  }

  GuiInt clip(ctx_->subVar(tclID+"-clip",false));
  if (clip.valid())
    drawinfo->check_clip = clip.get();
  drawinfo->init_clip(); // set clipping

  GuiInt cull(ctx_->subVar(tclID+"-cull",false));
  if (cull.valid())
    drawinfo->cull = cull.get();

  GuiInt dl(ctx_->subVar(tclID+"-dl",false));
  if (dl.valid())
    drawinfo->dl = dl.get();

  GuiInt fog(ctx_->subVar(tclID+"-fog",false));
  if (fog.valid())
    drawinfo->fog=fog.get();

  GuiInt lighting(ctx_->subVar(tclID+"-light",false));
  if (lighting.valid())
      drawinfo->lighting=lighting.get();

  drawinfo->currently_lit=drawinfo->lighting;
  drawinfo->init_lighting(drawinfo->lighting);
}



void
ViewWindow::setDI(DrawInfoOpenGL* drawinfo,string name)
{
  map<string,int>::iterator tag_iter = obj_tag_.find(name);
  if (tag_iter != obj_tag_.end()) { // if found
    setState(drawinfo,to_string((*tag_iter).second));
  }
}


// Set the bits for the clipping planes that are on.
void
ViewWindow::setClip(DrawInfoOpenGL* drawinfo)
{
  GuiString val(ctx_->subVar("clip-visible",false));
  GuiInt clipnum(ctx_->subVar("clip-num",false));
  int i = clipnum.get();

  drawinfo->clip_planes = 0; // set them all of for default
  if (val.valid() && clipnum.valid())
  {
    int cur_flag = CLIP_P5;
    if (i > 0 && i < 7)
    {
      while(i--)
      {
	const string istr = to_string(i+1);
	GuiInt clip_visible(ctx_->subVar("clip-visible-"+ istr,false));
		
	if (!clip_visible.valid()) continue;

	if (!clip_visible.get()) {
	  glDisable((GLenum)(GL_CLIP_PLANE0+i));
	}
	else 
	{
	  double plane[4];
	  GuiDouble x(ctx_->subVar("clip-normal-x-"+ istr,false));
	  GuiDouble y(ctx_->subVar("clip-normal-y-"+ istr,false));
	  GuiDouble z(ctx_->subVar("clip-normal-z-"+ istr,false));
	  GuiDouble d(ctx_->subVar("clip-normal-d-"+ istr,false));

	  if (!x.valid() || !y.valid() || ! z.valid() || !d.valid())
	  {
	    cerr << "Error: clipping plane " << i << " has invalid values.\n";
	    continue;
	  }
	  
	  plane[0] = x.get();
	  plane[1] = y.get();
	  plane[2] = z.get();
	  plane[3] = d.get();
	  
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
	}
	cur_flag >>= 1; // shift the bit we are looking at...
      }
    }
  }
}

void
ViewWindow::setMouse(DrawInfoOpenGL* drawinfo)
{
  drawinfo->mouse_action = mouse_action_;
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
if(glXGetConfig(x11_dpy_, &vinfo[i], attrib, &value) != 0){\
  args.error("Error getting attribute: " #attrib); \
  return; \
}


void
OpenGL::listvisuals(GuiArgs& args)
{
  gui_->lock();

  Thread::allow_sgi_OpenGL_page0_sillyness();
  Tk_Window topwin=Tk_NameToWindow(the_interp, ccast_unsafe(args[2]),
				   Tk_MainWindow(the_interp));
  if(!topwin)
  {
    cerr << "Unable to locate window!\n";
    gui_->unlock();
    return;
  }
  x11_dpy_=Tk_Display(topwin);
  int screen=Tk_ScreenNumber(topwin);
  vector<string> visualtags;
  vector<int> scores;
  x11_visuals_.clear();
  int nvis;
  XVisualInfo* vinfo=XGetVisualInfo(x11_dpy_, 0, NULL, &nvis);
  if(!vinfo)
  {
    args.error("XGetVisualInfo failed");
    gui_->unlock();
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
    x11_visuals_.push_back(&vinfo[i]);
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
	XVisualInfo* tmp3=x11_visuals_[i];
	x11_visuals_[i]=x11_visuals_[j];
	x11_visuals_[j]=tmp3;
      }
    }
  }
  args.result(GuiArgs::make_list(visualtags));
  gui_->unlock();
}



void
OpenGL::setvisual(const string& wname, unsigned int which, int wid, int height)
{
  if (which >= x11_visuals_.size())
  {
    cerr << "Invalid OpenGL visual, using default.\n";
    which = 0;
  }

  tkwin_=0;
  current_drawer=0;

  gui_->execute("opengl " + wname +
	       " -visual " + to_string((int)x11_visuals_[which]->visualid) +
	       " -direct true" +
	       " -geometry " + to_string(wid) + "x" + to_string(height));

  myname_ = wname;
}



void
OpenGL::deriveFrustum()
{
  double pmat[16];
  glGetDoublev(GL_PROJECTION_MATRIX, pmat);
  const double G = (pmat[10]-1)/(pmat[10]+1);
  frustum_.znear = -(pmat[14]*(G-1))/(2*G);
  frustum_.zfar = frustum_.znear*G;
  frustum_.left = frustum_.znear*(pmat[8]-1)/pmat[0];
  frustum_.right = frustum_.znear*(pmat[8]+1)/pmat[0];
  frustum_.bottom = frustum_.znear*(pmat[9]-1)/pmat[5];
  frustum_.top = frustum_.znear*(pmat[9]+1)/pmat[5];
  frustum_.width = frustum_.right - frustum_.left;
  frustum_.height = frustum_.top - frustum_.bottom;
}



void
OpenGL::setFrustumToWindowPortion()
{
  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  glFrustum(frustum_.left + frustum_.width / hi_res_.ncols * hi_res_.col,
	    frustum_.left + frustum_.width / hi_res_.ncols * (hi_res_.col+1),
	    frustum_.bottom + frustum_.height / hi_res_.nrows * hi_res_.row,
	    frustum_.bottom + frustum_.height / hi_res_.nrows *(hi_res_.row+1),
	    frustum_.znear, frustum_.zfar);
}



void
OpenGL::saveImage(const string& fname,
		  const string& type,
		  int x, int y)
{
  send_mailbox_.send(DO_IMAGE);
  img_mailbox_.send(ImgReq(fname,type,x,y));
}

void
OpenGL::getData(int datamask, FutureValue<GeometryData*>* result)
{
  send_mailbox_.send(DO_GETDATA);
  get_mailbox_.send(GetReq(datamask, result));
}

void
OpenGL::real_getData(int datamask, FutureValue<GeometryData*>* result)
{
  GeometryData* res = new GeometryData;
  if(datamask&GEOM_VIEW)
  {
    res->view=new View(cached_view_);
    res->xres=xres_;
    res->yres=yres_;
    res->znear=znear_;
    res->zfar=zfar_;
  }
  if(datamask&(GEOM_COLORBUFFER|GEOM_DEPTHBUFFER/*CollabVis*/|GEOM_MATRICES))
  {
    gui_->lock();
  }
  if(datamask&GEOM_COLORBUFFER)
  {
    ColorImage* img = res->colorbuffer = new ColorImage(xres_, yres_);
    float* data=new float[xres_*yres_*3];
    cerr << "xres_=" << xres_ << ", yres_=" << yres_ << "\n";
    WallClockTimer timer;
    timer.start();

    glReadPixels(0, 0, xres_, yres_, GL_RGB, GL_FLOAT, data);
    timer.stop();
    cerr << "done in " << timer.time() << " seconds\n";
    float* p=data;
    for(int y=0;y<yres_;y++)
    {
      for(int x=0;x<xres_;x++)
      {
	img->put_pixel(x, y, Color(p[0], p[1], p[2]));
	p+=3;
      }
    }
    delete[] data;
  }
  if(datamask&GEOM_DEPTHBUFFER)
  {
    //    DepthImage* img=res->depthbuffer=new DepthImage(xres_, yres_);
    unsigned int* data=new unsigned int[xres_*yres_*3];
    cerr << "reading depth...\n";
    WallClockTimer timer;
    timer.start();
    glReadPixels(0, 0,xres_,yres_, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, data);
    timer.stop();
    cerr << "done in " << timer.time() << " seconds\n";
  }

  if(datamask&(GEOM_COLORBUFFER|GEOM_DEPTHBUFFER/*CollabVis*/|GEOM_MATRICES))
  {
    GLenum errcode;
    while((errcode=glGetError()) != GL_NO_ERROR)
    {
      cerr << "We got an error from GL: " <<
	(char*)gluErrorString(errcode) << "\n";
    }
    gui_->unlock();
  }
  result->send(res);
}

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

#if defined(HAVE_PBUFFER)
  if( have_pbuffer_ && pbuffer_->is_valid() && pbuffer_->is_current() ){
    glXMakeCurrent( x11_dpy_, x11_win_, x11_gl_context_ ); 
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, width, 0.0, height, -10.0, 10.0);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glDrawBuffer(GL_BACK);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(width,height, GL_RGB, GL_UNSIGNED_BYTE,ptr);
    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glXSwapBuffers(x11_dpy_,x11_win_);
  }
#endif

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
  y = (yres_ - 1) - y;
  int index = x + (y * xres_);
  double z = depth_buffer_[index];
  if (p)
  {
    // Unproject the window point (x, y, z).
    GLdouble world_x, world_y, world_z;
    gluUnProject(x, y, z,
		 modelview_matrix_, projection_matrix_, viewport_matrix_,
		 &world_x, &world_y, &world_z);

    *p = Point(world_x, world_y, world_z);
  }

  // if z is close to 1, then assume no object was picked
  return (z < .999999);
}



bool
OpenGL::compute_depth(const View& view, double& znear, double& zfar)
{
  znear=MAXDOUBLE;
  zfar=-MAXDOUBLE;
  BBox bb;
  view_window_->get_bounds(bb);
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
    const double d = -Dot(eyep, dir);
    for (int i=0;i<8;i++)
    {
      Point p((i&1)?max.x():min.x(),
	      (i&2)?max.y():min.y(),
	      (i&4)?max.z():min.z());
      double dist=Dot(p, dir)+d;
      znear=Min(znear, dist);
      zfar=Max(zfar, dist);
    }
    znear *= 0.99;
    zfar  *= 1.01;

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


bool
OpenGL::compute_fog_depth(const View &view, double &znear, double &zfar,
			  bool visible_only)
{
  znear=MAXDOUBLE;
  zfar=-MAXDOUBLE;
  BBox bb;
  if (visible_only)
  {
    view_window_->get_bounds(bb);
  }
  else
  {
    view_window_->get_bounds_all(bb);
  }
  if(bb.valid())
  {
    // We have something to draw.
    Point eyep(view.eyep());
    Vector dir(view.lookat()-eyep);
    const double dirlen2 = dir.length2();
    if (dirlen2 < 1.0e-6 || dirlen2 != dirlen2)
      return false;
    dir.normalize();
    const double d = -Dot(eyep, dir);

    // Compute distance to center of bbox.
    double dist = Dot(bb.center(), dir);
    // Compute bbox view radius.
    double radius = bb.diagonal().length() * dir.length2() * 0.5;

    znear = d + dist - radius;
    zfar = d + dist + radius;

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


// i is the frame number, usually refers to left or right when do_stereo
// is set.

void
OpenGL::render_rotation_axis(const View &view,
			     bool do_stereo, int i, const Vector &eyesep)
{
  static GeomHandle axis_obj = 0;
  if (axis_obj.get_rep() == 0) axis_obj = view_window_->createGenAxes();

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
      if (!view_window_->gui_sr_.get())
	lookat-=eyesep;
    } else {
      eyep+=eyesep;
      if (!view_window_->gui_sr_.get())
	lookat+=eyesep;
    }
  }

  Vector up(view.up());
  gluLookAt(eyep.x(), eyep.y(), eyep.z(),
	    lookat.x(), lookat.y(), lookat.z(),
	    up.x(), up.y(), up.z());
  if(do_hi_res_)
  {
    // Draw in upper right hand corner of total image, not viewport image.
    const int xysize = Min(hi_res_.resx, hi_res_.resy) / 4;
    const int xoff = hi_res_.resx - hi_res_.col * viewport[2];
    const int yoff = hi_res_.resy - hi_res_.row * viewport[3];
    glViewport(xoff - xysize, yoff - xysize, xysize, xysize);
  }

  // Disable fog for the orientation axis.
  const bool fog = drawinfo_->fog;
  if (fog) { glDisable(GL_FOG); }
  drawinfo_->fog = false;

  // Set up Lighting
  glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
  const Lighting& l = viewer_->lighting_;
  int idx=0;
  int ii;
  for(ii=0;ii<l.lights.size();ii++)
  {
    LightHandle light=l.lights[ii];
    light->opengl_setup(view, drawinfo_, idx);
  }
  for(ii=0;ii<idx && ii<max_gl_lights_;ii++)
    glEnable((GLenum)(GL_LIGHT0+ii));
  for(;ii<max_gl_lights_;ii++)
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

  drawinfo_->viewwindow = view_window_;

  // Use depthrange to force the icon to move forward.
  // Ideally the rest of the scene should be drawn at 0.05 1.0,
  // so there was no overlap at all, but that would require
  // mucking about in the picking code.
  glDepthRange(0.0, 0.05);
  axis_obj->draw(drawinfo_, 0, current_time_);
  glDepthRange(0.0, 1.0);

  drawinfo_->fog = fog;  // Restore fog state.
  if (fog) { glEnable(GL_FOG); }

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
