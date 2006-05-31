

#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>
#include <sci_values.h>

#include <sci_defs/bits_defs.h>
#include <sci_defs/image_defs.h>

#include <Core/Events/OpenGLViewer.h>
#include <Core/Geom/Pbuffer.h>
#include <Core/Containers/StringUtil.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/Util/Environment.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/DrawInfoOpenGL.h>
#include <Core/Geom/GeomPick.h>
#include <Core/Geom/PointLight.h>
#include <Core/Math/Trig.h>
#include <Core/Geom/GeomScene.h>
#include <Core/Geom/GeomCone.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/GeomGroup.h>     
#include <Core/Geom/GeomSticky.h>
#include <Core/Geom/GeomRenderMode.h>
#include <Core/Geom/HeadLight.h>
#include <Core/Geom/DirectionalLight.h>


#if defined(HAVE_PNG) && HAVE_PNG
#  include <png.h>
#endif

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#ifdef _WIN32
#  include <windows.h>
#  include <winbase.h>
#  include <Core/Thread/Time.h>
#  undef near
#  undef far
#  undef min
#  undef max
#  define SCISHARE __declspec(dllimport)
#else
#  define SCISHARE
#  include <sys/time.h>
#endif


namespace SCIRun {

#define DO_REDRAW     0
#define DO_PICK       1
#define DO_GETDATA    2
#define REDRAW_DONE   4
#define PICK_DONE     5
#define DO_IMAGE      6
#define IMAGE_DONE    7

int CAPTURE_Z_DATA_HACK = 0;

static const int pick_buffer_size = 512;
static const double pick_window = 10.0;


OpenGLViewer::OpenGLViewer(OpenGLContext *oglc) :
  xres_(0),
  yres_(0),
  doing_image_p_(false),
  doing_movie_p_(false),
  make_MPEG_p_(false),
  current_movie_frame_(0),
  movie_name_("./movie.%04d"),
  gl_context_(oglc),
  myname_("Not Intialized"),
  // private member variables
  drawinfo_(scinew DrawInfoOpenGL),
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
  tm_("OpenGLViewer tool manager"),
  events_(EventManager::register_event_messages("OpenGLViewer")),
  pbuffer_(0),
  bgcolor_(Color(.0, .0, .0)),
  homeview_(Point(2.1, 1.6, 11.5), Point(.0, .0, .0), Vector(0,1,0), 20),
  view_(homeview_),
  ambient_scale_(1.0),	     
  diffuse_scale_(1.0),	     
  specular_scale_(0.4),	     
  shininess_scale_(1.0),	     
  emission_scale_(1.0),	     
  line_width_(1.0),	     
  point_size_(1.0),	     
  polygon_offset_factor_(1.0),
  polygon_offset_units_(0.0),
  eye_sep_base_(0.4),
  ortho_view_(false),
  fogusebg_(true),
  fogcolor_(Color(0.0,0.0,1.0)),
  fog_start_(0.0),
  fog_end_(0.714265),
  fog_visibleonly_(true),
  focus_sphere_(scinew GeomSphere),
  scene_graph_(0),
  visible_(),
  obj_tag_(),
  draw_type_(GOURAUD_E)
{
  fps_timer_.start();

  // Add a headlight
  lighting_.lights.add(scinew HeadLight("Headlight", Color(1,1,1)));
  for(int i = 1; i < 4; i++){ // only set up 3 more lights
    ostringstream str;
    str << "Light" << i;
    lighting_.lights.add(scinew DirectionalLight(str.str(), 
						 Vector(0,0,1), 
						 Color(1,1,1), false, false));
  }

  // 0 - Axes, visible
  internal_objs_.push_back(scinew GeomViewerItem(create_viewer_axes(),
						 "Axis",0));
  internal_objs_visible_p_.push_back(true);              

  // 1 - Unicam control sphere, not visible by default.
  MaterialHandle focus_color = scinew Material(Color(0.0, 0.0, 1.0));
  internal_objs_.push_back(scinew GeomMaterial(focus_sphere_, focus_color));
  internal_objs_visible_p_.push_back(false);

  default_material_ =
    scinew Material(Color(.1,.1,.1), Color(.6,0,0), Color(.7,.7,.7), 50);


  //rot = new ViewRotateTool();
  //tm_.add_tool(rot);

}


OpenGLViewer::~OpenGLViewer()
{
  // Finish up the mpeg that was in progress.
  if (encoding_mpeg_)
  {
    encoding_mpeg_ = false;
    EndMpeg();
  }
  fps_timer_.stop();

  delete drawinfo_;
  drawinfo_ = 0;

  if (gl_context_)
  {
    delete gl_context_;
    gl_context_ = 0;
  }

  if (pbuffer_)
  {
    pbuffer_->destroy();
    delete pbuffer_;
    pbuffer_ = 0;
  }
}

GeomHandle
OpenGLViewer::create_viewer_axes() 
{
  const Color black(0,0,0), grey(.5,.5,.5);
  MaterialHandle dk_red =   scinew Material(black, Color(.2,0,0), grey, 20);
  MaterialHandle dk_green = scinew Material(black, Color(0,.2,0), grey, 20);
  MaterialHandle dk_blue =  scinew Material(black, Color(0,0,.2), grey, 20);
  MaterialHandle lt_red =   scinew Material(black, Color(.8,0,0), grey, 20);
  MaterialHandle lt_green = scinew Material(black, Color(0,.8,0), grey, 20);
  MaterialHandle lt_blue =  scinew Material(black, Color(0,0,.8), grey, 20);

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

void
OpenGLViewer::redraw(double tbeg, double tend, int nframes, double framerate)
{
  if (dead_) return;
  animate_time_begin_ = tbeg;
  animate_time_end_ = tend;
  animate_num_frames_ = nframes;
  animate_framerate_ = framerate;
  //FIX_ME  make sure a redraw happens...
}

void
OpenGLViewer::run()
{
  TimeThrottle throttle;
  throttle.start();
  const double inc = 1. / 30; // the rate at which we refresh the monitor.
  double t = throttle.time();
#if 1 //SHOW_FRAME_RATE
  double tlast = t;
  int f = 0;
#endif
  while (!dead_) {
    t = throttle.time();
#if 1 //SHOW_FRAME_RATE
    f++;
    if (t - tlast > 1.0) {
      cerr << f << std::endl;
      f = 0;
      tlast = t;
    }
#endif
    throttle.wait_for_time(t + inc);
    // process the cached events since last draw.
    event_handle_t ev;
    while (events_ && events_->tryReceive(ev)) {
      // Tools will set up the appropriate rendering state.
      if (ev == 0) {
	// this is the terminate signal, so return.
	cerr << "Viewer exiting." << endl;
	dead_ = true;
	return;
      }

      tm_.propagate_event(ev);
    }

    if (do_hi_res_p()) {
      render_and_save_image(); // Image tool has the required parameters.
      do_hi_res_ = false;
    }
    //redraw(t, t+ inc
    redraw_frame();
    // replies should have been sent via events to the EventManager.
    //total_frames_+= total_frames_;

  } // end while(!dead_)
}


void
OpenGLViewer::render_and_save_image()
// FIX_ME these parameters come from the tool via the event.
// 				    int x, int y,
// 				    const string & fname, 
// 				    const string & ftype)
                               
{
#if 0
  bool use_convert = false;

#if defined(HAVE_PNG) && HAVE_PNG
  bool write_png = false;
  // Either the user specified the type to be ppm or raw (in that case
  // we create that type of image), or they specified the "by_extension"
  // type in which case we need to look at the extension and try to write
  // out a temporary png and then use convert if it is available to write
  // out the appropriate image.
  if (ftype != "ppm" && ftype != "raw")
    {
      // determine the extension
      string ext = fname.substr(fname.find(".", 0)+1, fname.length());

      // FIX ME convert ext to lower case
      for(unsigned int i=0; i<ext.size(); i++) {
	ext[i] = tolower(ext[i]);
      }

      if (ext != "png") {
	if (system("convert -version") != 0) {
          setMovieMessage(string("Error - Unsupported extension ") + ext + 
			  ". Program \"convert\" not found in the path", 
			  true);
	  return;
	} else {
	  use_convert = true;
	  write_png = true;
	}
      } else {
	write_png = true;
      }
    }
#endif

  cout << "Saving " + to_string(x) + "x" + to_string(y) +
    " image to '" + fname + "'.\n";

  gl_context_->raise();
  gl_context_->make_current();

  deriveFrustum();

  // Get Viewport dimensions
  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);

  hi_res_.resx = x;
  hi_res_.resy = y;
  // The EXACT # of screen rows and columns needed to render the image
  hi_res_.ncols = (double)hi_res_.resx/(double)vp[2];
  hi_res_.nrows = (double)hi_res_.resy/(double)vp[3];

  // The # of screen rows and columns that will be rendered to make the image
  const int nrows = (int)ceil(hi_res_.nrows);
  const int ncols = (int)ceil(hi_res_.ncols);

  ofstream *image_file = NULL;

#if defined(HAVE_PNG) && HAVE_PNG
  // Create the PNG struct.
  png_structp png;
  png_infop info;
  if (write_png)
  {
    png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (png == NULL) {
      setMovieMessage( "ERROR - Failed to create PNG write struct", true );
      return;
    }

    // Create the PNG info struct.
    info = png_create_info_struct(png);

    if (info == NULL) {
      setMovieMessage( "ERROR - Failed to create PNG info struct", true );
      png_destroy_write_struct(&png, NULL);
      return;
    }

    if (setjmp(png_jmpbuf(png))) {
      setMovieMessage( "ERROR - Initializing PNG.", true );
      png_destroy_write_struct(&png, &info);
      return;
    }
  }
#endif

  ASSERT(sci_getenv("SCIRUN_TMP_DIR"));

  const char * tmp_dir(sci_getenv("SCIRUN_TMP_DIR"));

  const string tmp_file = string (tmp_dir + string("/scirun_temp_png.png"));  


  int channel_bytes, num_channels;
  FILE *fp = NULL;

  if (ftype == "ppm" || ftype == "raw")
  {
    image_file = scinew ofstream(fname.c_str());
    if ( !image_file )
    {
      setMovieMessage( "Error Opening File: " + fname, true );
      return;
    }

    channel_bytes = 1;
    if (ftype != "raw")
      num_channels = 3;
    else
      num_channels = 4;

    if (ftype == "ppm")
    {
      (*image_file) << "P6" << std::endl;
      (*image_file) << hi_res_.resx << " " << hi_res_.resy << std::endl;
      (*image_file) << 255 << std::endl;
    }
    else if (ftype == "raw")
    {
      ofstream *nhdr_file = scinew ofstream((fname+string(".nhdr")).c_str());
      if ( !nhdr_file )
      {
	setMovieMessage( string("ERROR opening file: ") + fname + ".nhdr", true );
	return;
      }
      (*nhdr_file) << "NRRD0001" << std::endl;
      (*nhdr_file) << "type: unsigned char" << std::endl;
      (*nhdr_file) << "dimension: 3" << std::endl;
      (*nhdr_file) << "sizes: 4 "<<hi_res_.resx << " " << hi_res_.resy << std::endl;
      (*nhdr_file) << "encoding: raw" << std::endl;
      (*nhdr_file) << "data file: " << fname << std::endl;
      nhdr_file->close();
      delete nhdr_file;
    }
  }
  else
  {
#if defined(HAVE_PNG) && HAVE_PNG
    channel_bytes = 1;
    num_channels = 3;

    if (use_convert) {
      /* write out temporary png in /tmp with same root */
      // determine the extension      
      fp = fopen(tmp_file.c_str(), "wb");
    } else {
      fp = fopen(fname.c_str(), "wb");
    }

    /* initialize IO */
    png_init_io(png, fp);
    
    png_set_IHDR(png, info, hi_res_.resx, hi_res_.resy,
		 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
		 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    /* write header */
    png_write_info(png, info);      
#endif
  }

  const int pix_size = channel_bytes * num_channels;

  // Write out a screen height X image width chunk of pixels at a time
  unsigned char* pixels =
    scinew unsigned char[hi_res_.resx*vp[3]*pix_size];

  // Start writing image_file
  unsigned char* tmp_row = scinew unsigned char[hi_res_.resx*pix_size];

  for (hi_res_.row = nrows - 1; hi_res_.row >= 0; --hi_res_.row)
  {
    int read_height = hi_res_.resy - hi_res_.row * vp[3];
    read_height = (vp[3] < read_height) ? vp[3] : read_height;

    if (!pixels)
    {
      cerr << "No Memory! Aborting...\n";
      break;
    }

    for (hi_res_.col = 0; hi_res_.col < ncols; hi_res_.col++)
    {
      // Render the col and row in the hi_res struct.
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
#ifndef _WIN32
                   (num_channels == 3) ? GL_RGB : GL_BGRA,
#else
                   (num_channels == 3) ? GL_RGB : GL_RGBA,
#endif
                   (channel_bytes == 1) ? GL_UNSIGNED_BYTE : GL_UNSIGNED_SHORT,
                   pixels);
      gui_->unlock();
    }
    // OpenGL renders upside-down to image_file writing
    unsigned char *top_row, *bot_row;   
    int top, bot;
    for (top = read_height-1, bot = 0; bot < read_height/2; top--, bot++)
    {
      top_row = pixels + hi_res_.resx*top*pix_size;
      bot_row = pixels + hi_res_.resx*bot*pix_size;
      memcpy(tmp_row, top_row, hi_res_.resx*pix_size);
      memcpy(top_row, bot_row, hi_res_.resx*pix_size);
      memcpy(bot_row, tmp_row, hi_res_.resx*pix_size);
    }

#if defined(HAVE_PNG) && HAVE_PNG
    if (write_png)
    {
      // run loop to divide memory into "row" chunks
      png_bytep *rows = (png_bytep*)malloc(sizeof(png_bytep)*hi_res_.resy);
      for(int hi=0; hi < hi_res_.resy; hi++) {
	rows[hi] = &((png_bytep)pixels)[hi*hi_res_.resx*num_channels];
      }

      png_set_rows(png, info, rows);

      png_write_image(png, rows);
      free(rows);
    }
    else
#endif
    {
      image_file->write((char *)pixels, hi_res_.resx*read_height*pix_size);
    }
  }

  gui_->lock();

  // Set OpenGL back to nice PixelStore values for somebody else
  glPixelStorei(GL_PACK_SKIP_PIXELS,0);
  glPixelStorei(GL_PACK_ROW_LENGTH,0);

#if defined(HAVE_PNG) && HAVE_PNG
  if (write_png)
    {
      /* end write */
      if (setjmp(png_jmpbuf(png))) {
        setMovieMessage( "Error during end of PNG write", true );
	png_destroy_write_struct(&png, &info);
	return;
      }
      
      /* finish writing */
      png_write_end(png, NULL);
      
      /* more clean up */
      png_destroy_write_struct(&png, &info);
      fclose(fp);

      if (use_convert) {
	if (system(string("convert " + tmp_file + " " + fname).c_str()) != 0) {
	  cerr << "ERROR - Using convert to write image.\n";
	}
	system(string("rm " + tmp_file).c_str());
      }
  }
  else
#endif
  {
    image_file->close();
    delete image_file;
  }

  gui_->unlock();

  delete [] pixels;
  delete [] tmp_row;
#endif
  ASSERTFAIL("render_and_save_image unimplimented");
} // end render_and_save_image()


void
OpenGLViewer::redraw_frame()
{
  if (dead_) return;
  if (!gl_context_) return;

  // Make sure our GL context is current
  gl_context_->make_current();
  // Get the window size
  xres_ = gl_context_->width();
  yres_ = gl_context_->height();
  // Clear the screen.
  glViewport(0, 0, xres_, yres_);
  glClearColor(bgcolor().r(), bgcolor().g(), bgcolor().b(), 0);
  
  GLint data[1];
  glGetIntegerv(GL_MAX_LIGHTS, data);
  max_gl_lights_=data[0];
#ifdef __sgi
  // Look for multisample extension...
  if (strstr((char*)glGetString(GL_EXTENSIONS), "GL_SGIS_multisample"))
  {
    cerr << "Enabling multisampling...\n";
    glEnable(GL_MULTISAMPLE_SGIS);
    glSamplePatternSGIS(GL_1PASS_SGIS);
  }
#endif

  
  // Start polygon counter...
  WallClockTimer timer;
  timer.clear();
  timer.start();
  
  // Get a lock on the geometry database...
  //scene_graph_.readLock();

  // dump a frame if we are Saving an image, or Recording a movie.
  const bool dump_frame = doing_image_p_ || doing_movie_p_;

#if defined(HAVE_PBUFFER)
  // Set up a pbuffer associated with gl_context_ for image or movie making.
  if (dump_frame)
  {
    if (pbuffer_ &&
        (xres_ != pbuffer_->width() || yres_ != pbuffer_->height()))
    {
      pbuffer_->destroy();
      delete pbuffer_;
      pbuffer_ = 0;
    }

    if (!pbuffer_)
    {
      pbuffer_ = scinew Pbuffer(xres_, yres_, GL_INT, 8, false, GL_FALSE);
      if (!pbuffer_->create())
      {
        pbuffer_->destroy();
        delete pbuffer_;
        pbuffer_ = 0;
      }
    }
  }
#endif
  
  if (pbuffer_ && dump_frame)
  {
    pbuffer_->makeCurrent();
    glDrawBuffer(GL_FRONT);
  }
  else
  {
    gl_context_->make_current();
  }


  // Setup the view...
  cached_view_ = view_;
  const double aspect = double(xres_)/double(yres_);
  // XXX - UNICam change-- should be '1.0/aspect' not 'aspect' below.
  const double fovy = RtoD(2*Atan(1.0/aspect*Tan(DtoR(view_.fov()/2.))));

  drawinfo_->reset();

  if (do_stereo_p())
  {
    GLboolean supported;
    glGetBooleanv(GL_STEREO, &supported);
    if (!supported)
    {
      do_stereo_ = false;
      static bool warnonce = true;
      if (warnonce)
      {
        cout << "Stereo display selected but not supported.\n";
        warnonce = false;
      }
    }
  }

  drawinfo_->ambient_scale_         = ambient_scale_;	     
  drawinfo_->diffuse_scale_         = diffuse_scale_;	     
  drawinfo_->specular_scale_        = specular_scale_;	     
  drawinfo_->shininess_scale_       = shininess_scale_;	     
  drawinfo_->emission_scale_        = emission_scale_;	     
  drawinfo_->line_width_            = line_width_;	     
  drawinfo_->point_size_            = point_size_;	     
  drawinfo_->polygon_offset_factor_ = polygon_offset_factor_;
  drawinfo_->polygon_offset_units_  = polygon_offset_units_; 

  if (compute_depth(view_, znear_, zfar_))
  {
    // Set up graphics state.
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);
    glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
    set_state(drawinfo_);
    drawinfo_->pickmode_=0;

    //CHECK_OPENGL_ERROR("after setting up the graphics state: ");

    // Do the redraw loop for each time value.
    const double dt = (animate_time_end_ - animate_time_begin_)
      / animate_num_frames_;
    const double frametime = animate_framerate_==0?0:1.0/animate_framerate_;
    TimeThrottle throttle;
    throttle.start();
    Vector eyesep(0.0, 0.0, 0.0);
    if (do_stereo_p())
    {
      // gui_sr_ was always 1 as far as I could tell, no idea what it was for.
      // (gui_sr_.get() ? 0.048 : 0.0125);
      const double eye_sep_dist = eye_sep_base_ * 0.048;
      Vector u, v;
      view_.get_viewplane(aspect, 1.0, u, v);
      u.safe_normalize();
      const double zmid = (znear_ + zfar_) / 2.0;
      eyesep = u * eye_sep_dist * zmid;
    }

    for (int t = 0; t < 1 /*animate_num_frames_*/; t++)
    {
      int n = 1;
      if ( do_stereo_p() ) n = 2;
      for (int i = 0; i < n; i++)
      {
        if ( do_stereo_p() )
        {
          glDrawBuffer( (i == 0) ? GL_BACK_LEFT : GL_BACK_RIGHT);
        }
        else
        {
          if (pbuffer_ && dump_frame)
          {
            //ASSERT(pbuffer_->is_current());
            glDrawBuffer(GL_FRONT);
          }
          else
          {
            glDrawBuffer(GL_BACK);
          }
        }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0, 0, xres_, yres_);

        const double modeltime = t * dt + animate_time_begin_;
        //set_current_time(modeltime);
        
        // render normal
	// Setup view.
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	if (do_ortho_view_p())
	{
	  const double len = (view_.lookat() - view_.eyep()).length();
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
	Point eyep(view_.eyep());
	Point lookat(view_.lookat());

	if (do_stereo_p()) {
	  if (i == 0) {
	    eyep -= eyesep;
	  } else {
	    eyep += eyesep;
	  }
	}
	Vector up(view_.up());
	gluLookAt(eyep.x(), eyep.y(), eyep.z(),
		  lookat.x(), lookat.y(), lookat.z(),
		  up.x(), up.y(), up.z());
	if (do_hi_res_)
	{
	  setFrustumToWindowPortion();
	}
        
        
        // Set up Lighting
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
        const Lighting& lighting = lighting_;
        int idx=0;
        int ii;
        for (ii=0; ii<lighting.lights.size(); ii++)
        {
          LightHandle light = lighting.lights[ii];
          light->opengl_setup(view_, drawinfo_, idx);
        }
        for (ii=0; ii<idx && ii<max_gl_lights_; ii++)
        {
          glEnable((GLenum)(GL_LIGHT0 + ii));
        }
        for (;ii<max_gl_lights_;ii++)
        {
          glDisable((GLenum)(GL_LIGHT0 + ii));
        }

        // Now set up the fog stuff.
        double fognear, fogfar;
        compute_fog_depth(view_, fognear, fogfar, fog_visibleonly_p());
        glFogi(GL_FOG_MODE, GL_LINEAR);
        const float fnear =
          fognear + (fogfar - fognear) * fog_start_;
        glFogf(GL_FOG_START, fnear);
        const double ffar =
          fognear + (fogfar - fognear) /
          Max(fog_end_, 0.001);
        glFogf(GL_FOG_END, ffar);
        GLfloat bgArray[4];
        if (fogusebg_)
        {
          bgArray[0] = bgcolor().r();
          bgArray[1] = bgcolor().g();
          bgArray[2] = bgcolor().b();
        }
        else
        {
          Color fogcolor(fogcolor_);
          bgArray[0] = fogcolor.r();
          bgArray[1] = fogcolor.g();
          bgArray[2] = fogcolor.b();
        }       
        bgArray[3] = 1.0;
        glFogfv(GL_FOG_COLOR, bgArray);

        // FIX_ME clip tool and mouse_action in drawinfo
        //setClip(drawinfo_);
        //setMouse(drawinfo_);

        // UNICAM addition
        glGetDoublev (GL_MODELVIEW_MATRIX, modelview_matrix_);
        glGetDoublev (GL_PROJECTION_MATRIX, projection_matrix_);
        glGetIntegerv(GL_VIEWPORT, viewport_matrix_);

        // set up point size, line size, and polygon offset
        glPointSize(drawinfo_->point_size_);
        glLineWidth(drawinfo_->line_width_);
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
        if (drawinfo_->polygon_offset_factor_ ||
            drawinfo_->polygon_offset_units_)
        {
          glPolygonOffset(drawinfo_->polygon_offset_factor_,
                          drawinfo_->polygon_offset_units_);
          glEnable(GL_POLYGON_OFFSET_FILL);
        }
        else
        {
          glDisable(GL_POLYGON_OFFSET_FILL);
        }

        // Draw it all.
        current_time_ = modeltime;
        draw_visible_scene_graph();

        if (do_rotation_axis_p())
        {
          render_rotation_axis(view_, do_stereo_p(), i, eyesep);
        }
      }

      // Save z-buffer data.
      if (capture_z_data_)
      {
	depth_buffer_.resize(xres_ * yres_);
	capture_z_data_ = false;
	glReadPixels(0, 0, xres_, yres_, GL_DEPTH_COMPONENT, GL_FLOAT,
		     &depth_buffer_[0]);
      }
        
      // Wait for the right time before swapping buffers
      const double realtime = t * frametime;
      if (animate_num_frames_>1)
      {
        throttle.wait_for_time(realtime);
      }

      //total_frames_.set(total_frames_+1);

      // Show the pretty picture.
      if (!(pbuffer_ && dump_frame))
      {
        gl_context_->swap();
      }
    }
    throttle.stop();
    double fps;
    if (throttle.time() > 0)
    {
      fps = animate_num_frames_ / throttle.time();
    }
    else
    {
      fps = animate_num_frames_;
    }
    //set_current_time(animate_time_end_);
  }
  else
  {
    // Just show the cleared screen
    //set_current_time(animate_time_end_);
        
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (!(pbuffer_ && dump_frame))
    {
      gl_context_->swap();
    }
  }
  gl_context_->swap();
  gl_context_->release();

  //viewer_->geomlock_.readUnlock();

  // Look for errors.
  //CHECK_OPENGL_ERROR("OpenGLViewer::redraw after drawing objects: ");

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
  if (fps_timer_.time()>.33)
  {
    double fps = animate_num_frames_*frame_count_/fps_timer_.time();
    cerr.flush();
    frame_count_ = 1;
    // TODO: This looks incorrect, round and then convert to int?
    fps += 0.05;                // Round to nearest tenth
    const int fps_whole = (int)fps;
    const int fps_tenths = (int)((fps-fps_whole)*10);
    fps_timer_.clear();
    fps_timer_.start();         // Start it running for next time
    double pps;
    if (timer.time() > 0)
    {
      pps = drawinfo_->polycount_/timer.time();
    }
    else
    {
      pps = drawinfo_->polycount_;
    }
    // str << id_ << " updatePerf \"";
    str << drawinfo_->polycount_ << " polygons in " << timer.time()
        << " seconds\" \"" << pps
        << " polygons/second\"" << " \"" << fps_whole << "."
        << fps_tenths << " frames/sec\"" << '\0';
  }
  else if (fps_timer_.time() > 0)
  {
    frame_count_++;
    fps_timer_.start();
  }
  else
  {
    fps_timer_.start();
  }

  /*****************************************/
  /* movie-movie makin' movie-movie makin' */
  /*                                       */
  /* Only do this if we are making a movie */
  /* and we are dumping a frame.           */
  /*****************************************/
  if (dump_frame && doing_movie_p_)
  {
    if (make_MPEG_p_ )
    {
      if (!encoding_mpeg_)
      {
        string fname = movie_name_;

        // only add extension if not allready there
        if (!(fname.find(".mpg") != std::string::npos ||
              fname.find(".MPG") != std::string::npos ||
              fname.find(".mpeg") != std::string::npos ||
              fname.find(".MPEG") != std::string::npos))
        {
          fname = fname + string(".mpg");
        }
        
        // Dump the mpeg in the local dir ... ignoring any path since mpeg
        // can not handle it.
        //std::string::size_type pos = fname.find_last_of("/");
        //if ( pos != std::string::npos ) {
        //cerr << "Removing the mpeg path." << std::endl;
        //fname = fname.erase(0, pos+1);
        //}

        if ( fname.find("%") != std::string::npos )
        {
          string message = "Bad Format - Remove the Frame Format.";
          //setMovieMessage( message, true );
        }
        else
        {
          string message = "Dumping mpeg " + fname;
          //setMovieMessage( message );

          StartMpeg( fname );

          current_movie_frame_ = 0;
          encoding_mpeg_ = true;
        }
      }

      if (encoding_mpeg_)
      {
        const string message =
          "Adding Mpeg Frame " + to_string( current_movie_frame_ );

        //setMovieMessage( message );

        //setMovieFrame(current_movie_frame_);
        AddMpegFrame();

        current_movie_frame_++;
        //setMovieFrame(current_movie_frame_);
      }
    }
    else
    { // Dump each frame.
      if (encoding_mpeg_)
      {
        // Finish up mpeg that was in progress.
        encoding_mpeg_ = false;

        EndMpeg();
      }

      std::string::size_type pos = movie_name_.find_last_of("%0");

      if ( pos == std::string::npos ||
	   movie_name_[pos+2] != 'd' ||
	   movie_name_.find("%") != movie_name_.find_last_of("%") )
      {
        string message = "Bad Format - Illegal Frame Format.";
        //setMovieMessage( message, true );
      }
      else
      {
        char fname[256];
        sprintf(fname, movie_name_.c_str(), current_movie_frame_);

	ostringstream timestr;
#ifndef _WIN32
	timeval tv;
	gettimeofday(&tv, 0);
	timestr << "." << tv.tv_sec << ".";
	timestr.fill('0');
	timestr.width(6);
	timestr << tv.tv_usec;
#else
        long m_sec = Time::currentSeconds();
        timestr << "." << m_sec /1000 << ".";
	timestr.fill('0');
	timestr.width(3);
        timestr << m_sec % 1000;
#endif
        string fullpath = string(fname) + "." + movie_frame_extension_;
        
        string message = "Dumping " + fullpath;
        //setMovieMessage( message );
        dump_image(fullpath, movie_frame_extension_);

        current_movie_frame_++;
        //setMovieFrame(current_movie_frame_);
      }
    }
  }

  // End the mpeg if we are no longer recording a movie
  if (!doing_movie_p_)
  {
    if (encoding_mpeg_)
    { // Finish up mpeg that was in progress.
      encoding_mpeg_ = false;
      EndMpeg();
    }
  }
}


void
OpenGLViewer::get_pick(int x, int y,
		       GeomHandle& pick_obj, GeomPickHandle& pick_pick,
		       int& pick_index)
{
#if 0
  send_pick_x_ = x;
  send_pick_y_ = y;
  send_mailbox_.send(DO_PICK);
  for (;;)
  {
    const int r = recv_mailbox_.receive();
    if (r != PICK_DONE)
    {
      cerr << "WANTED A PICK!!! (got back " << r << "\n";
    }
    else
    {
      pick_obj = ret_pick_obj_;
      pick_pick = ret_pick_pick_;
      pick_index = ret_pick_index_;
      break;
    }
  }
#endif
}


void
OpenGLViewer::real_get_pick(int x, int y,
                      GeomHandle& pick_obj, GeomPickHandle& pick_pick,
                      int& pick_index)
{
#if 0
  pick_obj = 0;
  pick_pick = 0;
  pick_index = 0x12345678;
  // Make ourselves current

  // Make sure our GL context is current
  if ((tk_gl_context_ != old_tk_gl_context_))
  {
    old_tk_gl_context_ = tk_gl_context_;
    gui_->lock();
    tk_gl_context_->make_current();
    gui_->unlock();
  }

  // Setup the view...
  View view(view_);
  viewer_->geomlock_.readLock();

  // Compute znear and zfar.
  double znear;
  double zfar;
  if (compute_depth(view, znear, zfar))
  {
    // Setup picking.
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
    const double aspect = double(xres_)/double(yres_);
    // XXX - UNICam change-- should be '1.0/aspect' not 'aspect' below
    const double fovy = RtoD(2*Atan(1.0/aspect*Tan(DtoR(view.fov()/2.))));
    glViewport(0, 0, xres_, yres_);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    gluPickMatrix(x, viewport[3]-y, pick_window, pick_window, viewport);
    if (do_ortho_view_p())
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

    drawinfo_->lighting_=0;
    drawinfo_->set_drawtype(DrawInfoOpenGLViewer::Flat);
    drawinfo_->pickmode_=1;

    // Draw it all.
    do_for_visible(this, true);

#ifdef SCI_64BITS
    glPopName();
    glPopName();
    glPopName();
#else
    glPopName();
    glPopName();
#endif

    glFlush();
    const int hits = glRenderMode(GL_RENDER);

//    CHECK_OPENGL_ERROR("OpenGLViewer::real_get_pick");

    gui_->unlock();
    GLuint min_z;
#ifdef SCI_64BITS
    unsigned long hit_obj=0;
    //    GLuint hit_obj_index = 0x12345678;
    unsigned long hit_pick=0;
    //    GLuint hit_pick_index = 0x12345678;  // need for object indexing
#else
    GLuint hit_obj = 0;
    //GLuint hit_obj_index = 0x12345678;  // need for object indexing
    GLuint hit_pick = 0;
    //GLuint hit_pick_index = 0x12345678;  // need for object indexing
#endif
    if (hits >= 1)
    {
      int idx = 0;
      min_z = 0;
      bool have_one = false;
      for (int h=0; h<hits; h++)
      {
        int nnames = pick_buffer[idx++];
        GLuint z=pick_buffer[idx++];
        if (nnames > 1 && (!have_one || z < min_z))
        {
          min_z = z;
          have_one = true;
          idx++; // Skip Max Z
#ifdef SCI_64BITS
          idx += nnames - 5; // Skip to the last one.
          const unsigned int ho1 = pick_buffer[idx++];
          const unsigned int ho2 = pick_buffer[idx++];
          hit_pick = ((long)ho1<<32) | ho2;
          //hit_obj_index = pick_buffer[idx++];
          const unsigned int hp1 = pick_buffer[idx++];
          const unsigned int hp2 = pick_buffer[idx++];
          hit_obj = ((long)hp1<<32)|hp2;
          //hit_pick_index = pick_buffer[idx++];
          idx++;
#else
          // hit_obj=pick_buffer[idx++];
          // hit_obj_index=pick_buffer[idx++];
          //for (int i=idx; i<idx+nnames; ++i) cerr << pick_buffer[i] << "\n";
          idx += nnames - 3; // Skip to the last one.
          hit_pick = pick_buffer[idx++];
          hit_obj = pick_buffer[idx++];
          idx++;
          //hit_pick_index=pick_buffer[idx++];
#endif
        }
        else
        {
          idx += nnames + 1;
        }
      }

      pick_obj = (GeomObj*)hit_obj;
      pick_pick = (GeomPick*)hit_pick;
      pick_obj->getId(pick_index); //(int)hit_pick_index;
    }
  }
  viewer_->geomlock_.readUnlock();
#endif
}

void
OpenGLViewer::draw_visible_scene_graph()
{
  // Do internal objects first...
  unsigned int i;
  for (i = 0; i < internal_objs_.size(); i++){
    if (internal_objs_visible_p_[i] == 1) {
      if (do_picking_p()) {
	pick_draw_obj(default_material_, internal_objs_[i].get_rep());
      } else {
	redraw_obj(default_material_, internal_objs_[i].get_rep());
      }
    }
  }
  
  if (!scene_graph_) return;
  for (int pass=0; pass < 4; pass++)
  {
    GeomIndexedGroup::IterIntGeomObj iter = scene_graph_->getIter();
    for ( ; iter.first != iter.second; iter.first++) {
      GeomViewerItem *si = (GeomViewerItem*)((*iter.first).second.get_rep());
      // Look up the name to see if it should be drawn...
      if (item_visible_p(si))
      {
	const bool transparent =
	  strstr(si->getString().c_str(), "TransParent") ||
	  strstr(si->getString().c_str(), "Transparent");
	const bool culledtext = strstr(si->getString().c_str(), "Culled Text");
	const bool sticky = strstr(si->getString().c_str(), "Sticky");
	if ((pass == 0 && !transparent && !culledtext && !sticky) ||
	    (pass == 1 && transparent && !culledtext && !sticky) ||
	    (pass == 2 && culledtext && !sticky) ||
	    (pass == 3 && sticky))
	{
	  if(si->crowd_lock()){
	    si->crowd_lock()->readLock();
	  }
	    
	  if (do_picking_p()) {
	    pick_draw_obj(default_material_, si);
	  } else {
	    redraw_obj(default_material_, si);
	  }
	    
	  if(si->crowd_lock()) {
	    si->crowd_lock()->readUnlock();
	  }
	}
      } else {
	cerr << "Warning: Object " << si->getString() 
	     <<" not in visibility database." << endl;
      }
    }
  }
}

// Dump a ppm image.
void
OpenGLViewer::dump_image(const string& fname, const string& ftype)
{
  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);
  const int pix_size = 3;  // for RGB
  const int n = pix_size * vp[2] * vp[3];
  unsigned char* pixels = scinew unsigned char[n];
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadBuffer(GL_FRONT);
  glReadPixels(0, 0, vp[2], vp[3], GL_RGB, GL_UNSIGNED_BYTE, pixels);

  // TODO: This looks bogus. Copying pbuffer image to screen only works
  // if we aren't flushed anyway and the image size was the same as
  // the screen size.
  if (pbuffer_ && pbuffer_->is_current())
  {
    gl_context_->make_current();
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
    glDrawPixels(vp[2], vp[3], GL_RGB, GL_UNSIGNED_BYTE, pixels);
    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    gl_context_->swap();
  }

#if defined(HAVE_PNG) && HAVE_PNG
  if (ftype == "png")
  {
    // Create the PNG struct.
    png_structp png =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (png == NULL) {
      //setMovieMessage( "ERROR - Failed to create PNG write struct", true );
      return;
    }

    // Create the PNG info struct.
    png_infop info = png_create_info_struct(png);

    if (info == NULL) {
      //setMovieMessage( "ERROR - Failed to create PNG info struct", true );
      png_destroy_write_struct(&png, NULL);
      return;
    }

    if (setjmp(png_jmpbuf(png))) {
      //setMovieMessage( "ERROR - Initializing PNG.", true );
      png_destroy_write_struct(&png, &info);
      return;
    }

    // Initialize the PNG IO.
    FILE *fp = fopen(fname.c_str(), "wb");
    png_init_io(png, fp);
    
    png_set_IHDR(png, info, vp[2], vp[3],
		 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
		 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    // Write the PNG header.
    png_write_info(png, info);      

    // Run loop to divide memory into "row" chunks
    png_bytep *rows = (png_bytep*)malloc(sizeof(png_bytep) * vp[3]);
    for (int hi = 0; hi < vp[3]; hi++) {
      rows[hi] = &((png_bytep)pixels)[(vp[3] - hi - 1) * vp[2] * pix_size];
    }

    png_set_rows(png, info, rows);

    png_write_image(png, rows);

    /* end write */
    if (setjmp(png_jmpbuf(png))) {
      //setMovieMessage( "Error during end of PNG write", true );
      png_destroy_write_struct(&png, &info);
      return;
    }
      
    // Finish writing.
    png_write_end(png, NULL);
      
    // More clean up.
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    free(rows);
  }
  else
#endif
  {
    ofstream dumpfile(fname.c_str());
    if ( !dumpfile )
    {
      string errorMsg = "ERROR opening file: " + fname;
      //setMovieMessage( errorMsg, true );
      cerr << errorMsg << "\n";
      return;
    }

    // Print out the ppm  header.
    dumpfile << "P6" << std::endl;
    dumpfile << vp[2] << " " << vp[3] << std::endl;
    dumpfile << 255 << std::endl;

    // OpenGL renders upside-down to ppm_file writing.
    unsigned char *top_row, *bot_row;     
    unsigned char *tmp_row = scinew unsigned char[ vp[2] * pix_size];
    int top, bot;
    for ( top = vp[3] - 1, bot = 0; bot < vp[3]/2; top --, bot++){
      top_row = pixels + vp[2] * top * pix_size;
      bot_row = pixels + vp[2] * bot * pix_size;
      memcpy(tmp_row, top_row, vp[2] * pix_size);
      memcpy(top_row, bot_row, vp[2] * pix_size);
      memcpy(bot_row, tmp_row, vp[2] * pix_size);
    }
    // Now dump the file.
    dumpfile.write((const char *)pixels, n);
    delete [] tmp_row;
  }

  delete [] pixels;
}



void
OpenGLViewer::put_scanline(int y, int width, Color* scanline, int repeat)
{
  float* pixels = scinew float[width*3];
  float* p = pixels;
  int i;
  for (i=0; i<width; i++)
  {
    *p++ = scanline[i].r();
    *p++ = scanline[i].g();
    *p++ = scanline[i].b();
  }
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glTranslated(-1.0, -1.0, 0.0);
  glScaled(2.0 / xres_, 2.0 / yres_, 1.0);
  glDepthFunc(GL_ALWAYS);
  glDrawBuffer(GL_FRONT);
  for (i=0; i<repeat; i++)
  {
    glRasterPos2i(0, y + i);
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
OpenGLViewer::pick_draw_obj(MaterialHandle def, GeomHandle obj)
{
#ifdef SCI_64BITS
  unsigned long o = (unsigned long)(obj.get_rep());
  unsigned int o1 = (o>>32)&0xffffffff;
  unsigned int o2 = o&0xffffffff;
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
  obj->draw(drawinfo_, def.get_rep(), current_time_);
}



void
OpenGLViewer::redraw_obj(MaterialHandle def, GeomHandle obj)
{
  GeomViewerItem *gvi = dynamic_cast<GeomViewerItem *>(obj.get_rep());
  ASSERT(gvi);
  string name = gvi->getString();
  map<string,int>::iterator tag_iter = obj_tag_.find(name);
  if (tag_iter != obj_tag_.end()) {
    // if found
    set_state(drawinfo_);
  }
  obj->draw(drawinfo_, def.get_rep(), current_time_);
}

void
OpenGLViewer::deriveFrustum()
{
  double pmat[16];
  glGetDoublev(GL_PROJECTION_MATRIX, pmat);
  const double G = (pmat[10]-1)/(pmat[10]+1);
  frustum_.znear = -(pmat[14]*(G-1))/(2*G);
  frustum_.zfar = frustum_.znear*G;

  if (do_ortho_view_p())
  {
    frustum_.left = (pmat[8]-1)/pmat[0];
    frustum_.right = (pmat[8]+1)/pmat[0];
    frustum_.bottom = (pmat[9]-1)/pmat[5];
    frustum_.top = (pmat[9]+1)/pmat[5];
    frustum_.width = frustum_.right - frustum_.left;
    frustum_.height = frustum_.top - frustum_.bottom;
  }
  else
  {
    frustum_.left = frustum_.znear*(pmat[8]-1)/pmat[0];
    frustum_.right = frustum_.znear*(pmat[8]+1)/pmat[0];
    frustum_.bottom = frustum_.znear*(pmat[9]-1)/pmat[5];
    frustum_.top = frustum_.znear*(pmat[9]+1)/pmat[5];
    frustum_.width = frustum_.right - frustum_.left;
    frustum_.height = frustum_.top - frustum_.bottom;
  }
}



void
OpenGLViewer::setFrustumToWindowPortion()
{
  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  if (do_ortho_view_p())
  {
    glOrtho(frustum_.left + frustum_.width / hi_res_.ncols * hi_res_.col,
            frustum_.left + frustum_.width / hi_res_.ncols * (hi_res_.col+1),
            frustum_.bottom + frustum_.height / hi_res_.nrows * hi_res_.row,
            frustum_.bottom + frustum_.height / hi_res_.nrows *(hi_res_.row+1),
            znear_, zfar_);
  }
  else
  {
    glFrustum(frustum_.left + frustum_.width / hi_res_.ncols * hi_res_.col,
              frustum_.left + frustum_.width / hi_res_.ncols * (hi_res_.col+1),
              frustum_.bottom + frustum_.height / hi_res_.nrows* hi_res_.row,
              frustum_.bottom + frustum_.height /hi_res_.nrows*(hi_res_.row+1),
              frustum_.znear, frustum_.zfar);
  }
}


// save image tool..
#if (0)
void
OpenGLViewer::save_image(const string& fname,
			 const string& type,
			 int x, int y)
{
  send_mailbox_.send(DO_IMAGE);
  img_mailbox_.send(ImgReq(fname,type,x,y));
}

void
OpenGLViewer::scheduleSyncFrame()
{
  send_mailbox_.send(DO_SYNC_FRAME);
}

void
OpenGLViewer::getData(int datamask, FutureValue<GeometryData*>* result)
{
  send_mailbox_.send(DO_GETDATA);
  get_mailbox_.send(GetReq(datamask, result));
}


void
OpenGLViewer::real_getData(int datamask, FutureValue<GeometryData*>* result)
{
  GeometryData* res = new GeometryData;
  if (datamask&GEOM_VIEW)
  {
    res->view=new View(cached_view_);
    res->xres=xres_;
    res->yres=yres_;
    res->znear=znear_;
    res->zfar=zfar_;
  }
  if (datamask&(GEOM_COLORBUFFER|GEOM_DEPTHBUFFER/*CollabVis*/|GEOM_MATRICES))
  {
    gui_->lock();
  }
  if (datamask&GEOM_COLORBUFFER)
  {
    ColorImage* img = res->colorbuffer = new ColorImage(xres_, yres_);
    float* data=new float[xres_*yres_*3];
    WallClockTimer timer;
    timer.start();

    glReadPixels(0, 0, xres_, yres_, GL_RGB, GL_FLOAT, data);
    timer.stop();
    float* p = data;
    for (int y=0; y<yres_; y++)
    {
      for (int x=0; x<xres_; x++)
      {
        img->put_pixel(x, y, Color(p[0], p[1], p[2]));
        p += 3;
      }
    }
    delete[] data;
  }
  if (datamask&GEOM_DEPTHBUFFER)
  {
    unsigned int* data=new unsigned int[xres_*yres_*3];
    WallClockTimer timer;
    timer.start();
    glReadPixels(0, 0,xres_,yres_, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, data);
    timer.stop();
  }

  if (datamask&(GEOM_COLORBUFFER|GEOM_DEPTHBUFFER/*CollabVis*/|GEOM_MATRICES))
  {
//    CHECK_OPENGL_ERROR("OpenGLViewer::real_getData");
    gui_->unlock();
  }

  if (datamask&(GEOM_VIEW_BOUNDS))
  {
    get_bounds_all(res->view_bounds_);
  }
  result->send(res);
}
#endif

void
OpenGLViewer::StartMpeg(const string& fname)
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
  char *pattern = scinew char[4];
  pattern = "II\0";
  mpeg_options_.frame_pattern = pattern;
  mpeg_options_.search_range[1]=0;
  mpeg_options_.gop_size=1;
  mpeg_options_.IQscale=1;
  mpeg_options_.PQscale=1;
  mpeg_options_.BQscale=1;
  mpeg_options_.pixel_search=MPEGe_options::FULL;
  if ( !MPEGe_open(mpeg_file_, &mpeg_options_ ) )
  {
    cerr << "MPEGe library initialisation failure!:" <<
      mpeg_options_.error << "\n";
    return;
  }
#endif // HAVE_MPEG
}


void
OpenGLViewer::AddMpegFrame()
{
#ifdef HAVE_MPEG
  // Looks like we'll blow up if you try to make more than one movie
  // at a time, as the memory per frame is static.
  static ImVfb *image=NULL; /* Only alloc memory for these once. */
  int width, height;
  ImVfbPtr ptr;

  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);

  width = vp[2];
  height = vp[3];

  // Set up the ImVfb used to store the image.
  if ( !image )
  {
    image=MPEGe_ImVfbAlloc( width, height, IMVFBRGB, true );
    if ( !image )
    {
      cerr<<"Couldn't allocate memory for frame buffer\n";
      exit(2);
    }
  }

  // Get to the first pixel in the image.
  ptr = ImVfbQPtr( image, 0, 0 );
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadBuffer(GL_FRONT);
  glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, ptr);

  // TODO: This looks bogus. Copying pbuffer image to screen only works
  // if we aren't flushed anyway and the image size was the same as
  // the screen size.  Maybe these are always true for movie making?
  if (pbuffer_ && pbuffer_->is_current())
  {
    gl_context_->make_current();
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
    gl_context_->swap();
  }

  const int r = 3 * width;
  unsigned char* row = scinew unsigned char[r];
  unsigned char* p0, *p1;

  int k, j;
  for (k = height -1, j = 0; j < height/2; k--, j++)
  {
    p0 = ptr + r * j;
    p1 = ptr + r * k;
    memcpy( row, p0, r);
    memcpy( p0, p1, r);
    memcpy( p1, row, r);
  }
  delete[] row;

  if ( !MPEGe_image(image, &mpeg_options_) )
  {
    //setMovieMessage( string("ERROR creating MPEG frame: ") + mpeg_options_.error, true );
  }
#endif // HAVE_MPEG
}



void
OpenGLViewer::EndMpeg()
{
#ifdef HAVE_MPEG
  if ( !MPEGe_close(&mpeg_options_) )
  {
    string errorMsg = string("ERROR closing MPEG file: ") + mpeg_options_.error;
    //setMovieMessage( errorMsg, true );
  }
  else
  {
    string message = "Ending Mpeg.";
    //setMovieMessage( message );
  }

  //setMovieStopped();
#endif // HAVE_MPEG
}



// Return world-space depth to point under pixel (x, y).
bool
OpenGLViewer::pick_scene( int x, int y, Point *p )
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
OpenGLViewer::item_visible_p(GeomViewerItem* si)
{
  map<string, bool>::iterator viter = visible_.find(si->getString());
  if (viter != visible_.end()) { return (*viter).second; }
  return false;
}

void
OpenGLViewer::get_bounds(BBox &bbox, bool check_visible)
{
  bbox.reset();
  if (scene_graph_) {
    GeomIndexedGroup::IterIntGeomObj iter = scene_graph_->getIter();
    for ( ; iter.first != iter.second; iter.first++) {
      GeomViewerItem *si = (GeomViewerItem*)((*iter.first).second.get_rep());
      // Look up the name to see if it should be drawn...
      if (!check_visible || item_visible_p(si)) {

	if(si->crowd_lock()) si->crowd_lock()->readLock();
	si->get_bounds(bbox);
	if(si->crowd_lock()) si->crowd_lock()->readUnlock();

      } else {
	cerr << "Warning: object " << si->getString()
	     << " not in visibility database." << endl;
	si->get_bounds(bbox);
      }
    }
  }
  const unsigned int objs_size = internal_objs_.size();
  for(unsigned int i = 0; i < objs_size; i++) {
    if (!check_visible || internal_objs_visible_p_[i])
      internal_objs_[i]->get_bounds(bbox);
  }

  // If the bounding box is empty, make it default to sane view.
  if (! bbox.valid()) {
    bbox.extend(Point(-1.0, -1.0, -1.0));
    bbox.extend(Point(1.0, 1.0, 1.0));
  }
}

bool
OpenGLViewer::compute_depth(const View& view, double& znear, double& zfar)
{
  znear = DBL_MAX;
  zfar =- DBL_MAX;
  BBox bb;
  get_bounds(bb);
  if (bb.valid())
  {
    // We have something to draw.
    Point min(bb.min());
    Point max(bb.max());
    Point eyep(view.eyep());
    Vector dir(view.lookat()-eyep);
    const double dirlen2 = dir.length2();
    if (dirlen2 < 1.0e-6 || dirlen2 != dirlen2)
    {
      return false;
    }
    dir.safe_normalize();
    const double d = -Dot(eyep, dir);
    for (int i=0;i<8;i++)
    {
      const Point p((i&1)?max.x():min.x(),
                    (i&2)?max.y():min.y(),
                    (i&4)?max.z():min.z());
      const double dist = Dot(p, dir) + d;
      znear = Min(znear, dist);
      zfar = Max(zfar, dist);
    }
    znear *= 0.99;
    zfar  *= 1.01;

    if (znear <= 0.0)
    {
      if (zfar <= 0.0)
      {
        // Everything is behind us - it doesn't matter what we do.
        znear = 1.0;
        zfar = 2.0;
      }
      else
      {
        znear = zfar * 0.001;
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
OpenGLViewer::compute_fog_depth(const View &view, double &znear, double &zfar,
                          bool visible_only)
{
  znear = DBL_MAX;
  zfar = -DBL_MAX;
  BBox bb;
  if (visible_only)
  {
    get_bounds(bb);
  }
  else
  {
    get_bounds_all(bb);
  }
  if (bb.valid())
  {
    // We have something to draw.
    Point eyep(view.eyep());
    Vector dir(view.lookat()-eyep);
    const double dirlen2 = dir.length2();
    if (dirlen2 < 1.0e-6 || dirlen2 != dirlen2)
    {
      return false;
    }
    dir.safe_normalize();
    const double d = -Dot(eyep, dir);

    // Compute distance to center of bbox.
    const double dist = Dot(bb.center(), dir);
    // Compute bbox view radius.
    const double radius = bb.diagonal().length() * dir.length2() * 0.5;

    znear = d + dist - radius;
    zfar = d + dist + radius;

    return true;
  }
  else
  {
    return false;
  }
}


// i is the frame number, usually refers to left or right when do_stereo_p()
// is set.
void
OpenGLViewer::render_rotation_axis(const View &view,
				   bool do_stereo, int i, 
				   const Vector &eyesep)
{
  static GeomHandle axis_obj = 0;
  if (axis_obj.get_rep() == 0) axis_obj = create_viewer_axes();

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
  oldeye.safe_normalize();
  Point eyep((oldeye * eyedist).asPoint());
  Point lookat(0.0, 0.0, 0.0);

  if (do_stereo_p()) {
    if (i == 0) {
      eyep -= eyesep;
    } else {
      eyep += eyesep;
    }
  }

  Vector up(view.up());
  gluLookAt(eyep.x(), eyep.y(), eyep.z(),
            lookat.x(), lookat.y(), lookat.z(),
            up.x(), up.y(), up.z());
  if (do_hi_res_)
  {
    // Draw in upper right hand corner of total image, not viewport image.
    const int xysize = Min(hi_res_.resx, hi_res_.resy) / 4;
    const int xoff = hi_res_.resx - hi_res_.col * viewport[2];
    const int yoff = hi_res_.resy - hi_res_.row * viewport[3];
    glViewport(xoff - xysize, yoff - xysize, xysize, xysize);
  }

  set_state(drawinfo_);

  // Disable fog for the orientation axis.
  const bool fog = drawinfo_->fog_;
  if (fog) { glDisable(GL_FOG); }
  drawinfo_->fog_ = false;

  // Set up Lighting
  glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
  const Lighting& l = lighting_;
  int idx=0;
  int ii;
  for (ii=0;ii<l.lights.size();ii++)
  {
    LightHandle light=l.lights[ii];
    light->opengl_setup(view, drawinfo_, idx);
  }
  for (ii=0;ii<idx && ii<max_gl_lights_;ii++)
    glEnable((GLenum)(GL_LIGHT0+ii));
  for (;ii<max_gl_lights_;ii++)
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

  // Use depthrange to force the icon to move forward.
  // Ideally the rest of the scene should be drawn at 0.05 1.0,
  // so there was no overlap at all, but that would require
  // mucking about in the picking code.
  glDepthRange(0.0, 0.05);
  axis_obj->draw(drawinfo_, 0, current_time_);
  glDepthRange(0.0, 1.0);

  drawinfo_->fog_ = fog;  // Restore fog state.
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


void
OpenGLViewer::set_state(DrawInfoOpenGL* drawinfo)
{
  switch (draw_type_) {
  case WIRE_E :
    {
      drawinfo->set_drawtype(DrawInfoOpenGL::WireFrame);
      drawinfo->lighting_=0;
    }
    break;
  case FLAT_E :
    {
      drawinfo->set_drawtype(DrawInfoOpenGL::Flat);
      drawinfo->lighting_=0;
    }
    break;
  case DEFAULT_E  :
  case GOURAUD_E :
  default:
    {
      drawinfo->set_drawtype(DrawInfoOpenGL::Gouraud);
      drawinfo->lighting_=1;
    }
  };

  // Now see if they want a bounding box.
  drawinfo->show_bbox_ = do_bbox_p();

#if 0 // FIX_ME make a movie tool
  if (!doing_movie_p())
  {
    doing_movie_p_ = 0;
    make_MPEG_p_ = 0;
  } else {
    current_movie_frame_ = movie_tool_->frame();
    
    doing_movie_p_ = 1;
    if (movie == 1)
    {
      renderer_->make_MPEG_p_ = 0;
      renderer_->movie_frame_extension_ = "ppm";
    }
    if (movie == 3)
    {
      renderer_->make_MPEG_p_ = 0;
      renderer_->movie_frame_extension_ = "png";
    }
    else if (movie == 2)
    {
      renderer_->make_MPEG_p_ = 1;
    }
  }
#endif

#if 0 //FIX_ME make clipping tool.
  if (clip.valid())
  {
    drawinfo->check_clip_ = clip;
  }
  setClip(drawinfo);
#endif

  drawinfo->cull_            = do_backface_cull_p();
  drawinfo->display_list_p_  = do_display_list_p();
  drawinfo->fog_             = do_fog_p();     
  drawinfo->lighting_        = do_lighting_p();
  drawinfo->currently_lit_   = drawinfo->lighting_;
  drawinfo->init_lighting(drawinfo->lighting_);
}

} // End namespace SCIRun
