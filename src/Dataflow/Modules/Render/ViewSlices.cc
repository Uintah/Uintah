
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
 *  ViewSlices.cc
 *
 *  Written by:
 *   McKay Davis
 *   School of Computing
 *   University of Utah
 *   September, 2004
 *
 *  Copyright (C) 2004 SCI Group
 */

#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>

#include <tcl.h>
#include <tk.h>
#include <stdlib.h>

#include <Core/Containers/Array3.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Runnable.h>
#include <Core/Util/Timer.h>
#include <Core/Datatypes/Field.h>
#include <Core/GuiInterface/UIvar.h>
#include <Core/Geom/Material.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>

#include <Core/Geom/GeomSwitch.h>
#include <Core/Geom/GeomCull.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/TexSquare.h>
#include <Core/Algorithms/Visualization/RenderField.h>

#include <Dataflow/Ports/NrrdPort.h>
#include <Dataflow/Ports/Colormap2Port.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>

#include <Core/Geom/OpenGLContext.h>
#include <Core/Geom/OpenGLViewport.h>
#include <Core/Geom/FreeType.h>
#include <Core/Volume/CM2Widget.h>

#include <Core/Util/Environment.h>
#include <Core/Thread/Mutex.h>

#include <typeinfo>
#include <iostream>
#include <map>

extern Tcl_Interp* the_interp;


namespace SCIRun {


class RealDrawer;

// SWAP ----------------------------------------------------------------
template <class T>
inline void SWAP(T& a, T& b) {
  T temp;
  temp = a;
  a = b;
  b = temp;
}


class ViewSlices : public Module
{
  enum {
    SHIFT_E	= 1,
    CAPS_LOCK_E = 2,
    CONTROL_E	= 4,
    ALT_E	= 8,
    M1_E	= 16,
    M2_E	= 32,
    M3_E	= 64,
    M4_E	= 128,
    BUTTON_1_E	= 256,
    BUTTON_2_E  = 512,
    BUTTON_3_E  = 1024
  };

  enum DisplayMode_e {
    normal_e,
    slab_e,
    mip_e,
    num_display_modes_e
  };

  enum ToolMode_e {
    slice_e,
    crop_e,
    translate_e,
    zoom_e,
    clut_e,
    num_tool_modes_e
  };
  

  struct NrrdVolume { 
    NrrdVolume		(GuiContext*ctx);
    void		reset();
    NrrdDataHandle	nrrd_;
    UIdouble		opacity_;
    UIint		invert_;
    UIint		flip_x_;
    UIint		flip_y_;
    UIint		flip_z_;
    UIint		transpose_yz_;
    UIint		transpose_xz_;
    UIint		transpose_xy_;
  };

  struct NrrdSlice {
    NrrdSlice(int axis=0, int slice=0, NrrdVolume *volume=0);
    int			axis_;  // which axis
    int			slice_num_;   // which slice # along axis
    int			slab_min_;
    int			slab_max_;
    NrrdDataHandle      nrrd_;
    
    bool		nrrd_dirty_;
    bool		tex_dirty_;
    bool		geom_dirty_;
    unsigned int	mode_;
    unsigned int	tex_wid_;
    unsigned int	tex_hei_;

    unsigned int	wid_;
    unsigned int	hei_;

    float		opacity_;
    float		tex_coords_[8];  // s,t * 4 corners
    float		pos_coords_[12]; // x,y,z * 4 corners
    GLuint		tex_name_;

    NrrdVolume *	volume_;
    //    Mutex		lock_;
    //    Thread *		owner_;
    //    int			lock_count_;
    void		do_lock();
    void		do_unlock();
  };
  typedef vector<NrrdSlice *>		NrrdSlices;
  typedef vector<NrrdSlices>		NrrdVolumeSlices;



  struct SliceWindow { 
    SliceWindow() { ASSERT(0); };
    SliceWindow(GuiContext *ctx);
    void		reset();

    string		name_;

    OpenGLViewport *	viewport_;
    NrrdSlices		slices_;

    NrrdSlices		paint_slices_;
    NrrdSlices		gradient_slices_;
    NrrdSlices		new_paint_slices_;

    UIint		slice_num_;
    UIint		axis_;
    UIdouble		zoom_;
    UIint		slab_min_;
    UIint		slab_max_;
      
    UIdouble		x_;
    UIdouble		y_;

    bool		redraw_;
    bool		clut_dirty_;
    UIint		clut_ww_;
    UIint		clut_wl_;
      
    UIint		auto_levels_;
    UIint		mode_;
    UIint		crosshairs_;
    UIint		snoop_;
    UIint		invert_;
    UIint		reverse_;
      
    UIint		mouse_x_;
    UIint		mouse_y_;
    UIint		show_guidelines_;
    bool		mouse_in_window_;

    UIdouble		fusion_;
    
    int			colormap_generation_;
    int			colormap_size_;
    NrrdDataHandle	colormap_nrrd_;

    int			cursor_;

    GLdouble		gl_modelview_matrix_[16];
    GLdouble		gl_projection_matrix_[16];
    GLint		gl_viewport_[4];
  };
  typedef vector<SliceWindow *>	SliceWindows;

  struct WindowLayout {
    WindowLayout	(GuiContext *ctx);
    //    void		reset();
    TkOpenGLContext *	opengl_;
    int			mouse_x_;
    int			mouse_y_;
    SliceWindows	windows_;
    string		name_;
  };

  struct TextLabel {
    string		name_;
    SliceWindows	windows_;
    FreeTypeText	text_;
    std::map<SliceWindow *, GLuint> tex_name_;
  };

  typedef vector<TextLabel *>		Labels;
  typedef vector<FreeTypeFace *>	FreeTypeFaces;

  typedef vector<NrrdVolume *>		NrrdVolumes;
  typedef map<string, WindowLayout *>	WindowLayouts;

  typedef vector<BBox>			PickBoxes;

  WindowLayouts		layouts_;
  NrrdVolumes		volumes_;
  NrrdDataHandle	gradient_;
  ColorMap2Handle	cm2_;


  ColorMapHandle	colormap_;

  SliceWindow *		current_window_;

  Point			cursor_;

  NrrdSlice *		mip_slices_[3];

  SliceWindow *		window_level_;
  int			window_level_ww_;
  int			window_level_wl_;

  int			pick_;
  int			pick_x_;
  int			pick_y_;
  BBox			crop_bbox_;
  BBox			crop_draw_bbox_;
  PickBoxes		crop_pick_boxes_;

  int			max_slice_[3];
  int			cur_slice_[3];
  int			slab_width_[3];
  double		scale_[3];
  double		center_[3];
  UIint			probe_;
  UIint			painting_;
  UIint			crop_;
  

  UIint			crop_min_x_;
  UIint			crop_min_y_;
  UIint			crop_min_z_;

  UIint			crop_max_x_;
  UIint			crop_max_y_;
  UIint			crop_max_z_;

  UIint			crop_min_pad_x_;
  UIint			crop_min_pad_y_;
  UIint			crop_min_pad_z_;

  UIint			crop_max_pad_x_;
  UIint			crop_max_pad_y_;
  UIint			crop_max_pad_z_;

  UIint			texture_filter_;

  UIdouble		min_;
  UIdouble		max_;

  float *		temp_tex_data_;

  //! output port
  GeometryOPort *	ogeom_;
  map<string,TexSquare*>    tobjs_;
  map<string,GeomID>    gobjs_;

  FreeTypeLibrary *	freetype_lib_;
  map<string, FreeTypeFace *>		fonts_;
  Labels		labels_;
  
  RealDrawer *		runner_;
  Thread *		runner_thread_;

  void			redraw_all();
  void			redraw_window_layout(WindowLayout &);
  void			redraw_window(SliceWindow &);
  void			draw_slice(SliceWindow &, NrrdSlice &);

  void			draw_all_labels(SliceWindow &);
  void			draw_anatomical_labels(SliceWindow &);
  void			draw_position_label(SliceWindow &);
  void			draw_label(SliceWindow &, string, int, int, 
				   FreeTypeText::anchor_e, 
				   FreeTypeFace *font = 0);

  
  float *		apply_colormap(NrrdSlice &, double min, double max, 
				       float *);

  void			draw_guide_lines(SliceWindow &, float, float, float, 
					bool cursor=true);
  void			draw_slice_lines(SliceWindow &);

  // Crop Widget routines
  void			draw_crop_bbox(SliceWindow &, BBox &);
  PickBoxes		compute_crop_pick_boxes(SliceWindow &, BBox &);
  int			mouse_in_pick_boxes(SliceWindow &, PickBoxes &);
  void			set_window_cursor(SliceWindow &, int pick);
  pair<Vector, Vector>	get_crop_vectors(SliceWindow &, int pick);
  BBox			update_crop_bbox(SliceWindow &, int pick, int X, int Y);
  void			update_crop_bbox_from_gui();
  void			update_crop_bbox_to_gui();

  int			x_axis(SliceWindow &);
  int			y_axis(SliceWindow &);

  void			setup_gl_view(SliceWindow &);

  void			set_slice_coords(NrrdSlice &slice, bool origin=true);


  void			extract_window_slices(SliceWindow &);
  void			extract_slice(NrrdSlice &,int axis, int slice_num);
  void			extract_mip_slices(NrrdVolume *);
  void			send_mip_slices(SliceWindow &);
  void			send_all_geometry();

  void			set_axis(SliceWindow &, unsigned int axis);
  void			next_slice(SliceWindow &);
  void			prev_slice(SliceWindow &);
  void			zoom_in(SliceWindow &);
  void			zoom_out(SliceWindow &);

  Point			screen_to_world(SliceWindow &,
					unsigned int x, unsigned int y);

  Point			world_to_screen(SliceWindow &, Point &);

  unsigned int		pow2(const unsigned int) const;
  unsigned int		log2(const unsigned int) const;
  bool			mouse_in_window(SliceWindow &);

  void			handle_gui_motion(GuiArgs &args);
  void			handle_gui_button(GuiArgs &args);
  void			handle_gui_button_release(GuiArgs &args);
  void			handle_gui_keypress(GuiArgs &args);
  void			handle_gui_enter(GuiArgs &args);
  void			handle_gui_leave(GuiArgs &args);

  void			debug_print_state(int state);

  void			send_slice_geometry(NrrdSlice &slice);
  void			check_colormap_on_execute();

  void			fill_paint_slices(SliceWindow &window);
  void			rasterize_colormap2();
  void			do_paint(SliceWindow &);
  void			apply_paint(SliceWindow &);
public:
  ViewSlices(GuiContext* ctx);
  virtual ~ViewSlices();
  virtual void		execute();
  virtual void		tcl_command(GuiArgs& args, void*);
  void			real_draw_all();
  void			extract_all_slices();
  double		fps_;
};


class RealDrawer : public Runnable {
  ViewSlices *	module_;
  TimeThrottle	throttle_;
public:
  bool		dead_;
  RealDrawer(ViewSlices* module) : module_(module), throttle_(), dead_(0) {};
  virtual ~RealDrawer();
  virtual void run();
};

GLenum err; 
#define GL_ERROR() \
  while ((err = glGetError()) != GL_NO_ERROR) \
    ;//error("GL error #"+to_string(err)+" on line #"+to_string(__LINE__));





RealDrawer::~RealDrawer()
{
}


void
RealDrawer::run()
{
  throttle_.start();
  
  double t = throttle_.time();
  double t2;
  double frame_count_start = t;
  double time_since_frame_count_start;
  int frames;
  while (!dead_) {
    module_->extract_all_slices();
    throttle_.wait_for_time(t);
    if (dead_) continue;
    module_->real_draw_all();
    t2 = throttle_.time();
    frames++;
    time_since_frame_count_start = t2 - frame_count_start;
    if (frames > 30 || ((time_since_frame_count_start > 3.00) && frames)) {
      module_->fps_ = frames / time_since_frame_count_start;
      frames = 0;
      frame_count_start = t2;
    }
    
    while (t <= t2) t += 1.0/120;
  }
}

void
ViewSlices::extract_all_slices() {
  TCLTask::lock();
  WindowLayouts::iterator liter = layouts_.begin(), lend = layouts_.end();
  for (; liter != lend; ++liter) {
    WindowLayout &layout = *(liter->second);
    SliceWindows::iterator viter = layout.windows_.begin();
    SliceWindows::iterator vend = layout.windows_.end();
    for (; viter != vend; ++viter) {
      SliceWindow &window = **viter;
      for (unsigned int s = 0; s < window.slices_.size(); ++s) {
	NrrdSlice &slice = *window.slices_[s];
	if (slice.axis_ != window.axis_() || !slice.nrrd_dirty_) continue;
	extract_slice(slice, window.axis_, window.slice_num_());
      }
    }
  }
  TCLTask::unlock();
}
  

ViewSlices::NrrdSlice::NrrdSlice(int axis, int slice, NrrdVolume *volume) :
  axis_(axis),
  slice_num_(slice),
  slab_min_(0),
  slab_max_(0),
  nrrd_(0),
  nrrd_dirty_(true),
  tex_dirty_(false),
  geom_dirty_(false),
  mode_(0),
  tex_wid_(0),
  tex_hei_(0),
  wid_(0),
  hei_(0),
  opacity_(1.0),
  tex_name_(0),
  volume_(volume)
  ///,
  //  lock_("Slice lock")/,
  //owner_(0),
  //lock_count_(0)
{
  do_lock();
  int i;
  for (i = 0; i < 8; i++) tex_coords_[i] = 0;
  for (i = 0; i < 12; i++) pos_coords_[i] = 0;
  do_unlock();
}

void
ViewSlices::NrrdSlice::do_lock()
{
  TCLTask::lock();
//     ASSERT(Thread::self() != 0);
//     if(owner_ == Thread::self()){
//       lock_count_++;
//       return;
//     }
//     lock_.lock();
//     lock_count_=1;
//    owner_=Thread::self();
}

void
ViewSlices::NrrdSlice::do_unlock()
{
//     ASSERT(lock_count_>0);
//     ASSERT(Thread::self() == owner_);
//     if(--lock_count_ == 0){
// 	owner_=0;
// 	lock_.unlock();
//     } else {
//     }
  TCLTask::unlock();
}


ViewSlices::SliceWindow::SliceWindow(GuiContext *ctx) :  
  name_("INVALID"),
  viewport_(0),
  slices_(),
  slice_num_(ctx->subVar("slice"), 0),
  axis_(ctx->subVar("axis"), 2),
  zoom_(ctx->subVar("zoom"), 100.0),
  slab_min_(ctx->subVar("slab_min"), 0),
  slab_max_(ctx->subVar("slab_max"), 0),
  x_(ctx->subVar("posx"),0.0),
  y_(ctx->subVar("posy"),0.0),
  redraw_(true),
  clut_dirty_(false),
  clut_ww_(ctx->subVar("clut_ww")),
  clut_wl_(ctx->subVar("clut_wl")),
  auto_levels_(ctx->subVar("auto_levels"),0),
  mode_(ctx->subVar("mode"),0),
  crosshairs_(ctx->subVar("crosshairs"),0),
  snoop_(ctx->subVar("snoop"),0),
  invert_(ctx->subVar("invert"),0),
  reverse_(ctx->subVar("reverse"),0),
  mouse_x_(ctx->subVar("mouse_x"),0),
  mouse_y_(ctx->subVar("mouse_y"),0),
  show_guidelines_(ctx->subVar("show_guidelines"),1),
  mouse_in_window_(false),
  fusion_(ctx->subVar("fusion"), 1.0),
  colormap_generation_(-1),
  colormap_size_(32),
  cursor_(-1)
{
}


ViewSlices::WindowLayout::WindowLayout(GuiContext */*ctx*/) :  
  opengl_(0),
  mouse_x_(0),
  mouse_y_(0),
  windows_()
{
}


ViewSlices::NrrdVolume::NrrdVolume(GuiContext *ctx) :
  nrrd_(0),
  opacity_(ctx->subVar("opacity"), 1.0),
  invert_(ctx->subVar("invert"), 0),
  flip_x_(ctx->subVar("flip_x"),0),
  flip_y_(ctx->subVar("flip_y"),0),
  flip_z_(ctx->subVar("flip_z"),0),
  transpose_yz_(ctx->subVar("transpose_yz"),0),
  transpose_xz_(ctx->subVar("transpose_xz"),0),
  transpose_xy_(ctx->subVar("transpose_xy"),0)
{
}


DECLARE_MAKER(ViewSlices)

ViewSlices::ViewSlices(GuiContext* ctx) :
  Module("ViewSlices", ctx, Filter, "Render", "SCIRun"),
  layouts_(),
  volumes_(),
  colormap_(0),
  current_window_(0),
  cursor_(0.0, 0.0, 0.0),
  window_level_(0),
  window_level_ww_(1),
  window_level_wl_(0),
  pick_(0),
  pick_x_(0),
  pick_y_(0),
  crop_bbox_(Point(0, 0, 0), Point(1, 1, 1)),
  crop_draw_bbox_(crop_bbox_),
  crop_pick_boxes_(),
  probe_(ctx->subVar("probe"),0),
  painting_(ctx->subVar("painting"),0),
  crop_(ctx->subVar("crop"),0),
  crop_min_x_(ctx->subVar("crop_minAxis0"),0),
  crop_min_y_(ctx->subVar("crop_minAxis1"),0),
  crop_min_z_(ctx->subVar("crop_minAxis2"),0),
  crop_max_x_(ctx->subVar("crop_maxAxis0"),0),
  crop_max_y_(ctx->subVar("crop_maxAxis1"),0),
  crop_max_z_(ctx->subVar("crop_maxAxis2"),0),
  crop_min_pad_x_(ctx->subVar("crop_minPadAxis0"),10),
  crop_min_pad_y_(ctx->subVar("crop_minPadAxis1"),20),
  crop_min_pad_z_(ctx->subVar("crop_minPadAxis2"),30),
  crop_max_pad_x_(ctx->subVar("crop_maxPadAxis0"),40),
  crop_max_pad_y_(ctx->subVar("crop_maxPadAxis1"),50),
  crop_max_pad_z_(ctx->subVar("crop_maxPadAxis2"),60),
  texture_filter_(ctx->subVar("texture_filter"),1),
  min_(ctx->subVar("min"), -1.0),
  max_(ctx->subVar("max"), -1.0),
  temp_tex_data_(0),
  ogeom_(0),
  freetype_lib_(0),
  fonts_(),
  labels_(0),
  runner_(0),
  runner_thread_(0),
  fps_(0.0)
{
  for (int a = 0; a < 3; ++a)
    mip_slices_[a] = 0;
  try {
    freetype_lib_ = scinew FreeTypeLibrary();
  } catch (...) {
    error("Cannot Initialize FreeType Library.  Did you configure with --with-freetype= ?");
    error("Module will not render text.");
  }

  if (freetype_lib_) {
    try {
      freetype_lib_ = scinew FreeTypeLibrary();
      string sdir;
      const char *dir = sci_getenv("SCIRUN_FONT_PATH");
      if (dir) 
	sdir = dir;
      else
	sdir = string(sci_getenv("SCIRUN_SRCDIR"))+"/Fonts";
      
      fonts_["default"] = freetype_lib_->load_face(sdir+"/scirun.ttf");
      fonts_["anatomical"] = fonts_["default"];
      fonts_["patientname"] = fonts_["default"];
      fonts_["fps"] = fonts_["default"];
      fonts_["view"] = fonts_["default"];
      fonts_["position"] = fonts_["default"];
      
    } catch (...) {
      fonts_.clear();
      error("Error loading fonts.\n"
	    "Please set SCIRUN_FONT_PATH to a directory with scirun.ttf\n");
    }
  }
  runner_ = scinew RealDrawer(this);
  runner_thread_ = scinew Thread(runner_, string(id+" OpenGL drawer").c_str());
}

void
ViewSlices::SliceWindow::reset() {
  zoom_ = 0.0;

  x_ = 0.0;
  y_ = 0.0;

  clut_ww_ = -1;
  clut_wl_ = -1;
  
  auto_levels_ = 0;
  mode_ = 0;
  crosshairs_ = 1;
  snoop_ = 0;
  invert_ = 0;
  reverse_ = 0;
  //  for (unsigned int v=0; v < windows_.size(); ++v)
  //  windows_[v]->reset();

}

void
ViewSlices::NrrdVolume::reset() {
  opacity_ = 0.5;
  invert_ = 0;
  flip_x_ = 0;
  flip_y_ = 0;
  flip_z_ = 0;
  transpose_yz_ = 0;
  transpose_xz_ = 0;
  transpose_xy_ = 0;
}
  

    

ViewSlices::~ViewSlices()
{
  if (runner_thread_) {
    runner_->dead_ = true;
    runner_thread_->join();
    runner_thread_ = 0;
  }
}


// TODO: Query opengl max texture size
unsigned int
ViewSlices::pow2(const unsigned int dim) const {
  unsigned int val = 1;
  while (val < dim) { val = val << 1; };
  return val;
}

unsigned int
ViewSlices::log2(const unsigned int dim) const {
  unsigned int log = 0;
  unsigned int val = 1;
  while (val < dim) { val = val << 1; log++; };
  return log;
}



void
ViewSlices::real_draw_all()
{
  TCLTask::lock();
  WindowLayouts::iterator liter = layouts_.begin(), lend = layouts_.end();
  for (; liter != lend; ++liter) {
    if (!liter->second) continue;
    WindowLayout &layout = *(liter->second);
    SliceWindows::iterator viter = layout.windows_.begin();
    SliceWindows::iterator vend = layout.windows_.end();
    for (; viter != vend; ++viter) {
      if (!*viter) continue;
      SliceWindow &window = **viter;
      if (!window.redraw_) continue;
      window.redraw_ = false;

      window.viewport_->make_current();
      window.viewport_->clear();
      GL_ERROR();
      for (unsigned int s = 0; s < window.slices_.size(); ++s) {
	if (!window.slices_[s]) continue;
	draw_slice(window, *window.slices_[s]);
      }
      
      if (bool(window.show_guidelines_()) && mouse_in_window(window))
	draw_guide_lines(window, cursor_.x(), cursor_.y(), cursor_.z());
      draw_slice_lines(window);
      if (crop_) {
	draw_crop_bbox(window, crop_draw_bbox_);
	crop_pick_boxes_ = compute_crop_pick_boxes(window,crop_draw_bbox_);
	set_window_cursor(window,
			  mouse_in_pick_boxes(window, crop_pick_boxes_));
      } else {
	set_window_cursor(window, 0);
      }

      draw_all_labels(window);
      GL_ERROR();

      window.viewport_->release();

    }
  }

  for (liter = layouts_.begin(); liter != lend; ++liter) {
    if (!liter->second) continue;
    WindowLayout &layout = *(liter->second);
    SliceWindows::iterator viter = layout.windows_.begin();
    SliceWindows::iterator vend = layout.windows_.end();
    for (; viter != vend; ++viter) {
      if (!*viter) continue;
      SliceWindow &window = **viter;
      TCLTask::lock();
      window.viewport_->make_current();
      window.viewport_->swap();
      window.viewport_->release();
      TCLTask::unlock();
    }
  }
  TCLTask::unlock();
  send_all_geometry();
}

void
ViewSlices::send_all_geometry()
{
  if (window_level_) return;
  bool flush = false;

  // If any of the mips are going to be extracted, we need to flush
  for (int axis = 0; axis < 3; ++axis)
    if (mip_slices_[axis] && mip_slices_[axis]->tex_dirty_)
      flush = true;
      
  WindowLayouts::iterator liter = layouts_.begin(), lend = layouts_.end();
  for (; liter != lend; ++liter) {
    if (!liter->second) continue;
    WindowLayout &layout = *(liter->second);
    SliceWindows::iterator viter = layout.windows_.begin();
    SliceWindows::iterator vend = layout.windows_.end();
    for (; viter != vend; ++viter) {
      if (!*viter) continue;
      SliceWindow &window = **viter;
      send_mip_slices(window);
      for (unsigned int s = 0; s < window.slices_.size(); ++s) {
	if (!window.slices_[s]) continue;
	NrrdSlice &slice = *window.slices_[s];
	if (slice.geom_dirty_) {
	  send_slice_geometry(slice);
	  flush = true;
	}
      }
    }
  }
  if (flush)  {
    ogeom_->flush();
    gui->eval("set "+id+"-geom_flushed 1");
  }
}


void
ViewSlices::redraw_all()
{
  WindowLayouts::iterator pos = layouts_.begin();
  while (pos != layouts_.end()) {
    redraw_window_layout(*(*pos++).second);
  }
}


void
ViewSlices::redraw_window_layout(ViewSlices::WindowLayout &layout)
{
  SliceWindows::iterator viter, vend = layout.windows_.end();
  for (viter = layout.windows_.begin(); viter != vend; ++viter) {
    redraw_window(**viter);
  }
}

void
ViewSlices::redraw_window(SliceWindow &window) {
  window.redraw_ = true;
}

// draw_guide_lines
// renders vertical and horizontal bars that represent
// selected slices in other dimensions
// if x, y, or z < 0, then that dimension wont be rendered
void
ViewSlices::draw_guide_lines(SliceWindow &window, float x, float y, float z, 
			   bool cursor) {
  if (cursor && !current_window_) return;
  if (cursor && !current_window_->show_guidelines_()) return;

  setup_gl_view(window);
  GL_ERROR();
  const bool inside = mouse_in_window(window);

  float unscaled_one = 1.0;
  Vector tmp = screen_to_world(window, 1, 0) - screen_to_world(window, 0, 0);
  tmp[window.axis_] = 0;
  float screen_space_one = Max(fabs(tmp[0]), fabs(tmp[1]), fabs(tmp[2]));
  if (cursor || screen_space_one > unscaled_one) 
    unscaled_one = screen_space_one;

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  GLdouble blue[4] = { 0.1, 0.4, 1.0, 0.8 };
  GLdouble green[4] = { 0.5, 1.0, 0.1, 0.8 };
  GLdouble red[4] = { 0.8, 0.2, 0.4, 0.9 };
  GLdouble yellow[4] = { 1.0, 0.76, 0.1, 0.8 };
  GLdouble white[4] = { 1.0, 1.0, 1.0, 1.0 };

  const int axis = window.axis_;
  int p = (axis+1)%3;
  int s = (axis+2)%3;


  double one;
  double c[3];
  c[0] = x;
  c[1] = y;
  c[2] = z;
  for (int i = 0; i < 2; ++i) {
    if (cursor) {
      one = unscaled_one;
      glColor4dv(yellow);
    } else {
      one = unscaled_one*scale_[p];
      switch (p) {
      case 0: glColor4dv(red); break;
      case 1: glColor4dv(green); break;
      default:
      case 2: glColor4dv(blue); break;
      }
    }
    glBegin(GL_QUADS);    
     if (c[p] >= 0 && c[p] <= max_slice_[p]*scale_[p]) {
       glVertex3f(p==0?x:0.0, p==1?y:0.0, p==2?z:0.0);
       glVertex3f(p==0?x+one:0.0, p==1?y+one:0.0, p==2?z+one:0.0);
       glVertex3f(p==0?x+one:(axis==0?0.0:(max_slice_[s]+1)*scale_[s]),
		  p==1?y+one:(axis==1?0.0:(max_slice_[s]+1)*scale_[s]),
		  p==2?z+one:(axis==2?0.0:(max_slice_[s]+1)*scale_[s]));
       
       glVertex3f(p==0?x:(axis==0?0.0:(max_slice_[s]+1)*scale_[s]),
		  p==1?y:(axis==1?0.0:(max_slice_[s]+1)*scale_[s]),
		  p==2?z:(axis==2?0.0:(max_slice_[s]+1)*scale_[s]));
     }
     glEnd();
     SWAP(p,s);
  }




  if (!inside || !cursor) {
    return;
  }

    
  Point cvll = screen_to_world(window, 0, 0);
  Point cvur = screen_to_world(window, 
			       window.viewport_->max_width(), 
			       window.viewport_->max_height());
  cvll(window.axis_) = 0;
  cvur(window.axis_) = 0;
  glColor4dv(white);
  glBegin(GL_QUADS);
  for (int i = 0; i < 2; ++i) {
    if (c[p] < 0 || c[p] > max_slice_[p]*scale_[p]) {
      glVertex3f(p==0?x:cvll(0), 
		 p==1?y:cvll(1), 
		 p==2?z:cvll(2));
      glVertex3f(p==0?x+one:cvll(0), 
		 p==1?y+one:cvll(1), 
		 p==2?z+one:cvll(2));
      glVertex3f(p==0?x+one:(axis==0?cvll(0):cvur(0)),
		 p==1?y+one:(axis==1?cvll(1):cvur(1)),
		 p==2?z+one:(axis==2?cvll(2):cvur(2)));
      
      glVertex3f(p==0?x:(axis==0?cvll(0):cvur(0)),
		 p==1?y:(axis==1?cvll(1):cvur(1)),
		 p==2?z:(axis==2?cvll(2):cvur(2)));
    } else {
      glVertex3f(p==0?x:cvll(0), p==1?y:cvll(1), p==2?z:cvll(2));
      glVertex3f(p==0?x+one:cvll(0), p==1?y+one:cvll(1), p==2?z+one:cvll(2));
      glVertex3f(p==0?x+one:(axis==0?cvll(0):0.0),
		 p==1?y+one:(axis==1?cvll(1):0.0),
		 p==2?z+one:(axis==2?cvll(2):0.0));
      
      glVertex3f(p==0?x:(axis==0?cvll(0):0.0),
		 p==1?y:(axis==1?cvll(1):0.0),
		 p==2?z:(axis==2?cvll(2):0.0));


      glVertex3f(p==0?x:(axis==0?0.0:(max_slice_[s]+1)*scale_[s]), 
		 p==1?y:(axis==1?0.0:(max_slice_[s]+1)*scale_[s]),
		 p==2?z:(axis==2?0.0:(max_slice_[s]+1)*scale_[s]));

      glVertex3f(p==0?x+one:(axis==0?0.0:(max_slice_[s]+1)*scale_[s]), 
		 p==1?y+one:(axis==1?0.0:(max_slice_[s]+1)*scale_[s]), 
		 p==2?z+one:(axis==2?0.0:(max_slice_[s]+1)*scale_[s]));

      glVertex3f(p==0?x+one:(axis==0?(axis==0?0.0:(max_slice_[s]+1)*scale_[s]):cvur(0)),
		 p==1?y+one:(axis==1?(axis==1?0.0:(max_slice_[s]+1)*scale_[s]):cvur(1)),
		 p==2?z+one:(axis==2?(axis==2?0.0:(max_slice_[s]+1)*scale_[s]):cvur(2)));
      
      glVertex3f(p==0?x:(axis==0?(axis==0?0.0:(max_slice_[s]+1)*scale_[s]):cvur(0)),
		 p==1?y:(axis==1?(axis==1?0.0:(max_slice_[s]+1)*scale_[s]):cvur(1)),
		 p==2?z:(axis==2?(axis==2?0.0:(max_slice_[s]+1)*scale_[s]):cvur(2)));
    }
    SWAP(p,s);
  }

      
  glEnd();
}  



// renders vertical and horizontal bars that represent
// selected slices in other dimensions
void
ViewSlices::draw_slice_lines(SliceWindow &window)
{
  setup_gl_view(window);
  GL_ERROR();

  Vector tmp = screen_to_world(window, 1, 0) - screen_to_world(window, 0, 0);
  tmp[window.axis_] = 0;
  double screen_space_one = Max(fabs(tmp[0]), fabs(tmp[1]), fabs(tmp[2]));

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  GLdouble blue[4] = { 0.1, 0.4, 1.0, 0.8 };
  GLdouble green[4] = { 0.5, 1.0, 0.1, 0.8 };
  GLdouble red[4] = { 0.8, 0.2, 0.4, 0.9 };

  const int axis = window.axis_;
  int p = (axis+1)%3;
  int s = (axis+2)%3;


  double one;
  double xyz[3];
  int i;
  for (int i = 0; i < 3; ++i)
    xyz[i] = cur_slice_[i]*scale_[i];
  
  for (i = 0; i < 2; ++i) {
    if (!slab_width_[p]) continue;
    one = Max(screen_space_one, double(scale_[p]*slab_width_[p]));

    switch (p) {
    case 0: glColor4dv(red); break;
    case 1: glColor4dv(green); break;
    default:
    case 2: glColor4dv(blue); break;
    }
    glBegin(GL_QUADS);    
    if (xyz[p] >= 0 && xyz[p] <= max_slice_[p]*scale_[p]) {
      glVertex3f(p==0?xyz[0]:0.0, p==1?xyz[1]:0.0, p==2?xyz[2]:0.0);
      glVertex3f(p==0?xyz[0]+one:0.0, p==1?xyz[1]+one:0.0, p==2?xyz[2]+one:0.0);
      glVertex3f(p==0?xyz[0]+one:(axis==0?0.0:(max_slice_[s]+1)*scale_[s]),
		 p==1?xyz[1]+one:(axis==1?0.0:(max_slice_[s]+1)*scale_[s]),
		 p==2?xyz[2]+one:(axis==2?0.0:(max_slice_[s]+1)*scale_[s]));
      
      glVertex3f(p==0?xyz[0]:(axis==0?0.0:(max_slice_[s]+1)*scale_[s]),
		 p==1?xyz[1]:(axis==1?0.0:(max_slice_[s]+1)*scale_[s]),
		 p==2?xyz[2]:(axis==2?0.0:(max_slice_[s]+1)*scale_[s]));
    }
    glEnd();
    SWAP(p,s);
  }
}



void
ViewSlices::draw_crop_bbox(SliceWindow &window, BBox &bbox) {
  setup_gl_view(window);
  GL_ERROR();
  float unscaled_one = 1.0;
  Vector tmp = screen_to_world(window, 1, 0) - screen_to_world(window, 0, 0);
  tmp[window.axis_] = 0;
  float screen_space_one = Max(fabs(tmp[0]), fabs(tmp[1]), fabs(tmp[2]));
  if (screen_space_one > unscaled_one) 
    unscaled_one = screen_space_one;

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  const int axis = window.axis_;
  int p = x_axis(window);
  int s = y_axis(window);
  
  double ll[3], ur[3], lr[3], ul[3], upper[3], lower[3], left[3], right[3];
  ll[0] = bbox.min().x()*scale_[0];
  ll[1] = bbox.min().y()*scale_[1];
  ll[2] = bbox.min().z()*scale_[2];

  ur[0] = bbox.max().x()*scale_[0];
  ur[1] = bbox.max().y()*scale_[1];
  ur[2] = bbox.max().z()*scale_[2];

  ll[axis] = int(window.slice_num_)*scale_[axis];
  ur[axis] = int(window.slice_num_)*scale_[axis];
  int i;
  for (i = 0; i < 3; ++i) {
    lr[i] = p==i?ur[i]:ll[i];
    ul[i] = s==i?ur[i]:ll[i];
    upper[i] = (ur[i]+ul[i])/2.0;
    lower[i] = (lr[i]+ll[i])/2.0;
    left[i] = (ll[i]+ul[i])/2.0;
    right[i] = (lr[i]+ur[i])/2.0;
  }    
  
  GLdouble blue[4] = { 0.1, 0.4, 1.0, 0.8 };
  GLdouble green[4] = { 0.5, 1.0, 0.1, 0.7 };
  GLdouble lt_green[4] = { 0.5, 1.0, 0.1, 0.4 };
  GLdouble red[4] = { 0.8, 0.2, 0.4, 0.9 };
  GLdouble grey[4] = { 0.6, 0.6, 0.6, 0.6 }; 
  GLdouble white[4] = { 1.0, 1.0, 1.0, 1.0 }; 
  GLdouble black[4] = { 0.0, 0.0, 0.0, 1.0 }; 
  GLdouble yellow[4] = { 1.0, 0.76, 0.1, 1.0 };

  switch (axis) {
  case 0: glColor4dv(red); break;
  case 1: glColor4dv(green); break;
  default:
  case 2: glColor4dv(blue); break;
  }

  if (double(window.slice_num_) >= bbox.min()(window.axis_) &&
      double(window.slice_num_) <= (bbox.max()(window.axis_)-1.0))
    glColor4dv(green);
  else
    glColor4dv(lt_green);

  glBegin(GL_QUADS);
  {
    glVertex3dv(ll);
    glVertex3dv(lr);
    glVertex3dv(ur);
    glVertex3dv(ul);
  }
  glEnd();

  glColor4dv(black);
  glEnable(GL_LINE_SMOOTH);
  glLineWidth(5.0);
  glBegin(GL_LINE_LOOP);
  {
    glVertex3dv(ll);
    glVertex3dv(lr);
    glVertex3dv(ur);
    glVertex3dv(ul);
  }
  glEnd();
  glDisable(GL_LINE_SMOOTH);

  glColor4dv(grey);
  glEnable(GL_LINE_SMOOTH);
  glLineWidth(3.0);
  glBegin(GL_LINE_LOOP);
  {
    glVertex3dv(ll);
    glVertex3dv(lr);
    glVertex3dv(ur);
    glVertex3dv(ul);
  }
  glEnd();
  glDisable(GL_LINE_SMOOTH);

  glColor4dv(white);
  glEnable(GL_LINE_SMOOTH);
  glLineWidth(1.0);
  glBegin(GL_LINE_LOOP);
  {
    glVertex3dv(ll);
    glVertex3dv(lr);
    glVertex3dv(ur);
    glVertex3dv(ul);
  }
  glEnd();
  glDisable(GL_LINE_SMOOTH);


  glColor4dv(black);
  glEnable(GL_POINT_SMOOTH);
  glPointSize(8.0);
  glBegin(GL_POINTS);
  {
    glVertex3dv(ll);
    glVertex3dv(lr);
    glVertex3dv(ur);
    glVertex3dv(ul);
    //    glVertex3dv(upper);
    //glVertex3dv(lower);
    //glVertex3dv(left);
    //glVertex3dv(right);
  }
  glEnd();

  glPointSize(6.0);
  glColor4dv(yellow);
  glBegin(GL_POINTS);
  {
    glVertex3dv(ll);
    glVertex3dv(lr);
    glVertex3dv(ur);
    glVertex3dv(ul);
    //    glVertex3dv(upper);
    //glVertex3dv(lower);
    //glVertex3dv(left);
    //glVertex3dv(right);
  }
  glEnd();

  glDisable(GL_POINT_SMOOTH);
  glDisable(GL_BLEND);
}  



ViewSlices::PickBoxes
ViewSlices::compute_crop_pick_boxes(SliceWindow &window, BBox &bbox) 
{
  const int axis = window.axis_;
  int p = x_axis(window);
  int s = y_axis(window);
  
  Point ll, ur, lr, ul;
  ll(0) = bbox.min().x()*scale_[0];
  ll(1) = bbox.min().y()*scale_[1];
  ll(2) = bbox.min().z()*scale_[2];

  ur(0) = bbox.max().x()*scale_[0];
  ur(1) = bbox.max().y()*scale_[1];
  ur(2) = bbox.max().z()*scale_[2];

  ll(axis) = int(window.slice_num_)*scale_[axis];
  ur(axis) = int(window.slice_num_)*scale_[axis];
  int i;
  for (i = 0; i < 3; ++i) {
    lr(i) = p==i?ur(i):ll(i);
    ul(i) = s==i?ur(i):ll(i);
  }
  
  ll = world_to_screen(window, ll);
  lr = world_to_screen(window, lr);
  ur = world_to_screen(window, ur);
  ul = world_to_screen(window, ul);

  Vector delta(3.0, 3.0, 1.0);
  PickBoxes ret_val;
  ret_val.reserve(9);
  ret_val.push_back(BBox(ll-delta, ll+delta)); // Lower-Left 1
  ret_val.push_back(BBox(lr-delta, lr+delta)); // Lower-Right 2
  ret_val.push_back(BBox(ur-delta, ur+delta)); // Upper-right 3
  ret_val.push_back(BBox(ul-delta, ul+delta)); // Upper-Left 4
  ret_val.push_back(BBox(ll-delta, ul+delta)); // Left 5
  ret_val.push_back(BBox(ll-delta, lr+delta)); // Lower 6
  ret_val.push_back(BBox(lr-delta, ur+delta)); // Right 7
  ret_val.push_back(BBox(ul-delta, ur+delta)); // Upper 8
  ret_val.push_back(BBox(ll-delta, ur+delta)); // Entire Crop Box

  return ret_val;
}

void
ViewSlices::update_crop_bbox_from_gui() 
{
  crop_draw_bbox_ = 
    BBox(Point(double(crop_min_x_()),
	       double(crop_min_y_()),
	       double(crop_min_z_())),
	 Point(double(crop_max_x_()),
	       double(crop_max_y_()),
	       double(crop_max_z_())));
}

void
ViewSlices::update_crop_bbox_to_gui() 
{
  crop_min_x_ = Round(crop_draw_bbox_.min().x());
  crop_min_y_ = Round(crop_draw_bbox_.min().y());
  crop_min_z_ = Round(crop_draw_bbox_.min().z());
  crop_max_x_ = Round(crop_draw_bbox_.max().x()-1.0);
  crop_max_y_ = Round(crop_draw_bbox_.max().y()-1.0);
  crop_max_z_ = Round(crop_draw_bbox_.max().z()-1.0);
}

int
ViewSlices::mouse_in_pick_boxes(SliceWindow &window, PickBoxes &pick_boxes)
{
  if (!mouse_in_window(window)) return 0;
  Point mouse(window.mouse_x_, window.mouse_y_, 0.0);
  // Check the last pick box assuming it encloses all the previous pick boxes
  if (!pick_boxes.back().inside(mouse)) return 0;
  else for (unsigned int i = 0; i < pick_boxes.size(); ++i)
    if (pick_boxes[i].inside(mouse)) return i+1;
  return 0;//pick_boxes.size();
}

void
ViewSlices::set_window_cursor(SliceWindow &window, int cursor) 
{
  if (!mouse_in_window(window) || window.cursor_ == cursor) return;
  window.cursor_ = cursor;
  string cursor_name;
  switch (cursor) {
  case 1: cursor_name = "bottom_left_corner"; break;
  case 2: cursor_name = "bottom_right_corner"; break;
  case 3: cursor_name = "top_right_corner"; break;
  case 4: cursor_name = "top_left_corner"; break;
  case 5: cursor_name = "sb_h_double_arrow"; break;
  case 6: cursor_name = "sb_v_double_arrow"; break;
  case 7: cursor_name = "sb_h_double_arrow"; break;
  case 8: cursor_name = "sb_v_double_arrow"; break;
  case 9: cursor_name = "fleur"; break;
  case 0: cursor_name = "crosshair"; break;
  }
  Tk_Window &tkwin = layouts_[window.name_]->opengl_->tkwin_;
  Tk_DefineCursor(tkwin, Tk_GetCursor(the_interp, tkwin, 
				      ccast_unsafe(cursor_name)));
//  gui->eval(window.name_+" configure -cursor "+cursor_name);
}


pair<Vector, Vector>
ViewSlices::get_crop_vectors(SliceWindow &window, int pick) 
{

  Vector tmp = screen_to_world(window, 1, 0) - screen_to_world(window, 0, 0);
  tmp[window.axis_] = 0;
  const float one = Max(fabs(tmp[0]), fabs(tmp[1]), fabs(tmp[2]));
  const int x_ax = x_axis(window);
  const int y_ax = y_axis(window);
  Vector x_delta(0.0, 0.0, 0.0), y_delta(0.0, 0.0, 0.0);
  if (pick != 6 && pick != 8)
    x_delta[x_ax] = one/scale_[x_ax];
  if (pick != 5 && pick != 7)
    y_delta[y_ax] = -one/scale_[y_ax];
  
  return make_pair(x_delta, y_delta);
}

BBox
ViewSlices::update_crop_bbox(SliceWindow &window, int pick, int X, int Y)
{
  pair<Vector, Vector> crop_delta = get_crop_vectors(window, pick);
  Vector crop_delta_x = crop_delta.first*(X - pick_x_);
  Vector crop_delta_y = crop_delta.second*(Y - pick_y_);
  Point min = crop_bbox_.min();
  Point max = crop_bbox_.max();
  const int p = x_axis(window);
  const int s = y_axis(window);

  //  UIint *uimin[3] = { &crop_min_x_, &crop_min_y_, &crop_min_z_ };
  // UIint *uimax[3] = { &crop_min_x_, &crop_min_y_, &crop_min_z_ };
  int uiminpad[3] = {crop_min_pad_x_(), crop_min_pad_y_(), crop_min_pad_z_()};
  int uimaxpad[3] = {crop_max_pad_x_(), crop_max_pad_y_(), crop_max_pad_z_()};

  switch (pick) {
  case 1: 
    min += crop_delta_x; 
    min += crop_delta_y; 
    break;
  case 2: 
    max += crop_delta_x; 
    min += crop_delta_y; 
    break;
  case 3: 
    max += crop_delta_x; 
    max += crop_delta_y; 
    break;
  case 4: 
    min += crop_delta_x; 
    max += crop_delta_y; 
    break;
  case 5:
    min += crop_delta_x; 
    break;
  case 6:
    min += crop_delta_y; 
    break;
  case 7:
    max += crop_delta_x; 
    break;
  case 8:
    max += crop_delta_y; 
    break;
  case 9:

    if (min(p)+crop_delta_x[p] < -uiminpad[p])
      crop_delta_x[p] = -min(p)-uiminpad[p];
    if (min(s)+crop_delta_y[s] < -uiminpad[s])
      crop_delta_y[s] = -min(s)-uiminpad[s];
    if (max(p)+crop_delta_x[p] > (max_slice_[p]+uimaxpad[p]+1.0))
      crop_delta_x[p] = (max_slice_[p]+uimaxpad[p]+1.0)-max(p);
    if (max(s)+crop_delta_y[s] > (max_slice_[s]+uimaxpad[s]+1.0))
      crop_delta_y[s] = (max_slice_[s]+uimaxpad[s]+1.0)-max(s);

    min += crop_delta_x;
    min += crop_delta_y; 
    max += crop_delta_x; 
    max += crop_delta_y; 
    break;
  default: break;
  }
  int i;
  for (i = 0; i < 3; ++i) 
    if (min(i) > max(i)) 
      SWAP(min(i), max(i));

  for (i = 0; i < 3; ++i) {
    if (min(i) > 0)
      min(i) = Round(min(i));
    else
      min(i) = -Round(-min(i)+0.5);
    max(i) = Round(max(i));
  }

  for (int i = 0; i < 3; ++i) {
    if (min(i) < -uiminpad[i]) 
      min(i) = -uiminpad[i];
    if (max(i) > (max_slice_[i]+uimaxpad[i]+1.0)) 
      max(i) = (max_slice_[i]+uimaxpad[i]+1.0);    
  }

  for (i = 0; i < 3; ++i)
    if (fabs(min(i)-max(i)) < 0.0001)  // floating point equal
      if (min(i)+uiminpad[i] > 0.0001) 
	min(i) = max(i)-1.0;
      else
	max(i) = -uiminpad[i]+1.0;

  return BBox(min, max);
}



// Returns and index to the axis that is most parallel and in the direction of
// +X in the screen.  
// 0 for +x, 1 for +y, and 2 for +z
// 3 for -x, 4 for -y, and 5 for -z
int
ViewSlices::x_axis(SliceWindow &window)
{
  Vector dir = screen_to_world(window,1,0)-screen_to_world(window,0,0);
  Vector adir = Abs(dir);
  int m = adir[0] > adir[1] ? 0 : 1;
  m = adir[m] > adir[2] ? m : 2;
  return dir[m] < 0.0 ? m+3 : m;
}

// Returns and index to the axis that is most parallel and in the direction of
// +Y in the screen.  
// 0 for +x, 1 for +y, and 2 for +z
// 3 for -x, 4 for -y, and 5 for -z
int
ViewSlices::y_axis(SliceWindow &window)
{
  Vector dir = screen_to_world(window,0,1)-screen_to_world(window,0,0);
  Vector adir = Abs(dir);
  int m = adir[0] > adir[1] ? 0 : 1;
  m = adir[m] > adir[2] ? m : 2;
  return dir[m] < 0.0 ? m+3 : m;
}

void
ViewSlices::setup_gl_view(SliceWindow &window)
{
  // bad hack to resolve infinite loop in screen_to_world
  static int here = 0;
  if (here) return;
  here = 1;
  GL_ERROR();
  glMatrixMode(GL_MODELVIEW);
  GL_ERROR();
  glLoadIdentity();
  GL_ERROR();

  int axis = window.axis_;

  if (axis == 0) { // screen +X -> +Y, screen +Y -> +Z
    glRotated(-90,0.,1.,0.);
    glRotated(-90,1.,0.,0.);
  } else if (axis == 1) { // screen +X -> +X, screen +Y -> +Z
    glRotated(-90,1.,0.,0.);
  }
  GL_ERROR();
  
  glTranslated((axis==0)?-double(window.slice_num_):0.0,
  	       (axis==1)?-double(window.slice_num_):0.0,
  	       (axis==2)?-double(window.slice_num_):0.0);
  GL_ERROR();
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  GL_ERROR();
  glGetDoublev(GL_MODELVIEW_MATRIX, window.gl_modelview_matrix_);
  glGetDoublev(GL_PROJECTION_MATRIX, window.gl_projection_matrix_);
  glGetIntegerv(GL_VIEWPORT, window.gl_viewport_);
  GL_ERROR();
  double hwid = (window.viewport_->width()/(*window.zoom_/100.0))/2;
  double hhei = (window.viewport_->height()/(*window.zoom_/100.0))/2;
  
  double cx = double(*window.x_) + center_[x_axis(window) % 3];
  double cy = double(*window.y_) + center_[y_axis(window) % 3];

  double minz = -max_slice_[axis]*scale_[axis];
  double maxz = max_slice_[axis]*scale_[axis];
  if (fabs(minz - maxz) < 0.01) {
    minz = -1.0;
    maxz = 1.0;
  }
  glOrtho(cx - hwid, cx + hwid, cy - hhei, cy + hhei, minz, maxz);
  GL_ERROR();
  glGetDoublev(GL_PROJECTION_MATRIX, window.gl_projection_matrix_);
  GL_ERROR();
  here = 0;

}





// A,P   R, L   S, I
void
ViewSlices::draw_position_label(SliceWindow &window)
{
  FreeTypeFace *font = fonts_["position"];
  if (!font) return;

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glScaled(2.0, 2.0, 2.0);
  glTranslated(-.5, -.5, -.5);

  BBox label_bbox;
  //  BBox line_bbox;
  font->set_points(15.0);
  FreeTypeText zoom("Zoom: "+to_string(window.zoom_())+string("%"), font);
  zoom.get_bounds(label_bbox);
  Point pos(0, label_bbox.max().y()+2, 0);
  FreeTypeText cursor("X: "+to_string(Ceil(cursor_.x()))+
		      " Y: "+to_string(Ceil(cursor_.y()))+
		      " Z: "+to_string(Ceil(cursor_.z())),
		      font, &pos);
  cursor.get_bounds(label_bbox);
  unsigned int wid = pow2(Round(label_bbox.max().x()));
  unsigned int hei = pow2(Round(label_bbox.max().y()));
  GLubyte *buf = scinew GLubyte[wid*hei*4];
  memset(buf, 0, wid*hei*4);
  zoom.render(wid, hei, buf);
  cursor.render(wid, hei, buf);
  
  GLuint tex_id;
  glGenTextures(1, &tex_id);

  glBindTexture(GL_TEXTURE_2D, tex_id);
  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1); 
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glPixelTransferi(GL_MAP_COLOR, 0);
  
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, wid, hei, 0, GL_RGBA, GL_UNSIGNED_BYTE, buf);
  delete[] buf;
  
  const double dx = 1.0/window.viewport_->width();
  const double dy = 1.0/window.viewport_->height();
  
  glBegin(GL_QUADS);

  glTexCoord2f(0.0, 0.0);
  glVertex3f(0.0, 0.0, 0.0);

  glTexCoord2f(1.0, 0.0);
  glVertex3f(dx*wid, 0.0, 0.0);

  glTexCoord2f(1.0, 1.0);
  glVertex3f(dx*wid, dy*hei , 0.0);

  glTexCoord2f(0.0, 1.0);
  glVertex3f(0.0, dy*hei, 0.0);

  glEnd();

  glDeleteTextures(1, &tex_id);
  //  glRasterPos2d(0.0, 0.0);
  //  glDrawPixels (wid, hei, GL_RGBA, GL_UNSIGNED_BYTE, buf);
}  




// Right	= -X
// Left		= +X
// Posterior	= -Y
// Anterior	= +X
// Inferior	= -Z
// Superior	= +Z
void
ViewSlices::draw_anatomical_labels(SliceWindow &window)
{
  FreeTypeFace *font = fonts_["anatomical"];
  if (!font) return;


  int prim = x_axis(window);
  int sec = y_axis(window);
  
  string ltext, rtext, ttext, btext;
  switch (prim % 3) {
  case 0: ltext = "R"; rtext = "L"; break;
  case 1: ltext = "P"; rtext = "A"; break;
  default:
  case 2: ltext = "I"; rtext = "S"; break;
  }
  if (prim >= 3) SWAP (ltext, rtext);

  switch (sec % 3) {
  case 0: btext = "R"; ttext = "L"; break;
  case 1: btext = "P"; ttext = "A"; break;
  default:
  case 2: btext = "I"; ttext = "S"; break;
  }
  if (sec >= 3) SWAP (ttext, btext);

  font->set_points(25.0);
  draw_label(window, ltext, 2, window.viewport_->height()/2, 
	     FreeTypeText::w, font);
  draw_label(window, rtext, window.viewport_->width()-2, 
	     window.viewport_->height()/2, FreeTypeText::e, font);
  draw_label(window, btext, window.viewport_->width()/2, 2, 
	     FreeTypeText::s, font);
  draw_label(window, ttext, window.viewport_->width()/2, 
	     window.viewport_->height()-2, FreeTypeText::n, font);

}


void
ViewSlices::draw_label(SliceWindow &window, string text, int x, int y,
		      FreeTypeText::anchor_e anchor, 
		      FreeTypeFace *font)
{
  if (!font && fonts_.size()) 
    font = fonts_["default"];
  if (!font) return;
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glScaled(2.0, 2.0, 2.0);
  glTranslated(-.5, -.5, -.5);

  BBox bbox;
  FreeTypeText fttext(text, font);
  fttext.get_bounds(bbox);
  if (bbox.min().y() < 0) {
    fttext.set_position(Point(0, -bbox.min().y(), 0));
    bbox.reset();
    fttext.get_bounds(bbox);
  }

  unsigned int wid = pow2(Round(bbox.max().x()));
  unsigned int hei = pow2(Round(bbox.max().y()));
  GLubyte *buf = scinew GLubyte[wid*hei*4];
  memset(buf, 0, wid*hei*4);
  fttext.render(wid, hei, buf);
  
  GLuint tex_id;
  glGenTextures(1, &tex_id);

  glBindTexture(GL_TEXTURE_2D, tex_id);
  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1); 
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glPixelTransferi(GL_MAP_COLOR, 0);
  
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, wid, hei, 0, GL_RGBA, GL_UNSIGNED_BYTE, buf);
  delete [] buf;
  
  double px = x;
  double py = y;

  switch (anchor) {
  case FreeTypeText::s:  px -= bbox.max().x()/2.0; break;
  case FreeTypeText::se: px -= bbox.max().x();     break;
  case FreeTypeText::e:  px -= bbox.max().x();     py -= bbox.max().y()/2.0; break;
  case FreeTypeText::ne: px -= bbox.max().x();     py -= bbox.max().y();     break;
  case FreeTypeText::n:  px -= bbox.max().x()/2.0; py -= bbox.max().y();     break;
  case FreeTypeText::nw: py -= bbox.max().y();     break;
  case FreeTypeText::w:  py -= bbox.max().y()/2.0; break;
  default: // lowerleft do noting
  case FreeTypeText::sw: break;
  }

  
  const double dx = 1.0/window.viewport_->width();
  const double dy = 1.0/window.viewport_->height();
  
  glBegin(GL_QUADS);

  glTexCoord2f(0.0, 0.0);
  glVertex3f(px*dx, py*dy, 0.0);

  glTexCoord2f(1.0, 0.0);
  glVertex3f(dx*(px+wid), py*dy, 0.0);

  glTexCoord2f(1.0, 1.0);
  glVertex3f(dx*(px+wid), dy*(py+hei) , 0.0);

  glTexCoord2f(0.0, 1.0);
  glVertex3f(px*dx, dy*(py+hei), 0.0);

  glEnd();
  glDeleteTextures(1, &tex_id);
}  


float *
ViewSlices::apply_colormap(NrrdSlice &slice, 
			   double min, double max, float *data) {
  const int sizeof4floats = 4*sizeof(float);
  unsigned int x, y;
  int val, pos, resolution;  
  const float *rgba;
  float *greyscale_rgba;
  if (!colormap_.get_rep()) {
    resolution = 256;
    greyscale_rgba = scinew float[256*4];
    for (int c = 0; c < 256*4; ++c)
      greyscale_rgba[c/4] = (c/4)/255.0;
    rgba = greyscale_rgba;
  } else {
    resolution = colormap_->resolution();
    rgba = colormap_->get_rgba();
  }
    
  const double scale = (resolution-1)/double(max-min);

  switch (slice.nrrd_->nrrd->type) {
  case nrrdTypeChar: {
    char *slicedata = (char *)slice.nrrd_->nrrd->data;
    for (y = 0; y < slice.tex_hei_; ++y)
      for (x = 0; x < slice.tex_wid_; ++x) {
	pos = y*slice.tex_wid_+x;
	val = Round(scale*(slicedata[pos] - min));
	val = Clamp(val, 0, resolution-1);
	memcpy(data+pos*4, rgba+val*4, sizeof4floats);
      }
  } break;
    
  case nrrdTypeUChar: {
    unsigned char *slicedata = (unsigned char *)slice.nrrd_->nrrd->data;
    for (y = 0; y < slice.tex_hei_; ++y)
      for (x = 0; x < slice.tex_wid_; ++x) {
	pos = y*slice.tex_wid_+x;
	val = Round(scale*(slicedata[pos] - min));
	val = Clamp(val, 0, resolution-1);
	memcpy(data+pos*4, rgba+val*4, sizeof4floats);
      }
  } break;

  case nrrdTypeShort: {
    short *slicedata = (short *)slice.nrrd_->nrrd->data;
    for (y = 0; y < slice.tex_hei_; ++y)
      for (x = 0; x < slice.tex_wid_; ++x) {
	pos = y*slice.tex_wid_+x;
	val = Round(scale*(slicedata[pos] - min));
	val = Clamp(val, 0, resolution-1);
	memcpy(data+pos*4, rgba+val*4, sizeof4floats);
      }
  } break;

  case nrrdTypeUShort: {
    unsigned short *slicedata = (unsigned short *)slice.nrrd_->nrrd->data;
    for (y = 0; y < slice.tex_hei_; ++y)
      for (x = 0; x < slice.tex_wid_; ++x) {
	pos = y*slice.tex_wid_+x;
	val = Round(scale*(slicedata[pos] - min));
	val = Clamp(val, 0, resolution-1);
	memcpy(data+pos*4, rgba+val*4, sizeof4floats);
      }
  } break;

  case nrrdTypeInt: {
    int *slicedata = (int *)slice.nrrd_->nrrd->data;
    for (y = 0; y < slice.tex_hei_; ++y)
      for (x = 0; x < slice.tex_wid_; ++x) {
	pos = y*slice.tex_wid_+x;
	val = Round(scale*(slicedata[pos] - min));
	val = Clamp(val, 0, resolution-1);
	memcpy(data+pos*4, rgba+val*4, sizeof4floats);
      }
  } break;

  case nrrdTypeUInt: {
    unsigned int *slicedata = (unsigned int *)slice.nrrd_->nrrd->data;
    for (y = 0; y < slice.tex_hei_; ++y)
      for (x = 0; x < slice.tex_wid_; ++x) {
	pos = y*slice.tex_wid_+x;
	val = Round(scale*(slicedata[pos] - min));
	val = Clamp(val, 0, resolution-1);
	memcpy(data+pos*4, rgba+val*4, sizeof4floats);
      }
  } break;

  case nrrdTypeLLong: {
    signed long long *slicedata = (signed long long *)slice.nrrd_->nrrd->data;
    for (y = 0; y < slice.tex_hei_; ++y)
      for (x = 0; x < slice.tex_wid_; ++x) {
	pos = y*slice.tex_wid_+x;
	val = Round(scale*(slicedata[pos] - min));
	val = Clamp(val, 0, resolution-1);
	memcpy(data+pos*4, rgba+val*4, sizeof4floats);
      }
  } break;

  case nrrdTypeULLong: {
    unsigned long long *slicedata = (unsigned long long *)slice.nrrd_->nrrd->data;
    for (y = 0; y < slice.tex_hei_; ++y)
      for (x = 0; x < slice.tex_wid_; ++x) {
	pos = y*slice.tex_wid_+x;
	val = Round(scale*(slicedata[pos] - min));
	val = Clamp(val, 0, resolution-1);
	memcpy(data+pos*4, rgba+val*4, sizeof4floats);
      }
  } break;

  case nrrdTypeFloat: {
    float *slicedata = (float *)slice.nrrd_->nrrd->data;
    for (y = 0; y < slice.tex_hei_; ++y)
      for (x = 0; x < slice.tex_wid_; ++x) {
	pos = y*slice.tex_wid_+x;
	val = Round(scale*(slicedata[pos] - min));
	val = Clamp(val, 0, resolution-1);
	memcpy(data+pos*4, rgba+val*4, sizeof4floats);
      }
  } break;

  case nrrdTypeDouble: {
    double *slicedata = (double *)slice.nrrd_->nrrd->data;
    for (y = 0; y < slice.tex_hei_; ++y)
      for (x = 0; x < slice.tex_wid_; ++x) {
	pos = y*slice.tex_wid_+x;
	val = Round(scale*(slicedata[pos] - min));
	val = Clamp(val, 0, resolution-1);
	memcpy(data+pos*4, rgba+val*4, sizeof4floats);
      }
  } break;


  default: error("Unsupported data type: "+slice.nrrd_->nrrd->type);
  }

  if (!colormap_.get_rep())
    delete[] greyscale_rgba;
    
  return data;
}



void
ViewSlices::draw_slice(SliceWindow &window, NrrdSlice &slice)
{
  if (slice.axis_ != window.axis_) return;
  //  if (slice.nrrd_dirty_) 
    //    cerr << window.name_ << " slice dirty\n";
  if (slice.nrrd_dirty_ && slice.volume_)
    extract_slice(slice, window.axis_(), window.slice_num_());
  
  ASSERT(slice.nrrd_.get_rep());

  slice.do_lock();

  GL_ERROR();
  setup_gl_view(window);

  glRasterPos2d(0.0, 0.0);

  // set material color to be white
  GLfloat ones[4] = {1.0, 1.0, 1.0, 1.0};
  glColor4fv(ones);

  // Setup the opacity of the slice to be drawn
  GLfloat opacity = slice.opacity_*window.fusion_();
  if (volumes_.size() == 2 && 
      slice.volume_->nrrd_.get_rep() == volumes_[1]->nrrd_.get_rep()) {
    opacity = slice.opacity_*(1.0 - window.fusion_); 
  }
  glColor4f(1.0, 1.0, 1.0, opacity);//, opacity, opacity, opacity);

  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glEnable(GL_TEXTURE_2D);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA); 
 
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glShadeModel(GL_FLAT);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  glTexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, ones);

  const bool bound = glIsTexture(slice.tex_name_);
  if (!bound) {
    glGenTextures(1, &slice.tex_name_);
  }

  glBindTexture(GL_TEXTURE_2D, slice.tex_name_);

  if (!bound || slice.tex_dirty_) {
    int min = window.clut_wl_ - window.clut_ww_/2;
    int max = window.clut_wl_ + window.clut_ww_/2;
    apply_colormap(slice, double(min), double(max), temp_tex_data_);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);  
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    const GLint filter_mode = (texture_filter_()?GL_LINEAR:GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter_mode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter_mode);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, slice.tex_wid_, slice.tex_hei_, 
    		 0, GL_RGBA, GL_FLOAT, temp_tex_data_);
    slice.tex_dirty_ = false;
    slice.geom_dirty_ = true;
  }

  unsigned int i;
  if (!painting_) {
    glBegin( GL_QUADS );
    for (i = 0; i < 4; i++) {
      glTexCoord2fv(&slice.tex_coords_[i*2]);
      glVertex3fv(&slice.pos_coords_[i*3]);
    }
    glEnd();
  }

  glBindTexture(GL_TEXTURE_2D, 0);

  glColor4fv(ones);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  if (painting_ && window.paint_slices_.size() && window.paint_slices_[window.axis_]) {
    NrrdSlice &paint = *window.paint_slices_[window.axis_];
    glGenTextures(1, &paint.tex_name_);
    glBindTexture(GL_TEXTURE_2D, paint.tex_name_);    
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);  
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, slice.tex_wid_, slice.tex_hei_, 
    		 0, GL_RGBA, GL_FLOAT, paint.nrrd_->nrrd->data);

    glBegin( GL_QUADS );
    for (i = 0; i < 4; i++) {
      glTexCoord2fv(&slice.tex_coords_[i*2]);
      glVertex3fv(&slice.pos_coords_[i*3]);
    }
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);
  }

  glFlush();
  slice.do_unlock();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glDisable(GL_TEXTURE_2D);
  glDisable(GL_BLEND);
    
  GL_ERROR();
}



void
ViewSlices::draw_all_labels(SliceWindow &window) {
    // set material color to be white
  GLfloat ones[4] = {1.0, 1.0, 1.0, 1.0};
  glColor4fv(ones);
  glEnable(GL_BLEND);
  glEnable(GL_TEXTURE_2D);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

  draw_position_label(window);
  draw_anatomical_labels(window);
  FreeTypeFace *font = fonts_["patientname"];
  if (0 && font) {
    font->set_points(20.0);
    draw_label(window, "Patient Name: Anonymous", 
	       window.viewport_->width() - 2,
	       window.viewport_->height() - 2,
	       FreeTypeText::ne, font);
  }

  font = fonts_["view"];
  if (font) {
    font->set_points(20.0);
    string text;
    switch (window.axis_) {
    case 0: text = "Sagittal"; break;
    case 1: text = "Coronal"; break;
    default:
    case 2: text = "Axial"; break;
    }
    if (window.mode_ == slab_e) text = "SLAB - "+text;
    if (window.mode_ == mip_e) text = "MIP - "+text;
    draw_label(window, text, window.viewport_->width() - 2, 0, 
	       FreeTypeText::se, font);
  }
  
  font = fonts_["fps"];
  if (font && string(sci_getenv("USER")) == string("mdavis")) {
    font->set_points(20.0);
        draw_label(window, "fps: "+to_string(fps_), 
		   0, window.viewport_->height() - 2,
		   FreeTypeText::nw, font);
  }
}
  



void
ViewSlices::set_slice_coords(NrrdSlice &slice, bool origin) {
  slice.do_lock();

  // Texture coordinates
  const double tex_x = double(slice.wid_)/double(slice.tex_wid_);
  const double tex_y = double(slice.hei_)/double(slice.tex_hei_);
  unsigned int i = 0;
  slice.tex_coords_[i++] = 0.0; 
  slice.tex_coords_[i++] = 0.0;
  slice.tex_coords_[i++] = tex_x;
  slice.tex_coords_[i++] = 0.0;
  slice.tex_coords_[i++] = tex_x;
  slice.tex_coords_[i++] = tex_y;
  slice.tex_coords_[i++] = 0.0;
  slice.tex_coords_[i++] = tex_y;


  
  // Position Coordinates
  int axis = slice.axis_;
  int slice_num = slice.slice_num_;
  double x_pos=0,y_pos=0,z_pos=0;
  double x_wid=0,y_wid=0,z_wid=0;
  double x_hei=0,y_hei=0,z_hei=0;
  
  if (axis == 0) {
    x_pos = slice_num;
    y_wid = slice.volume_->nrrd_->nrrd->axis[1].size;
    z_hei = slice.volume_->nrrd_->nrrd->axis[2].size; 
  } else if (axis == 1) {
    y_pos = slice_num;
    x_wid = slice.volume_->nrrd_->nrrd->axis[0].size;
    z_hei = slice.volume_->nrrd_->nrrd->axis[2].size;
  } else /*if (axis == 2)*/ {
    z_pos = slice_num;
    x_wid = slice.volume_->nrrd_->nrrd->axis[0].size;
    y_hei = slice.volume_->nrrd_->nrrd->axis[1].size;
  }

  if (*slice.volume_->flip_x_) {
    x_pos += x_wid+x_hei;
    x_wid *= -1.0;
    x_hei *= -1.0;
  }

  if (*slice.volume_->flip_y_) {
    y_pos += y_wid+y_hei;
    y_wid *= -1.0;
    y_hei *= -1.0;
  }

  if (*slice.volume_->flip_z_) {
    z_pos += z_wid+z_hei;
    z_wid *= -1.0;
    z_hei *= -1.0;
  }

  if (*slice.volume_->transpose_yz_) {
    SWAP(y_wid, z_wid);
    SWAP(y_hei, z_hei);
  } else if (*slice.volume_->transpose_xz_) {
    SWAP(x_wid, z_wid);
    SWAP(x_hei, z_hei);
  } else if (*slice.volume_->transpose_xy_) {
    SWAP(x_wid, y_wid);
    SWAP(x_hei, y_hei);
  } 

  i = 0;
  slice.pos_coords_[i++] = x_pos;
  slice.pos_coords_[i++] = y_pos;
  slice.pos_coords_[i++] = z_pos;

  slice.pos_coords_[i++] = x_pos+x_wid;
  slice.pos_coords_[i++] = y_pos+y_wid;
  slice.pos_coords_[i++] = z_pos+z_wid;

  slice.pos_coords_[i++] = x_pos+x_wid+x_hei;
  slice.pos_coords_[i++] = y_pos+y_wid+y_hei;
  slice.pos_coords_[i++] = z_pos+z_wid+z_hei;

  slice.pos_coords_[i++] = x_pos+x_hei;
  slice.pos_coords_[i++] = y_pos+y_hei;
  slice.pos_coords_[i++] = z_pos+z_hei;

  for (i = 0; i < 12; ++i) {
    slice.pos_coords_[i] *= scale_[i%3];
    if (!origin) 
      slice.pos_coords_[i] += slice.volume_->nrrd_->nrrd->axis[i%3].min;
  }

  slice.do_unlock();
}


void
ViewSlices::extract_window_slices(SliceWindow &window) {
  unsigned int v, s;
  int a;

  if (!window.slices_.size()) {
    for (v = 0; v < volumes_.size(); ++v) {
      for (a = 0; a < 3; a++)
	window.slices_.push_back
	  (scinew NrrdSlice(a, window.slice_num_(), volumes_[v]));
    }
  } 

  s = 0;
  for (v = 0; v < volumes_.size(); ++v) {
    for (a = 0; a < 3; a++) {
      NrrdSlice &slice = *window.slices_[s];
      slice.do_lock();
      slice.nrrd_dirty_ = true;
      slice.mode_ = window.mode_();
      slice.volume_ = volumes_[v];
      slice.slice_num_ = window.slice_num_();
      slice.slab_min_ = window.slab_min_();
      slice.slab_max_ = window.slab_max_();
      slice.do_unlock();	
      s++;
    }
  }
}



void
ViewSlices::extract_slice(NrrdSlice &slice, int axis, int slice_num)
{
  if (!slice.nrrd_dirty_) return;

  ASSERT(slice.volume_);
  ASSERT(slice.volume_->nrrd_.get_rep());

  slice.do_lock();

  slice.nrrd_ = 0;
  slice.nrrd_ = scinew NrrdData;
  slice.nrrd_->nrrd = nrrdNew();

  NrrdDataHandle temp1 = scinew NrrdData;
  temp1->nrrd = nrrdNew();  

  NrrdDataHandle temp2 = scinew NrrdData;
  temp2->nrrd = nrrdNew();
  
  if (slice.mode_ == mip_e || slice.mode_ == slab_e) {
    int min[3], max[3];
    for (int i = 0; i < 3; i++) {
      min[i] = 0;
      max[i] = max_slice_[i];
    }
    slab_width_[axis] = 0;
    if (slice.mode_ == slab_e) {
      min[axis] = Min(max_slice_[axis], Max(0, slice.slab_min_));
      max[axis] = Max(0, Min(slice.slab_max_, max_slice_[axis]));
      slab_width_[axis] = slice.slab_max_ - slice.slab_min_ + 1;
    }
    if (max[axis] < min[axis]) SWAP(min[axis], max[axis]);
    cur_slice_[axis] = min[axis];

    if (nrrdCrop(temp2->nrrd, slice.volume_->nrrd_->nrrd, min, max)) {
      char *err = biffGetDone(NRRD);
      error(string("Error MIP cropping nrrd: ") + err);
      free(err);
    }
      
    if (nrrdProject(temp1->nrrd, temp2->nrrd, axis, 2, nrrdTypeDefault)) {
      char *err = biffGetDone(NRRD);
      error(string("Error Projecting nrrd: ") + err);
      free(err);
    }
  } else {
    slice_num = Clamp(slice_num, 0, max_slice_[axis]);
    cur_slice_[axis] = slice_num;
    slab_width_[axis] = 1;
    if (nrrdSlice(temp1->nrrd, slice.volume_->nrrd_->nrrd, axis, slice_num)) {
      char *err = biffGetDone(NRRD);
      error(string("Error Slicing nrrd: ") + err);
      free(err);
    }
  }

  slice.axis_ = axis;
  slice.slice_num_ = slice_num;
  slice.nrrd_dirty_ = false;
  slice.tex_dirty_ = true;
  slice.geom_dirty_ = false;
  slice.wid_     = temp1->nrrd->axis[0].size;
  slice.hei_     = temp1->nrrd->axis[1].size;
  slice.tex_wid_ = pow2(slice.wid_);
  slice.tex_hei_ = pow2(slice.hei_);
  slice.opacity_ = slice.volume_->opacity_;

  int minp[2] = { 0, 0 };
  int maxp[2] = { slice.tex_wid_-1, slice.tex_hei_-1 };

  if (nrrdPad(slice.nrrd_->nrrd, temp1->nrrd, minp,maxp,nrrdBoundaryBleed)) 
  {
    char *err = biffGetDone(NRRD);
    error(string("Trouble resampling: ") + err);
    free(err);
  }

  set_slice_coords(slice);

  slice.do_unlock();
}


void
ViewSlices::extract_mip_slices(NrrdVolume *volume)
{
  if (!volume || !volume->nrrd_.get_rep()) { return; }
  for (int axis = 0; axis < 3; ++axis) {
    //    cerr << "Extracting MIP " << axis << std::endl;
    if (!mip_slices_[axis])
      mip_slices_[axis] = scinew NrrdSlice(axis, 0, volume);
      
    NrrdSlice &slice = *mip_slices_[axis];
    slice.do_lock();
    slice.volume_ = volume;

    slice.nrrd_ = scinew NrrdData;
    slice.nrrd_->nrrd = nrrdNew();

    NrrdDataHandle temp1 = scinew NrrdData;
    temp1->nrrd = nrrdNew();  
    
    int min[3], max[3];
    for (int i = 0; i < 3; i++) {
      min[i] = 0;
      max[i] = max_slice_[i];
    }
    
    if (nrrdProject(temp1->nrrd, slice.volume_->nrrd_->nrrd, 
		    axis, 2, nrrdTypeDefault)) {
      char *err = biffGetDone(NRRD);
      error(string("Error Projecting MIP nrrd: ") + err);
      free(err);
    }
    
    slice.axis_ = axis;
    slice.slice_num_ = 0;
    slice.nrrd_dirty_ = false;
    slice.tex_dirty_ = true;
    slice.geom_dirty_ = false;
    slice.wid_     = temp1->nrrd->axis[0].size;
    slice.hei_     = temp1->nrrd->axis[1].size;
    slice.tex_wid_ = pow2(slice.wid_);
    slice.tex_hei_ = pow2(slice.hei_);
    slice.opacity_ = slice.volume_->opacity_;
    
    int minp[2] = { 0, 0 };
    int maxp[2] = { slice.tex_wid_-1, slice.tex_hei_-1 };
    
    if (nrrdPad(slice.nrrd_->nrrd, temp1->nrrd, 
		minp,maxp,nrrdBoundaryPad, 0.0)) 
    {
      char *err = biffGetDone(NRRD);
      error(string("Trouble resampling: ") + err);
      free(err);
    }
        
    set_slice_coords(slice);
    slice.do_unlock();
  }
}


void
ViewSlices::send_mip_slices(SliceWindow &window)
{
  GeometryOPort *geom = (GeometryOPort *)get_oport("Geometry");
  if (!geom) {
    error("Cannot find port Geometry!\n");
    return;
  }
  for (int axis = 0; axis < 3; ++axis)
  {
    if (!mip_slices_[axis]) continue;
    NrrdSlice &slice = *mip_slices_[axis];
    if (!slice.tex_dirty_) continue;
    //    cerr << "Sending MIP " << axis << std::endl;

    slice.do_lock();
    slice.tex_dirty_ = false;
    window.viewport_->make_current();
    bool bound = glIsTexture(slice.tex_name_);
    if (!bound) {
      glGenTextures(1, &slice.tex_name_);
    }

    glBindTexture(GL_TEXTURE_2D, slice.tex_name_);
    int min = window.clut_wl_ - window.clut_ww_/2;
    int max = window.clut_wl_ + window.clut_ww_/2;
    apply_colormap(slice, double(min), double(max), temp_tex_data_);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);  
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, slice.tex_wid_, slice.tex_hei_, 
    		 0, GL_RGBA, GL_FLOAT, temp_tex_data_);
    glBindTexture(GL_TEXTURE_2D, 0);
    window.viewport_->release();
    string name = "MIP Slice"+to_string(slice.axis_);

    TexSquare *mintex = scinew TexSquare();

    slice.slice_num_ = 0;
    set_slice_coords(slice, false);
    mintex->set_coords(slice.tex_coords_, slice.pos_coords_);
    mintex->set_texname(slice.tex_name_);

    Vector *minvec = scinew 
      Vector(axis==0?1.0:0.0, axis==1?1.0:0.0, axis==2?1.0:0.0);
    GeomCull *mincull = scinew GeomCull(mintex, minvec);

    TexSquare *maxtex = scinew TexSquare();
    slice.slice_num_ = max_slice_[axis];
    set_slice_coords(slice, false);
    maxtex->set_coords(slice.tex_coords_, slice.pos_coords_);
    maxtex->set_texname(slice.tex_name_);

    Vector *maxvec = scinew 
      Vector(axis==0?-1.0:0.0, axis==1?-1.0:0.0, axis==2?-1.0:0.0);
    GeomCull *maxcull = scinew GeomCull(maxtex, maxvec);

    GeomGroup *group = scinew GeomGroup();
    group->add(mincull);
    group->add(maxcull);
    
    GeomHandle gobj = group;
    slice.do_unlock();
    if (gobjs_[name]) geom->delObj(gobjs_[name]);
    gobjs_[name] = geom->addObj(gobj, name);
  }
}



void
ViewSlices::set_axis(SliceWindow &window, unsigned int axis) {
  window.axis_ = axis;
  extract_window_slices(window);
  redraw_window(window);
}

void
ViewSlices::prev_slice(SliceWindow &window)
{
  if (window.slice_num_() == 0) 
    return;
  if (window.slice_num_ < 1)
    window.slice_num_ = 0;
  else 
    window.slice_num_--;
  cursor_ = screen_to_world(window, window.mouse_x_, window.mouse_y_);
  extract_window_slices(window);
  redraw_all();
}

void
ViewSlices::next_slice(SliceWindow &window)
{
  if (*window.slice_num_() == max_slice_[window.axis_()]) 
    return;
  if (window.slice_num_ > max_slice_[window.axis_])
    window.slice_num_ = max_slice_[window.axis_];
  else
    window.slice_num_++;
  cursor_ = screen_to_world(window, window.mouse_x_, window.mouse_y_);
  extract_window_slices(window);
  redraw_all();
}

void
ViewSlices::zoom_in(SliceWindow &window)
{
  window.zoom_ *= 1.1;
  redraw_window(window);
}

void
ViewSlices::zoom_out(SliceWindow &window)
{
  window.zoom_ /= 1.1;
  redraw_window(window);
}

  
Point
ViewSlices::screen_to_world(SliceWindow &window, 
			   unsigned int x, unsigned int y) {
  GLdouble xyz[3];
  gluUnProject(double(x), double(y), double(window.slice_num_),
	       window.gl_modelview_matrix_, 
	       window.gl_projection_matrix_,
	       window.gl_viewport_,
	       xyz+0, xyz+1, xyz+2);
  xyz[window.axis_] = double(window.slice_num_)*scale_[window.axis_];

  return Point(xyz[0], xyz[1], xyz[2]);
}


Point
ViewSlices::world_to_screen(SliceWindow &window, Point &world)
{
  GLdouble xyz[3];
  
  gluProject(double(world(0)), double(world(1)), double(world(2)),
	       window.gl_modelview_matrix_, 
	       window.gl_projection_matrix_,
	       window.gl_viewport_,
	       xyz+0, xyz+1, xyz+2);
  //  xyz[window.axis_] = double(window.slice_num_)*scale_[window.axis_];

  return Point(xyz[0], xyz[1], xyz[2]);
}
  

void
ViewSlices::check_colormap_on_execute() {
  ColorMapIPort *color_iport = (ColorMapIPort *)get_iport("Nrrd1ColorMap");  
  if (!color_iport) 
  {
    error("Unable to initialize iport ColorMap.");
    return;
  }
  color_iport->get(colormap_);
}



void
ViewSlices::execute()
{
  //  cerr << "Execute\n";
  update_state(Module::JustStarted);
  NrrdIPort *nrrd1_port = (NrrdIPort*)get_iport("Nrrd1");
  NrrdIPort *nrrd2_port = (NrrdIPort*)get_iport("Nrrd2");
  NrrdIPort *nrrdGradient_port = (NrrdIPort*)get_iport("NrrdGradient");
  ColorMap2IPort* cmap_iport = (ColorMap2IPort*)get_iport("InputColorMap2");

  ogeom_ = (GeometryOPort *)get_oport("Geometry");

  if (!nrrd1_port) 
  {
    error("Unable to initialize iport Nrrd1.");
    return;
  }

  if (!nrrd2_port) 
  {
    error("Unable to initialize iport Nrrd.");
    return;
  }

  if (!ogeom_) {
    error("Unable to initialize oport Scene Graph.");
    return;
  }

  if (cmap_iport) {
    cmap_iport->get(cm2_);
  }
  
  if (nrrdGradient_port) {
    nrrdGradient_port->get(gradient_);
  }

  check_colormap_on_execute();

  update_state(Module::NeedData);
  NrrdDataHandle nrrd1, nrrd2;
  nrrd1_port->get(nrrd1);
  nrrd2_port->get(nrrd2);
  
  if (!nrrd1.get_rep() && !nrrd2.get_rep())
  {
    error ("Unable to get a nrrd from Nrrd1 or Nrrd2.");
    return;
  }
  unsigned int i;

					 
  for (i = 0; i < 3; i++)
    max_slice_[i] = -1;

  if (nrrd1.get_rep() && nrrd2.get_rep()) 
    for (i = 0; i < 3; i++)
      if (nrrd1->nrrd->axis[i].size != nrrd2->nrrd->axis[i].size) {
	error("Both input nrrds must have same dimensions.");
	error("  Only rendering first inpput.");
	nrrd2 = 0;
      } else
	max_slice_[i] = nrrd1->nrrd->axis[i].size-1;
  else if (nrrd1.get_rep()) 
    for (i = 0; i < 3; i++)
      max_slice_[i] = nrrd1->nrrd->axis[i].size-1;
  else if (nrrd2.get_rep()) 
    for (i = 0; i < 3; i++)
      max_slice_[i] = nrrd2->nrrd->axis[i].size-1;

  if (temp_tex_data_) delete[] temp_tex_data_;
  // Temporary space to hold the colormapped texture
  temp_tex_data_ = scinew float[Max(pow2(max_slice_[0]), pow2(max_slice_[1]),
                                    pow2(max_slice_[2]))*Mid(pow2(max_slice_[0]),
                                    pow2(max_slice_[1]), pow2(max_slice_[2]))*4];

  TCLTask::lock();
  // Clear the volumes
  for (i = 0; i < volumes_.size(); ++i) 
    delete volumes_[i];
  volumes_.clear();

  if (nrrd1.get_rep()) {
    NrrdRange *range =nrrdRangeNewSet(nrrd1->nrrd,0);
    min_ = Min(range->min, range->max);
    max_ = Max(range->min, range->max);

    volumes_.push_back(scinew NrrdVolume(ctx->subVar("nrrd1",0)));
    volumes_.back()->nrrd_ = nrrd1;
  }

  if (nrrd2.get_rep()) {
    volumes_.push_back(scinew NrrdVolume(ctx->subVar("nrrd2",0)));
    volumes_.back()->nrrd_ = nrrd2;
  }

  for (unsigned int v = 0; v < volumes_.size(); ++v) {
    NrrdVolume *volume = volumes_[v];
    for (i = 0; i < 3; ++i) {
      scale_[i] = (airExists_d(volume->nrrd_->nrrd->axis[i].spacing) ?
		   volume->nrrd_->nrrd->axis[i].spacing : 1.0);
      center_[i] = (volume->nrrd_->nrrd->axis[i].size*scale_[i])/2;
    }
  }
  
  if (volumes_.size())
    extract_mip_slices(volumes_.back());


  WindowLayouts::iterator pos = layouts_.begin();
  while (pos != layouts_.end()) {
    WindowLayout &layout = *(*pos++).second;
    SliceWindows::iterator viter, vend = layout.windows_.end();
    for (viter = layout.windows_.begin(); viter != vend; ++viter) {
      extract_window_slices(**viter);
    }
  }

  TCLTask::unlock();

  update_state(Module::Executing);  
  redraw_all();
  update_state(Module::Completed);  
}

bool
ViewSlices::mouse_in_window(SliceWindow &window) {
  return (window.mouse_in_window_ &&
	  window.mouse_x_ >= window.viewport_->x() && 
	  window.mouse_x_ < window.viewport_->x()+window.viewport_->width() &&
	  window.mouse_y_ >= window.viewport_->y() &&
	  window.mouse_y_ < window.viewport_->y()+window.viewport_->height());
}

  


void
ViewSlices::handle_gui_motion(GuiArgs &args) {
  int state;
  if (!string_to_int(args[5], state)) {
    args.error ("Cannot convert motion state");
    return;
  }

  if (layouts_.find(args[2]) == layouts_.end()) {
    error ("Cannot handle motion on "+args[2]);
    return;
  }

  SliceWindow *inside_window = 0;

  WindowLayout &layout = *layouts_[args[2]];
  int x, y;
  string_to_int(args[3], x);
  string_to_int(args[4], y);
  y = layout.opengl_->height() - 1 - y;
  for (unsigned int w = 0; w < layout.windows_.size(); ++w) {
    SliceWindow &window = *layout.windows_[w];
    window.mouse_x_ = x;
    window.mouse_y_ = y;
    if (mouse_in_window(window)) {
      inside_window = &window;
      cursor_ = screen_to_world(window, window.mouse_x_, window.mouse_y_);
    }
  }

  int X, Y;
  string_to_int(args[7], X);
  string_to_int(args[8], Y);

  if (state & BUTTON_1_E && painting_ && inside_window) { 
    do_paint(*inside_window);
  } else if ((state & BUTTON_1_E) && crop_ && inside_window && pick_) {
    crop_draw_bbox_ = update_crop_bbox(*inside_window, pick_, X, Y);
    crop_pick_boxes_ = compute_crop_pick_boxes(*inside_window, crop_draw_bbox_);
    update_crop_bbox_to_gui();
  } else if (inside_window && probe_()) {
    WindowLayouts::iterator liter = layouts_.begin();
    while (liter != layouts_.end()) {
      for (unsigned int v = 0; v < (*liter).second->windows_.size(); ++v) {
	SliceWindow &window = *(*liter).second->windows_[v];
	const int axis = window.axis_();
	window.slice_num_ = Round(cursor_(axis)/scale_[axis]);
	if (window.slice_num_ < 0) 
	  window.slice_num_ = 0;
	if (window.slice_num_ > max_slice_[axis]) 
	  window.slice_num_ = max_slice_[axis];
	
	extract_window_slices(window);
      }
      ++liter;
    }
    extract_all_slices();
  } else if (window_level_) {
    for (int axis = 0; axis < 3; ++axis)
    {
      if (!mip_slices_[axis]) continue;
      NrrdSlice &slice = *mip_slices_[axis];
      slice.do_lock();
      slice.tex_dirty_ = true;
      slice.do_unlock();
    }

    if (state & SHIFT_E) {
      SliceWindow &window = *window_level_;
      window.clut_ww_ = window_level_ww_ + (X - pick_x_)*2;
      if (window.clut_ww_ < 1) window.clut_ww_ = 1;
      window.clut_wl_ = window_level_wl_ + (pick_y_ - Y)*2;
      window.clut_dirty_ = true;
      for (unsigned int s = 0; s < window.slices_.size(); ++s)
	window.slices_[s]->tex_dirty_ = true;
    } else {
      WindowLayouts::iterator liter = layouts_.begin();
      while (liter != layouts_.end()) {
	for (unsigned int v = 0; v < (*liter).second->windows_.size(); ++v) {
	  SliceWindow &window = *(*liter).second->windows_[v];
	  window.clut_ww_ = window_level_ww_ + (X - pick_x_)*2;
	  if (window.clut_ww_ < 1) window.clut_ww_ = 1;
	  window.clut_wl_ = window_level_wl_ + (pick_y_ - Y)*2;
	  window.clut_dirty_ = true;
	  for (unsigned int s = 0; s < window.slices_.size(); ++s)
	    window.slices_[s]->tex_dirty_ = true;
	}
	++liter;
      }
    }
  }
  redraw_all();
}



void
ViewSlices::handle_gui_enter(GuiArgs &args) {
  if (layouts_.find(args[2]) == layouts_.end()) {
    error ("Cannot handle enter on "+args[2]);
    return;
  }
  WindowLayout &layout = *layouts_[args[2]];
  for (unsigned int w = 0; w < layout.windows_.size(); ++w) {
    SliceWindow &window = *layout.windows_[w];
    window.mouse_in_window_ = true;
    current_window_ = &window;
  }
}


void
ViewSlices::handle_gui_leave(GuiArgs &args) {
  if (layouts_.find(args[2]) == layouts_.end()) {
    error ("Cannot handle enter on "+args[2]);
    return;
  }
  WindowLayout &layout = *layouts_[args[2]];
  for (unsigned int w = 0; w < layout.windows_.size(); ++w) {
    SliceWindow &window = *layout.windows_[w];
    window.mouse_in_window_ = false;
    current_window_ = false;
  }
}



void
ViewSlices::debug_print_state(int state) {
  cerr << "State: " << state;
  vector<string> modstrings;
  modstrings.push_back("Shift"); // 1
  modstrings.push_back("Caps Lock"); // 2
  modstrings.push_back("Control"); // 8
  modstrings.push_back("Alt"); // 4
  modstrings.push_back("M1"); // 16
  modstrings.push_back("M2"); // 32
  modstrings.push_back("M3"); // 64
  modstrings.push_back("M4"); // 128
  modstrings.push_back("Button1"); // 256
  modstrings.push_back("Button2"); // 512
  modstrings.push_back("Button3"); // 1024
  int i = 1;
  unsigned int power = 0;
  while (i < state) {
    if (state & i)
      if (power < modstrings.size())
	cerr << modstrings[power] << "-";
      else
	cerr << i << "-";  
    i <<= 1;
    ++power;
  }
  cerr << std::endl;
}

void
ViewSlices::handle_gui_button_release(GuiArgs &args) {
  int button;
  int state;
  int x;
  int y;

  if (args.count() != 7) {
    args.error(args[0]+" "+args[1]+" expects a window #, button #, and state");
    return;
  }

  if (!string_to_int(args[3], button)) {
    args.error ("Cannot convert window #");
    return;
  }

  if (!string_to_int(args[4], state)) {
    args.error ("Cannot convert button state");
    return;
  }

  if (!string_to_int(args[5], x)) {
    args.error ("Cannot convert X");
    return;
  }

  if (!string_to_int(args[6], y)) {
    args.error ("Cannot convert Y");
    return;
  }

  if (layouts_.find(args[2]) == layouts_.end()) {
    error ("Cannot handle button release on "+args[2]);
    return;
  }

  switch (button) {
  case 1:
    window_level_ = 0;
    if (pick_) crop_bbox_ = crop_draw_bbox_;
    pick_ = 0;
    break;
  case 2:
    probe_ = 0;
    break;
  default:
    break;
  }

}

void
ViewSlices::handle_gui_button(GuiArgs &args) {
  int button;
  int state;
  int x;
  int y;

  if (args.count() != 7) {
    args.error(args[0]+" "+args[1]+" expects a window #, button #, and state");
    return;
  }

  if (!string_to_int(args[3], button)) {
    args.error ("Cannot convert window #");
    return;
  }

  if (!string_to_int(args[4], state)) {
    args.error ("Cannot convert button state");
    return;
  }

  if (!string_to_int(args[5], x)) {
    args.error ("Cannot convert X");
    return;
  }

  if (!string_to_int(args[6], y)) {
    args.error ("Cannot convert Y");
    return;
  }

  if (layouts_.find(args[2]) == layouts_.end()) {
    error ("Cannot handle motion on "+args[2]);
    return;
  }

  WindowLayout &layout = *layouts_[args[2]];
  
  for (unsigned int w = 0; w < layout.windows_.size(); ++w) {
    SliceWindow &window = *layout.windows_[w];
    if (!mouse_in_window(window)) continue;
    switch (button) {
    case 1:
      if (painting_) { do_paint(window); continue; }
      crop_pick_boxes_ = compute_crop_pick_boxes(window, crop_bbox_);
      pick_ = mouse_in_pick_boxes(window, crop_pick_boxes_);
      pick_x_ = x;
      pick_y_ = y;
      if (!pick_) {
	window_level_ = layout.windows_[w];
	window_level_ww_ = window.clut_ww_();
	window_level_wl_ = window.clut_wl_();
      }
      break;
    case 2:
      probe_ = 1;
      break;


    case 4:
      if (state & CONTROL_E == CONTROL_E) 
	zoom_in(window);
      else
	next_slice(window);
      break;
      
    case 5:
      if (state & SHIFT_E == SHIFT_E) 
	zoom_out(window);
      else
	prev_slice(window);
      break;
      
    default: 
      break;
    }
  }
}

void
ViewSlices::handle_gui_keypress(GuiArgs &args) {
  int keycode;

  if (false)
    for (int i = 0; i < args.count(); ++i)
      cerr << args[i] << (i == args.count()-1)?"\n":" ";
  
  if (args.count() != 6) {
    args.error(args[0]+" "+args[1]+" expects a win #, keycode, keysym,& time");
    return;
  }
  

  if (!string_to_int(args[3], keycode)) {
    args.error ("Cannot convert keycode");
    return;
  }

  if (layouts_.find(args[2]) == layouts_.end()) {
    error ("Cannot handle motion on "+args[2]);
    return;
  }

  WindowLayout &layout = *layouts_[args[2]];

  for (unsigned int w = 0; w < layout.windows_.size(); ++w) {
    SliceWindow &window = *layout.windows_[w];
    double pan_delta = Round(3.0/(window.zoom_()/100.0));
    if (pan_delta < 1.0) pan_delta = 1.0;
    if (!mouse_in_window(window)) continue;
    if (args[4] == "equal" || args[4] == "plus") {
      zoom_in(window);
    } else if (args[4] == "minus" || args[4] == "underscore") {
      zoom_out(window);
    } else if (args[4] == "0") {
      set_axis(window, 0);
    } else if (args[4] == "1") {
      set_axis(window, 1);
    } else if (args[4] == "2") {
      set_axis(window, 2);
    } else if (args[4] == "i") {
      window.invert_ = window.invert_?0:1;
      redraw_window(window);
    } else if (args[4] == "p") {
      if (painting_())
	painting_ = 0;
      else {
	painting_ = 1;
	fill_paint_slices(window);
      }
      redraw_window(window);
    } else if (args[4] == "m") {
      window.mode_ = (window.mode_+1)%num_display_modes_e;
      extract_window_slices(window);
      redraw_window(window);
    } else if (args[4] == "Left") {
      window.x_ += pan_delta;
      redraw_window(window);
    } else if (args[4] == "Right") {
      window.x_ -= pan_delta;
      redraw_window(window);
    } else if (args[4] == "Down") {
      window.y_ += pan_delta;
      redraw_window(window);
    } else if (args[4] == "Up") {
      window.y_ -= pan_delta;
      redraw_window(window);
    } else if (args[4] == "less" || args[4] == "comma") {
      prev_slice(window);
    } else if (args[4] == "greater" || args[4] == "period") {
      next_slice(window);
    } 
  }
}


void
ViewSlices::tcl_command(GuiArgs& args, void* userdata) {
  if(args.count() < 2) {
    args.error("ViewSlices needs a minor command");
    return;
  } else if (args[1] == "motion") handle_gui_motion(args);
  else if (args[1] == "button")   handle_gui_button(args);
  else if (args[1] == "release")  handle_gui_button_release(args);
  else if (args[1] == "keypress") handle_gui_keypress(args);
  else if (args[1] == "enter")    handle_gui_enter(args);
  else if (args[1] == "leave")    handle_gui_leave(args);
  else if(args[1] == "setgl") {
    int visualid = 0;
    if (args.count() == 5 && !string_to_int(args[4], visualid))
      error("setgl bad visual id: "+args[4]);

    TkOpenGLContext *context = scinew TkOpenGLContext(args[2], visualid, 512, 512);
    XSync(context->display_, 0);
    ASSERT(layouts_.find(args[2]) == layouts_.end());
    layouts_[args[2]] = scinew WindowLayout(ctx->subVar(args[3],0));
    layouts_[args[2]]->opengl_ = context;
  } else if(args[1] == "destroygl") {
    WindowLayouts::iterator pos = layouts_.find(args[2]);
    ASSERT(pos != layouts_.end());
    delete (pos->second);
    layouts_.erase(pos);
  } else if(args[1] == "add_viewport") {
    ASSERT(layouts_.find(args[2]) != layouts_.end());
    WindowLayout *layout = layouts_[args[2]];
    SliceWindow *window = scinew SliceWindow(ctx->subVar(args[3],0));
    window->name_ = args[2];
    window->viewport_ = scinew OpenGLViewport(layout->opengl_);
    window->axis_ = (layouts_.size())%3;
    layout->windows_.push_back(window);
  } else if(args[1] == "redraw") {
    ASSERT(layouts_.find(args[2]) != layouts_.end());
    redraw_window_layout(*layouts_[args[2]]);
  } else if(args[1] == "redrawall") {
    redraw_all();
  } else if(args[1] == "rebind") {
    ASSERT(layouts_.find(args[2]) != layouts_.end());
    WindowLayout &layout = *layouts_[args[2]];
    for (unsigned int w = 0; w < layout.windows_.size(); ++w) {
      SliceWindow &window = *layout.windows_[w];
      extract_window_slices(window);
    }
    redraw_window_layout(layout);
  } else if(args[1] == "texture_rebind") {
    ASSERT(layouts_.find(args[2]) != layouts_.end());
    WindowLayout &layout = *layouts_[args[2]];
    for (unsigned int w = 0; w < layout.windows_.size(); ++w) {
      SliceWindow &window = *layout.windows_[w];
      unsigned int v, s;
      int a;      
      s = 0;
      for (v = 0; v < volumes_.size(); ++v) {
	for (a = 0; a < 3; a++) {
	  window.slices_[s]->do_lock();
	  window.slices_[s]->tex_dirty_ = true;
	  window.slices_[s]->do_unlock();	
	  s++;
	}
      }
    }
    redraw_window_layout(layout);
  } else if(args[1] == "startcrop") {
    crop_ = 1;
    crop_bbox_ = BBox
      (Point(0,0,0), 
       Point(max_slice_[0]+1, max_slice_[1]+1, max_slice_[2]+1));
    crop_draw_bbox_ = crop_bbox_;
    update_crop_bbox_to_gui();
    redraw_all();
  } else if(args[1] == "stopcrop") {
    crop_ = 0;
    redraw_all();
  } else if(args[1] == "updatecrop") {
    update_crop_bbox_from_gui();
    redraw_all();
  } else if (args[1] == "setclut") {
    WindowLayouts::iterator liter = layouts_.begin();
    while (liter != layouts_.end()) {
      for (unsigned int v = 0; v < (*liter).second->windows_.size(); ++v) {
	SliceWindow &window = *(*liter).second->windows_[v];
	window.clut_ww_ = window.clut_ww_();
	window.clut_wl_ = window.clut_wl_();
	window.clut_dirty_ = true;
	for (unsigned int s = 0; s < window.slices_.size(); ++s)
	  window.slices_[s]->tex_dirty_ = true;
      }
      ++liter;
    }
  } else Module::tcl_command(args, userdata);
}



void
ViewSlices::send_slice_geometry(NrrdSlice &slice) {

  slice.do_lock();
  string name = "Slice"+to_string(slice.axis_);
  if (tobjs_[name] == 0) 
    tobjs_[name] = scinew TexSquare();
  set_slice_coords(slice, false);
  tobjs_[name]->set_coords(slice.tex_coords_, slice.pos_coords_);
  set_slice_coords(slice, true);
  tobjs_[name]->set_texname(slice.tex_name_);
  GeomHandle gobj = tobjs_[name];
  slice.geom_dirty_ = false;
  slice.do_unlock();
  
  if (gobjs_[name]) ogeom_->delObj(gobjs_[name]);
  gobjs_[name] = ogeom_->addObj(gobj, name);
  
}

typedef vector<CM2WidgetHandle> CM2Widgets;

void
ViewSlices::rasterize_colormap2() {
//   if (!cm2_.get_rep()) return;
//   CM2Widgets &widgets = cm2_->widgets();
//   cm2_buffer_.initialize(0.0);
//   for (unsigned int i = 0; i < widgets.size(); ++i) {
//     if (!widgets[i]->is_empty()) {
//       widgets[i]->rasterize(cm2_buffer_, false);
//     }
//  }
}



void
ViewSlices::fill_paint_slices(SliceWindow &window) {
  if (!gradient_.get_rep()) return;
  if (!cm2_.get_rep()) return;

  CM2Widgets &widgets = cm2_->widgets();

  Array3<float> cm2_buffer_(256, 256, 4);

  cm2_buffer_.initialize(0.0);
  for (unsigned int i = 0; i < widgets.size(); ++i) {
    if (!widgets[i]->is_empty()) {
      widgets[i]->rasterize(cm2_buffer_, false);
    }
  }

  unsigned int a, x, y;
  if (!window.gradient_slices_.size()) {
    for (a = 0; a < 3; ++a)
      window.gradient_slices_.push_back
	(scinew NrrdSlice(a, window.slice_num_(), scinew NrrdVolume(ctx)));
  }

  for (a = 0; a < 3; a++) {
    window.gradient_slices_[a]->do_lock();
    window.gradient_slices_[a]->nrrd_dirty_ = true;
    window.gradient_slices_[a]->mode_ = normal_e;
    window.gradient_slices_[a]->volume_->nrrd_ = gradient_;
    window.gradient_slices_[a]->slab_min_ = window.slab_min_();
    window.gradient_slices_[a]->slab_max_ = window.slab_max_();
    if (a == (unsigned int)window.axis_)
      extract_slice(*window.gradient_slices_[a], a, window.slice_num_());
    window.gradient_slices_[a]->do_unlock();	
  }


  if (!window.paint_slices_.size()) {
    for (a = 0; a < 3; ++a) {
      window.paint_slices_.push_back
	(scinew NrrdSlice(a, window.slice_num_(), 0));
      window.paint_slices_.back()->nrrd_ = scinew NrrdData;
      window.paint_slices_.back()->nrrd_->nrrd = nrrdNew();
      window.paint_slices_.back()->nrrd_->nrrd->data = 0;
      
    }
  }


  if (!window.new_paint_slices_.size()) {
    for (a = 0; a < 3; ++a) {
      window.new_paint_slices_.push_back
	(scinew NrrdSlice(a, window.slice_num_(), 0));
      window.new_paint_slices_.back()->nrrd_ = scinew NrrdData;
      window.new_paint_slices_.back()->nrrd_->nrrd = nrrdNew();
      window.new_paint_slices_.back()->nrrd_->nrrd->data = 0;
      
    }
  }



  for (a = 0; a < 3; ++a) {
    if (a != (unsigned int)window.axis_) continue;
    NrrdDataHandle &paint = window.paint_slices_[a]->nrrd_;
    if (!paint->nrrd->data) {
      nrrdAlloc(paint->nrrd, nrrdTypeFloat, 3, 4, 
		window.gradient_slices_[a]->tex_wid_,
		window.gradient_slices_[a]->tex_hei_);
    }
    NrrdDataHandle &newpaint = window.new_paint_slices_[a]->nrrd_;
    if (!newpaint->nrrd->data) {
      nrrdAlloc(newpaint->nrrd, nrrdTypeFloat, 3, 4, 
		window.gradient_slices_[a]->tex_wid_,
		window.gradient_slices_[a]->tex_hei_);
    }


    //    unsigned char *graddata = 
    //      (unsigned char *)window.gradient_slices_[a]->nrrd_->nrrd->data;
    float *paintdata = 
      (float *)window.paint_slices_[a]->nrrd_->nrrd->data;
    unsigned char val;
    int pos;
    double min = double(window.clut_wl_) - double(window.clut_ww_);
    double max = double(window.clut_wl_) + double(window.clut_ww_);
    double scale = 255.0 / (max - min);
    for (y = 0; y < window.gradient_slices_[a]->tex_hei_; ++y) {
      for (x = 0; x < window.gradient_slices_[a]->tex_wid_; ++x) {
	pos = y*window.gradient_slices_[a]->tex_wid_+x;
	switch (window.slices_[a]->nrrd_->nrrd->type) {
	case nrrdTypeChar: {
	  char *slicedata = (char *)(window.slices_[a]->nrrd_->nrrd->data);
	  val = Round(scale*(slicedata[pos] - min));
	}
	case nrrdTypeUChar: {
	  unsigned char *slicedata = (unsigned char *)(window.slices_[a]->nrrd_->nrrd->data);
	  val = Round(scale*(slicedata[pos] - min));
	}
	case nrrdTypeShort: {
	  short *slicedata = (short *)(window.slices_[a]->nrrd_->nrrd->data);
	  val = Round(scale*(slicedata[pos] - min));
	} break;
	case nrrdTypeUShort: { 	  
	  unsigned short*slicedata = (unsigned short*)(window.slices_[a]->nrrd_->nrrd->data);
	  val = Round(scale*(slicedata[pos] - min));
	} break;
	case nrrdTypeInt: {
	  int *slicedata = (int *)(window.slices_[a]->nrrd_->nrrd->data);
	  val = Round(scale*(slicedata[pos] - min));
	} break;
	case nrrdTypeUInt: {
	  unsigned int *slicedata = (unsigned int *)(window.slices_[a]->nrrd_->nrrd->data);
	  val = Round(scale*(slicedata[pos] - min));
	} break;
	case nrrdTypeLLong: {
	  long long  *slicedata = (long long *)(window.slices_[a]->nrrd_->nrrd->data);
	  val = Round(scale*(slicedata[pos] - min));
	} break;
	case nrrdTypeULLong: {
	  unsigned long long  *slicedata = (unsigned long long *)(window.slices_[a]->nrrd_->nrrd->data);
	  val = Round(scale*(slicedata[pos] - min));
	} break;
	case nrrdTypeFloat: {
	  float *slicedata = (float *)(window.slices_[a]->nrrd_->nrrd->data);
	  val = Round(scale*(slicedata[pos] - min));
	} break;
	case nrrdTypeDouble: {
	  double *slicedata = (double *)(window.slices_[a]->nrrd_->nrrd->data);
	  val = Round(scale*(slicedata[pos] - min));
	} break;
	default: break;
	}
	//	memcpy(paintdata+pos*4, &cm2_buffer_(graddata[pos], val,/* graddata[pos],*/ 0), 4*sizeof(float));
	memcpy(paintdata+pos*4, &cm2_buffer_(y, x, 0), 4*sizeof(float));
// 	paintdata[pos*4+0] = Round(*255.0);
// 	paintdata[pos*4+1] = Round(cm2_buffer_(val, graddata[pos], 1)*255.0);
// 	paintdata[pos*4+2] = Round(cm2_buffer_(val, graddata[pos], 2)*255.0);
// 	paintdata[pos*4+3] = Round(cm2_buffer_(val, graddata[pos], 3)*255.0);
	//	cerr << "v: " << int(val) << " g: " << int(graddata[pos]) << " d: " << Round(cm2_buffer_(val, graddata[pos], 3)*255.0) << "   - ";
      }
    }
  }
  
}


void
ViewSlices::do_paint(SliceWindow &window) {
  TCLTask::lock();

  NrrdSlice &ps = *window.paint_slices_[window.axis_];
  NrrdSlice &gs = *window.gradient_slices_[window.axis_];
  float *paintdata = 
    (float *)ps.nrrd_->nrrd->data;
  if (cursor_(0) >= 0 && cursor_(0) <= max_slice_[0] &&
      cursor_(1) >= 0 && cursor_(1) <= max_slice_[1] &&
      cursor_(2) >= 0 && cursor_(2) <= max_slice_[2]) 
  {
      
    unsigned int pos = 4*(Round(cursor_(y_axis(window)))*gs.tex_wid_+
			  Round(cursor_(x_axis(window))));
    
    paintdata[pos+0] = 1.0;
    paintdata[pos+1] = 1.0;
    paintdata[pos+2] = 1.0;
    paintdata[pos+3] = 1.0;
  }
    
  TCLTask::unlock();
}


void
ViewSlices::apply_paint(SliceWindow &window) {
#if 0
  TCLTask::lock();
  for (y = 0; y < ps->tex_hei_; ++y)  
    for (x = 0; x < ps->tex_hei_; ++x)
    {
      
      

  
  fill_paint_slices(window);
  TCLTask::unlock();

#endif 
}


  
  


} // End namespace SCIRun
