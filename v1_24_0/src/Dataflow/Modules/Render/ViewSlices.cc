
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

#include <Core/Exceptions/GuiException.h>
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

  struct SliceWindow;
  struct WindowLayout;

  struct NrrdSlice {
    NrrdSlice(NrrdVolume *, SliceWindow *);
    string		name_;
    NrrdVolume *	volume_;
    SliceWindow	*	window_;
    NrrdDataHandle      nrrd_;

    int			axis_;
    int			slice_num_;
    int			slab_min_;
    int			slab_max_;
    
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

    //    Mutex		lock_;
    //    Thread *	owner_;
    //    int		lock_count_;
    void		do_lock();
    void		do_unlock();
  };
  typedef vector<NrrdSlice *>		NrrdSlices;
  typedef vector<NrrdSlices>		NrrdVolumeSlices;

  struct SliceWindow { 
    SliceWindow(GuiContext *ctx);

    string		name_;
    WindowLayout *	layout_;
    OpenGLViewport *	viewport_;
    NrrdSlices		slices_;

    NrrdSlice		paint_under_;
    NrrdSlice		paint_;
    NrrdSlice		paint_over_;

    UIint		slice_num_;
    UIint		axis_;
    UIdouble		zoom_;
    UIint		slab_min_;
    UIint		slab_max_;
      
    UIdouble		x_;
    UIdouble		y_;

    bool		redraw_;
      
    UIint		auto_levels_;
    UIint		mode_;
    UIint		crosshairs_;
    UIint		snoop_;
    UIint		invert_;
    UIint		reverse_;
      
    UIint		mouse_x_;
    UIint		mouse_y_;
    UIint		show_guidelines_;
    bool		cursor_moved_;
    UIdouble		fusion_;
    
    int			cursor_pixmap_;

    GLdouble		gl_modelview_matrix_[16];
    GLdouble		gl_projection_matrix_[16];
    GLint		gl_viewport_[4];
  };
  typedef vector<SliceWindow *>	SliceWindows;

  struct WindowLayout {
    WindowLayout	(GuiContext *ctx);
    TkOpenGLContext *	opengl_;
    int			mouse_x_;
    int			mouse_y_;
    SliceWindows	windows_;
    string		name_;
  };
  typedef map<string, WindowLayout *>	WindowLayouts;

  typedef vector<NrrdVolume *>		NrrdVolumes;

  typedef vector<BBox>			PickBoxes;

  WindowLayouts		layouts_;
  NrrdVolumes		volumes_;
  NrrdDataHandle	gradient_;
  ColorMap2Handle	cm2_;


  vector<int>		nrrd_generations_;
  int			cm2_generation_;
  Array3<float>		cm2_buffer_under_;
  Array3<float>		cm2_buffer_;
  Array3<float>		cm2_buffer_over_;


  ColorMapHandle	colormap_;
  int			colormap_generation_;

  SliceWindow *		zooming_;
  SliceWindow *		panning_;
  double		original_zoom_;
  pair<double,double>	original_pan_;

  Point			cursor_;

  NrrdSlice *		mip_slices_[3];

  SliceWindow *		window_level_;
  UIdouble		clut_ww_;
  UIdouble		clut_wl_;
  double		original_ww_;
  double		original_wl_;

  int			pick_;
  int			pick_x_;
  int			pick_y_;
  SliceWindow *		pick_window_;
  BBox			crop_bbox_;
  BBox			crop_draw_bbox_;
  PickBoxes		crop_pick_boxes_;

  int			max_slice_[3];
  int			cur_slice_[3];
  int			slab_width_[3];
  double		scale_[3];
  double		center_[3];
  UIint			probe_;
  UIint			show_colormap2_;
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
  UIint			anatomical_coordinates_;
  UIint			show_text_;
  UIdouble		font_r_;
  UIdouble		font_g_;
  UIdouble		font_b_;
  UIdouble		font_a_;

  UIdouble		min_;
  UIdouble		max_;

  UIint			dim0_;
  UIint			dim1_;
  UIint			dim2_;
  UIint			geom_flushed_;

  UIdouble		background_threshold_;
  UIdouble		gradient_threshold_;

  PaintCM2Widget *	paint_widget_;
  Mutex			paint_lock_;

  float *		temp_tex_data_;

  //! Ports
  ColorMap2IPort*	cmap2_iport_;
  ColorMapIPort *	n1_cmap_iport_;
  ColorMapIPort *	n2_cmap_iport_;
  NrrdIPort *		grad_iport_;
  GeometryOPort *	geom_oport_;
  ColorMap2OPort*	cmap2_oport_;

  typedef		map<string,TexSquare*> TexSquareMap;
  TexSquareMap		tobjs_;
  map<string,GeomID>    gobjs_;

  FreeTypeLibrary *	freetype_lib_;
  typedef		map<string, FreeTypeFace *> FontMap;
  FontMap		fonts_;
  UIdouble		font_size_;
  
  RealDrawer *		runner_;
  Thread *		runner_thread_;
  CrowdMonitor		slice_lock_;

  // Methods for drawing to the GL window
  void			redraw_all();
  int			redraw_window(SliceWindow &);
  void			setup_gl_view(SliceWindow &);
  int			draw_slice(NrrdSlice &);
  void			bind_slice(NrrdSlice &,float *tex=0,bool filter=true);
  bool			bind_nrrd(Nrrd &);
  void			draw_slice_quad(NrrdSlice &);
  void			draw_guide_lines(SliceWindow &, float, float, float);
  void			draw_slice_lines(SliceWindow &);
  void			draw_slice_arrows(SliceWindow &);
  void			draw_dark_slice_regions(SliceWindow &);


  // Methods to render TrueType text labels
  void			initialize_fonts();
  void			delete_all_fonts();
  void			set_font_sizes(double size);
  void			draw_all_labels(SliceWindow &);
  void			draw_window_label(SliceWindow &);
  void			draw_orientation_labels(SliceWindow &);
  void			draw_position_label(SliceWindow &);
  void			draw_label(SliceWindow &, string, int, int, 
				   FreeTypeText::anchor_e, 
				   FreeTypeFace *font = 0); 

  // Crop Widget routines
  void			draw_crop_bbox(SliceWindow &, BBox &);
  PickBoxes		compute_crop_pick_boxes(SliceWindow &, BBox &);
  int			mouse_in_pick_boxes(SliceWindow &, PickBoxes &);
  void			set_window_cursor(SliceWindow &, int pick);
  pair<Vector, Vector>	get_crop_vectors(SliceWindow &, int pick);
  BBox			update_crop_bbox(SliceWindow &, int pick, int X,int Y);
  void			update_crop_bbox_from_gui();
  void			update_crop_bbox_to_gui();

  // Slice extraction and colormapping
  float *		apply_colormap(NrrdSlice &, float *);
  template <class T> 
  void			apply_colormap_to_raw_data(float *, T *, int, int,
						   const float *, int,
						   double, double);
  void			set_slice_coords(NrrdSlice &slice, bool origin);
  int			extract_window_slices(SliceWindow &);
  int			extract_slice(NrrdSlice &);
  int			extract_mip_slices(NrrdVolume *);

  // Methods to send geometry to Viewer
  int			send_mip_textures(SliceWindow &);
  int			send_slice_textures(NrrdSlice &slice);
  void			send_all_geometry();

  // Misc
  void			update_background_threshold();
  double		get_value(const Nrrd *, int, int, int);

  // Methods for navigating around the slices
  void			set_axis(SliceWindow &, unsigned int axis);
  void			next_slice(SliceWindow &);
  void			prev_slice(SliceWindow &);
  void			zoom_in(SliceWindow &);
  void			zoom_out(SliceWindow &);

  // Methods for cursor/coordinate projection and its inverse
  int			x_axis(SliceWindow &);
  int			y_axis(SliceWindow &);
  bool			mouse_in_window(SliceWindow &);
  Point			world_to_screen(SliceWindow &, Point &);
  Point			screen_to_world(SliceWindow &,
					unsigned int x, unsigned int y);

  // Methods called by tcl_command
  void			handle_gui_motion(GuiArgs &args);
  void			handle_gui_button(GuiArgs &args);
  void			handle_gui_button_release(GuiArgs &args);
  void			handle_gui_keypress(GuiArgs &args);
  void			handle_gui_enter(GuiArgs &args);
  void			handle_gui_leave(GuiArgs &args);

  // Methods for Painting into 2D Transfer Function
  void			undo_paint_stroke();
  int			extract_window_paint(SliceWindow &window);
  int			extract_current_paint(SliceWindow &window);
  void			apply_colormap2_to_slice(Array3<float> &, NrrdSlice &);
  bool			rasterize_colormap2();
  void			rasterize_widgets_to_cm2(int min, int max, Array3<float> &);
  void			do_paint(SliceWindow &);

  typedef int (SCIRun::ViewSlices::* SliceWindowFunc)(SliceWindow &);
  typedef int (SCIRun::ViewSlices::* WindowLayoutFunc)(WindowLayout &);
  typedef int (SCIRun::ViewSlices::* NrrdSliceFunc)(NrrdSlice &);
  template <class T> int			for_each(T);
  int			for_each(SliceWindow &, NrrdSliceFunc);
  int			for_each(WindowLayout &, NrrdSliceFunc);
  int			for_each(WindowLayout &, SliceWindowFunc);

  int			render_window(SliceWindow &);
  int			swap_window(SliceWindow &);
  int			set_probe(SliceWindow &);
  int			set_paint_dirty(SliceWindow &);
  int			autoview(SliceWindow &);

  int			setup_slice_nrrd(NrrdSlice &);
  int			rebind_slice(NrrdSlice &);
  int			set_slice_nrrd_dirty(NrrdSlice &);
  int			update_slice_from_window(NrrdSlice &);


public:
  ViewSlices(GuiContext* ctx);
  virtual ~ViewSlices();
  virtual void		execute();
  virtual void		tcl_command(GuiArgs& args, void*);
  void			real_draw_all();
  double		fps_;
  WindowLayout *	current_layout_;
  int			executing_;
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

#define NRRD_EXEC(__nrrd_command__) \
  if (__nrrd_command__) { \
    char *err = biffGetDone(NRRD); \
    error(string("Error on line #")+to_string(__LINE__) + \
	  string(" executing nrrd command: \n")+ \
          string(#__nrrd_command__)+string("\n")+ \
          string("Message: ")+string(err)); \
    free(err); \
    return 0; \
  }


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
  int frames = -1;
  while (!dead_) {
    if (!module_->current_layout_) throttle_.wait_for_time(t);
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
    
    while (t <= t2) t += 1.0/30.0;
  }
}

int
ViewSlices::for_each(SliceWindow &window, NrrdSliceFunc func) {
  int value = 0;
  for (unsigned int slice = 0; slice < window.slices_.size(); ++slice) {
    ASSERT(window.slices_[slice]);
    value += (this->*func)(*window.slices_[slice]);
  }
  return value;
}

int
ViewSlices::for_each(WindowLayout &layout, NrrdSliceFunc func) {
  int value = 0;
  for (unsigned int window = 0; window < layout.windows_.size(); ++window) {
    ASSERT(layout.windows_[window]);
    value += for_each(*layout.windows_[window], func);
  }
  return value;
}
  
int
ViewSlices::for_each(WindowLayout &layout, SliceWindowFunc func)
{
  int value = 0;
  for (unsigned int window = 0; window < layout.windows_.size(); ++window) {
    ASSERT(layout.windows_[window]);
    value += (this->*func)(*layout.windows_[window]);
  }
  return value;
}

template <class T>
int			
ViewSlices::for_each(T func) {
  int value = 0;
  WindowLayouts::iterator liter = layouts_.begin(), lend = layouts_.end();
  for (; liter != lend; ++liter) {
    ASSERT(liter->second);
    value += for_each(*(liter->second), func);
  }
  return value;
}


ViewSlices::NrrdSlice::NrrdSlice(NrrdVolume *volume, SliceWindow *window) :
  name_("INVALID"),
  volume_(volume),
  window_(window),
  nrrd_(0),
  axis_(0),
  slice_num_(0),
  slab_min_(0),
  slab_max_(0),
  nrrd_dirty_(true),
  tex_dirty_(false),
  geom_dirty_(false),
  mode_(0),
  tex_wid_(0),
  tex_hei_(0),
  wid_(0),
  hei_(0),
  opacity_(1.0),
  tex_name_(0)
  //lock_("Slice lock"),
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
  layout_(0),
  viewport_(0),
  slices_(),
  paint_under_(0, this),
  paint_(0, this),
  paint_over_(0, this),
  slice_num_(ctx->subVar("slice"), 0),
  axis_(ctx->subVar("axis"), 2),
  zoom_(ctx->subVar("zoom"), 100.0),
  slab_min_(ctx->subVar("slab_min"), 0),
  slab_max_(ctx->subVar("slab_max"), 0),
  x_(ctx->subVar("posx"),0.0),
  y_(ctx->subVar("posy"),0.0),
  redraw_(true),
  auto_levels_(ctx->subVar("auto_levels"),0),
  mode_(ctx->subVar("mode"),0),
  crosshairs_(ctx->subVar("crosshairs"),0),
  snoop_(ctx->subVar("snoop"),0),
  invert_(ctx->subVar("invert"),0),
  reverse_(ctx->subVar("reverse"),0),
  mouse_x_(ctx->subVar("mouse_x"),0),
  mouse_y_(ctx->subVar("mouse_y"),0),
  show_guidelines_(ctx->subVar("show_guidelines"),1),
  cursor_moved_(true),
  fusion_(ctx->subVar("fusion"), 1.0),
  cursor_pixmap_(-1)
{
  paint_under_.name_ = "Paint Under";
  paint_.name_ = "Paint";
  paint_over_.name_ = "Paint Over";

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
  nrrd_generations_(),
  cm2_generation_(-1),
  cm2_buffer_under_(256, 256, 4),
  cm2_buffer_(256, 256, 4),
  cm2_buffer_over_(256, 256, 4),
  colormap_(0),
  colormap_generation_(-1),
  zooming_(0),
  panning_(0),
  cursor_(0.0, 0.0, 0.0),
  window_level_(0),
  clut_ww_(ctx->subVar("clut_ww"), 1.0),
  clut_wl_(ctx->subVar("clut_wl"), 0.0),
  pick_(0),
  pick_x_(0),
  pick_y_(0),
  pick_window_(0),
  crop_bbox_(Point(0, 0, 0), Point(1, 1, 1)),
  crop_draw_bbox_(crop_bbox_),
  crop_pick_boxes_(),
  probe_(ctx->subVar("probe"),0),
  show_colormap2_(ctx->subVar("show_colormap2"),0),
  painting_(ctx->subVar("painting"),0),
  crop_(ctx->subVar("crop"),0),
  crop_min_x_(ctx->subVar("crop_minAxis0"),0),
  crop_min_y_(ctx->subVar("crop_minAxis1"),0),
  crop_min_z_(ctx->subVar("crop_minAxis2"),0),
  crop_max_x_(ctx->subVar("crop_maxAxis0"),0),
  crop_max_y_(ctx->subVar("crop_maxAxis1"),0),
  crop_max_z_(ctx->subVar("crop_maxAxis2"),0),
  crop_min_pad_x_(ctx->subVar("crop_minPadAxis0"),0),
  crop_min_pad_y_(ctx->subVar("crop_minPadAxis1"),0),
  crop_min_pad_z_(ctx->subVar("crop_minPadAxis2"),0),
  crop_max_pad_x_(ctx->subVar("crop_maxPadAxis0"),0),
  crop_max_pad_y_(ctx->subVar("crop_maxPadAxis1"),0),
  crop_max_pad_z_(ctx->subVar("crop_maxPadAxis2"),0),
  texture_filter_(ctx->subVar("texture_filter"),1),
  anatomical_coordinates_(ctx->subVar("anatomical_coordinates"), 1),
  show_text_(ctx->subVar("show_text"), 1),
  font_r_(ctx->subVar("color_font-r"), 1.0),
  font_g_(ctx->subVar("color_font-g"), 1.0),
  font_b_(ctx->subVar("color_font-b"), 1.0),
  font_a_(ctx->subVar("color_font-a"), 1.0),
  min_(ctx->subVar("min"), -1.0),
  max_(ctx->subVar("max"), -1.0),
  dim0_(ctx->subVar("dim0"), 0),
  dim1_(ctx->subVar("dim1"), 0),
  dim2_(ctx->subVar("dim2"), 0),
  geom_flushed_(ctx->subVar("geom_flushed"), 0),
  background_threshold_(ctx->subVar("background_threshold"), 0.0),
  gradient_threshold_(ctx->subVar("gradient_threshold"), 0.0),
  paint_widget_(0),
  paint_lock_("ViewSlices paint lock"),
  temp_tex_data_(0),
  cmap2_iport_((ColorMap2IPort*)get_iport("InputColorMap2")),
  n1_cmap_iport_((ColorMapIPort *)get_iport("Nrrd1ColorMap")),
  n2_cmap_iport_((ColorMapIPort *)get_iport("Nrrd2ColorMap")),
  grad_iport_((NrrdIPort*)get_iport("NrrdGradient")),
  geom_oport_((GeometryOPort *)get_oport("Geometry")),
  cmap2_oport_((ColorMap2OPort*)get_oport("ColorMap2")),
  freetype_lib_(0),
  fonts_(),
  font_size_(ctx->subVar("font_size"),15.0),
  runner_(0),
  runner_thread_(0),
  slice_lock_("Slice lock"),
  fps_(0.0),
  current_layout_(0),
  executing_(0)
{
  nrrd_generations_.resize(2);
  nrrd_generations_[0] = -1;
  nrrd_generations_[1] = -1;

  for (int a = 0; a < 3; ++a) {
    mip_slices_[a] = 0;
    max_slice_[a] = -1;
  }

  runner_ = scinew RealDrawer(this);
  runner_thread_ = scinew Thread(runner_, string(id+" OpenGL drawer").c_str());

  initialize_fonts();
}

ViewSlices::~ViewSlices()
{
  if (runner_thread_) {
    runner_->dead_ = true;
    runner_thread_->join();
    runner_thread_ = 0;
  }
}


int
ViewSlices::render_window(SliceWindow &window) {
  if (!window.redraw_) return 0;
  window.redraw_ = false;
  
  window.viewport_->make_current();
  window.viewport_->clear();
  setup_gl_view(window);
  GL_ERROR();
  
  if (window.cursor_moved_)
  {
    cursor_ = screen_to_world(window, window.mouse_x_, window.mouse_y_);
    window.cursor_moved_ = false;
  }
  
  for_each(window, &ViewSlices::draw_slice);

  //  draw_dark_slice_regions(window);
  draw_slice_lines(window);
  //  draw_slice_arrows(window);
  draw_guide_lines(window, cursor_.x(), cursor_.y(), cursor_.z());
  
  if (crop_) {
    draw_crop_bbox(window, crop_draw_bbox_);
    crop_pick_boxes_ = compute_crop_pick_boxes(window,crop_draw_bbox_);
    set_window_cursor(window,
		      mouse_in_pick_boxes(window, crop_pick_boxes_));
  } else {
    set_window_cursor(window, 0);
  }
  draw_all_labels(window);
  window.viewport_->release();
  GL_ERROR();
  return 1;
}

int
ViewSlices::swap_window(SliceWindow &window) {
  window.viewport_->make_current();
  window.viewport_->swap();
  window.viewport_->release();
  return 1;
}


void
ViewSlices::real_draw_all()
{
  TCLTask::lock();
  //  for_each(&ViewSlices::extract_slice);
  if (for_each(&ViewSlices::render_window)) 
    for_each(&ViewSlices::swap_window);
  TCLTask::unlock();
  send_all_geometry();
}

  
void
ViewSlices::send_all_geometry()
{
  if (window_level_) return;
  TCLTask::lock();
  slice_lock_.writeLock();
  int flush = for_each(&ViewSlices::send_mip_textures);
  flush += for_each(&ViewSlices::send_slice_textures);  
  slice_lock_.writeUnlock();
  TCLTask::unlock();
  if (flush) 
  {
    geom_oport_->flushViewsAndWait();
    geom_flushed_ = 1;
  }
}


void
ViewSlices::redraw_all()
{
  for_each(&ViewSlices::redraw_window);
}

int
ViewSlices::redraw_window(SliceWindow &window) {
  window.redraw_ = true;
  return 1;
}

// draw_guide_lines
// renders vertical and horizontal bars that represent
// selected slices in other dimensions
// if x, y, or z < 0, then that dimension wont be rendered
void
ViewSlices::draw_guide_lines(SliceWindow &window, float x, float y, float z) {
  if (!window.show_guidelines_()) return;
  if (!mouse_in_window(window)) return;
  Vector tmp = screen_to_world(window, 1, 0) - screen_to_world(window, 0, 0);
  tmp[window.axis_] = 0;
  const float one = Max(fabs(tmp[0]), fabs(tmp[1]), fabs(tmp[2]));

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  GLdouble yellow[4] = { 1.0, 0.76, 0.1, 0.8 };
  GLdouble white[4] = { 1.0, 1.0, 1.0, 1.0 };

  const int axis = window.axis_;
  int p = (axis+1)%3;
  int s = (axis+2)%3;

  double c[3];
  c[0] = x;
  c[1] = y;
  c[2] = z;
  glColor4dv(yellow);
  for (int i = 0; i < 2; ++i) {
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
  if (max_slice_[window.axis_] <= 0) return;
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
  GL_ERROR();
}




// renders vertical and horizontal bars that represent
// selected slices in other dimensions
void
ViewSlices::draw_slice_arrows(SliceWindow &window)
{
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
  double min, max;
  double xyz[3];
  int i;



  for (int i = 0; i < 3; ++i)
    xyz[i] = cur_slice_[i]*scale_[i];
  
  for (i = 0; i < 2; ++i) {
    if (!slab_width_[p]) continue;
    one = Max(screen_space_one, double(scale_[p]*slab_width_[p]));
    
    
    min = double(scale_[p]*window.slab_min_);
    max = double(scale_[p]*window.slab_max_);
    
    if (fabs(max-min) < screen_space_one) {
      double mid = (min+max)/2.0;
      min = mid-screen_space_one/2.0;
      max = mid+screen_space_one/2.0;
    }

    switch (p) {
    case 0: glColor4dv(red); break;
    case 1: glColor4dv(green); break;
    default:
    case 2: glColor4dv(blue); break;
    }

    glBegin(GL_QUADS);    

    glVertex3f(p==0?xyz[0]:0.0, p==1?xyz[1]:0.0, p==2?xyz[2]:0.0);
    glVertex3f(p==0?xyz[0]+one:0.0, p==1?xyz[1]+one:0.0, p==2?xyz[2]+one:0.0);
    glVertex3f(p==0?xyz[0]+one:(axis==0?0.0:(max_slice_[s]+1)*scale_[s]),
	       p==1?xyz[1]+one:(axis==1?0.0:(max_slice_[s]+1)*scale_[s]),
	       p==2?xyz[2]+one:(axis==2?0.0:(max_slice_[s]+1)*scale_[s]));
    
    glVertex3f(p==0?xyz[0]:(axis==0?0.0:(max_slice_[s]+1)*scale_[s]),
	       p==1?xyz[1]:(axis==1?0.0:(max_slice_[s]+1)*scale_[s]),
	       p==2?xyz[2]:(axis==2?0.0:(max_slice_[s]+1)*scale_[s]));
    glEnd();
    SWAP(p,s);
  }
  GL_ERROR();
}



void
ViewSlices::draw_crop_bbox(SliceWindow &window, BBox &bbox) {
  if (!crop_) return;
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
  if (0) {
    crop_min_x_ = Clamp(crop_min_x_(), 0, max_slice_[0]);
    crop_min_y_ = Clamp(crop_min_y_(), 0, max_slice_[1]);
    crop_min_z_ = Clamp(crop_min_z_(), 0, max_slice_[2]);
    crop_max_x_ = Clamp(crop_max_x_(), 0, max_slice_[0]);
    crop_max_y_ = Clamp(crop_max_y_(), 0, max_slice_[1]);
    crop_max_z_ = Clamp(crop_max_z_(), 0, max_slice_[2]);
  }

  crop_draw_bbox_ = 
    BBox(Point(double(Min(crop_min_x_(), crop_max_x_())),
	       double(Min(crop_min_y_(), crop_max_y_())),
	       double(Min(crop_min_z_(), crop_max_z_()))),
	 Point(double(Max(crop_min_x_(), crop_max_x_())+1),
	       double(Max(crop_min_y_(), crop_max_y_())+1),
	       double(Max(crop_min_z_(), crop_max_z_())+1)));
  crop_bbox_ = crop_draw_bbox_;
}

void
ViewSlices::update_crop_bbox_to_gui() 
{
  crop_min_x_ = int(crop_draw_bbox_.min().x());
  crop_min_y_ = int(crop_draw_bbox_.min().y());
  crop_min_z_ = int(crop_draw_bbox_.min().z());
  crop_max_x_ = int(crop_draw_bbox_.max().x()-1.0);
  crop_max_y_ = int(crop_draw_bbox_.max().y()-1.0);
  crop_max_z_ = int(crop_draw_bbox_.max().z()-1.0);
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
  if (!mouse_in_window(window) || window.cursor_pixmap_ == cursor) return;
  window.cursor_pixmap_ = cursor;
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
  for (int n = 0; n < 3; n++)
    uimaxpad[n] += uiminpad[n];
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
    if (min(i) < -uiminpad[i]) {
      min(i) = -uiminpad[i];
    }
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
  if ((adir[0] > adir[1]) && (adir[0] > adir[2])) return 0;
  if ((adir[1] > adir[0]) && (adir[1] > adir[2])) return 1;
  return 2;

  //  int m = adir[0] > adir[1] ? 0 : 1;
  //  m = adir[m] > adir[2] ? m : 2;
  //  return dir[m] < 0.0 ? m+3 : m;
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
  if ((adir[0] > adir[1]) && (adir[0] > adir[2])) return 0;
  if ((adir[1] > adir[0]) && (adir[1] > adir[2])) return 1;
  return 2;
  //  int m = adir[0] > adir[1] ? 0 : 1;
  //  m = adir[m] > adir[2] ? m : 2;
  //  return dir[m] < 0.0 ? m+3 : m;
}

void
ViewSlices::setup_gl_view(SliceWindow &window)
{
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  GL_ERROR();

  int axis = window.axis_;

  if (axis == 0) { // screen +X -> +Y, screen +Y -> +Z
    glRotated(-90,0.,1.,0.);
    glRotated(-90,1.,0.,0.);
  } else if (axis == 1) { // screen +X -> +X, screen +Y -> +Z
    glRotated(-90,1.,0.,0.);
  }
  
  glTranslated((axis==0)?-double(window.slice_num_)*scale_[0]:0.0,
  	       (axis==1)?-double(window.slice_num_)*scale_[1]:0.0,
  	       (axis==2)?-double(window.slice_num_)*scale_[2]:0.0);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glGetDoublev(GL_MODELVIEW_MATRIX, window.gl_modelview_matrix_);
  glGetDoublev(GL_PROJECTION_MATRIX, window.gl_projection_matrix_);
  glGetIntegerv(GL_VIEWPORT, window.gl_viewport_);
  
  double hwid = (window.viewport_->width()/(window.zoom_()/100.0))/2.0;
  double hhei = (window.viewport_->height()/(window.zoom_()/100.0))/2.0;
  
  double cx = double(*window.x_) + center_[x_axis(window) % 3];
  double cy = double(*window.y_) + center_[y_axis(window) % 3];

  double minz = -max_slice_[axis]*scale_[axis];
  double maxz = max_slice_[axis]*scale_[axis];
  if (fabs(minz - maxz) < 0.01) {
    minz = -1.0;
    maxz = 1.0;
  }
  glOrtho(cx - hwid, cx + hwid, cy - hhei, cy + hhei, minz, maxz);
  glGetDoublev(GL_PROJECTION_MATRIX, window.gl_projection_matrix_);
  GL_ERROR();
}


void
ViewSlices::draw_window_label(SliceWindow &window)
{
  string text;
  if (anatomical_coordinates_()) { 
    switch (window.axis_) {
    case 0: text = "Sagittal"; break;
    case 1: text = "Coronal"; break;
    default:
    case 2: text = "Axial"; break;
    }
  } else {
    switch (window.axis_) {
    case 0: text = "YZ Plane"; break;
    case 1: text = "XZ Plane"; break;
    default:
    case 2: text = "XY Plane"; break;
    }
  }

  if (window.mode_ == slab_e) text = "SLAB - "+text;
  if (window.mode_ == mip_e) text = "MIP - "+text;
  draw_label(window, text, window.viewport_->width() - 2, 0, 
	     FreeTypeText::se, fonts_["view"]);

  if (string(sci_getenv("USER")) == string("mdavis"))
    draw_label(window, "fps: "+to_string(fps_), 
	       0, window.viewport_->height() - 2,
	       FreeTypeText::nw, fonts_["default"]);
}


void
ViewSlices::draw_position_label(SliceWindow &window)
{
  FreeTypeFace *font = fonts_["default"];
  if (!font) return;
  
  const string zoom_text = "Zoom: "+to_string(window.zoom_())+"%";
  string position_text;
  if (anatomical_coordinates_()) {
    position_text = ("S: "+to_string(Floor(cursor_.x()/scale_[0]))+
		     " C: "+to_string(Floor(cursor_.y()/scale_[1]))+
		     " A: "+to_string(Floor(cursor_.z()/scale_[2])));
  } else {
    position_text = ("X: "+to_string(Ceil(cursor_.x()))+
		     " Y: "+to_string(Ceil(cursor_.y()))+
		     " Z: "+to_string(Ceil(cursor_.z())));
  }
    
  FreeTypeText position(position_text, font);
  BBox bbox;
  position.get_bounds(bbox);
  int y_pos = Ceil(bbox.max().y())+2;
  draw_label(window, position_text, 0, 0, FreeTypeText::sw, font);
  draw_label(window, zoom_text, 0, y_pos, FreeTypeText::sw, font);
}  


// Right	= -X
// Left		= +X
// Posterior	= -Y
// Anterior	= +X
// Inferior	= -Z
// Superior	= +Z
void
ViewSlices::draw_orientation_labels(SliceWindow &window)
{
  FreeTypeFace *font = fonts_["orientation"];
  if (!font) return;

  int prim = x_axis(window);
  int sec = y_axis(window);
  
  string ltext, rtext, ttext, btext;

  if (anatomical_coordinates_()) {
    switch (prim % 3) {
    case 0: ltext = "R"; rtext = "L"; break;
    case 1: ltext = "P"; rtext = "A"; break;
    default:
    case 2: ltext = "I"; rtext = "S"; break;
    }
    
    switch (sec % 3) {
    case 0: btext = "R"; ttext = "L"; break;
    case 1: btext = "P"; ttext = "A"; break;
    default:
    case 2: btext = "I"; ttext = "S"; break;
    }
  } else {
    switch (prim % 3) {
    case 0: ltext = "-X"; rtext = "+X"; break;
    case 1: ltext = "-Y"; rtext = "+Y"; break;
    default:
    case 2: ltext = "-Z"; rtext = "+Z"; break;
    }
    switch (sec % 3) {
    case 0: btext = "-X"; ttext = "+X"; break;
    case 1: btext = "-Y"; ttext = "+Y"; break;
    default:
    case 2: btext = "-Z"; ttext = "+Z"; break;
    }
  }    


  if (prim >= 3) SWAP (ltext, rtext);
  if (sec >= 3) SWAP (ttext, btext);

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

  unsigned int wid = Pow2(Ceil(bbox.max().x()));
  unsigned int hei = Pow2(Ceil(bbox.max().y()));
  GLubyte *buf = scinew GLubyte[wid*hei];
  memset(buf, 0, wid*hei);
  fttext.render(wid, hei, buf);
  
  glEnable(GL_TEXTURE_2D);
  GLuint tex_id;
  glGenTextures(1, &tex_id);
  
  glBindTexture(GL_TEXTURE_2D, tex_id);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1); 
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_BLEND);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  
  glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, wid, hei, 0, 
	       GL_ALPHA, GL_UNSIGNED_BYTE, buf);
  delete [] buf;
  
  double px = x;
  double py = y;

  switch (anchor) {
  case FreeTypeText::s:  px -= bbox.max().x()/2.0; break;
  case FreeTypeText::se: px -= bbox.max().x();     break;
  case FreeTypeText::e:  px -= bbox.max().x();     
                         py -= bbox.max().y()/2.0; break;
  case FreeTypeText::ne: px -= bbox.max().x();     
                         py -= bbox.max().y();     break;
  case FreeTypeText::n:  px -= bbox.max().x()/2.0; 
                         py -= bbox.max().y();     break;
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
  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_TEXTURE_2D);
}  


template <class T>
void
ViewSlices::apply_colormap_to_raw_data(float *data, T *slicedata, 
				       int wid, int hei, 
				       const float *rgba, int ncolors,
				       double offset, double scale)
{
  const int sizeof3floats = 3*sizeof(float);
  int val;
  double fval;
  const int resolution = ncolors-1;
  for (int pos = 0; pos < wid*hei; ++pos) {
    fval = scale*(slicedata[pos] + offset);
    val = Clamp(Round(fval*resolution), 0, resolution);
    memcpy(data+pos*4, rgba+val*4, sizeof3floats);
    data[pos*4+3] = Clamp(fval, 0.0, 1.0);
  }
}


float *
ViewSlices::apply_colormap(NrrdSlice &slice, float *data) 
{
  const double min = clut_wl_ - clut_ww_/2.0;
  const double max = clut_wl_ + clut_ww_/2.0;
  void *slicedata = slice.nrrd_->nrrd->data;
  const int wid = slice.tex_wid_, hei = slice.tex_hei_;
  const double scale = 1.0/double(max-min);
  int ncolors;  
  const float *rgba;
  if (!colormap_.get_rep()) {
    ncolors = 256;
    float *nrgba = scinew float[256*4];
    for (int c = 0; c < 256*4; ++c) nrgba[c] = (c/4)/255.0;
    rgba = nrgba;
  } else {
    ncolors = colormap_->resolution();
    rgba = colormap_->get_rgba();
  }

  switch (slice.nrrd_->nrrd->type) {
  case nrrdTypeChar: {
    apply_colormap_to_raw_data(data, (char *)slicedata, wid, hei,
			       rgba, ncolors, -min, scale);
  } break;
  case nrrdTypeUChar: {
    apply_colormap_to_raw_data(data, (unsigned char *)slicedata, wid, hei,
			       rgba, ncolors, -min, scale);
  } break;
  case nrrdTypeShort: {
    apply_colormap_to_raw_data(data, (short *)slicedata, wid, hei,
			       rgba, ncolors, -min, scale);
  } break;
  case nrrdTypeUShort: {
    apply_colormap_to_raw_data(data, (unsigned short *)slicedata, wid, hei,
			       rgba, ncolors, -min, scale);
  } break;
  case nrrdTypeInt: {
    apply_colormap_to_raw_data(data, (int *)slicedata, wid, hei,
			       rgba, ncolors, -min, scale);
  } break;
  case nrrdTypeUInt: {
    apply_colormap_to_raw_data(data, (unsigned int *)slicedata, wid, hei,
			       rgba, ncolors, -min, scale);
  } break;
  case nrrdTypeLLong: {
    apply_colormap_to_raw_data(data, (signed long long *)slicedata, wid, hei,
			       rgba, ncolors, -min, scale);
  } break;
  case nrrdTypeULLong: {
    apply_colormap_to_raw_data(data, (unsigned long long *)slicedata, wid, hei,
			       rgba, ncolors, -min, scale);
  } break;
  case nrrdTypeFloat: {
    apply_colormap_to_raw_data(data, (float *)slicedata, wid, hei,
			       rgba, ncolors, -min, scale);
  } break;
  case nrrdTypeDouble: {
    apply_colormap_to_raw_data(data, (double *)slicedata, wid, hei,
			       rgba, ncolors, -min, scale);
  } break;
  default: error("Unsupported data type: "+slice.nrrd_->nrrd->type);
  }

  if (!colormap_.get_rep())
    delete[] rgba;
    
  return data;
}


double
ViewSlices::get_value(const Nrrd *nrrd, int x, int y, int z) {
  ASSERT(nrrd->dim == 3);
  const int position = nrrd->axis[0].size*(z*nrrd->axis[1].size+y)+x;
  switch (nrrd->type) {
  case nrrdTypeChar: {
    char *slicedata = (char *)nrrd->data;
    return (double)slicedata[position];
    break;
  }
  case nrrdTypeUChar: {
    unsigned char *slicedata = (unsigned char *)nrrd->data;
    return (double)slicedata[position];
    break;
  }
  case nrrdTypeShort: {
    short *slicedata = (short *)nrrd->data;
    return (double)slicedata[position];
    break;
  }
  case nrrdTypeUShort: {
    unsigned short *slicedata = (unsigned short *)nrrd->data;
    return (double)slicedata[position];
    break;
  }
  case nrrdTypeInt: {
    int *slicedata = (int *)nrrd->data;
    return (double)slicedata[position];
    break;
  }
  case nrrdTypeUInt: {
    unsigned int *slicedata = (unsigned int *)nrrd->data;
    return (double)slicedata[position];
    break;
  }
  case nrrdTypeLLong: {
    signed long long *slicedata = (signed long long *)nrrd->data;
    return (double)slicedata[position];
    break;
  }
  case nrrdTypeULLong: {
    unsigned long long *slicedata = (unsigned long long *)nrrd->data;
    return (double)slicedata[position];
    break;
  }
  case nrrdTypeFloat: {
    float *slicedata = (float *)nrrd->data;
    return (double)slicedata[position];
    break;
  }
  case nrrdTypeDouble: {
    double *slicedata = (double *)nrrd->data;
    return (double)slicedata[position];
    break;
  }
  default: error("Unsupported data type: "+nrrd->type);
  }
  return 0.0;
}


bool
ViewSlices::bind_nrrd(Nrrd &nrrd) {
  int prim = 1;
  GLenum pixtype;
  if (nrrd.axis[0].size == 1) 
    pixtype = GL_ALPHA;
  if (nrrd.axis[0].size == 2) 
    pixtype = GL_LUMINANCE_ALPHA;
  else if (nrrd.axis[0].size == 3) 
    pixtype = GL_RGB;
  else if (nrrd.axis[0].size == 4) 
    pixtype = GL_RGBA;
  else {
    prim = 0;
    pixtype = GL_ALPHA;
  }
  GLenum type = 0;
  switch (nrrd.type) {
  case nrrdTypeChar: type = GL_BYTE; break;
  case nrrdTypeUChar: type = GL_UNSIGNED_BYTE; break;
  case nrrdTypeShort: type = GL_SHORT; break;
  case nrrdTypeUShort: type = GL_UNSIGNED_SHORT; break;	
  case nrrdTypeInt: type = GL_INT; break;
  case nrrdTypeUInt: type = GL_UNSIGNED_INT; break;
  case nrrdTypeFloat: type = GL_FLOAT; break;
  default: error("Cant bind nrrd"); return false; break;
  }
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
	       nrrd.axis[prim].size, nrrd.axis[prim+1].size, 
	       0, pixtype, type, nrrd.data);
  return true;
}
  

void
ViewSlices::bind_slice(NrrdSlice &slice, float *tex, bool filter)
{
  const bool bound = glIsTexture(slice.tex_name_);
  if (!bound) {
    glGenTextures(1, &slice.tex_name_);
  }

  glBindTexture(GL_TEXTURE_2D, slice.tex_name_);

  if (!bound || slice.tex_dirty_) {
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);  
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    const GLint filter_mode=((filter&&texture_filter_())?GL_LINEAR:GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter_mode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter_mode);
    if (tex)
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, slice.tex_wid_, slice.tex_hei_, 
		   0, GL_RGBA, GL_FLOAT, tex);
    else
      bind_nrrd(*slice.nrrd_->nrrd);
    
    slice.tex_dirty_ = false;
    slice.geom_dirty_ = true;
  }
}


void
ViewSlices::draw_slice_quad(NrrdSlice &slice) {
  unsigned int i;
  glBegin( GL_QUADS );
  for (i = 0; i < 4; i++) {
    glTexCoord2fv(&slice.tex_coords_[i*2]);
    glVertex3fv(&slice.pos_coords_[i*3]);
  }
  glEnd();
}
  

int
ViewSlices::draw_slice(NrrdSlice &slice)
{
  extract_slice(slice); // Returns immediately if slice is current
  ASSERT(slice.window_);  
  ASSERT(slice.axis_ == slice.window_->axis_);
  ASSERT(slice.nrrd_.get_rep());

  slice.do_lock();

  // Setup the opacity of the slice to be drawn
  GLfloat opacity = slice.opacity_*slice.window_->fusion_();
  if (volumes_.size() == 2 && 
      slice.volume_->nrrd_.get_rep() == volumes_[1]->nrrd_.get_rep()) {
    opacity = slice.opacity_*(1.0 - slice.window_->fusion_); 
  }
  glColor4f(opacity, opacity, opacity, opacity);

  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glEnable(GL_TEXTURE_2D);
  glEnable(GL_BLEND);
  glEnable(GL_COLOR_MATERIAL);
  glBlendFunc(GL_ONE, GL_ONE); 
 
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glShadeModel(GL_FLAT);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  GLfloat ones[4] = {1.0, 1.0, 1.0, 1.0};
  glTexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, ones);

  if (slice.tex_dirty_)
    apply_colormap(slice, temp_tex_data_);
  
  bind_slice(slice, temp_tex_data_);
  draw_slice_quad(slice);

  if ((painting_ || show_colormap2_) && (slice.mode_ == normal_e) &&
      gradient_.get_rep() && cm2_.get_rep())
  {
    extract_window_paint(*slice.window_);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    bind_slice(slice.window_->paint_under_, 0, 0);
    draw_slice_quad(slice);
    bind_slice(slice.window_->paint_, 0, 0);
    draw_slice_quad(slice);
    bind_slice(slice.window_->paint_over_, 0, 0);
    draw_slice_quad(slice);
  }
  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_TEXTURE_2D);
  glDisable(GL_BLEND);
  GL_ERROR();  
  slice.do_unlock();

  return 1;
}



void
ViewSlices::draw_all_labels(SliceWindow &window) {
  if (!show_text_) return;
  glColor4d(font_r_, font_g_, font_b_, font_a_);
  draw_position_label(window);
  draw_orientation_labels(window);
  draw_window_label(window);

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
  

  if (origin) {
    if (axis == 0) {
      x_pos = slice_num+0.5;
      y_wid = slice.volume_->nrrd_->nrrd->axis[1].size;
      z_hei = slice.volume_->nrrd_->nrrd->axis[2].size; 
    } else if (axis == 1) {
      y_pos = slice_num+0.5;
      x_wid = slice.volume_->nrrd_->nrrd->axis[0].size;
      z_hei = slice.volume_->nrrd_->nrrd->axis[2].size;
    } else /*if (axis == 2)*/ {
      z_pos = slice_num+0.5;
      x_wid = slice.volume_->nrrd_->nrrd->axis[0].size;
      y_hei = slice.volume_->nrrd_->nrrd->axis[1].size;
    }
  } else {
    if (axis == 0) {
      x_pos = (slice_num+0.5)*(slice.volume_->nrrd_->nrrd->axis[0].max - 
			       slice.volume_->nrrd_->nrrd->axis[0].min)/
	slice.volume_->nrrd_->nrrd->axis[0].size;
      y_wid = slice.volume_->nrrd_->nrrd->axis[1].max - 
	slice.volume_->nrrd_->nrrd->axis[1].min;
      z_hei = slice.volume_->nrrd_->nrrd->axis[2].max - 
	slice.volume_->nrrd_->nrrd->axis[2].min;
    } else if (axis == 1) {
      y_pos = (slice_num+0.5)*(slice.volume_->nrrd_->nrrd->axis[1].max - 
			       slice.volume_->nrrd_->nrrd->axis[1].min)/
	slice.volume_->nrrd_->nrrd->axis[1].size;

      x_wid = slice.volume_->nrrd_->nrrd->axis[0].max - 
	slice.volume_->nrrd_->nrrd->axis[0].min;
      z_hei = slice.volume_->nrrd_->nrrd->axis[2].max - 
	slice.volume_->nrrd_->nrrd->axis[2].min;
    } else /*if (axis == 2)*/ {
      z_pos = (slice_num+0.5)*(slice.volume_->nrrd_->nrrd->axis[2].max - 
			       slice.volume_->nrrd_->nrrd->axis[2].min)/
	slice.volume_->nrrd_->nrrd->axis[2].size;

      x_wid = slice.volume_->nrrd_->nrrd->axis[0].max - 
	slice.volume_->nrrd_->nrrd->axis[0].min;
      y_hei = slice.volume_->nrrd_->nrrd->axis[1].max - 
	slice.volume_->nrrd_->nrrd->axis[1].min;
    }
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


int
ViewSlices::extract_window_slices(SliceWindow &window) {
  for (unsigned int s = window.slices_.size(); s < volumes_.size(); ++s)
    window.slices_.push_back(scinew NrrdSlice(volumes_[s], &window));
  for_each(&ViewSlices::update_slice_from_window);
  for_each(window, &ViewSlices::set_slice_nrrd_dirty);
  set_slice_nrrd_dirty(window.paint_);

  return window.slices_.size();
}

int
ViewSlices::set_slice_nrrd_dirty(NrrdSlice &slice) {
  slice.do_lock();
  slice.nrrd_dirty_ = true;
  slice.do_unlock();
  return 0;
}


int
ViewSlices::extract_slice(NrrdSlice &slice)
{
  if (!slice.nrrd_dirty_) return 0;

  ASSERT(slice.volume_);
  ASSERT(slice.volume_->nrrd_.get_rep());

  slice.do_lock();
  setup_slice_nrrd(slice);

  int axis = slice.axis_;
  Nrrd *volume = slice.volume_->nrrd_->nrrd;
  NrrdDataHandle tmp1 = scinew NrrdData;
  NrrdDataHandle tmp2 = scinew NrrdData;
  if (slice.mode_ == mip_e || slice.mode_ == slab_e) {
    int min[3], max[3];
    for (int i = 0; i < 3; i++) {
      min[i] = 0;
      max[i] = max_slice_[i];
    }
    if (slice.mode_ == slab_e) {
      min[axis] = slice.slab_min_;
      max[axis] = slice.slab_max_;
    }
    NRRD_EXEC(nrrdCrop(tmp2->nrrd, volume, min, max));
    NRRD_EXEC(nrrdProject(tmp1->nrrd, tmp2->nrrd, axis, 2, nrrdTypeDefault));
  } else {
    NRRD_EXEC(nrrdSlice(tmp1->nrrd, volume, axis, slice.slice_num_));
  }
  
  int minp[2] = { 0, 0 };
  int maxp[2] = { slice.tex_wid_-1, slice.tex_hei_-1 };
  NRRD_EXEC(nrrdPad(slice.nrrd_->nrrd,tmp1->nrrd,minp,maxp,nrrdBoundaryBleed));

  slice.nrrd_dirty_ = false;
  slice.tex_dirty_ = true;
  slice.geom_dirty_ = false;
  slice.do_unlock();

  return 1;
}


int
ViewSlices::extract_mip_slices(NrrdVolume *volume)
{
  if (!volume || !volume->nrrd_.get_rep()) { return 0; }
  for (int axis = 0; axis < 3; ++axis) {
    if (!mip_slices_[axis]) {
      mip_slices_[axis] = scinew NrrdSlice(volume, 0);
      mip_slices_[axis]->name_ = "MIP";
    }
      
    NrrdSlice &slice = *mip_slices_[axis];
    slice.do_lock();
    slice.volume_ = volume;

    slice.nrrd_ = scinew NrrdData;

    NrrdDataHandle temp1 = scinew NrrdData;
    
    int min[3], max[3];
    for (int i = 0; i < 3; i++) {
      min[i] = 0;
      max[i] = max_slice_[i];
    }
    
    NRRD_EXEC(nrrdProject(temp1->nrrd, slice.volume_->nrrd_->nrrd, 
			  axis, 2, nrrdTypeDefault));
    slice.axis_ = axis;
    slice.slice_num_ = 0;
    slice.nrrd_dirty_ = false;
    slice.tex_dirty_ = true;
    slice.geom_dirty_ = false;
    slice.wid_     = temp1->nrrd->axis[0].size;
    slice.hei_     = temp1->nrrd->axis[1].size;
    slice.tex_wid_ = Pow2(slice.wid_);
    slice.tex_hei_ = Pow2(slice.hei_);
    slice.opacity_ = slice.volume_->opacity_;
    
    int minp[2] = { 0, 0 };
    int maxp[2] = { slice.tex_wid_-1, slice.tex_hei_-1 };
    
    NRRD_EXEC(nrrdPad(slice.nrrd_->nrrd, temp1->nrrd, 
		      minp,maxp,nrrdBoundaryPad, 0.0));
    set_slice_coords(slice, true);
    slice.do_unlock();
  }
  return 1;
}


int
ViewSlices::send_mip_textures(SliceWindow &window)
{
  int value = 0;
  for (int axis = 0; axis < 3; ++axis)
  {
    if (!mip_slices_[axis]) continue;
    NrrdSlice &slice = *mip_slices_[axis];
    if (!slice.tex_dirty_) continue;
    value++;
    slice.do_lock();

    apply_colormap(slice, temp_tex_data_);

    window.viewport_->make_current();
    bind_slice(slice, temp_tex_data_);
    window.viewport_->release();

    string name = "MIP Slice"+to_string(slice.axis_);

    TexSquare *mintex = scinew TexSquare();
    tobjs_[name+"-min"] = mintex;
    slice.slice_num_ = 0;
    set_slice_coords(slice, false);
    mintex->set_coords(slice.tex_coords_, slice.pos_coords_);
    mintex->set_texname(slice.tex_name_);
    Vector normal(axis==0?1.0:0.0, axis==1?-1.0:0.0, axis==2?1.0:0.0);
    mintex->set_normal(normal);

    Vector *minvec = scinew 
      Vector(axis==0?1.0:0.0, axis==1?1.0:0.0, axis==2?1.0:0.0);
    GeomCull *mincull = scinew GeomCull(mintex, minvec);

    TexSquare *maxtex = scinew TexSquare();
    tobjs_[name+"-max"] = maxtex;
    slice.slice_num_ = max_slice_[axis];
    set_slice_coords(slice, false);
    maxtex->set_coords(slice.tex_coords_, slice.pos_coords_);
    maxtex->set_texname(slice.tex_name_);
    maxtex->set_normal(normal);

    Vector *maxvec = scinew 
      Vector(axis==0?-1.0:0.0, axis==1?-1.0:0.0, axis==2?-1.0:0.0);
    GeomCull *maxcull = scinew GeomCull(maxtex, maxvec);

    GeomGroup *group = scinew GeomGroup();
    group->add(mincull);
    group->add(maxcull);
    
    GeomHandle gobj = group;
    slice.do_unlock();
    if (gobjs_[name]) geom_oport_->delObj(gobjs_[name]);
    gobjs_[name] = geom_oport_->addObj(gobj, name, &slice_lock_);
  }
  return value;
}


void
ViewSlices::set_axis(SliceWindow &window, unsigned int axis) {
  window.axis_ = axis;
  extract_window_slices(window);
  window.redraw_ = true;
}

void
ViewSlices::prev_slice(SliceWindow &window)
{
  if (window.slice_num_() == 0) 
    return;
  window.slice_num_--;
  if (window.slice_num_ < 0)
    window.slice_num_ = 0;
  window.cursor_moved_ = true;
  extract_window_slices(window);
  redraw_all();
}

void
ViewSlices::next_slice(SliceWindow &window)
{
  if (window.slice_num_() == max_slice_[window.axis_()]) 
    return;
  window.slice_num_++;
  if (window.slice_num_ > max_slice_[window.axis_])
    window.slice_num_ = max_slice_[window.axis_];
  window.cursor_moved_ = true;
  extract_window_slices(window);
  redraw_all();
}

void
ViewSlices::zoom_in(SliceWindow &window)
{
  window.zoom_ *= 1.1;
  window.cursor_moved_ = true;
  window.redraw_ = true;
}

void
ViewSlices::zoom_out(SliceWindow &window)
{
  window.zoom_ /= 1.1;
  window.cursor_moved_ = true;
  window.redraw_ = true;
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
  return Point(xyz[0], xyz[1], xyz[2]);
}
  
void
ViewSlices::execute()
{
  update_state(Module::JustStarted);
  unsigned int a, n = 1;
  vector<NrrdIPort *> nrrd_iports;
  NrrdIPort *nrrd_iport;
  do {
    try {
      nrrd_iport = (NrrdIPort*)get_iport("Nrrd"+to_string(n));
    } catch (...) {
      nrrd_iport = 0;
    }
    if (nrrd_iport) {
      nrrd_iports.push_back(nrrd_iport);
    }
    ++n;
  } while (nrrd_iport);


  update_state(Module::NeedData);


  vector<NrrdDataHandle> nrrds;
  for (n = 0; n < nrrd_iports.size(); ++n) {
    NrrdDataHandle nrrdH;
    nrrd_iports[n]->get(nrrdH);

    if (nrrdH.get_rep())
    {
      if (nrrdH->nrrd->dim < 3)
      {
        warning("Nrrd with dim < 3, skipping.");
      }
      else
      {
        nrrds.push_back(nrrdH);
      }
    }
  }

  if (!nrrds.size()) {
    error ("Unable to get an input nrrd.");
    return;
  }

  if (painting_() == 2) {
    ASSERT(cm2_.get_rep());
    cm2_=scinew ColorMap2(*cm2_.get_rep());
    painting_ = 1;
  } 
  cmap2_oport_->send_intermediate(cm2_);
  cmap2_iport_->get(cm2_);

  paint_lock_.lock();
  gradient_ = 0;
  grad_iport_->get(gradient_);

  paint_widget_ = 0;
  if (cm2_.get_rep() && (cm2_->selected() >= 0) && 
      (cm2_->selected() < int(cm2_->widgets().size()))) {
    CM2Widget *current = cm2_->widgets()[cm2_->selected()].get_rep();
    paint_widget_ = dynamic_cast<SCIRun::PaintCM2Widget*>(current);
  }

  painting_ = paint_widget_ && gradient_.get_rep() && cm2_.get_rep();

  if ((show_colormap2_() || painting_) && 
      cm2_.get_rep() && cm2_generation_ != cm2_->generation)
    for_each(&ViewSlices::set_paint_dirty);
  paint_lock_.unlock();
  
      
  n1_cmap_iport_->get(colormap_);
  bool re_extract = 
    colormap_.get_rep()?(colormap_generation_ != colormap_->generation):false;
  if (colormap_.get_rep())
    colormap_generation_ = colormap_->generation;

  update_state(Module::Executing);
  TCLTask::lock();

  while(volumes_.size() < nrrds.size())
    volumes_.push_back(0);

  double min_value = airNaN();
  double max_value = airNaN();

  for (a = 0; a < 3; a++) {
    max_slice_[a] = -1;
    scale_[a] = airNaN();
  }

  NrrdRange *range = scinew NrrdRange;
  for (n = 0; n < nrrds.size(); ++n) {
    NrrdDataHandle nrrdH = nrrds[n];

    nrrdRangeSet(range, nrrdH->nrrd, 1);
    if (range && (airIsNaN(min_value) || range->min < min_value))
      min_value = range->min;
    if (range && (airIsNaN(max_value) || range->max < max_value))
      max_value = range->max;      

    for (a = 0; a < 3; ++a) {
      if (nrrdH->nrrd->axis[a].min > nrrdH->nrrd->axis[a].max)
	SWAP(nrrdH->nrrd->axis[a].min,nrrdH->nrrd->axis[a].max);
      if (nrrdH->nrrd->axis[a].spacing < 0.0)
	nrrdH->nrrd->axis[a].spacing *= -1.0;
      const double spacing = nrrdH->nrrd->axis[a].spacing;
      if (airIsNaN(scale_[a]))
	scale_[a] = (airExists(spacing) ? spacing : 1.0);
      else if (scale_[a] != spacing) {
	error("Nrrd #"+to_string(n)+
	      " has spacing different than a previous nrrd. Stopping.");
	return;
      }
      
      const int size = nrrdH->nrrd->axis[a].size;
      if (max_slice_[a] == -1)
	max_slice_[a] = size-1;
      else if (max_slice_[a] != size-1) {
	error("Nrrd #"+to_string(n)+
	      " has different dimensions than a previous nrrd.");
	error("Both input nrrds must have same dimensions. Stopping.");
	return;
      }     
      center_[a] = size*scale_[a]/2.0;
    }
    dim0_ = max_slice_[0]+1;
    dim1_ = max_slice_[1]+1;
    dim2_ = max_slice_[2]+1;
      
    if (nrrdH.get_rep() && nrrdH->generation != nrrd_generations_[n]) {
      re_extract = true;
      nrrd_generations_[n] = nrrdH->generation;
      if (volumes_[n]) {
	delete volumes_[n];
	volumes_[n] = 0;
      }
      volumes_[n] = scinew NrrdVolume(ctx->subVar("nrrd"+to_string(n), false));
      volumes_[n]->nrrd_ = nrrdH;
      if (n == 0) 
	extract_mip_slices(volumes_[n]);	
    }
  }

  delete range;
  if (!airIsNaN(min_value)) min_ = min_value;
  if (!airIsNaN(max_value)) max_ = max_value;

  // Temporary space to hold the colormapped texture
  if (temp_tex_data_) delete[] temp_tex_data_;
  const int max_dim = 
    Max(Pow2(max_slice_[0]+1), Pow2(max_slice_[1]+1), Pow2(max_slice_[2]+1));
  const int mid_dim = 
    Mid(Pow2(max_slice_[0]+1), Pow2(max_slice_[1]+1), Pow2(max_slice_[2]+1));
  temp_tex_data_ = scinew float[max_dim*mid_dim*4];

  // Mark all windows slices dirty
  if (re_extract) {
    for_each(&ViewSlices::extract_window_slices);
    for (int n = 0; n < 3; ++n)
      if (mip_slices_[n])
	rebind_slice(*mip_slices_[n]);
    for_each(&ViewSlices::autoview);
  }
  redraw_all();
  TCLTask::unlock();

  cmap2_oport_->send(cm2_);
  cmap2_iport_->get(cm2_);
  update_state(Module::Completed);
  TCLTask::lock();
  if (executing_) --executing_;
  TCLTask::unlock();

}

bool
ViewSlices::mouse_in_window(SliceWindow &window) {
  return (window.layout_ == current_layout_ &&
	  window.mouse_x_ >= window.viewport_->x() && 
	  window.mouse_x_ < window.viewport_->x()+window.viewport_->width() &&
	  window.mouse_y_ >= window.viewport_->y() &&
	  window.mouse_y_ < window.viewport_->y()+window.viewport_->height());
}


int
ViewSlices::set_probe(SliceWindow &window) {
  const int axis = window.axis_();
  window.slice_num_ = Floor(cursor_(axis)/scale_[axis]);
  if (window.slice_num_ < 0) 
    window.slice_num_ = 0;
  if (window.slice_num_ > max_slice_[axis]) 
    window.slice_num_ = max_slice_[axis];  
  extract_window_slices(window);
  return 1;
}


int
ViewSlices::rebind_slice(NrrdSlice &slice) {
  slice.do_lock();
  slice.tex_dirty_ = true;
  slice.do_unlock();
  return 1;
}

  
void
ViewSlices::handle_gui_motion(GuiArgs &args) {
  ASSERT(layouts_.find(args[2]) != layouts_.end());
  int state = args.get_int(5);
  WindowLayout &layout = *layouts_[args[2]];

  int x = args.get_int(3);
  int y = args.get_int(4);
  int X = args.get_int(7);
  int Y = args.get_int(8);
  y = layout.opengl_->height() - 1 - y;

  // Take care of zooming/panning first because it can affect cursor position
  if (zooming_) {
    const double dy = (Y-pick_y_);
    const double dx = (X-pick_x_);
    const double zoom = original_zoom_+(dx+dy);
    zooming_->zoom_ = Clamp(zoom, 1.0, 3200.0);
  } else if (panning_) {
    panning_->x_ = original_pan_.first+(pick_x_ - X)/(panning_->zoom_/100.0);
    panning_->y_ = original_pan_.second+(Y - pick_y_)/(panning_->zoom_/100.0);
  }

  SliceWindow *inside_window = 0;
  for (unsigned int w = 0; w < layout.windows_.size(); ++w) {
    SliceWindow &window = *layout.windows_[w];
    window.mouse_x_ = x;
    window.mouse_y_ = y;
    if (mouse_in_window(window)) {
      inside_window = &window;
      window.cursor_moved_ = true;
    }
  }

  if (panning_ || zooming_) {
    for_each(layout, &ViewSlices::redraw_window);
    return;
  }

  const bool button1 = state & BUTTON_1_E;
  if (button1 && !crop_ && painting_ && 
      inside_window && (inside_window->mode_ == normal_e)) { 
    do_paint(*inside_window);
  } else if (button1 && crop_ && pick_window_ && pick_) {
    crop_draw_bbox_ = update_crop_bbox(*pick_window_, pick_, X, Y);
    crop_pick_boxes_ = compute_crop_pick_boxes(*pick_window_,
					       crop_draw_bbox_);
    update_crop_bbox_to_gui();
  } else if (inside_window && probe_()) {
    inside_window->viewport_->make_current();
    setup_gl_view(*inside_window);
    cursor_ = screen_to_world(*inside_window, x, y);
    inside_window->viewport_->release();
    
    for_each(&ViewSlices::set_probe);
  } else if (window_level_) {
      //      WindowLayouts::iterator liter = layouts_.begin();
    const double diagonal = 
      sqrt(double(layout.opengl_->width()*layout.opengl_->width()) +
	   double(layout.opengl_->height()*layout.opengl_->height()));
    const double scale = (max_ - min_)/(2*diagonal);
    clut_ww_ = Max(0.0,original_ww_ + (X - pick_x_)*scale);
    clut_wl_ = Clamp(original_wl_ + (pick_y_ - Y)*scale, min_, max_);
    for_each(&ViewSlices::rebind_slice);
    for (int axis = 0; axis < 3; ++axis)
      if (mip_slices_[axis])
	rebind_slice(*mip_slices_[axis]);
  }
  redraw_all();
}



void
ViewSlices::handle_gui_enter(GuiArgs &args) {
  ASSERTMSG(layouts_.find(args[2]) != layouts_.end(),
	    ("Cannot handle enter on "+args[2]).c_str());
  ASSERTMSG(current_layout_ == 0,
	    "Haven't left window");
  current_layout_ = layouts_[args[2]];
}


void
ViewSlices::handle_gui_leave(GuiArgs &args) {
  current_layout_ = 0;
  redraw_all();
}



void
ViewSlices::handle_gui_button_release(GuiArgs &args) {
  if (args.count() != 7)
    SCI_THROW(GuiException(args[0]+" "+args[1]+
			   " expects a window #, button #, and state"));

  int button = args.get_int(3);
  window_level_ = 0;
  pick_window_ = 0;
  probe_ = 0;
  zooming_ = 0;
  panning_ = 0;

  switch (button) {
  case 1:
    if (pick_) {
      crop_bbox_ = crop_draw_bbox_;
      pick_ = 0;
    }
    if (!crop_ && painting_) { 
      painting_ = 2;
      if (!executing_) {
	++executing_;
	want_to_execute();
      }
    }
    break;
  case 2:
  case 3:
  default:
    break;
  }

}

void
ViewSlices::handle_gui_button(GuiArgs &args) {
  if (args.count() != 7)
    SCI_THROW(GuiException(args[0]+" "+args[1]+
			   " expects a window #, button #, and state"));

  int button = args.get_int(3);
  int state = args.get_int(4);
  int x = args.get_int(5);
  int y = args.get_int(6);
  pick_x_ = x;
  pick_y_ = y;
  
  ASSERT(layouts_.find(args[2]) != layouts_.end());
  WindowLayout &layout = *layouts_[args[2]];
  

  for (unsigned int w = 0; w < layout.windows_.size(); ++w) {
    SliceWindow &window = *layout.windows_[w];
    if (!mouse_in_window(window)) continue;
    switch (button) {
    case 1:
      if (state & SHIFT_E == SHIFT_E) {
	panning_ = &window;
	original_pan_ = make_pair(double(window.x_),double(window.y_));
	continue;
      }

      if (!crop_ && painting_ && paint_widget_ && (window.mode_==normal_e)) { 
	if (!paint_lock_.tryLock()) continue;
	paint_widget_->add_stroke();
	do_paint(window); 
	paint_lock_.unlock();
	continue;
      }

      crop_pick_boxes_ = compute_crop_pick_boxes(window, crop_draw_bbox_);
      pick_ = mouse_in_pick_boxes(window, crop_pick_boxes_);
      pick_window_ = layout.windows_[w];
      if (!pick_) {
	window_level_ = layout.windows_[w];
	original_ww_ = clut_ww_();
	original_wl_ = clut_wl_();
      }
      break;
    case 2:
      if (state & SHIFT_E == SHIFT_E)
	autoview(window);
      else
	probe_ = 1;
      break;
    case 3:
      if (state & SHIFT_E == SHIFT_E) {
	zooming_ = &window;
	original_zoom_ = window.zoom_;
      }
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
  if (args.count() != 6)
    SCI_THROW(GuiException(args[0]+" "+args[1]+
			   " expects a win #, keycode, keysym,& time"));

  ASSERT(layouts_.find(args[2]) != layouts_.end());
  WindowLayout &layout = *layouts_[args[2]];

  for (unsigned int w = 0; w < layout.windows_.size(); ++w) {
    SliceWindow &window = *layout.windows_[w];
    double pan_delta = Round(3.0/(window.zoom_()/100.0));
    if (pan_delta < 1.0) pan_delta = 1.0;
    if (!mouse_in_window(window)) continue;
    if (args[4] == "equal" || args[4] == "plus") zoom_in(window);
    else if (args[4] == "minus" || args[4] == "underscore") zoom_out(window);
    else if (args[4] == "less" || args[4] == "comma") prev_slice(window);
    else if (args[4] == "greater" || args[4] == "period") next_slice(window);
    else if (args[4] == "0") set_axis(window, 0);
    else if (args[4] == "1") set_axis(window, 1);
    else if (args[4] == "2") set_axis(window, 2);
    else if (args[4] == "i") {
      window.invert_ = window.invert_?0:1;
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
    }
  }
}
  

void
ViewSlices::tcl_command(GuiArgs& args, void* userdata) {
  if(args.count() < 2) {
    args.error("ViewSlices needs a minor command");
    return;
  }

  //  cerr << ":";
  //  for (int a = 0; a < args.count(); ++a)
  //    cerr << args[a] << " ";
  //  cerr << "\n";

  if (args[1] == "motion")	  handle_gui_motion(args);
  else if (args[1] == "button")   handle_gui_button(args);
  else if (args[1] == "release")  handle_gui_button_release(args);
  else if (args[1] == "keypress") handle_gui_keypress(args);
  else if (args[1] == "enter")    handle_gui_enter(args);
  else if (args[1] == "leave")    handle_gui_leave(args);
  else if (args[1] == "background_thresh") update_background_threshold();
  else if (args[1] == "gradient_thresh")for_each(&ViewSlices::set_paint_dirty);
  else if (args[1] == "undo") undo_paint_stroke();
  else if (args[1] == "set_font_sizes") set_font_sizes(font_size_());
  else if(args[1] == "setgl") {
    TkOpenGLContext *context = \
      scinew TkOpenGLContext(args[2], args.get_int(4), 512, 512);
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
    window->layout_ = layout;
    window->name_ = args[2];
    window->viewport_ = scinew OpenGLViewport(layout->opengl_);
    window->axis_ = (layouts_.size())%3;
    layout->windows_.push_back(window);
    autoview(*window);
  } else if(args[1] == "redraw") {
    ASSERT(layouts_.find(args[2]) != layouts_.end());
    for_each(*layouts_[args[2]], &ViewSlices::redraw_window);
  } else if(args[1] == "resize") {
    ASSERT(layouts_.find(args[2]) != layouts_.end());
    for_each(*layouts_[args[2]], &ViewSlices::autoview);
  } else if(args[1] == "redrawall") {
    redraw_all();
  } else if(args[1] == "rebind") {
    ASSERT(layouts_.find(args[2]) != layouts_.end());
    for_each(*layouts_[args[2]], &ViewSlices::extract_window_slices);
    for_each(*layouts_[args[2]], &ViewSlices::redraw_window);
  } else if(args[1] == "texture_rebind") {
    if (args.count() == 2) { 
      for_each(&ViewSlices::rebind_slice);
      for_each(&ViewSlices::redraw_window);
    } else {
      for_each(*layouts_[args[2]], &ViewSlices::rebind_slice);
      for_each(*layouts_[args[2]], &ViewSlices::redraw_window);
    }
  } else if(args[1] == "startcrop") {
    crop_ = 1;
    if (args.count() == 3 && args.get_int(2)) {
      crop_bbox_ = BBox
	(Point(0,0,0), 
	 Point(max_slice_[0]+1, max_slice_[1]+1, max_slice_[2]+1));
      crop_draw_bbox_ = crop_bbox_;
      update_crop_bbox_to_gui();
    } else {
      update_crop_bbox_from_gui();
      crop_bbox_ = crop_draw_bbox_;
    }
    redraw_all();
  } else if(args[1] == "stopcrop") {
    crop_ = 0;
    redraw_all();
  } else if(args[1] == "updatecrop") {
    update_crop_bbox_from_gui();
    redraw_all();
  } else if (args[1] == "setclut") {
    const double cache_ww = clut_ww_;
    const double cache_wl = clut_wl_;
    clut_ww_(); // resets gui context
    clut_wl_(); // resets gui context
    if (fabs(cache_ww - clut_ww_) < 0.0001 && 
	fabs(cache_wl - clut_wl_) < 0.0001) return;
    for_each(&ViewSlices::rebind_slice);
    for_each(&ViewSlices::redraw_window);
    for (int n = 0; n < 3; ++n)
      if (mip_slices_[n])
	rebind_slice(*mip_slices_[n]);
  } else 
    Module::tcl_command(args, userdata);
}



int
ViewSlices::send_slice_textures(NrrdSlice &slice) {
  if (!slice.geom_dirty_) return 0;
  slice.do_lock();
  string name = "Slice"+to_string(slice.axis_);
  if (tobjs_[name] == 0) 
    tobjs_[name] = scinew TexSquare();
  tobjs_[name]->set_texname(slice.tex_name_);
  set_slice_coords(slice, false);
  tobjs_[name]->set_coords(slice.tex_coords_, slice.pos_coords_);
  int axis = slice.axis_;
  Vector normal(axis==0?1.0:0.0, axis==1?-1.0:0.0, axis==2?1.0:0.0);
  tobjs_[name]->set_normal(normal);
  set_slice_coords(slice, true);
  GeomHandle gobj = tobjs_[name];
  slice.geom_dirty_ = false;
  slice.do_unlock();

  if (gobjs_[name]) geom_oport_->delObj(gobjs_[name]);
  gobjs_[name] = geom_oport_->addObj(gobj, name, &slice_lock_);
  return 1;
}

int
ViewSlices::set_paint_dirty(SliceWindow &window) 
{
  window.paint_.nrrd_dirty_ = true;
  return 1;
}

void
ViewSlices::rasterize_widgets_to_cm2(int min, int max, Array3<float> &buffer) 
{
  buffer.initialize(0.0);
  for (int i = min; i <= max; ++i)
    cm2_->widgets()[i]->rasterize(buffer);
}


bool
ViewSlices::rasterize_colormap2() {
  if (!cm2_.get_rep()) return false;
  if (cm2_generation_ == cm2_->generation) return false;
  cm2_generation_ = cm2_->generation;

  const int last_widget = cm2_->widgets().size() - 1;
  int selected = cm2_->selected();
  if (selected < 0 || selected > int(cm2_->widgets().size())) 
    selected = last_widget;

  rasterize_widgets_to_cm2(0, selected-1, cm2_buffer_under_);
  rasterize_widgets_to_cm2(selected, selected, cm2_buffer_);
  rasterize_widgets_to_cm2(selected+1, last_widget, cm2_buffer_over_);

  for_each(&ViewSlices::set_paint_dirty);

  return true;
}


void
ViewSlices::apply_colormap2_to_slice(Array3<float> &cm2, NrrdSlice &slice)
{
  slice.do_lock();
  setup_slice_nrrd(slice);
  slice.tex_dirty_ = true;

  const unsigned int y_ax = y_axis(*slice.window_);
  const unsigned int x_ax = x_axis(*slice.window_);
  const unsigned int z = slice.slice_num_;
  const Nrrd* value_nrrd = volumes_[0]->nrrd_->nrrd;
  const Nrrd* grad_nrrd = gradient_->nrrd;
  const double min = clut_wl_ - clut_ww_/2.0;
  const double scale = 255.999 / clut_ww_;
  float *paintdata =  (float *)slice.nrrd_->nrrd->data;  
  const int grad_thresh = Round(gradient_threshold_()*255.0);
  int grad, cval, pos;
  double val;

  for (int y = 0; y <= max_slice_[y_ax]; ++y) {
    for (int x = 0; x < max_slice_[x_ax]; ++x) {
      pos = (y*slice.tex_wid_+x)*4;
      grad = int(get_value(grad_nrrd,
			   (x_ax==0)?x:((y_ax==0)?y:z),
			   (x_ax==1)?x:((y_ax==1)?y:z),
			   (x_ax==2)?x:((y_ax==2)?y:z)));
      if (grad < grad_thresh) {
	paintdata[pos] = 0.0;
	paintdata[pos+1] = 0.0;
	paintdata[pos+2] = 0.0;
	paintdata[pos+3] = 1.0;
      } else {
	val = get_value(value_nrrd, 
			(x_ax==0)?x:((y_ax==0)?y:z),
			(x_ax==1)?x:((y_ax==1)?y:z),
			(x_ax==2)?x:((y_ax==2)?y:z));
	cval = Floor((val-min)*scale);
	
	if (cval >= 0 && cval <= 255) {
	  memcpy(paintdata+pos, 
		 &cm2(grad, cval, 0), 
		 4*sizeof(float));
	} else {
	  paintdata[pos] = 0.0;
	  paintdata[pos+1] = 0.0;
	  paintdata[pos+2] = 0.0;
	  paintdata[pos+3] = 0.0;
	}
      }
    }
  }
  slice.do_unlock();
}

int
ViewSlices::update_slice_from_window(NrrdSlice &slice) {
  ASSERT(slice.window_);
  slice.do_lock();
  SliceWindow &window = *slice.window_;
  const int axis = window.axis_();
  slice.axis_ = axis;
  slice.slice_num_ = Clamp(window.slice_num_(), 0, max_slice_[axis]);
  slice.slab_min_ = Min(max_slice_[axis], Max(0, window.slab_min_()));
  slice.slab_max_ = Max(0, Min(window.slab_max_(), max_slice_[axis]));
  slice.mode_ = window.mode_();
  if (slice.mode_ == mip_e || slice.mode_ == slab_e) {
    cur_slice_[slice.axis_] = slice.slab_min_;
    if (slice.mode_ == mip_e)
      slab_width_[slice.axis_] = 0;
    if (slice.mode_ == slab_e)
      slab_width_[slice.axis_] = slice.slab_max_ - slice.slab_min_ + 1;
  } else {
    cur_slice_[slice.axis_] = slice.slice_num_;
    slab_width_[axis] = 1;
  }

  const int xaxis = x_axis(window);
  const int yaxis = y_axis(window);
  Nrrd *volume = slice.volume_->nrrd_->nrrd;
  if (volume) {
    ASSERT(xaxis < volume->dim);
    ASSERT(yaxis < volume->dim);
    slice.wid_ = volume->axis[xaxis].size;
    slice.hei_ = volume->axis[yaxis].size;
  }
  slice.tex_wid_ = Pow2(slice.wid_);
  slice.tex_hei_ = Pow2(slice.hei_);
  slice.opacity_ = slice.volume_->opacity_();

  slice.do_unlock();
  return 1;
}



int
ViewSlices::extract_window_paint(SliceWindow &window) {
  if (!gradient_.get_rep() || !cm2_.get_rep()) return 0;
  if (window.slices_.empty()) return 0;
  rasterize_colormap2();

  if (!window.paint_.nrrd_dirty_) return 0;
  window.paint_.nrrd_dirty_ = false;

  window.paint_under_.volume_ = volumes_[0];
  window.paint_.volume_ = volumes_[0];
  window.paint_over_.volume_ = volumes_[0];

  apply_colormap2_to_slice(cm2_buffer_under_, window.paint_under_);
  apply_colormap2_to_slice(cm2_buffer_, window.paint_);
  apply_colormap2_to_slice(cm2_buffer_over_, window.paint_over_);
  
  return 3;
}


void
ViewSlices::do_paint(SliceWindow &window) {
  if (!paint_lock_.tryLock()) return;
  if (!paint_widget_ || !gradient_.get_rep() || !paint_widget_) {
    paint_lock_.unlock();
    return;
  }
  int xyz[3];
  for (int i = 0; i < 3; ++i) {
    xyz[i] = Floor(cursor_(i)/scale_[i]);
    if (xyz[i] < 0 || xyz[i] > max_slice_[i]) return;
  }
  const double gradient = 
    get_value(gradient_->nrrd, xyz[0], xyz[1], xyz[2])/255.0;
  if (gradient < gradient_threshold_) return;
  const double value = 
    get_value(volumes_[0]->nrrd_->nrrd,xyz[0],xyz[1],xyz[2]);
  paint_widget_->add_coordinate(make_pair(value, gradient));
  rasterize_widgets_to_cm2(cm2_->selected(), cm2_->selected(), cm2_buffer_);
  for_each(&ViewSlices::extract_current_paint);
  paint_lock_.unlock();
}

int
ViewSlices::extract_current_paint(SliceWindow &window)
{
  apply_colormap2_to_slice(cm2_buffer_, window.paint_);
  window.redraw_ = true;
  return 1;
}


void
ViewSlices::update_background_threshold() {
  const double min = clut_wl_ - clut_ww_/2.0;
  double alpha = (background_threshold_() - min) / clut_ww_;
  TexSquareMap::iterator iter = tobjs_.begin();
  while (iter != tobjs_.end()) {
    iter->second->set_alpha_cutoff(alpha);
    iter++;
  }  
  if (geom_oport_) geom_oport_->flushViews();
}

void
ViewSlices::undo_paint_stroke() {
  if (!paint_widget_) return;
  if (paint_widget_->pop_stroke()) {
    painting_ = 2;
    want_to_execute();
  }
}
  
void
ViewSlices::delete_all_fonts() {
  FontMap::iterator fiter = fonts_.begin();
  const FontMap::iterator fend = fonts_.end();
  while (fiter != fend)
    if (fiter->second)
      delete fiter++->second;
  fonts_.clear();
}

void
ViewSlices::initialize_fonts() {
  if (!freetype_lib_) {
    try {
      freetype_lib_ = scinew FreeTypeLibrary();
    } catch (...) {
      freetype_lib_ = 0;
      error("Cannot Initialize FreeType Library.");
      error("Did you configure with --with-freetype= ?");
      error("Will not render text in windows.");
    }
  }

  if (!freetype_lib_) return;

  delete_all_fonts();

  string font_dir;
  const char *dir = sci_getenv("SCIRUN_FONT_PATH");
  if (dir) 
    font_dir = dir;
  if (gui->eval("validDir "+font_dir) == "0")
    font_dir = string(sci_getenv("SCIRUN_SRCDIR"))+"/pixmaps";
  string fontfile = font_dir+"/scirun.ttf";
  
  try {
    fonts_["default"] = freetype_lib_->load_face(fontfile);
    fonts_["orientation"] = freetype_lib_->load_face(fontfile);
    fonts_["view"] = freetype_lib_->load_face(fontfile);
    set_font_sizes(font_size_());
  } catch (...) {
    delete_all_fonts();
    error("Error loading font file: "+fontfile);
    error("Please set SCIRUN_FONT_PATH to a directory with scirun.ttf\n");
  }
}

void
ViewSlices::set_font_sizes(double size) {
  show_text_();
  font_r_();
  font_g_();
  font_b_();
  font_a_();
  try {
    if (fonts_["default"]) 
      fonts_["default"]->set_points(size);
    if (fonts_["orientation"]) 
      fonts_["orientation"]->set_points(size+10.0);
    if (fonts_["view"]) 
      fonts_["view"]->set_points(size+5.0);
    redraw_all();
  } catch (...) {
    delete_all_fonts();
    error("Error calling set_points on FreeTypeFont.");
    error("No fonts will be rendered.");
  }
}


int
ViewSlices::autoview(SliceWindow &window) {
  int wid = window.viewport_->width();
  int hei = window.viewport_->height();
  int xtr = 0;
  int ytr = 0;
  FreeTypeFace *font = fonts_["orientation"];
  if (font)
  {
    FreeTypeText dummy("X", font);
    BBox bbox;
    dummy.get_bounds(bbox);
    xtr = Ceil(bbox.max().x() - bbox.min().x())+2;
    ytr = Ceil(bbox.max().y() - bbox.min().y())+2;
    wid -= 2*xtr;
    hei -= 2*ytr;
  }
  
  int xax = x_axis(window);
  int yax = y_axis(window);

  double w_ratio = wid/((max_slice_[xax]+1)*scale_[xax]*1.1);
  double h_ratio = hei/((max_slice_[yax]+1)*scale_[yax]*1.1);

  window.zoom_ = Min(w_ratio*100.0, h_ratio*100.0);
  if (window.zoom_ < 1.0) window.zoom_ = 100.0;
  window.x_ = 0.0;
  window.y_ = 0.0;
  redraw_window(window);
  return 1;
}
   
int
ViewSlices::setup_slice_nrrd(NrrdSlice &slice)
{
  slice.do_lock();
  update_slice_from_window(slice);

  if (!slice.nrrd_.get_rep()) {
    slice.nrrd_ = scinew NrrdData;
    nrrdAlloc(slice.nrrd_->nrrd, nrrdTypeFloat, 3, // 3 dim = RGBA x X x Y
	      4, slice.tex_wid_, slice.tex_hei_);
    slice.tex_dirty_ = true;
  }

  if (slice.nrrd_->nrrd && 
      slice.nrrd_->nrrd->dim >= 3 &&
      slice.nrrd_->nrrd->data &&
      (int(slice.tex_wid_) != slice.nrrd_->nrrd->axis[1].size ||
       int(slice.tex_hei_) != slice.nrrd_->nrrd->axis[2].size)) {
    nrrdEmpty(slice.nrrd_->nrrd);
    slice.nrrd_->nrrd->data = 0;
    slice.tex_dirty_ = true;
  }
  
  if (!slice.nrrd_->nrrd->data) {
    nrrdAlloc(slice.nrrd_->nrrd, nrrdTypeFloat, 3, // 3 dim = RGBA x X x Y
	      4, slice.tex_wid_, slice.tex_hei_);
    slice.tex_dirty_ = true;
  }
  set_slice_coords(slice, true);
  slice.do_unlock();

  return 1;
}
} // End namespace SCIRun
