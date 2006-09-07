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
 *  Painter.cc
 *
 *  Written by:
 *   McKay Davis
 *   School of Computing
 *   University of Utah
 *   November, 2005
 *
 *  Copyright (C) 2005 SCI Group
 */

#include <Dataflow/Modules/Render/Painter.h>
#include <sci_comp_warn_fixes.h>
#include <tcl.h>
#include <tk.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <typeinfo>
#include <iostream>
#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>
#include <sci_algorithm.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Scheduler.h>
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Ports/GeometryPort.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <Core/Algorithms/Visualization/RenderField.h>
#include <Core/Bundle/Bundle.h>
#include <Core/Containers/Array3.h>
#include <Core/Datatypes/Field.h> 
#include <Core/Exceptions/GuiException.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/ColorMappedNrrdTextureObj.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geom/GeomCull.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/TexSquare.h>
#include <Dataflow/GuiInterface/TkOpenGLContext.h>
#include <Core/Geom/OpenGLViewport.h>
#include <Core/Geom/FreeType.h>
#include <Dataflow/GuiInterface/UIvar.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/CleanupManager.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/Environment.h>
#include <Core/Volume/CM2Widget.h>
#include <Core/Geom/TextRenderer.h>
#include <Core/Util/SimpleProfiler.h>
#include <Dataflow/GuiInterface/TCLKeysyms.h>

#ifdef _WIN32
#define snprintf _snprintf
#define SCISHARE __declspec(dllimport)
#else
#define SCISHARE
#endif

extern "C" SCISHARE Tcl_Interp* the_interp;

namespace SCIRun {

  /* Todo:
     X - Persistent volume state when re-executing
     X - Fix Non-origin world_to_index conversion
     X - Use FreeTypeTextTexture class for text
     X - Send Bundles correctly
     X - Ability to render 2D nrrds
     X - Automatic world space grid
     X - Add tool support for ITK filters
     X - Show other windows slice correctly
     X - Remove TCLTask::lock and replace w/ volume lock
     X - Optimize build_index_to_world_matrix out
     X - Add support for RGBA nrrds 
     X - Change tools to store error codes
     X - Add keybooard to tools
     X - Faster painting, using current window buffer
     X - vertical text for grid
       - compute bounding box for fast volume rendering
       - help mode
     W - Better tool mechanism to allow for customization & event fallthrough 
       - Add support for time axis in nrrds
       - Add back in MIP mode
       - Geometry output port
       - Automatic index space grid
       - View to choose tools
       - Migrate all operations to tools (next_siice, zoom, etc)
       - Use GPU/3DTextures for applying colormap when supported
       - Remove clever offseting for non-power-of-2 suported machines
       - Removal of for_each
       - Support applying CM2
       - Move freetype font initialization to static global singleton
       - Multi-rez texture map for MipMapping & drawing subregions


     ITK Filters:
     X - confidence connected image filter
     X - gradient magnitude image filter
     X - binary dilate/erode filters
       - *aniosotropicimagediffusionfilters* (vector if supported)
       - discrete gaussian image filter
       - watershed
  */



Painter::RealDrawer::~RealDrawer()
{
}


void
Painter::RealDrawer::run()
{
  throttle_.start();

  double t = throttle_.time();
  double frame_count_start = t;
  double time_since_frame_count_start;
  int frames = -1;
  while (!dead_) {
    if (!module_->current_layout_) throttle_.wait_for_time(t);
    if (dead_) continue;
    module_->real_draw_all();
    double t2 = throttle_.time();
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
Painter::for_each(SliceWindow &window, NrrdSliceFunc func) {
  int value = 0;
  for (unsigned int slice = 0; slice < window.slices_.size(); ++slice)
    {
      ASSERT(window.slices_[slice]);
      value += (this->*func)(*window.slices_[slice]);
    }
  return value;
}

int
Painter::for_each(WindowLayout &layout, NrrdSliceFunc func) {
  int value = 0;
  for (unsigned int window = 0; 
       window < layout.windows_.size(); ++window)
    {
      ASSERT(layout.windows_[window]);
      value += for_each(*layout.windows_[window], func);
    }
  return value;
}

int
Painter::for_each(WindowLayout &layout, SliceWindowFunc func)
{
  int value = 0;
  for (unsigned int window = 0; 
       window < layout.windows_.size(); ++window)
    {
      ASSERT(layout.windows_[window]);
      value += (this->*func)(*layout.windows_[window]);
    }
  return value;
}

template <class T>
int			
Painter::for_each(T func) {
  int value = 0;
  WindowLayouts::iterator liter = layouts_.begin();
  WindowLayouts::iterator lend = layouts_.end();
  for (; liter != lend; ++liter) {
    ASSERT(liter->second);
    value += for_each(*(liter->second), func);
  }
  return value;
}


Painter::NrrdSlice::NrrdSlice(Painter *painter,
                              NrrdVolume *volume, 
                              Point &p, Vector &n) :
  painter_(painter),
  volume_(volume),
  nrrd_dirty_(true),
  tex_dirty_(false),
  geom_dirty_(false),
  pos_(),
  xdir_(),
  ydir_(),
  plane_(p,n),
  texture_(0)
{
}


Painter::SliceWindow::SliceWindow(Painter *painter, GuiContext *ctx) :  
  painter_(painter),
  name_("INVALID"),
  layout_(0),
  viewport_(0),
  slices_(),
  slice_map_(),
  paint_layer_(0),
  center_(0,0,0),
  normal_(0,0,0),
  slice_num_(ctx->subVar("slice"), 0),
  axis_(ctx->subVar("axis")),
  zoom_(ctx->subVar("zoom"), 100.0),
  slab_min_(ctx->subVar("slab_min"), 0),
  slab_max_(ctx->subVar("slab_max"), 0),
  redraw_(true),
  mode_(ctx->subVar("mode"),0),
  show_guidelines_(ctx->subVar("show_guidelines"),1),
  cursor_pixmap_(-1)
{
}


Painter::WindowLayout::WindowLayout(GuiContext */*ctx*/) :  
  opengl_(0),
  windows_()
{
}


Painter::NrrdVolume::NrrdVolume(GuiContext *ctx,
                                const string &name,
                                NrrdDataHandle &nrrd) :
  nrrd_handle_(0),
  gui_context_(ctx),
  name_(gui_context_->subVar("name"), name),
  name_prefix_(""),
  opacity_(gui_context_->subVar("opacity"), 1.0),
  clut_min_(gui_context_->subVar("clut_min"), 0.0),
  clut_max_(gui_context_->subVar("clut_max"), 1.0),
  mutex_(gui_context_->getfullname().c_str()),
  data_min_(0),
  data_max_(1.0),
  colormap_(ctx->subVar("colormap")),
  stub_axes_(),
  transform_(),
  keep_(true)
{
  if (!colormap_.valid()) colormap_.set(0);
  set_nrrd(nrrd);
}



Painter::NrrdVolume::~NrrdVolume() {
  mutex_.lock();
  nrrd_handle_ = 0;
  //  delete gui_context_;
  mutex_.unlock();

}


int
nrrd_type_size(Nrrd *nrrd)
{
  int val = 0;
  switch (nrrd->type) {
  case nrrdTypeChar: val = sizeof(char); break;
  case nrrdTypeUChar: val = sizeof(unsigned char); break;
  case nrrdTypeShort: val = sizeof(short); break;
  case nrrdTypeUShort: val = sizeof(unsigned short); break;
  case nrrdTypeInt: val = sizeof(int); break;
  case nrrdTypeUInt: val = sizeof(unsigned int); break;
  case nrrdTypeLLong: val = sizeof(signed long long); break;
  case nrrdTypeULLong: val = sizeof(unsigned long long); break;
  case nrrdTypeFloat: val = sizeof(float); break;
  case nrrdTypeDouble: val = sizeof(double); break;
  default: throw "Unsupported data type: "+to_string(nrrd->type);
  }
  return val;
}


int
nrrd_data_size(Nrrd *nrrd)
{
  if (!nrrd->dim) return 0;
  unsigned int size = nrrd->axis[0].size;
  for (unsigned int a = 1; a < nrrd->dim; ++a)
    size *= nrrd->axis[a].size;
  return size*nrrd_type_size(nrrd);
}


Painter::NrrdVolume::NrrdVolume(NrrdVolume *copy, 
                                const string &name,
                                int clear) :
  nrrd_handle_(0),
  gui_context_(copy->gui_context_->get_parent()->subVar(name,0)),
  name_(gui_context_->subVar("name"), name),
  name_prefix_(copy->name_prefix_),
  opacity_(gui_context_->subVar("opacity"), copy->opacity_.get()),
  clut_min_(gui_context_->subVar("clut_min"), copy->clut_min_.get()),
  clut_max_(gui_context_->subVar("clut_max"), copy->clut_max_.get()),
  mutex_(gui_context_->getfullname().c_str()),
  data_min_(copy->data_min_),
  data_max_(copy->data_max_),
  colormap_(gui_context_->subVar("colormap"), copy->colormap_.get()),
  stub_axes_(copy->stub_axes_),
  transform_(),
  keep_(copy->keep_)
{
  copy->mutex_.lock();
  mutex_.lock();

  if (clear == 2) {
    nrrd_handle_ = copy->nrrd_handle_;
  } else {
    nrrd_handle_ = scinew NrrdData();
    nrrdCopy(nrrd_handle_->nrrd_, copy->nrrd_handle_->nrrd_);
    if (clear) 
      memset(nrrd_handle_->nrrd_->data, 0, nrrd_data_size(nrrd_handle_->nrrd_));
  }

  mutex_.unlock();
  //  set_nrrd(nrrd_);
  build_index_to_world_matrix();
  copy->mutex_.unlock();


}



void
Painter::NrrdVolume::set_nrrd(NrrdDataHandle &nrrd_handle) 
{
  mutex_.lock();
  nrrd_handle_ = nrrd_handle;
  nrrd_handle_.detach();
  //  nrrdBasicInfoCopy(nrrd_handle_->nrrd_, nrrd->nrrd,0);
  //  nrrdAxisInfoCopy(nrrd_handle_->nrrd_, nrrd->nrrd, 0,0);
  //  nrrdCopy(nrrd_handle_->nrrd_, nrrd->nrrd);

  stub_axes_.clear();
  if (nrrd_handle_->nrrd_->axis[0].size > 4) {
    nrrdAxesInsert(nrrd_handle_->nrrd_, nrrd_handle_->nrrd_, 0);
    nrrd_handle_->nrrd_->axis[0].min = 0.0;
    nrrd_handle_->nrrd_->axis[0].max = 1.0;
    nrrd_handle_->nrrd_->axis[0].spacing = 1.0;
    stub_axes_.push_back(0);
  }

  if (nrrd_handle_->nrrd_->dim == 3) {
    nrrdAxesInsert(nrrd_handle_->nrrd_, nrrd_handle_->nrrd_, 3);
    nrrd_handle_->nrrd_->axis[3].min = 0.0;
    nrrd_handle_->nrrd_->axis[3].max = 1.0;
    nrrd_handle_->nrrd_->axis[3].spacing = 1.0;
    stub_axes_.push_back(3);
  }


  for (unsigned int a = 0; a < nrrd_handle_->nrrd_->dim; ++a) {
    if (nrrd_handle_->nrrd_->axis[a].center == nrrdCenterUnknown)
      nrrd_handle_->nrrd_->axis[a].center = nrrdCenterNode;
    if (nrrd_handle_->nrrd_->axis[a].min > nrrd_handle_->nrrd_->axis[a].max)
      SWAP(nrrd_handle_->nrrd_->axis[a].min,nrrd_handle_->nrrd_->axis[a].max);
    if (nrrd_handle_->nrrd_->axis[a].spacing < 0.0)
      nrrd_handle_->nrrd_->axis[a].spacing *= -1.0;
  }

  NrrdRange range;
  nrrdRangeSet(&range, nrrd_handle_->nrrd_, 0);
  if (data_min_ != range.min || data_max_ != range.max) {
    data_min_ = range.min;
    data_max_ = range.max;
    clut_min_ = range.min;
    clut_max_ = range.max;
    opacity_ = 1.0;
  }
  mutex_.unlock();
  build_index_to_world_matrix();

}


NrrdDataHandle
Painter::NrrdVolume::get_nrrd() 
{
  NrrdDataHandle nrrd_handle = nrrd_handle_;
  nrrd_handle.detach();
  NrrdDataHandle nrrd2_handle = scinew NrrdData();

  //   nrrdBasicInfoCopy(nrrd->nrrd, nrrd_handle_->nrrd_,0);
  //   nrrdAxisInfoCopy(nrrd->nrrd, nrrd_handle_->nrrd_, 0,0);
  //   nrrd->nrrd->data = nrrd_handle_->nrrd_->data;

  for (int s = stub_axes_.size()-1; s >= 0 ; --s) {
    nrrdAxesDelete(nrrd2_handle->nrrd_, nrrd_handle->nrrd_, stub_axes_[s]);
    nrrd_handle = nrrd2_handle;
  }
  nrrdKeyValueCopy(nrrd_handle->nrrd_, nrrd_handle_->nrrd_);
  
  //  unsigned long ptr = (unsigned long)(&painter_);
  //  nrrdKeyValueAdd(nrrd_handle->nrrd_, 
  //                  "progress_ptr", to_string(ptr).c_str());

  return nrrd_handle;
}

DECLARE_MAKER(Painter)

Painter::Painter(GuiContext* ctx) :
  Module("Painter", ctx, Filter, "Render", "SCIRun"),
  layouts_(),
  volumes_(),
  volume_map_(),
  volume_order_(),
  current_volume_(0),
  undo_volume_(0),
  colormaps_(),
  tools_(),
  anatomical_coordinates_(ctx->subVar("anatomical_coordinates"), 1),
  show_grid_(ctx->subVar("show_grid"), 1),
  show_text_(ctx->subVar("show_text"), 1),
  font_r_(ctx->subVar("color_font_r"), 1.0),
  font_g_(ctx->subVar("color_font_g"), 1.0),
  font_b_(ctx->subVar("color_font_b"), 1.0),
  font_a_(ctx->subVar("color_font_a"), 1.0),
  bundle_oport_((BundleOPort *)get_oport("Paint Data")),
  freetype_lib_(0),
  fonts_(),
  font_size_(ctx->subVar("font_size"),17.0),
  font1_(0),
  font2_(0),
  font3_(0),
  runner_(0),
  runner_thread_(0),
  filter_(0),
  fps_(0.0),
  current_layout_(0),
  executing_(0)
{
  runner_ = scinew RealDrawer(this);
  runner_thread_ = scinew Thread(runner_, string(get_id()+" OpenGL drawer").c_str());
  event_.window_ = 0;
  event_.position_ = Point(0,0,0);
  initialize_fonts();
}

Painter::~Painter()
{
  if (runner_thread_) {
    runner_->dead_ = true;
    runner_thread_->join();
    runner_thread_ = 0;
  }
}

bool
Painter::static_callback(void *this_ptr) {
  Painter *painter = static_cast<Painter *>(this_ptr);
  painter->filter_ = 0;
  painter->filter_text_ = "";
  painter->redraw_all();
  return true;
}

void
Painter::set_context(Network *net) {
  Module::set_context(net);
  sched_->add_callback(this->static_callback, this);
}

static SimpleProfiler profiler("RenderWindow", sci_getenv_p("SCIRUN_PROFILE"));

int
Painter::render_window(SliceWindow &window) {
  if (!window.redraw_) return 0;
  window.redraw_ = false;
  window.viewport_->make_current();
  //  profiler.disable();
  profiler.enter("Render window");

  //  window.viewport_->clear();
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT);
  window.setup_gl_view();
  if (window.autoview_) {
    window.autoview_ = false;
    // with EXPERIMENTAL_TCL_LOCK, we need to unlock/lock here to avoid deadlock
    // in the case where you're trying to draw the window while you're creating it
    get_gui()->unlock();
    autoview(window);
    get_gui()->lock();
    window.setup_gl_view();
  }
  CHECK_OPENGL_ERROR();

  profiler("setup gl");

  for (unsigned int s = 0; s < window.slices_.size(); ++s) {

    if (window.paint_layer_ &&
        window.slices_[s]->volume_ == window.paint_layer_->volume_)
      window.paint_layer_->draw();
    else 
      window.slices_[s]->draw();
  }

  profiler("draw slices");

  draw_slice_lines(window);

  profiler("draw slice lines");

  if (show_grid_()) 
    window.render_grid();

  profiler("render grid");

  if (event_.window_ == &window) {
    Point windowpos(event_.x_, event_.y_, 0);
    window.render_guide_lines(windowpos);
  }

  profiler("render_guide_lines");

  for (unsigned int t = 0; t < tools_.size(); ++t) {
    tools_[t]->draw(window);
    if (event_.window_ == &window)
      tools_[t]->draw_mouse_cursor(event_);
  }


  profiler("tool draw");

  window.render_text();

  profiler("render_text");

  if (filter_text_.length()) 
    window.render_progress_bar();

  profiler.leave();
  profiler.print();

  CHECK_OPENGL_ERROR();
  window.viewport_->release();


  return 1;
}

int
Painter::swap_window(SliceWindow &window) {
  window.viewport_->make_current();
  window.viewport_->swap();
  window.viewport_->release();
  return 1;
}


void
Painter::real_draw_all()
{
  get_gui()->lock();
  if (for_each(&Painter::render_window)) {
    for_each(&Painter::swap_window);
  }
  
  get_gui()->unlock();
}


void
Painter::redraw_all()
{
  for_each(&Painter::redraw_window);
}

int
Painter::redraw_window(SliceWindow &window) {
  window.redraw_ = true;
  return 1;
}

void
Painter::SliceWindow::push_gl_2d_view() {
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glScaled(2.0, 2.0, 2.0);
  glTranslated(-.5, -.5, -.5);
  double vpw = viewport_->width();
  double vph = viewport_->height();
  glScaled(1.0/vpw, 1.0/vph, 1.0);
  CHECK_OPENGL_ERROR();
}


void
Painter::SliceWindow::pop_gl_2d_view() {
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  CHECK_OPENGL_ERROR();
}


void
Painter::SliceWindow::render_guide_lines(Point mouse) {
  if (!show_guidelines_()) return;

  //  GLdouble yellow[4] = { 1.0, 0.76, 0.1, 0.8 };
  GLdouble white[4] = { 1.0, 1.0, 1.0, 1.0 };

  push_gl_2d_view();
  double vpw = viewport_->width();
  double vph = viewport_->height();

  glColor4dv(white);
  glBegin(GL_LINES); 
  glVertex3d(0, mouse.y(), mouse.z());
  glVertex3d(vpw, mouse.y(), mouse.z());
  glVertex3d(mouse.x(), 0, mouse.z());
  glVertex3d(mouse.x(), vph, mouse.z());
  glEnd();
  CHECK_OPENGL_ERROR();

  pop_gl_2d_view();

}



// renders vertical and horizontal bars that represent
// selected slices in other dimensions
void
Painter::draw_slice_lines(SliceWindow &window)
{
  if (!current_volume_) return;
  profiler.enter("draw_slice_lines");
  double upp = 100.0 / window.zoom_;    // World space units per one pixel

  //  GLdouble blue[4] = { 0.1, 0.4, 1.0, 0.8 };
  //  GLdouble green[4] = { 0.5, 1.0, 0.1, 0.8 };
  GLdouble red[4] = { 0.8, 0.2, 0.4, 0.9 };

  // Vector scale = current_volume_->scale();
  profiler("scale");
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  vector<int> zero_idx(current_volume_->nrrd_handle_->nrrd_->dim, 0);
  WindowLayouts::iterator liter = layouts_.begin();
  WindowLayouts::iterator lend = layouts_.end();
  while (liter != lend) {
    ASSERT(liter->second);
    WindowLayout &layout = *(liter->second);
    liter++;
    for (unsigned int win = 0; 
         win < layout.windows_.size(); ++win) {
      ASSERT(layout.windows_[win]);
      SliceWindow &window2 = *(layout.windows_[win]);
      if (&window == &window2) continue;
      if (Dot(window.normal_, window2.normal_) > 0.999) continue;
      Vector span = Cross(window.normal_, window2.normal_);
      vector<double> span_index = current_volume_->vector_to_index(span);
      int span_axis = max_vector_magnitude_index(span_index);

      vector<int> pos_idx = current_volume_->world_to_index(window2.center_);
      vector<int> min_idx = pos_idx;
      min_idx[span_axis] = 0;
      vector<int> max_idx = pos_idx;
      max_idx[span_axis] = current_volume_->nrrd_handle_->nrrd_->axis[span_axis].size;
      Point min = current_volume_->index_to_world(min_idx);
      Point max = current_volume_->index_to_world(max_idx);
      vector<int> one_idx = zero_idx;
      one_idx[window2.axis_+1] = 1;
      double scale = (current_volume_->index_to_world(one_idx) - 
                      current_volume_->index_to_world(zero_idx)).length();
      Vector wid = window2.normal_;
      wid.normalize();
      wid *= Max(upp, scale);
      glColor4dv(red);
      glBegin(GL_QUADS);    
      glVertex3dv(&min(0));
      glVertex3dv(&max(0));
      min = min + wid;
      max = max + wid;
      glVertex3dv(&max(0));
      glVertex3dv(&min(0));
      glEnd();
    }
  }
  profiler("done");
  profiler.leave();
  
}




double div_d(double dividend, double divisor) {
  return Floor(dividend/divisor);
}

double mod_d(double dividend, double divisor) {
  return dividend - Floor(dividend/divisor)*divisor;
}


void
Painter::SliceWindow::render_frame(double x,
                                   double y,
                                   double border_wid,
                                   double border_hei,
                                   double *color1,
                                   double *color2)
{
  const double vw = viewport_->width();
  const double vh = viewport_->height();
  if (color1)
    glColor4dv(color1);

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  CHECK_OPENGL_ERROR();

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glScaled(2.0, 2.0, 2.0);
  glTranslated(-.5, -.5, -.5);
  glScaled(1.0/vw, 1.0/vh, 1.0);
  CHECK_OPENGL_ERROR();

  glBegin(GL_QUADS);
  glVertex3d(x,y,0);
  glVertex3d(vw-x,y,0);
  glVertex3d(vw-x,y+border_hei,0);
  glVertex3d(x,y+border_hei,0);

  glVertex3d(vw-x-border_wid,y+border_hei,0);
  glVertex3d(vw-x,y+border_hei,0);
  glVertex3d(vw-x,vh-y,0);
  glVertex3d(vw-x-border_wid,vh-y,0);

  if (color2)
    glColor4dv(color2);

  glVertex3d(x,vh-y,0);
  glVertex3d(vw-x,vh-y,0);
  glVertex3d(vw-x-border_wid,vh-y-border_hei,0);
  glVertex3d(x,vh-y-border_hei,0);

  glVertex3d(x,y,0);
  glVertex3d(x+border_wid,y+border_hei,0);
  glVertex3d(x+border_wid,vh-y-border_hei,0);
  glVertex3d(x,vh-y-border_hei,0);

  glEnd();
  CHECK_OPENGL_ERROR();

  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  CHECK_OPENGL_ERROR();
}

  


// renders vertical and horizontal bars that represent
// selected slices in other dimensions
void
Painter::SliceWindow::render_grid()
{
  profiler.enter("render_grid");
  //  double one = 100.0 / zoom_;    // World space units per one pixel
  double units = zoom_ / 100.0;  // Pixels per world space unit
  const double pixels = 100.0;    // Number of target pixels for grid gap

  vector<double> gaps(1, 1.0);
  gaps.push_back(1/2.0);
  gaps.push_back(1/5.0);

  double realdiff = 10000;
  int selected = 0;
  for (unsigned int i = 0; i < gaps.size(); ++i) {
    bool done = false;
    double diff = fabs(gaps[i]*units-pixels);
    while (!done) {
      if (fabs((gaps[i]*10.0)*units - pixels) < diff) 
        gaps[i] *= 10.0;
      else if (fabs((gaps[i]/10.0)*units - pixels) < diff) 
        gaps[i] /= 10.0;
      else
        done = true;
      diff = fabs(gaps[i]*units-pixels);
    }

    if (diff < realdiff) {
      realdiff = diff;
      selected = i;
    }
  }
  double gap = gaps[selected];

  profiler("gaps");
  const int vw = viewport_->width();
  const int vh = viewport_->height();

  double grey1[4] = { 0.75, 0.75, 0.75, 1.0 };
  double grey2[4] = { 0.5, 0.5, 0.5, 1.0 };
  double grey3[4] = { 0.25, 0.25, 0.25, 1.0 };
  double white[4] = { 1,1,1,1 };
  render_frame(0,0, 15, 15, grey1);
  render_frame(15,15, 3, 3, white, grey2 );
  render_frame(17,17, 2, 2, grey3);
  profiler("render_frame");
  double grid_color = 0.25;

  glDisable(GL_TEXTURE_2D);
  CHECK_OPENGL_ERROR();

  Point min = screen_to_world(0,0);
  Point max = screen_to_world(vw-1, vh-1);

  int xax = x_axis();
  int yax = y_axis();
  min(xax) = div_d(min(xax), gap)*gap;
  min(yax) = div_d(min(yax), gap)*gap;

  vector<string> lab;
  lab.push_back("X: ");
  lab.push_back("Y: ");
  lab.push_back("Z: ");

  int num = 0;
  Point linemin = min;
  Point linemax = min;
  linemax(yax) = max(yax);
  TextRenderer &font1 = *painter_->font1_;
  string str;
  profiler("start");
  while (linemin(xax) < max(xax)) {
    linemin(xax) = linemax(xax) = min(xax) + gap*num;
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);

    glColor4d(grid_color, grid_color, grid_color, 0.25);
    glBegin(GL_LINES);
    glVertex3dv(&linemin(0));
    glVertex3dv(&linemax(0));
    glEnd();

    //    str = lab[xax]+to_string(linemin(xax));
    str = to_string(linemin(xax));
    Point pos = world_to_screen(linemin);
    font1.render(str, pos.x()+1, 2, 
                 TextRenderer::SHADOW | 
                 TextRenderer:: SW | 
                 TextRenderer::REVERSE);
    
    //    pos = world_to_screen(linemax);
    font1.render(str, pos.x()+1, vh-2, 
                 TextRenderer::SHADOW | 
                 TextRenderer:: NW | 
                 TextRenderer::REVERSE);
    num++;
  }
  profiler("horizontal");
  //  int wid = text.width();

  num = 0;
  linemin = linemax = min;
  linemax(xax) = max(xax);
  while (linemin(yax) < max(yax)) {
    linemin(yax) = linemax(yax) = min(yax) + gap*num;
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);

    glColor4d(grid_color, grid_color, grid_color, 0.25);
    glBegin(GL_LINES);
    glVertex3dv(&linemin(0));
    glVertex3dv(&linemax(0));
    glEnd();

    str = to_string(linemin(yax));
    Point pos = world_to_screen(linemin);
    font1.render(str, 2, pos.y(), 
                 TextRenderer::SHADOW | 
                 TextRenderer::NW | 
                 TextRenderer::VERTICAL | 
                 TextRenderer::REVERSE);

    font1.render(str, vw-2, pos.y(), 
                 TextRenderer::SHADOW | 
                 TextRenderer::NE | 
                 TextRenderer::VERTICAL | 
                 TextRenderer::REVERSE);
    num++;
  }
  profiler("vertical");
  profiler.leave();
  CHECK_OPENGL_ERROR();
}


Point
Painter::NrrdVolume::center(int axis, int slice) {
  vector<int> index(nrrd_handle_->nrrd_->dim,0);
  for (unsigned int a = 0; a < index.size(); ++a) 
    index[a] = nrrd_handle_->nrrd_->axis[a].size/2;
  if (axis >= 0 && axis < int(index.size()))
    index[axis] = Clamp(slice, 0, nrrd_handle_->nrrd_->axis[axis].size-1);
  ASSERT(index_valid(index));
  return index_to_world(index);
}


Point
Painter::NrrdVolume::min(int axis, int slice) {
  vector<int> index(nrrd_handle_->nrrd_->dim,0);
  if (axis >= 0 && axis < int(index.size()))
    index[axis] = Clamp(slice, 0, nrrd_handle_->nrrd_->axis[axis].size-1);
  ASSERT(index_valid(index));
  return index_to_world(index);
}

Point
Painter::NrrdVolume::max(int axis, int slice) {
  vector<int> index = max_index();
  if (axis >= 0 && axis < int(index.size()))
    index[axis] = Clamp(slice, 0, nrrd_handle_->nrrd_->axis[axis].size-1);
  ASSERT(index_valid(index));
  return index_to_world(index);
}



Vector
Painter::NrrdVolume::scale() {
  vector<int> index_zero(nrrd_handle_->nrrd_->dim,0);
  vector<int> index_one(nrrd_handle_->nrrd_->dim,1);
  return index_to_world(index_one) - index_to_world(index_zero);
}


double
Painter::NrrdVolume::scale(unsigned int axis) {
  ASSERT(axis >= 0 && (unsigned int) axis < nrrd_handle_->nrrd_->dim);
  return scale()[axis];
}



vector<int>
Painter::NrrdVolume::max_index() {
  vector<int> max_index(nrrd_handle_->nrrd_->dim,0);
  for (unsigned int a = 0; a < nrrd_handle_->nrrd_->dim; ++a)
    max_index[a] = nrrd_handle_->nrrd_->axis[a].size;
  return max_index;
}

int
Painter::NrrdVolume::max_index(unsigned int axis) {
  ASSERT(axis >= 0 && (unsigned int) axis < nrrd_handle_->nrrd_->dim);
  return max_index()[axis];
}

bool
Painter::NrrdVolume::inside_p(const Point &p) {
  return index_valid(world_to_index(p));
}



// Returns the index to the axis coordinate that is most parallel and 
// in the direction of X in the screen.  
// 0 for x, 1 for y, and 2 for z
int
Painter::SliceWindow::x_axis()
{
  Vector adir = Abs(x_dir());
  if ((adir[0] > adir[1]) && (adir[0] > adir[2])) return 0;
  if ((adir[1] > adir[0]) && (adir[1] > adir[2])) return 1;
  return 2;
}

// Returns the index to the axis coordinate that is most parallel and 
// in the direction of Y in the screen.  
// 0 for x, 1 for y, and 2 for z
int
Painter::SliceWindow::y_axis()
{
  Vector adir = Abs(y_dir());
  if ((adir[0] > adir[1]) && (adir[0] > adir[2])) return 0;
  if ((adir[1] > adir[0]) && (adir[1] > adir[2])) return 1;
  return 2;
}

void
Painter::SliceWindow::setup_gl_view()
{
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  CHECK_OPENGL_ERROR();
  if (!painter_->current_volume_) return;
  
  if (axis_ == 0) { // screen +X -> +Y, screen +Y -> +Z
    glRotated(-90,0.,1.,0.);
    glRotated(-90,1.,0.,0.);
  } else if (axis_ == 1) { // screen +X -> +X, screen +Y -> +Z
    glRotated(-90,1.,0.,0.);
  }
  CHECK_OPENGL_ERROR();
  
  // Do this here because x_axis and y_axis functions use these matrices
  glGetIntegerv(GL_VIEWPORT, gl_viewport_);
  glGetDoublev(GL_MODELVIEW_MATRIX, gl_modelview_matrix_);
  glGetDoublev(GL_PROJECTION_MATRIX, gl_projection_matrix_);
  CHECK_OPENGL_ERROR();

  double hwid = viewport_->width()*50.0/zoom_();
  double hhei = viewport_->height()*50.0/zoom_();
  int vol_x_fastest_axis = max_vector_magnitude_index
    (painter_->current_volume_->vector_to_index(x_dir()));
  int vol_y_fastest_axis = max_vector_magnitude_index
    (painter_->current_volume_->vector_to_index(y_dir()));

  double cx = center_(vol_x_fastest_axis);
  double cy = center_(vol_y_fastest_axis);
  cx = center_(x_axis());
  cy = center_(y_axis());
  
  double maxz = center_(axis_) + Max(hwid, hhei);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(cx - hwid, cx + hwid, cy - hhei, cy + hhei, -2*maxz, 2*maxz);
  glGetIntegerv(GL_VIEWPORT, gl_viewport_);
  glGetDoublev(GL_MODELVIEW_MATRIX, gl_modelview_matrix_);
  glGetDoublev(GL_PROJECTION_MATRIX, gl_projection_matrix_);
  CHECK_OPENGL_ERROR();
}


string
double_to_string(double val)
{
  char s[50];
  snprintf(s, 49, "%1.2f", val);
  return string(s);
}


// Right	= -X
// Left		= +X
// Posterior	= -Y
// Anterior	= +X
// Inferior	= -Z
// Superior	= +Z
void
Painter::SliceWindow::render_orientation_text()
{
  TextRenderer *text = painter_->font3_;
  if (!text) return;

  profiler.enter("render_orientation_text");
  int prim = x_axis();
  int sec = y_axis();
  
  string ltext, rtext, ttext, btext;

  if (painter_->anatomical_coordinates_()) {
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
  profiler("start render");
  text->set_shadow_offset(2,-2);
  text->render(ltext, 2, viewport_->height()/2,
            TextRenderer::W | TextRenderer::SHADOW | TextRenderer::REVERSE);
  profiler("ltext");

  text->render(rtext,viewport_->width()-2, viewport_->height()/2,
               TextRenderer::E | TextRenderer::SHADOW | TextRenderer::REVERSE);
  profiler("rtext");

  text->render(btext,viewport_->width()/2, 2,
               TextRenderer::S | TextRenderer::SHADOW | TextRenderer::REVERSE);
  profiler("btext");  

  text->render(ttext,viewport_->width()/2, viewport_->height()-2, 
               TextRenderer::N | TextRenderer::SHADOW | TextRenderer::REVERSE);
  profiler("ttext");

  profiler.leave();
}


unsigned int 
Painter::NrrdSlice::axis() {
  ASSERT(volume_);
  return max_vector_magnitude_index(volume_->vector_to_index(plane_.normal()));
}


void
Painter::NrrdSlice::bind()
{
  if (texture_ && tex_dirty_) { 
    delete texture_; 
    texture_ = 0; 
  }


  if (!texture_) {
    vector<int> index = volume_->world_to_index(plane_.project(Point(0,0,0)));
    unsigned int ax = axis();
    unsigned int slice = index[ax];
    volume_->mutex_.lock();
    if (slice>=0 && slice < volume_->nrrd_handle_->nrrd_->axis[ax].size)
      texture_ = scinew ColorMappedNrrdTextureObj(volume_->nrrd_handle_, 
                                                  ax,
                                                  slice, 
                                                  slice);
    volume_->mutex_.unlock();

    tex_dirty_ = true;
  }


  if (texture_ && tex_dirty_) {
    texture_->set_clut_minmax(volume_->clut_min_, volume_->clut_max_);
    ColorMapHandle cmap = painter_->get_colormap(volume_->colormap_.get());
    texture_->set_colormap(cmap);
  }


  tex_dirty_ = false;
  geom_dirty_ = true;

  return;
}


void
Painter::SliceWindow::render_progress_bar() {
  GLdouble grey[4] = { 0.6, 0.6, 0.6, 0.6 }; 
  GLdouble white[4] = { 1.0, 1.0, 1.0, 1.0 }; 
  GLdouble black[4] = { 0.0, 0.0, 0.0, 1.0 }; 
  GLdouble yellow[4] = { 1.0, 0.76, 0.1, 1.0 };
  GLdouble lt_yellow[4] = { 0.8, 0.5, 0.1, 1.0 };  
  
  GLdouble *colors[5] = { lt_yellow, yellow, black, grey, white };
  GLdouble widths[5] = { 11, 9.0, 7.0, 5.0, 1.0 }; 

  push_gl_2d_view();

  double vpw = viewport_->width();
  double vph = viewport_->height();
  double x_off = 50;
  double h = 50;
  double gap = 5;
  //  double y_off = 20;

  Point ll(x_off, vph/2.0 - h/2, 0);
  Point lr(vpw-x_off, vph/2.0 - h/2, 0);
  Point ur(vpw-x_off, vph/2.0 + h/2, 0);
  Point ul(x_off, vph/2.0 + h/2, 0);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glEnable(GL_LINE_SMOOTH);
  for (int pass = 2; pass < 5; ++pass) {
    glColor4dv(colors[pass]);
    glLineWidth(widths[pass]);    

    glBegin(GL_LINE_LOOP);
    {
      glVertex3dv(&ll(0));
      glVertex3dv(&lr(0));
      glVertex3dv(&ur(0));
      glVertex3dv(&ul(0));
    }
    glEnd();
  }
  glLineWidth(1.0);
  glDisable(GL_LINE_SMOOTH);
  CHECK_OPENGL_ERROR();

  Vector right = Vector(vpw - 2 *x_off - 2*gap, 0, 0);
  Vector up = Vector(0, h - gap * 2, 0);

  ll = ll + Vector(gap, gap, 0);
  lr = ll + right;
  ur = lr + up;
  ul = ll + up;

  glColor4dv(yellow);
  glBegin(GL_QUADS);
  glVertex3dv(&ll(0));
  glVertex3dv(&lr(0));
  glVertex3dv(&ur(0));
  glVertex3dv(&ul(0));
  glEnd();
  CHECK_OPENGL_ERROR();
  


  pop_gl_2d_view();
}




ColorMapHandle
Painter::get_colormap(int id)
{
  if (id > 0 && id <= int(colormap_names_.size()) &&
      colormaps_.find(colormap_names_[id - 1]) != colormaps_.end())
    return colormaps_[colormap_names_[id - 1]];
  return 0;
}
    

void
Painter::NrrdSlice::draw()
{
  if (nrrd_dirty_) {
    set_coords();
    nrrd_dirty_ = false;
    tex_dirty_ = true;
    geom_dirty_ = false;
  }

  float a = volume_->opacity_;
  glColor4f(a,a,a,a);
  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glShadeModel(GL_FLAT);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  CHECK_OPENGL_ERROR();  

  bind();
  if (texture_)
    texture_->draw_quad(pos_, xdir_, ydir_);
}

#if 0
void
Painter::SliceWindow::render_vertical_text(FreeTypeTextTexture *text,
                                           double x, double y)
{
  string str = text->get_string();
  int hei = text->height();
  for (unsigned int i = 0; i < str.length(); ++i) 
  {
    text->set(str.substr(i, 1));
    text->draw(x,y, FreeTypeTextTexture::n);
    y -= 2 + hei;
  }
}
#endif


static int dots = 0;

void
Painter::SliceWindow::render_text()
{
  if (!painter_->show_text_()) return;
  profiler.enter("render_text");  
  if (!(painter_->font1_ && painter_->font2_ && painter_->font3_)) return;
  TextRenderer &font1 = *painter_->font1_;
  TextRenderer &font2 = *painter_->font2_;
  //TextRenderer &font3 = *painter_->font3_;
  
  const int yoff = 19;
  const int xoff = 19;
  const int vw = viewport_->width();
  const int vh = viewport_->height();
  
  const int y_pos = font1.height("X")+2;
  font1.render("Zoom: "+to_string(zoom_())+"%", xoff, yoff, 
               TextRenderer::SHADOW | TextRenderer::SW);
  profiler("zoom");


  NrrdVolume *vol = painter_->current_volume_;
  
  for (unsigned int s = 0; s < slices_.size(); ++s) {
    string str = slices_[s]->volume_->name_prefix_ + 
      slices_[s]->volume_->name_.get();
    if (slices_[s]->volume_ == vol) {
      font1.set_color(240/255.0, 1.0, 0.0, 1.0);
      str = "->" + str;
    } else {
      font1.set_color(1.0, 1.0, 1.0, 1.0);
    }

    font1.render(str,
                 vw-2-xoff, vh-2-yoff-(y_pos*(slices_.size()-1-s)),
                 TextRenderer::NE | TextRenderer::SHADOW);
  }
  profiler("volumes");
  font1.set_color(1.0, 1.0, 1.0, 1.0);


  if (painter_->filter_text_.length()) {
    string filter = painter_->filter_text_;
    for (unsigned int d = 0; d < (dots/(painter_->layouts_.size()*3))%5; ++d)
      filter = filter + ".";
    ++dots;
    filter = filter + ".";
    font1.render(filter, xoff+2+1, vh-2-yoff-1,
                 TextRenderer::NW | TextRenderer::SHADOW);
  } else if (painter_->tools_.size())
    font1.render(painter_->tools_.back()->get_name(), 
                 xoff+2+1, vh-2-yoff-1, 
                 TextRenderer::NW | TextRenderer::SHADOW);

  profiler("tool");

  if (vol) {
    const float ww = vol->clut_max_ - vol->clut_min_;
    const float wl = vol->clut_min_ + ww/2.0;
    font1.render("WL: " + to_string(wl) +  " -- WW: " + to_string(ww),
                 xoff, y_pos+yoff,
                 TextRenderer::SHADOW | TextRenderer::SW);

    profiler("WLWW");
    font1.render("Min: " + to_string(vol->clut_min_) + 
                 " -- Max: " + to_string(vol->clut_max_),
                 xoff, y_pos*2+yoff, TextRenderer::SHADOW | TextRenderer::SW);
    profiler("Min/Max");
    if (this == painter_->event_.window_) {
      font1.render("X: "+double_to_string(painter_->event_.position_.x())+
                   " Y: "+double_to_string(painter_->event_.position_.y())+
                   " Z: "+double_to_string(painter_->event_.position_.z()),
                   viewport_->width()-2-xoff, yoff,
                   TextRenderer::SHADOW | TextRenderer::SE);
      profiler("XYZ");
      vector<int> index = vol->world_to_index(painter_->event_.position_);
      if (vol->index_valid(index)) {
        font1.render("S: "+to_string(index[1])+
                     " C: "+to_string(index[2])+
                     " A: "+to_string(index[3]),
                     viewport_->width()-2-xoff, yoff+y_pos,
                     TextRenderer::SHADOW | TextRenderer::SE);
        profiler("SCA");
        double val = 1.0;
        vol->get_value(index, val);
        font1.render("Value: " + to_string(val),
                     viewport_->width()-2-xoff,y_pos*2+yoff, 
                     TextRenderer::SHADOW | TextRenderer::SE);
        profiler("VALUE");
      }
    }
  }
    
  render_orientation_text();

  string str;
  if (!painter_->anatomical_coordinates_()) { 
    switch (axis_) {
    case 0: str = "Sagittal"; break;
    case 1: str = "Coronal"; break;
    default:
    case 2: str = "Axial"; break;
    }
  } else {
    switch (axis_) {
    case 0: str = "YZ Plane"; break;
    case 1: str = "XZ Plane"; break;
    default:
    case 2: str = "XY Plane"; break;
    }
  }

  if (mode_ == slab_e) str = "SLAB - "+str;
  else if (mode_ == mip_e) str = "MIP - "+str;

  font2.set_shadow_offset(2,-2);
  font2.render(str, viewport_->width() - 2, 2,
               TextRenderer::SHADOW | 
               TextRenderer::SE | 
               TextRenderer::REVERSE);
  profiler("plane");
  profiler.leave();
}


void
Painter::NrrdSlice::set_coords() {  
  vector<int> sindex = volume_->world_to_index(plane_.project(Point(0,0,0)));
  unsigned int ax = axis();
  pos_ = volume_->min(ax, sindex[ax]);
  vector<int> index(volume_->nrrd_handle_->nrrd_->dim,0);

  int primary = (ax == 1) ? 2 : 1;
  index[primary] = volume_->nrrd_handle_->nrrd_->axis[primary].size;
  xdir_ = volume_->index_to_world(index) - pos_;
  index[primary] = 0;

  int secondary = (ax == 3) ? 2 : 3;
  index[secondary] = volume_->nrrd_handle_->nrrd_->axis[secondary].size;
  ydir_ = volume_->index_to_world(index) - pos_;
}

int
Painter::extract_window_slices(SliceWindow &window) {
  for (unsigned int s = volumes_.size(); s < window.slices_.size(); ++s) {
    delete window.slices_[s];
  }

  if (window.slices_.size() > volumes_.size())
    window.slices_.resize(volumes_.size());
  
  for (unsigned int s = window.slices_.size(); s < volumes_.size(); ++s) {
    //    if (volumes_[s] == current_volume_) 
    //      window.center_ = current_volume_->center();
    window.slices_.push_back(scinew NrrdSlice(this, volumes_[s], 
                                              window.center_, window.normal_));
  }
  window.slice_map_.clear();
  for (unsigned int s = 0; s < volumes_.size(); ++s) {
    window.slice_map_[volumes_[s]] = window.slices_[s];
    window.slices_[s]->volume_ = volumes_[s];
    window.slices_[s]->plane_ = Plane(window.center_, window.normal_);
  }

  for_each(window, &Painter::set_slice_nrrd_dirty);

  if (window.paint_layer_) {
    delete window.paint_layer_;
    window.paint_layer_ = 0;
  }
  return window.slices_.size();
}


#if 0
int
Painter::extract_window_slices(SliceWindow &window) {
  for (unsigned int s = 0; s < window.slices_.size(); ++s)
    delete window.slices_[s];

  window.slices_.clear();

  for (unsigned int s = 0; s < volumes_.size(); ++s) {
    window.slices_.push_back
      (scinew NrrdSlice(this, volumes_[s], 
                        window.center_, window.normal_));
  }
    
//   if (window.paint_layer_) {
//     delete window.paint_layer_;
//     window.paint_layer_ = 0;
//   }
  return window.slices_.size();
}
#endif
  
int
Painter::set_slice_nrrd_dirty(NrrdSlice &slice) {
  slice.nrrd_dirty_ = true;
  return 0;
}




void
Painter::set_axis(SliceWindow &window, unsigned int axis) {
  window.axis_ = axis;
  window.normal_ = Vector(axis == 0 ? 1 : 0,
                          axis == 1 ? 1 : 0,
                          axis == 2 ? 1 : 0);
  extract_window_slices(window);
  window.redraw_ = true;
}


void
Painter::SliceWindow::prev_slice()
{
  NrrdVolume *volume = painter_->current_volume_;
  if (!volume) return;
  vector<double> delta = volume->vector_to_index(normal_);
  unsigned int index = max_vector_magnitude_index(delta);
  delta[index] /= fabs(delta[index]);
  Point new_center = center_ - volume->index_to_vector(delta);
  if (!painter_->current_volume_->inside_p(new_center)) return;
  center_ = new_center;
  painter_->extract_window_slices(*this);
  painter_->redraw_all();
}


void
Painter::SliceWindow::next_slice()
{
  NrrdVolume *volume = painter_->current_volume_;
  if (!volume) return;
  vector<double> delta = volume->vector_to_index(-normal_);
  unsigned int index = max_vector_magnitude_index(delta);
  delta[index] /= fabs(delta[index]);
  Point new_center = center_ - volume->index_to_vector(delta);
  if (!painter_->current_volume_->inside_p(new_center)) return;
  center_ = new_center;
  painter_->extract_window_slices(*this);
  painter_->redraw_all();
}

void
Painter::layer_up()
{
  if (!current_volume_) return;
  unsigned int i = 0;
  for (i = 0; i < volumes_.size(); ++i)
    if (volumes_[i] == current_volume_) break;
  ASSERT(volumes_[i] == current_volume_);
  if (i == volumes_.size()-1) return;

  NrrdVolumeOrder::iterator voiter1 = std::find(volume_order_.begin(), 
                                                volume_order_.end(), 
                                                volumes_[i]->name_.get());

  //    volume_order_.find();
  NrrdVolumeOrder::iterator voiter2 =std::find(volume_order_.begin(),
                                               volume_order_.end(),
                                               volumes_[i+1]->name_.get());
  //    volume_order_.find(volumes_[i+1]->name_.get());
  ASSERT(voiter1 != volume_order_.end());
  ASSERT(voiter2 != volume_order_.end());

  NrrdVolume *tempvol = volumes_[i+1];
  volumes_[i+1] = volumes_[i];
  volumes_[i] = tempvol;
  
  NrrdVolumeOrder::value_type temporder = *voiter2;
  *voiter2 = *voiter1;
  *voiter1 = temporder;
  
  for_each(&Painter::extract_window_slices);
  redraw_all();
}

void
Painter::layer_down()
{
  if (!current_volume_) return;
  unsigned int i = 0;
  for (i = 0; i < volumes_.size(); ++i)
    if (volumes_[i] == current_volume_) break;
  ASSERT(volumes_[i] == current_volume_);
  if (i == 0) return;


  NrrdVolumeOrder::iterator voiter1 = 
    std::find(volume_order_.begin(), volume_order_.end(), volumes_[i]->name_.get());

  //    volume_order_.find();
  NrrdVolumeOrder::iterator voiter2 =
    std::find(volume_order_.begin(),volume_order_.end(),volumes_[i-1]->name_.get());

  ASSERT(voiter1 != volume_order_.end());
  ASSERT(voiter2 != volume_order_.end());

  NrrdVolumeOrder::value_type temporder = *voiter2;
  *voiter2 = *voiter1;
  *voiter1 = temporder;

  NrrdVolume *temp = volumes_[i-1];
  volumes_[i-1] = volumes_[i];
  volumes_[i] = temp;

  for_each(&Painter::extract_window_slices);
  redraw_all();
}




void
Painter::SliceWindow::zoom_in()
{
  zoom_ *= 1.1;
  redraw_ = true;
}

void
Painter::SliceWindow::zoom_out()
{
  zoom_ /= 1.1;
  redraw_ = true;
}
  
Point
Painter::SliceWindow::screen_to_world(unsigned int x, unsigned int y) {
  GLdouble xyz[3];
  gluUnProject(double(x)+0.5, double(y)+0.5, 0,
	       gl_modelview_matrix_, 
	       gl_projection_matrix_,
	       gl_viewport_,
	       xyz+0, xyz+1, xyz+2);
  xyz[axis_] = center_(axis_);
  return Point(xyz[0], xyz[1], xyz[2]);
}


Point
Painter::SliceWindow::world_to_screen(const Point &world)
{
  GLdouble xyz[3];
  gluProject(world(0), world(1), world(2),
             gl_modelview_matrix_, 
             gl_projection_matrix_,
	     gl_viewport_,
             xyz+0, xyz+1, xyz+2);
  return Point(xyz[0], xyz[1], xyz[2]);
}


Vector
Painter::SliceWindow::x_dir()
{
  return screen_to_world(1,0) - screen_to_world(0,0);
}

Vector
Painter::SliceWindow::y_dir()
{
  return screen_to_world(0,1) - screen_to_world(0,0);
}



Point
Painter::NrrdVolume::index_to_world(const vector<int> &index) {
  unsigned int dim = index.size()+1;
  ColumnMatrix index_matrix(dim);
  ColumnMatrix world_coords(dim);
  for (unsigned int i = 0; i < dim-1; ++i)
    index_matrix[i] = index[i];
  index_matrix[dim-1] = 1.0;
  DenseMatrix transform = transform_;
  int tmp1, tmp2;
  transform.mult(index_matrix, world_coords, tmp1, tmp2);
  Point return_val;
  for (int i = 1; i < 4; ++i) 
    return_val(i-1) = world_coords[i];
  return return_val;
}


Point
Painter::NrrdVolume::index_to_point(const vector<double> &index) {
  unsigned int dim = index.size()+1;
  ColumnMatrix index_matrix(dim);
  ColumnMatrix world_coords(dim);
  for (unsigned int i = 0; i < dim-1; ++i)
    index_matrix[i] = index[i];
  index_matrix[dim-1] = 1.0;
  DenseMatrix transform = transform_;
  int tmp1, tmp2;
  transform.mult(index_matrix, world_coords, tmp1, tmp2);
  Point return_val;
  for (int i = 1; i < 4; ++i) 
    return_val(i-1) = world_coords[i];
  return return_val;
}


vector<int> 
Painter::NrrdVolume::world_to_index(const Point &p) {
  DenseMatrix transform = transform_;
  ColumnMatrix index_matrix(transform.ncols());
  ColumnMatrix world_coords(transform.nrows());
  for (int i = 0; i < transform.nrows(); ++i)
    if (i > 0 && i < 4) 
      world_coords[i] = p(i-1)-transform.get(i,transform.ncols()-1);
    else       
      world_coords[i] = 0.0;;
  transform.solve(world_coords, index_matrix, 1);
  vector<int> return_val(index_matrix.nrows()-1);
  for (unsigned int i = 0; i < return_val.size(); ++i) {
    return_val[i] = Floor(index_matrix[i]);
  }
  return return_val;
}

vector<double> 
Painter::NrrdVolume::point_to_index(const Point &p) {
  DenseMatrix transform = transform_;
  ColumnMatrix index_matrix(transform.ncols());
  ColumnMatrix world_coords(transform.nrows());
  for (int i = 0; i < transform.nrows(); ++i)
    if (i > 0 && i < 4) 
      world_coords[i] = p(i-1)-transform.get(i,transform.ncols()-1);
    else       
      world_coords[i] = 0.0;;
  transform.solve(world_coords, index_matrix, 1);
  vector<double> return_val(index_matrix.nrows()-1);
  for (unsigned int i = 0; i < return_val.size(); ++i) {
    return_val[i] = index_matrix[i];
  }
  return return_val;
}




vector<double> 
Painter::NrrdVolume::vector_to_index(const Vector &v) {
  Point zero(0,0,0);
  vector<double> zero_idx = point_to_index(zero);
  vector<double> idx = point_to_index(v.asPoint());
  for (unsigned int i = 0; i < zero_idx.size(); ++i) 
    idx[i] = idx[i] - zero_idx[i];
  return idx;
    
//   DenseMatrix transform = transform_;
//   ColumnMatrix index_matrix(transform.ncols());
//   ColumnMatrix world_coords(transform.nrows());
//   for (int i = 0; i < transform.nrows(); ++i)
//     if (i > 0 && i < 4) 
//       world_coords[i] = v[i-1];
//     else       
//       world_coords[i] = 0.0;;
//   int tmp, tmp2;
//   transform.mult_transpose(world_coords, index_matrix, tmp, tmp2);
//   vector<double> return_val(index_matrix.nrows()-1);
//   for (unsigned int i = 0; i < return_val.size(); ++i)
//     return_val[i] = index_matrix[i];
//   return return_val;
}


Vector 
Painter::NrrdVolume::index_to_vector(const vector<double> &index) {
  vector<double> zero_index(index.size(),0.0);
  return index_to_point(index) - index_to_point(zero_index);
}



void
Painter::NrrdVolume::build_index_to_world_matrix() {
  Nrrd *nrrd = nrrd_handle_->nrrd_;
  int dim = nrrd->dim+1;
  DenseMatrix matrix(dim, dim);
  matrix.zero();
  for (int i = 0; i < dim-1; ++i) {
    if (airExists(nrrd->axis[i].spacing))
      matrix.put(i,i,nrrd->axis[i].spacing);
    else 
      matrix.put(i,i,((nrrd->axis[i].max-nrrd->axis[i].min+1.0)/
                      nrrd->axis[i].size));
    if (airExists(nrrd->axis[i].min))
      matrix.put(i, nrrd->dim, nrrd->axis[i].min);
  }

  if (nrrd->axis[0].size != 1) {
    matrix.put(2,nrrd->dim, nrrd->axis[2].min+nrrd->axis[2].size*matrix.get(2,2));
    matrix.put(2,2,-matrix.get(2,2));
  }


  matrix.put(dim-1, dim-1, 1.0);
    
  transform_ = matrix;
}

bool
Painter::NrrdVolume::index_valid(const vector<int> &index) {
  unsigned int dim = nrrd_handle_->nrrd_->dim;
  if (index.size() != dim) return false;
  for (unsigned int a = 0; a < dim; ++a) 
    if (index[a] < 0 ||
	(unsigned int) index[a] >= nrrd_handle_->nrrd_->axis[a].size) {
      return false;
    }
  return true;
}
  
void
Painter::send_data()
{
  BundleHandle bundle = new Bundle();
  NrrdVolumeMap::iterator viter = volume_map_.begin();
  NrrdVolumeMap::iterator vend = volume_map_.end();
  for (; viter != vend; ++viter)
    if (viter->second) {
      NrrdDataHandle nrrd = viter->second->get_nrrd();
      bundle->setNrrd(viter->first, nrrd);
    }

  
//   for (unsigned int v = 0; v < volumes_.size(); ++v) {
//     string name = volumes_[v]->name_.get();
//     NrrdDataHandle nrrd = volumes_[v]->get_nrrd();
//     bundle->setNrrd(name, nrrd);
//   }

  BundleOPort *oport = (BundleOPort *)get_oport("Paint Data");
  ASSERT(oport);
  oport->send(bundle);  
}



void
Painter::receive_filter_bundles(Bundles &bundles)
{   
  BundleIPort *filter_port = (BundleIPort *)get_iport("Filter Data");
  ASSERT(filter_port);
  BundleHandle bundle = 0;
  filter_port->get(bundle);
  if (bundle.get_rep()) 
    bundles.push_back(bundle);
}

void
Painter::receive_normal_bundles(Bundles &bundles)
{
  BundleIPort *filter_port = (BundleIPort *)get_iport("Paint Data");
  ASSERT(filter_port);
  BundleHandle bundle = 0;
  filter_port->get(bundle);
  if (bundle.get_rep()) 
    bundles.push_back(bundle);
}
  
void
Painter::extract_data_from_bundles(Bundles &bundles)
{
  vector<NrrdDataHandle> nrrds;
  vector<string> nrrd_names;
  colormaps_.clear();
  colormap_names_.clear();

  for (unsigned int b = 0; b < bundles.size(); ++b) {
    int numNrrds = bundles[b]->numNrrds();
    for (int n = 0; n < numNrrds; n++) {
      string name = bundles[b]->getNrrdName(n);
      NrrdDataHandle nrrdH = bundles[b]->getNrrd(name);
      if (!nrrdH.get_rep()) continue;
      if (nrrdH->nrrd_->dim < 2)
      {
        warning("Nrrd with dim < 2, skipping.");
        continue;
      }
      nrrds.push_back(nrrdH);
      nrrd_names.push_back(name);
    }
    
    int numColormaps = bundles[b]->numColorMaps();
    for (int n = 0; n < numColormaps; n++) {
      const string name = bundles[b]->getColorMapName(n);
      ColorMapHandle cmap = bundles[b]->getColorMap(name);
      if (cmap.get_rep()) {
        colormaps_[name] = cmap;
        colormap_names_.push_back(name);
      }
    }
  }
  
  update_state(Module::Executing);

  get_gui()->lock();
  NrrdVolumeMap::iterator viter = volume_map_.begin();
  NrrdVolumeMap::iterator vend = volume_map_.end();

  for (unsigned int n = 0; n < nrrds.size(); ++n) {
    string name = nrrd_names[n];
    viter = volume_map_.find(name);
    if (viter == vend || viter->second == 0) {
      volume_map_[name] = 
        new NrrdVolume(get_ctx()->subVar(name), name, nrrds[n]);
      show_volume(name);
    } else {
      viter->second->set_nrrd(nrrds[n]);
    }
    volume_map_[name]->keep_ = 1;
  }
  get_gui()->unlock();
}



void
Painter::execute()
{
  update_state(Module::JustStarted);
  update_state(Module::NeedData);

  Bundles bundles;
  if (filter_) {
    send_data();
    receive_filter_bundles(bundles);
  }
  receive_normal_bundles(bundles);
  extract_data_from_bundles(bundles);
  recompute_volume_list();
  if (!filter_)
    send_data();

  update_state(Module::Completed);
}


int
Painter::set_probe(SliceWindow &window) {
  if (event_.window_ != &window &&
      current_volume_->inside_p(event_.position_)) {
    window.center_(window.axis_) = event_.position_(window.axis_);
    extract_window_slices(window);
  }
  return 1;
}


int
Painter::rebind_slice(NrrdSlice &slice) {
  slice.tex_dirty_ = true;
  return 1;
}

  


void
Painter::handle_gui_mouse_enter(GuiArgs &args) {
  ASSERTMSG(layouts_.find(args[3]) != layouts_.end(),
	    ("Cannot handle enter on "+args[3]).c_str());
  ASSERTMSG(current_layout_ == 0, "Haven't left window");
  current_layout_ = layouts_[args[3]];
  if (event_.window_)
    redraw_window(*event_.window_);
}


void
Painter::handle_gui_mouse_leave(GuiArgs &args) {
  current_layout_ = 0;
  if (event_.last_window_)
    redraw_window(*event_.last_window_);
}

bool
Painter::Event::button(unsigned int button)
{
  const int mask = Event::BUTTON_1_E << (button-1);
  return (state_ & mask) ? true : false;
}

bool
Painter::Event::shift()
{
  return (state_ & Event::SHIFT_E) ? true : false;
}

bool
Painter::Event::control()
{
  return (state_ & Event::CONTROL_E) ? true : false;
}

bool
Painter::Event::alt()
{
  return (state_ & Event::ALT_E) ? true : false;
}

void
Painter::Event::update_state(GuiArgs &args, Painter &painter) {
  ASSERT(painter.layouts_.find(args[3]) != painter.layouts_.end());
  Painter::WindowLayout &layout = *painter.layouts_[args[3]];

  state_ = args.get_int(5);

  if (args[2] == "motion")
    type_ = MOUSE_MOTION_E;
  else if (args[2] == "button")
    type_ = BUTTON_PRESS_E;
  else if (args[2] == "release")
    type_ = BUTTON_RELEASE_E;
  else if (args[2] == "keydown")
    type_ = KEY_PRESS_E;
  else if (args[2] == "keyup")
    type_ = KEY_RELEASE_E;
  else if (args[2] == "enter")
    type_ = FOCUS_IN_E;
  else if (args[2] == "leave")
    type_ = FOCUS_OUT_E;


  if (args[2] == "keydown" || args[2] == "keyup") {
    TCLKeysym_t::iterator keysym = tcl_keysym.find(args[4]);

    if (keysym != tcl_keysym.end()) {
      key_ = "";
      key_.push_back(keysym->second);
    } else {
      key_ = args[4];
    }


    if (args[2] == "keydown") 
      keys_.insert(key_);
    else {
      set<string>::iterator pos = keys_.find(key_);
      if (pos != keys_.end())
        keys_.erase(pos);
      else {
        return;
      }
    }
    return;
  }
  
  
  if (args[2] != "motion" && args[2] != "enter" && args[2] != "leave") {
    // The button parameter may be invalid on motion events (mainly OS X)
    // Button presses don't set state correctly, so manually set state_ here
    // to make Event::button() method work on press events
    button_ = args.get_int(4);
    switch (button_) {
    case 1: state_ |= BUTTON_1_E; break;
    case 2: state_ |= BUTTON_2_E; break;
    case 3: state_ |= BUTTON_3_E; break;
    case 4: state_ |= BUTTON_4_E; break;
    case 5: state_ |= BUTTON_5_E; break;
    default: break;
    }
  }

  X_ = args.get_int(6);
  Y_ = args.get_int(7);
  x_ = args.get_int(8);
  y_ = layout.opengl_->height() - 1 - args.get_int(9);

  last_window_ = window_;
  window_ = 0;
  SliceWindows::iterator window = layout.windows_.begin();
  while (window != layout.windows_.end() && !window_) {
    if (x_ >= (*window)->viewport_->x() && 
        x_ <  (*window)->viewport_->x() + (*window)->viewport_->width() &&
        y_ >= (*window)->viewport_->y() &&
        y_ <  (*window)->viewport_->y() + (*window)->viewport_->height()) {
      
      window_ = (*window);
      position_ = window_->screen_to_world(x_, y_);
    }
    ++window;
  }
}



void
Painter::handle_gui_mouse_button_press(GuiArgs &args) {
  if (!event_.window_) return;

  switch (event_.button_) {
  case 1:
    if (event_.shift())
      tools_.push_back(new PanTool(this));
    else
      tools_.push_back(new CLUTLevelsTool(this));
    break;
    
  case 2:
    if (event_.shift())
      tools_.push_back(new AutoviewTool(this));
    else
      tools_.push_back(new ProbeTool(this));
    break;
  case 3:
    if (event_.shift())
      tools_.push_back(new ZoomTool(this));
    break;
  case 4:
    if (event_.control()) 
      event_.window_->zoom_in();
    else
      event_.window_->next_slice();
    break;
    
  case 5:
    if (event_.shift())
      event_.window_->zoom_out();
    else
      event_.window_->prev_slice();
    break;
    
  default: 
    break;
  }

  int tool = tools_.size();
  while (tool > 0) {
    --tool;
    switch (tools_[tool]->do_event(event_)) {
    case PainterTool::HANDLED_E: {
      return;
    } break;
    case PainterTool::QUIT_E: { 
      delete tools_[tool];
      tools_.erase(tools_.begin()+tool);
      return;
    } break;
    case PainterTool::ERROR_E: { 
      cerr << tools_[tool]->get_name() << " Tool Error: " 
           << tools_[tool]->err() << std::endl;
      return;
    } break;
    default: // nothing, continue to next tool 
      break;
    }
  }
}


  
void
Painter::handle_gui_keypress(GuiArgs &args) {
  ASSERT(layouts_.find(args[3]) != layouts_.end());
  if (!event_.window_) return;
  SliceWindow &window = *event_.window_;
  string &key = event_.key_;
  unsigned int numtools = tools_.size();
  if (key == "=" || key == "+") window.zoom_in();
  else if (key == "-" || key == "_") window.zoom_out();
  else if (key == "<" || key == ",") window.prev_slice();
  else if (key == ">" || key == ".") window.next_slice();
  else if (key == "[") {
    if (current_volume_) {
      current_volume_->colormap_.set(Max(0, current_volume_->colormap_.get()-1));
      for_each(&Painter::rebind_slice);
      for_each(&Painter::redraw_window);
    }
  }
  else if (key == "]") {
    if (current_volume_) {
      current_volume_->colormap_.set(Min(int(colormap_names_.size()), 
                                         current_volume_->colormap_.get()+1));
      for_each(&Painter::rebind_slice);
      for_each(&Painter::redraw_window);
    }
  }
  else if (key == "s") want_to_execute();
  else if (key == "u") undo_volume();
  else if (key == "q") { 
    for (unsigned int t = 0; t < tools_.size(); ++t)
      delete tools_[t];
    tools_.clear();
  }
  else if (key == "x") {
    if (current_volume_) {
      current_volume_->keep_ = 0;
      recompute_volume_list();
    }    
  }
  else if (key == "c") {
    if (current_volume_) {
      string base = current_volume_->name_.get();
      string::size_type pos = base.find_last_not_of(" 0123456789");
      base = base.substr(0, pos+1);
      int i = 0;
      string name = base + " "+to_string(++i);
      while (volume_map_.find(name) != volume_map_.end())
        name = base + " "+to_string(++i);
      current_volume_ = copy_current_volume(name,0);
    }
  }
  else if (key == "v") {
    if (current_volume_) {
      string base = "New Layer";
      int i = 0;
      string name = base + " "+to_string(++i);
      while (volume_map_.find(name) != volume_map_.end())
        name = base + " "+to_string(++i);
      current_volume_ = copy_current_volume(name,1);
    }
  } else if (key == "a") {
    if (tools_.empty())
      tools_.push_back(new CropTool(this));
  } else if (key == "f") {
    tools_.push_back(new FloodfillTool(this));
  } else if (key == "b") {
    if (tools_.empty())
      tools_.push_back(new BrushTool(this));
  } else if (key == "h") {
    if (tools_.empty()) {
      tools_.push_back(new BrushTool(this));
      tools_.push_back(new ITKThresholdTool(this));
    }
  }
  else if (key == "k") {
    if (tools_.empty())
      tools_.push_back(new ITKConfidenceConnectedImageFilterTool(this));
  }
  else if (key == "g") ITKGradientMagnitudeTool temp(this);
  else if (key == "j") ITKBinaryDilateErodeTool temp(this);
  else if (key == "d") ITKCurvatureAnisotropicDiffusionTool temp(this);
  else if (key == "m") 
    LayerMergeTool temp(this);
  else if (key == "l") { 
    tools_.push_back(new StatisticsTool(this));
  }

  else if (key == "p") { 
    if (current_volume_) {
      current_volume_->opacity_ = 
        Clamp(current_volume_->opacity_+0.05, 0.0, 1.0);
      
      for_each(&Painter::redraw_window);
    }
  }
  else if (key == "o") { 
    if (current_volume_) {
      current_volume_->opacity_ = 
        Clamp(current_volume_->opacity_-0.05, 0.0, 1.0);
      for_each(&Painter::redraw_window);
    }
  }
  else if (key == "r") {
    if (current_volume_) {
      current_volume_->clut_min_ = current_volume_->data_min_;
      current_volume_->clut_max_ = current_volume_->data_max_;
      for_each(&Painter::rebind_slice);
      for_each(&Painter::redraw_window);
    }
  } 
  else if (key == "Left") layer_down();
  else if (key == "Right") layer_up(); 
  else if (key == "Down") {
    if (volumes_.size() < 2 || current_volume_ == volumes_[0]) 
      return;
    for (unsigned int i = 1; i < volumes_.size(); ++i)
      if (current_volume_ == volumes_[i]) {
        current_volume_ = volumes_[i-1];
        redraw_all();
        return;
      }
          
  } else if (key == "Up") {
    if (volumes_.size() < 2 || current_volume_ == volumes_.back()) 
      return;
    for (unsigned int i = 0; i < volumes_.size()-1; ++i)
      if (current_volume_ == volumes_[i]) {
        current_volume_ = volumes_[i+1];
        redraw_all();
        return;
      }
  }
  if (numtools != tools_.size())
    redraw_all();
}


void
Painter::tcl_command(GuiArgs& args, void* userdata) {
  if(args.count() < 2) {
    args.error("Painter needs a minor command");
    return;
  }

  if (sci_getenv_p("TCL_DEBUG")) {
    cerr << ":";
    for (int a = 0; a < args.count(); ++a)
      cerr << args[a] << " ";
    cerr << "\n";
  }
    

  if (args[1] == "event") {
    event_.update_state(args, *this);
    if (event_.type_ == Event::MOUSE_MOTION_E&& event_.window_)
      redraw_window(*event_.window_);

    int tool = tools_.size();
    while (tool > 0) {
      --tool;
      switch (tools_[tool]->do_event(event_)) {
      case PainterTool::HANDLED_E: {
        return;
      } break;
      case PainterTool::QUIT_E: { 
        delete tools_[tool];
        tools_.erase(tools_.begin()+tool);
        return;
      } break;
      case PainterTool::ERROR_E: { 
        cerr << tools_[tool]->get_name() << " Tool Error: " 
             << tools_[tool]->err() << std::endl;
        return;
      } break;
      default: // nothing, continue to next tool 
        break;
      }
    }
    

    if (args[2] == "enter")        handle_gui_mouse_enter(args);
    else if (args[2] == "leave")   handle_gui_mouse_leave(args);
    else if (args[2] == "button")  handle_gui_mouse_button_press(args);
    else if (args[2] == "keydown") handle_gui_keypress(args);
  }
  else if (args[1] == "set_font_sizes") set_font_sizes(font_size_());
  else if (args[1] == "setgl") {
    TkOpenGLContext *context = \
      scinew TkOpenGLContext(args[2], args.get_int(4), 256, 256);
    XSync(context->display_, 0);
    ASSERT(layouts_.find(args[2]) == layouts_.end());
    layouts_[args[2]] = scinew WindowLayout(get_ctx()->subVar(args[3],0));
    layouts_[args[2]]->opengl_ = context;
    char z = 0;
    Tk_Cursor cursor = 
      Tk_GetCursorFromData(the_interp, context->tkwin_, &z, &z, 1,1,0,0,
                           //source, mask, 8, 8, 0, 0, 
                           Tk_GetUid("black"), Tk_GetUid("black"));
    Tk_DefineCursor(context->tkwin_, cursor);

  } else if(args[1] == "destroygl") {
    WindowLayouts::iterator pos = layouts_.find(args[2]);
    ASSERT(pos != layouts_.end());
    delete (pos->second);
    layouts_.erase(pos);
  } else if(args[1] == "add_viewport") {
    ASSERT(layouts_.find(args[2]) != layouts_.end());
    WindowLayout *layout = layouts_[args[2]];
    SliceWindow *window = scinew SliceWindow(this, get_ctx()->subVar(args[3],0));
    window->layout_ = layout;
    window->name_ = args[2];
    window->viewport_ = scinew OpenGLViewport(layout->opengl_);
    set_axis(*window, layouts_.size()%3);
    layout->windows_.push_back(window);
    for_each(&Painter::mark_autoview);
  } else if(args[1] == "redraw") {
    ASSERT(layouts_.find(args[2]) != layouts_.end());
    for_each(*layouts_[args[2]], &Painter::redraw_window);
  } else if(args[1] == "resize") {
    //    ASSERT(layouts_.find(args[2]) != layouts_.end());
    //    for_each(*layouts_[args[2]], &Painter::autoview);
  } else
    Module::tcl_command(args, userdata);
}



void
Painter::delete_all_fonts() {
  FontMap::iterator fiter = fonts_.begin();
  const FontMap::iterator fend = fonts_.end();
  while (fiter != fend)
    if (fiter->second)
      delete fiter++->second;
  fonts_.clear();
}
 
void
Painter::initialize_fonts() {
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
  if (get_gui()->eval("validDir "+font_dir) == "0")
    font_dir = string(sci_getenv("SCIRUN_SRCDIR"))+"/pixmaps";
  string fontfile = font_dir+"/scirun.ttf";
  
  try {
    fonts_["default"] = freetype_lib_->load_face(fontfile);
    fonts_["orientation"] = freetype_lib_->load_face(fontfile);
    fonts_["view"] = freetype_lib_->load_face(fontfile);
    set_font_sizes(font_size_());
    font1_ = scinew TextRenderer(fonts_["default"]);
    font2_ = scinew TextRenderer(fonts_["view"]);
    font3_ = scinew TextRenderer(fonts_["orientation"]);

  } catch (...) {
    delete_all_fonts();
    error("Error loading font file: "+fontfile);
    error("Please set SCIRUN_FONT_PATH to a directory with scirun.ttf");
  }
}

void
Painter::set_font_sizes(double size) {
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
Painter::mark_autoview(SliceWindow &window) {
  if (current_volume_)
    window.center_ = current_volume_->center();

  for (unsigned int s = 0; s < window.slices_.size(); ++s) {
    window.slices_[s]->plane_ = Plane(window.center_, window.normal_);
  }

  window.autoview_ = true;
  window.redraw_ = true;
  return 1;
}
  

int
Painter::autoview(SliceWindow &window) {
  if (!current_volume_) return 0;
  int wid = window.viewport_->width();
  int hei = window.viewport_->height();
  FreeTypeFace *font = fonts_["orientation"];
  if (font)
  {
    FreeTypeText dummy("X", font);
    BBox bbox;
    dummy.get_bounds(bbox);
    wid -= 2*Ceil(bbox.max().x() - bbox.min().x())+4;
    hei -= 2*Ceil(bbox.max().y() - bbox.min().y())+4;
  }
  
  int xax = window.x_axis();
  int yax = window.y_axis();

  vector<int> zero(current_volume_->nrrd_handle_->nrrd_->dim,0);
  vector<int> index = zero;
  index[xax+1] = current_volume_->nrrd_handle_->nrrd_->axis[xax+1].size;
  double w_wid = (current_volume_->index_to_world(index) - 
                  current_volume_->index_to_world(zero)).length();
  double w_ratio = wid/w_wid;

  index = zero;
  index[yax+1] = current_volume_->nrrd_handle_->nrrd_->axis[yax+1].size;
  double w_hei = (current_volume_->index_to_world(index) - 
                  current_volume_->index_to_world(zero)).length();
  double h_ratio = hei/w_hei;

  window.zoom_ = Min(w_ratio*100.0, h_ratio*100.0);
  if (window.zoom_ < 1.0) window.zoom_ = 100.0;
  window.center_(xax) = current_volume_->center()(xax);
  window.center_(yax) = current_volume_->center()(yax);
  redraw_window(window);
  return 1;
}
   



void
Painter::create_undo_volume() {
  if (undo_volume_) 
    delete undo_volume_;
  string newname = current_volume_->name_.get();
  undo_volume_ = scinew NrrdVolume(current_volume_, newname, 0);
}

void
Painter::undo_volume() {
  if (!undo_volume_) return;
  NrrdVolume *vol = volume_map_[undo_volume_->name_.get()];
  if (!vol) return;
  vol->nrrd_handle_ = undo_volume_->nrrd_handle_;
  vol->nrrd_handle_.detach();
  for_each(&Painter::extract_window_slices);
  redraw_all();
  //  delete undo_volume_;
  //  undo_volume_ = 0;
}


void
Painter::recompute_volume_list()
{
  get_gui()->lock();
  string currentname = "";
  if (current_volume_)
    currentname = current_volume_->name_.get();

  NrrdVolumeMap newmap;
  vector<NrrdVolume *> todelete;
  NrrdVolumeMap::iterator viter = volume_map_.begin();
  NrrdVolumeMap::iterator vend = volume_map_.end();
  for (; viter != vend; ++viter)
    if (viter->second->keep_)
      newmap[viter->first] = viter->second;
    else
      todelete.push_back(viter->second);

  current_volume_ = 0;
  volume_map_ = newmap;

  for (unsigned int v = 0; v < todelete.size(); ++v)
    delete todelete[v];

  volumes_.clear();

  NrrdVolume *newcurrent = 0;
  bool found = false;
  NrrdVolumeOrder::iterator voiter = volume_order_.begin();
  NrrdVolumeOrder::iterator voend = volume_order_.end();
  for (; voiter != voend; ++voiter) {

    if (*voiter == currentname)
      found = true;

    viter = volume_map_.find(*voiter);
    if (viter != vend && viter->second != 0) {
      if (!found || !newcurrent)
        newcurrent = viter->second;
          
      volumes_.push_back(viter->second);
    }
  }

  if (!current_volume_)
    current_volume_ = newcurrent;
  
  for_each(&Painter::extract_window_slices);

  if (newcurrent && currentname.empty())
    for_each(&Painter::mark_autoview);

  redraw_all();  
  get_gui()->unlock();
}


void
Painter::show_volume(const string &name)
{
  NrrdVolumeOrder::iterator voiter;  
  voiter =  std::find(volume_order_.begin(), volume_order_.end(), name);
  if (voiter == volume_order_.end())
    volume_order_.push_back(name);
}

void
Painter::hide_volume(const string &name)
{
  NrrdVolumeOrder::iterator voiter;  
  voiter = std::find(volume_order_.begin(), volume_order_.end(), name);
  if (voiter != volume_order_.end())
    volume_order_.erase(voiter);
}


pair<double, double>
Painter::compute_mean_and_deviation(Nrrd *nrrd, Nrrd *mask) {
  double mean = 0;
  double squared = 0;
  unsigned int n = 0;
  ASSERT(nrrd->dim > 3 && mask->dim > 3 && 
         nrrd->axis[0].size == mask->axis[0].size &&
         nrrd->axis[1].size == mask->axis[1].size &&
         nrrd->axis[2].size == mask->axis[2].size &&
         nrrd->axis[3].size == mask->axis[3].size &&
         nrrd->type == nrrdTypeFloat &&
         mask->type == nrrdTypeFloat);

  unsigned int size = nrrd->axis[0].size;
  for (unsigned int a = 1; a < nrrd->dim; ++a)
    size *= nrrd->axis[a].size;

  float *src = (float *)nrrd->data;
  float *test = (float *)mask->data;
  
  for (unsigned int i = 0; i < size; ++i)
    if (test[i] > 0.0) {
      //      cerr << test[i] << std::endl;
      mean += src[i];
      squared += src[i]*src[i];
      ++n;
    }

  mean = mean / n;
  double deviation = sqrt(squared/n-mean*mean);
  cerr << "size: " << size << " n: " << n << std::endl;
  cerr << "mean: " << mean << " dev: " << deviation << std::endl;
  return make_pair(mean, deviation);
}
  
         

Painter::NrrdVolume *
Painter::copy_current_volume(const string &name, int mode) {
  if (!current_volume_) return 0;
  NrrdVolume *vol = new NrrdVolume(current_volume_, name, mode);
  volume_map_[name] = vol;
  volumes_.push_back(vol);
  show_volume(name);
  for_each(&Painter::extract_window_slices);
  redraw_all();
  return vol;
}
  

  
} // end namespace SCIRun
