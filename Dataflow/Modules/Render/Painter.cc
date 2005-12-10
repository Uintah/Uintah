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
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/BundlePort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/Algorithms/Visualization/RenderField.h>
#include <Core/Bundle/Bundle.h>
#include <Core/Containers/Array3.h>
#include <Core/Datatypes/Field.h> 
#include <Core/Exceptions/GuiException.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/ColorMappedNrrdTextureObj.h>
#include <Core/Geom/FreeTypeTextTexture.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geom/GeomCull.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/TexSquare.h>
#include <Core/Geom/OpenGLContext.h>
#include <Core/Geom/OpenGLViewport.h>
#include <Core/Geom/FreeType.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/UIvar.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Util/Timer.h>
#include <Core/Util/Environment.h>
#include <Core/Volume/CM2Widget.h>

namespace SCIRun {


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
                              SliceWindow *window) :
  painter_(painter),
  volume_(volume),
  window_(window),
  nrrd_dirty_(true),
  tex_dirty_(false),
  geom_dirty_(false),
  texture_(0)
  //lock_("Slice lock"),
  //owner_(0),
  //lock_count_(0)
{
}

void
Painter::NrrdSlice::do_lock()
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
Painter::NrrdSlice::do_unlock()
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


Painter::SliceWindow::SliceWindow(Painter &painter, GuiContext *ctx) :  
  painter_(painter),
  name_("INVALID"),
  layout_(0),
  viewport_(0),
  slices_(),
  paint_under_(0,0, this),
  paint_(0,0, this),
  paint_over_(0,0, this),
  center_(0,0,0),
  normal_(1,0,0),
  slice_num_(ctx->subVar("slice"), 0),
  axis_(ctx->subVar("axis"), 2),
  zoom_(ctx->subVar("zoom"), 100.0),
  slab_min_(ctx->subVar("slab_min"), 0),
  slab_max_(ctx->subVar("slab_max"), 0),
  x_(ctx->subVar("posx"),0.0),
  y_(ctx->subVar("posy"),0.0),
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


Painter::NrrdVolume::NrrdVolume(GuiContext *ctx) :
  nrrd_(0),
  name_(ctx->subVar("name")),
  opacity_(ctx->subVar("opacity"), 1.0),
  clut_min_(ctx->subVar("clut_min"), 0.0),
  clut_max_(ctx->subVar("clut_max"), 1.0),
  semaphore_(ctx->getfullname().c_str(), 1),
  data_min_(0),
  data_max_(1.0),
  colormap_(0)
            
{
}


DECLARE_MAKER(Painter)

Painter::Painter(GuiContext* ctx) :
  Module("Painter", ctx, Filter, "Render", "SCIRun"),
  layouts_(),
  volumes_(),
  current_volume_(0),
  nrrd_generations_(),
  cm2_generation_(-1),
  cm2_buffer_under_(256, 256, 4),
  cm2_buffer_(256, 256, 4),
  cm2_buffer_over_(256, 256, 4),
  colormaps_(),
  colormap_generation_(-1),
  tool_(0),
  show_colormap2_(ctx->subVar("show_colormap2"),0),
  painting_(ctx->subVar("painting"),0),
  texture_filter_(ctx->subVar("texture_filter"),0),
  anatomical_coordinates_(ctx->subVar("anatomical_coordinates"), 1),
  show_text_(ctx->subVar("show_text"), 1),
  font_r_(ctx->subVar("color_font-r"), 1.0),
  font_g_(ctx->subVar("color_font-g"), 1.0),
  font_b_(ctx->subVar("color_font-b"), 1.0),
  font_a_(ctx->subVar("color_font-a"), 1.0),
  geom_flushed_(ctx->subVar("geom_flushed"), 0),
  background_threshold_(ctx->subVar("background_threshold"), 0.0),
  gradient_threshold_(ctx->subVar("gradient_threshold"), 0.0),
  paint_widget_(0),
  paint_lock_("Painter paint lock"),
  bundle_oport_((BundleOPort *)get_oport("Paint Data")),
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
  }

  runner_ = scinew RealDrawer(this);
  runner_thread_ = scinew Thread(runner_, string(id+" OpenGL drawer").c_str());
  mouse_.window_ = 0;
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


int
Painter::render_window(SliceWindow &window) {
  if (!window.redraw_) return 0;
  window.redraw_ = false;
  window.viewport_->make_current();
  window.viewport_->clear();
  window.setup_gl_view();
  CHECK_OPENGL_ERROR();
  

  for (unsigned int s = 0; s < window.slices_.size(); ++s)
    window.slices_[s]->draw();


  //  draw_dark_slice_regions(window);
  draw_slice_lines(window);
  //  draw_slice_arrows(window);
  if (mouse_.window_ == &window)
    mouse_.position_ = screen_to_world(window, mouse_.x_, mouse_.y_);
  draw_guide_lines(window, mouse_.position_.x(), 
                   mouse_.position_.y(), mouse_.position_.z());
  
  if (tool_) 
    tool_->draw(window);

  //  if (!crop_) {
  //  set_window_cursor(window, 0);
  draw_all_labels(window);
  window.viewport_->release();
  CHECK_OPENGL_ERROR();
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
  TCLTask::lock();
  if (for_each(&Painter::render_window)) 
    for_each(&Painter::swap_window);
  TCLTask::unlock();
  send_all_geometry();
}

  
void
Painter::send_all_geometry()
{
  if (tool_) return;
  TCLTask::lock();
  slice_lock_.writeLock();
  int flush = for_each(&Painter::send_mip_textures);
  flush += for_each(&Painter::send_slice_textures);  
  slice_lock_.writeUnlock();
  TCLTask::unlock();
  if (flush) 
  {
    //    geom_oport_->flushViewsAndWait();
    //    geom_flushed_ = 1;
  }
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

// draw_guide_lines
// renders vertical and horizontal bars that represent
// selected slices in other dimensions
// if x, y, or z < 0, then that dimension wont be rendered
void
Painter::draw_guide_lines(SliceWindow &window, float x, float y, float z) {
  if (!window.show_guidelines_()) return;
  if (mouse_.window_ != &window) return;
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

  vector<int> max_slice = current_volume_->max_index();
  Vector scale = current_volume_->scale();

  glColor4dv(yellow);
  for (int i = 0; i < 2; ++i) {
    glBegin(GL_QUADS);    
    if (c[p] >= 0 && c[p] <= max_slice[p]*scale[p]) {
      glVertex3f(p==0?x:0.0, p==1?y:0.0, p==2?z:0.0);
      glVertex3f(p==0?x+one:0.0, p==1?y+one:0.0, p==2?z+one:0.0);
      glVertex3f(p==0?x+one:(axis==0?0.0:(max_slice[s]+1)*scale[s]),
		 p==1?y+one:(axis==1?0.0:(max_slice[s]+1)*scale[s]),
		 p==2?z+one:(axis==2?0.0:(max_slice[s]+1)*scale[s]));
      
      glVertex3f(p==0?x:(axis==0?0.0:(max_slice[s]+1)*scale[s]),
		 p==1?y:(axis==1?0.0:(max_slice[s]+1)*scale[s]),
		 p==2?z:(axis==2?0.0:(max_slice[s]+1)*scale[s]));
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
    if (c[p] < 0 || c[p] > max_slice[p]*scale[p]) {
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


      glVertex3f(p==0?x:(axis==0?0.0:(max_slice[s]+1)*scale[s]), 
		 p==1?y:(axis==1?0.0:(max_slice[s]+1)*scale[s]),
		 p==2?z:(axis==2?0.0:(max_slice[s]+1)*scale[s]));

      glVertex3f(p==0?x+one:(axis==0?0.0:(max_slice[s]+1)*scale[s]), 
		 p==1?y+one:(axis==1?0.0:(max_slice[s]+1)*scale[s]), 
		 p==2?z+one:(axis==2?0.0:(max_slice[s]+1)*scale[s]));

      glVertex3f(p==0?x+one:(axis==0?(axis==0?0.0:(max_slice[s]+1)*scale[s]):cvur(0)),
		 p==1?y+one:(axis==1?(axis==1?0.0:(max_slice[s]+1)*scale[s]):cvur(1)),
		 p==2?z+one:(axis==2?(axis==2?0.0:(max_slice[s]+1)*scale[s]):cvur(2)));
      
      glVertex3f(p==0?x:(axis==0?(axis==0?0.0:(max_slice[s]+1)*scale[s]):cvur(0)),
		 p==1?y:(axis==1?(axis==1?0.0:(max_slice[s]+1)*scale[s]):cvur(1)),
		 p==2?z:(axis==2?(axis==2?0.0:(max_slice[s]+1)*scale[s]):cvur(2)));
    }
    SWAP(p,s);
  }

      
  glEnd();
}  



// renders vertical and horizontal bars that represent
// selected slices in other dimensions
void
Painter::draw_slice_lines(SliceWindow &window)
{
  if (!current_volume_) return;
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

  vector<int> max_slice = current_volume_->max_index();
  Vector scale = current_volume_->scale();

  double one;
  double xyz[3];
  int i;
  for (int i = 0; i < 3; ++i)
    xyz[i] = cur_slice_[i]*scale[i];
  

  for (i = 0; i < 2; ++i) {
    if (!slab_width_[p]) continue;
    one = Max(screen_space_one, double(scale[p]*slab_width_[p]));

    switch (p) {
    case 0: glColor4dv(red); break;
    case 1: glColor4dv(green); break;
    default:
    case 2: glColor4dv(blue); break;
    }
    glBegin(GL_QUADS);    
    if (xyz[p] >= 0 && xyz[p] <= max_slice[p]*scale[p]) {
      glVertex3f(p==0?xyz[0]:0.0, p==1?xyz[1]:0.0, p==2?xyz[2]:0.0);
      glVertex3f(p==0?xyz[0]+one:0.0, p==1?xyz[1]+one:0.0, p==2?xyz[2]+one:0.0);
      glVertex3f(p==0?xyz[0]+one:(axis==0?0.0:(max_slice[s]+1)*scale[s]),
		 p==1?xyz[1]+one:(axis==1?0.0:(max_slice[s]+1)*scale[s]),
		 p==2?xyz[2]+one:(axis==2?0.0:(max_slice[s]+1)*scale[s]));
      
      glVertex3f(p==0?xyz[0]:(axis==0?0.0:(max_slice[s]+1)*scale[s]),
		 p==1?xyz[1]:(axis==1?0.0:(max_slice[s]+1)*scale[s]),
		 p==2?xyz[2]:(axis==2?0.0:(max_slice[s]+1)*scale[s]));
    }
    glEnd();
    SWAP(p,s);
  }
  CHECK_OPENGL_ERROR();
}





Point
Painter::NrrdVolume::center(int axis, int slice) {
  vector<int> index(nrrd_->nrrd->dim,0);
  for (unsigned int a = 0; a < index.size(); ++a) 
    index[a] = nrrd_->nrrd->axis[a].size/2;
  if (axis >= 0 && axis < int(index.size()))
    index[axis] = Clamp(slice, 0, nrrd_->nrrd->axis[axis].size-1);
  ASSERT(index_valid(index));
  return index_to_world(index);
}


Point
Painter::NrrdVolume::min(int axis, int slice) {
  vector<int> index(nrrd_->nrrd->dim,0);
  if (axis >= 0 && axis < int(index.size()))
    index[axis] = Clamp(slice, 0, nrrd_->nrrd->axis[axis].size-1);
  ASSERT(index_valid(index));
  return index_to_world(index);
}

Point
Painter::NrrdVolume::max(int axis, int slice) {
  vector<int> index = max_index();
  if (axis >= 0 && axis < int(index.size()))
    index[axis] = Clamp(slice, 0, nrrd_->nrrd->axis[axis].size-1);
  ASSERT(index_valid(index));
  return index_to_world(index);
}



Vector
Painter::NrrdVolume::scale() {
  vector<int> index_zero(nrrd_->nrrd->dim,0);
  vector<int> index_one(nrrd_->nrrd->dim,1);
  return index_to_world(index_one) - index_to_world(index_zero);
}

double
Painter::NrrdVolume::scale(int axis) {
  ASSERT(axis >= 0 && axis < nrrd_->nrrd->dim);
  return scale()[axis];
}


vector<int>
Painter::NrrdVolume::max_index() {
  vector<int> max_index(nrrd_->nrrd->dim,0);
  for (int a = 0; a < nrrd_->nrrd->dim; ++a)
    max_index[a] = nrrd_->nrrd->axis[a].size-1;
  return max_index;
}

int
Painter::NrrdVolume::max_index(int axis) {
  ASSERT(axis >= 0 && axis < nrrd_->nrrd->dim);
  return max_index()[axis];
}

bool
Painter::NrrdVolume::inside_p(const Point &p) {
  return index_valid(world_to_index(p));
}



// Returns and index to the axis that is most parallel and in the direction of
// +X in the screen.  
// 0 for +x, 1 for +y, and 2 for +z
// 3 for -x, 4 for -y, and 5 for -z
int
Painter::x_axis(SliceWindow &window)
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
Painter::y_axis(SliceWindow &window)
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
Painter::SliceWindow::setup_gl_view()
{
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  CHECK_OPENGL_ERROR();
  if (!painter_.current_volume_) return;
  
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

  const Point center = center_;
  //  const Point center = painter_.current_volume_->center(axis_, slice_num_);
  double hwid = viewport_->width()*50.0/zoom_();
  double hhei = viewport_->height()*50.0/zoom_();
  double cx = double(x_()) + center(painter_.x_axis(*this));
  double cy = double(y_()) + center(painter_.y_axis(*this));
  double maxz = center(axis_) + Max(hwid, hhei);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(cx - hwid, cx + hwid, cy - hhei, cy + hhei, -maxz, maxz);
  glGetIntegerv(GL_VIEWPORT, gl_viewport_);
  glGetDoublev(GL_MODELVIEW_MATRIX, gl_modelview_matrix_);
  glGetDoublev(GL_PROJECTION_MATRIX, gl_projection_matrix_);
  CHECK_OPENGL_ERROR();
}


void
Painter::draw_window_label(SliceWindow &window)
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


string
double_to_string(double val)
{
  char s[50];
  sprintf(s, "%.2g", val);
  return string(s);
}


void
Painter::draw_position_label(SliceWindow &window)
{
  FreeTypeFace *font = fonts_["default"];
  if (!font) return;
  
  const string zoom_text = "Zoom: "+to_string(window.zoom_())+"%";
  string position_text;
  position_text = ("X: "+double_to_string(mouse_.position_.x())+
                   " Y: "+double_to_string(mouse_.position_.y())+
                   " Z: "+double_to_string(mouse_.position_.z()));
    
  FreeTypeText position(position_text, font);
  BBox bbox;
  position.get_bounds(bbox);
  int y_pos = Ceil(bbox.max().y())+2;
  draw_label(window, position_text, 0, 0, FreeTypeText::sw, font);
  draw_label(window, zoom_text, 0, y_pos, FreeTypeText::sw, font);
  if (current_volume_) {
    const float ww = (current_volume_->clut_max_ - current_volume_->clut_min_);
    const float wl = current_volume_->clut_min_ + ww/2.0;
    const string clut = "WL: " + to_string(wl) +  " -- WW: " + to_string(ww);
    draw_label(window, clut, 0, y_pos*2, FreeTypeText::sw, font);
    const string minmax = "Min: " + to_string(current_volume_->clut_min_) + 
      " -- Max: " + to_string(current_volume_->clut_max_);
    draw_label(window, minmax, 0, y_pos*3, FreeTypeText::sw, font);
    if (mouse_.window_ && current_volume_) {
      vector<int> index = current_volume_->world_to_index(mouse_.position_);

      if (current_volume_->index_valid(index)) {
        Point pos(index[0], index[1], index[2]);
        double val;
        current_volume_->get_value(index, val);
        const string value = "Value: " + to_string(val);
        draw_label(window, value, 0, y_pos*4, FreeTypeText::sw, font);
        string pos_text = ("S: "+to_string(index[0])+
                           " C: "+to_string(index[1])+
                           " A: "+to_string(index[2]));
        draw_label(window, pos_text, 0, y_pos*5, FreeTypeText::sw, font);
      }
        
    }
  }
      
}  


// Right	= -X
// Left		= +X
// Posterior	= -Y
// Anterior	= +X
// Inferior	= -Z
// Superior	= +Z
void
Painter::draw_orientation_labels(SliceWindow &window)
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
Painter::draw_label(SliceWindow &window, string text, int x, int y,
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


void
Painter::NrrdSlice::bind()
{
  do_lock();
  if (texture_ && tex_dirty_) { 
    delete texture_; 
    texture_ = 0; 
  }

  if (!texture_) {
    vector<int> index = volume_->world_to_index(window_->center_);
    texture_ = scinew ColorMappedNrrdTextureObj(volume_->nrrd_, 
                                                window_->axis_,
                                                index[window_->axis_],
                                                index[window_->axis_]);
    tex_dirty_ = true;
  }


  if (tex_dirty_) {
    texture_->set_clut_minmax(volume_->clut_min_, volume_->clut_max_);
    ColorMapHandle cmap = painter_->get_colormap(volume_->colormap_);
    texture_->set_colormap(cmap);
  }

  tex_dirty_ = false;
  geom_dirty_ = true;

  do_unlock();
  return;
}


ColorMapHandle
Painter::get_colormap(int id)
{
  if (id >= 0 && id < int(colormap_names_.size()) &&
      colormaps_.find(colormap_names_[id]) != colormaps_.end())
    return colormaps_[colormap_names_[id]];
  return 0;
}
    

void
Painter::NrrdSlice::draw()
{
  if (nrrd_dirty_) {
    do_lock();
    //    painter_->cur_slice_[window_->axis_] = window_->slice_num_;
    painter_->slab_width_[window_->axis_] = 
      window_->slab_max_ - window_->slab_min_ + 1;

    set_coords();
    nrrd_dirty_ = false;
    tex_dirty_ = true;
    geom_dirty_ = false;
    do_unlock();
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
  texture_->draw_quad(pos_, xdir_, ydir_);
}



void
Painter::draw_all_labels(SliceWindow &window) {
  if (!show_text_) return;
  glColor4d(font_r_, font_g_, font_b_, font_a_);
  draw_position_label(window);
  draw_orientation_labels(window);
  draw_window_label(window);

}


void
Painter::NrrdSlice::set_coords() {
  do_lock();

  vector<int> centerindex = volume_->world_to_index(window_->center_);
  pos_ = volume_->min(window_->axis_, centerindex[window_->axis_]);

  const int axis = window_->axis_;
  vector<int> index(3,0);
  int prim = (axis == 1) ? 0 : (axis+1)%3;
  index[prim] = volume_->nrrd_->nrrd->axis[prim].size;
  xdir_ = volume_->index_to_world(index) - pos_;
  index[prim] = 0;

  int sec = (axis == 1) ? 2: (axis+2)%3;
  index[sec] = volume_->nrrd_->nrrd->axis[sec].size;
  ydir_ = volume_->index_to_world(index) - pos_;

  do_unlock();
}


int
Painter::extract_window_slices(SliceWindow &window) {
  for (unsigned int s = window.slices_.size(); s < volumes_.size(); ++s)
    window.slices_.push_back(scinew NrrdSlice(this, volumes_[s], &window));
  for_each(window, &Painter::set_slice_nrrd_dirty);
  set_slice_nrrd_dirty(window.paint_);

  return window.slices_.size();
}

int
Painter::set_slice_nrrd_dirty(NrrdSlice &slice) {
  slice.do_lock();
  slice.nrrd_dirty_ = true;
  slice.do_unlock();
  return 0;
}



int
Painter::extract_mip_slices(NrrdVolume *volume)
{
  return 0;
#if 0
  if (!volume || !volume->nrrd_.get_rep()) { return 0; }
  for (int axis = 0; axis < 3; ++axis) {
    if (!mip_slices_[axis]) {
      mip_slices_[axis] = scinew NrrdSlice(this, volume, 0);
    }
      
    NrrdSlice &slice = *mip_slices_[axis];
    slice.do_lock();
    slice.volume_ = volume;

    slice.nrrd_ = scinew NrrdData;

    NrrdDataHandle temp1 = scinew NrrdData;
    
    int max[3];
    for (int i = 0; i < 3; i++) {
      max[i] = max_slice_[i];
    }
    
    NRRD_EXEC(nrrdProject(temp1->nrrd, slice.volume_->nrrd_->nrrd, 
			  axis, 2, nrrdTypeDefault));
    slice.nrrd_dirty_ = false;
    slice.tex_dirty_ = true;
    slice.geom_dirty_ = false;
    
//     int minp[2] = { 0, 0 };
//     int maxp[2] = { slice.tex_wid_-1, slice.tex_hei_-1 };
    
//     NRRD_EXEC(nrrdPad(slice.nrrd_->nrrd, temp1->nrrd, 
// 		      minp,maxp,nrrdBoundaryPad, 0.0));
    set_slice_coords(slice, true);
    slice.do_unlock();
  }
  return 1;
#endif
}


int
Painter::send_mip_textures(SliceWindow &window)
{
  return 0;
#if 0
  int value = 0;
  for (int axis = 0; axis < 3; ++axis)
  {
    if (!mip_slices_[axis]) continue;
    NrrdSlice &slice = *mip_slices_[axis];
    if (!slice.tex_dirty_) continue;
    value++;
    slice.do_lock();

    //    apply_colormap(slice, temp_tex_data_);

    window.viewport_->make_current();
    //    bind_slice(slice);
    window.viewport_->release();

    string name = "MIP Slice"+to_string(slice.axis_);

    TexSquare *mintex = scinew TexSquare();
    tobjs_[name+"-min"] = mintex;
    slice.slice_num_ = 0;
    set_slice_coords(slice, false);
    mintex->set_coords(slice.tex_coords_, slice.pos_coords_);
    //    mintex->set_texname(slice.tex_name_);
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
    //    maxtex->set_texname(slice.tex_name_);
    maxtex->set_normal(normal);

    Vector *maxvec = scinew 
      Vector(axis==0?-1.0:0.0, axis==1?-1.0:0.0, axis==2?-1.0:0.0);
    GeomCull *maxcull = scinew GeomCull(maxtex, maxvec);

    GeomGroup *group = scinew GeomGroup();
    group->add(mincull);
    group->add(maxcull);
    
    GeomHandle gobj = group;
    slice.do_unlock();
    //    if (gobjs_[name]) geom_oport_->delObj(gobjs_[name]);
    //    gobjs_[name] = geom_oport_->addObj(gobj, name, &slice_lock_);
  }
  return value;
#endif
}


void
Painter::set_axis(SliceWindow &window, unsigned int axis) {
  window.axis_ = axis;
  window.normal_ = Vector(0,0,0);
  window.normal_[axis] = 1.0;
  extract_window_slices(window);
  window.redraw_ = true;
}

void
Painter::SliceWindow::prev_slice()
{
  if (!painter_.current_volume_) return;
  Point cached = center_;
  center_ = center_ + normal_;
  if (!painter_.current_volume_->inside_p(center_)) {
    center_ = cached;
    return;
  }
  painter_.extract_window_slices(*this);
  painter_.redraw_all();
}

void
Painter::SliceWindow::next_slice()
{
  if (!painter_.current_volume_) return;
  Point cached = center_;
  center_ = center_ - normal_;
  if (!painter_.current_volume_->inside_p(center_)) {
    center_ = cached;
    return;
  }
  painter_.extract_window_slices(*this);
  painter_.redraw_all();
}


void
Painter::zoom_in(SliceWindow &window)
{
  window.zoom_ *= 1.1;
  window.redraw_ = true;
}

void
Painter::zoom_out(SliceWindow &window)
{
  window.zoom_ /= 1.1;
  window.redraw_ = true;
}
  
Point
Painter::screen_to_world(SliceWindow &window, 
                         unsigned int x, unsigned int y) {
  Point center(0,0,0);
//   if (current_volume_) {
//     center = current_volume_->center(window.axis_, window.slice_num_);
//     center = world_to_screen(window, center);
//  }
  GLdouble xyz[3];

  gluUnProject(double(x), double(y), 0,//window.center_(window.axis_),
	       window.gl_modelview_matrix_, 
	       window.gl_projection_matrix_,
	       window.gl_viewport_,
	       xyz+0, xyz+1, xyz+2);
  xyz[window.axis_] = window.center_(window.axis_);
  return Point(xyz[0], xyz[1], xyz[2]);
}


Point
Painter::world_to_screen(SliceWindow &window, Point &world)
{
  GLdouble xyz[3];
  gluProject(double(world(0)), double(world(1)), double(world(2)),
	       window.gl_modelview_matrix_, 
	       window.gl_projection_matrix_,
	       window.gl_viewport_,
	       xyz+0, xyz+1, xyz+2);
  return Point(xyz[0], xyz[1], xyz[2]);
}


Point
Painter::NrrdVolume::index_to_world(const vector<int> &index) {
  unsigned int dim = index.size()+1;
  ColumnMatrix index_matrix(dim);
  ColumnMatrix world_coords(dim);
  for (unsigned int i = 0; i < dim-1; ++i)
    index_matrix[i] = index[i];
  index_matrix[dim-1] = 1.0;
  DenseMatrix transform = build_index_to_world_matrix();
  int tmp1, tmp2;
  transform.mult(index_matrix, world_coords, tmp1, tmp2);
  Point return_val;
  for (int i = 0; i < 3; ++i) 
    return_val(i) = world_coords[i];
  return return_val;
}



vector<int> 
Painter::NrrdVolume::world_to_index(const Point &p) {
  DenseMatrix transform = build_index_to_world_matrix();
  ColumnMatrix index_matrix(transform.ncols());
  ColumnMatrix world_coords(transform.nrows());
  for (int i = 0; i < transform.nrows(); ++i)
    if (i < 3) 
      world_coords[i] = p(i);
    else       
      world_coords[i] = 0.0;;
  //  cerr << "world2index: " << p << std::endl;
  //  transform.print();
  transform.solve(world_coords, index_matrix, 1);
  vector<int> return_val(index_matrix.nrows()-1);
  for (unsigned int i = 0; i < return_val.size(); ++i) {
    return_val[i] = Floor(index_matrix[i]);
  }
  
  return return_val;
}


DenseMatrix
Painter::NrrdVolume::build_index_to_world_matrix() {
  Nrrd *nrrd = nrrd_->nrrd;
  int dim = nrrd->dim+1;
  DenseMatrix matrix(dim, dim);
  matrix.zero();
  for (int i = 0; i < dim-1; ++i)
    matrix.put(i,i,nrrd->axis[i].spacing);
  matrix.put(dim-1, dim-1, 1.0);
  
  for (int i = 0; i < 3; ++i)
    matrix.put(i, nrrd->dim, nrrd->axis[i].min);
  return matrix;
      
}

bool
Painter::NrrdVolume::index_valid(const vector<int> &index) {
  for (int a = 0; a < nrrd_->nrrd->dim; ++a) 
    if (index[a] < 0 || index[a] >= nrrd_->nrrd->axis[a].size)
      return false;
  return true;
}
  

  
void
Painter::execute()
{
  update_state(Module::JustStarted);

  update_state(Module::NeedData);

  bundles_.clear();
  port_range_type prange = get_iports("Paint Data");
  port_map_type::iterator pi = prange.first;
  while (pi != prange.second) {
    BundleIPort *iport = (BundleIPort *)get_iport(pi++->second);
    BundleHandle bundle = 0;
    if (iport && iport->get(bundle) && bundle.get_rep())
      bundles_.push_back(bundle);
  }

  vector<NrrdDataHandle> nrrds;
  colormaps_.clear();
  colormap_names_.clear();

  for (unsigned int b = 0; b < bundles_.size(); ++b) {
    int numNrrds = bundles_[b]->numNrrds();
    for (int n = 0; n < numNrrds; n++) {
      NrrdDataHandle nrrdH = bundles_[b]->getNrrd(bundles_[b]->getNrrdName(n));
      if (!nrrdH.get_rep()) continue;
      if (nrrdH->nrrd->dim < 3)
      {
        warning("Nrrd with dim < 3, skipping.");
        continue;
      }
      nrrds.push_back(nrrdH);
    }

    int numColormaps = bundles_[b]->numColormaps();
    for (int n = 0; n < numColormaps; n++) {
      const string name = bundles_[b]->getColormapName(n);
      ColorMapHandle cmap = bundles_[b]->getColormap(name);
      if (cmap.get_rep()) {
        colormaps_[name] = cmap;
        colormap_names_.push_back(name);
      }
    }
  }

  if (!nrrds.size()) {
    error ("Unable to get an input nrrd.");
    return;
  }

  
  


  bool re_extract = true;

//     colormap_.get_rep()?(colormap_generation_ != colormap_->generation):false;
//   if (colormap_.get_rep())
//     colormap_generation_ = colormap_->generation;

  update_state(Module::Executing);
  TCLTask::lock();

  while(volumes_.size() < nrrds.size())
    volumes_.push_back(0);

  bool do_autoview = false;

  NrrdRange *range = scinew NrrdRange;
  for (unsigned int n = 0; n < nrrds.size(); ++n) {
    NrrdDataHandle nrrdH = nrrds[n];
    nrrdRangeSet(range, nrrdH->nrrd, 0);
    //    Nrrd *nrrd = nrrdH->nrrd;
    for (int a = 0; a < nrrdH->nrrd->dim; ++a) {
      if (nrrdH->nrrd->axis[a].min > nrrdH->nrrd->axis[a].max)
	SWAP(nrrdH->nrrd->axis[a].min,nrrdH->nrrd->axis[a].max);
      if (nrrdH->nrrd->axis[a].spacing < 0.0)
	nrrdH->nrrd->axis[a].spacing *= -1.0;
    }
      
    if (nrrdH.get_rep()) { // && nrrdH->generation != nrrd_generations_[n]) {

      re_extract = true;
      nrrd_generations_[n] = nrrdH->generation;
      if (volumes_[n]) {
	delete volumes_[n];
	volumes_[n] = 0;
      }
      volumes_[n] = scinew NrrdVolume(ctx->subVar("nrrd"+to_string(n), false));
      volumes_[n]->nrrd_ = nrrdH;
      volumes_[n]->data_min_ = range->min;
      volumes_[n]->data_max_ = range->max;
      volumes_[n]->clut_min_ = range->min;
      volumes_[n]->clut_max_ = range->max;
      volumes_[n]->name_.set("nrrd"+to_string(n));

      current_volume_ = volumes_[n];      
      //      if (n == 0) 
      //	extract_mip_slices(volumes_[n]);	
    }
  }

  delete range;

  // Mark all windows slices dirty
  if (re_extract) {
    for_each(&Painter::extract_window_slices);
    for (int n = 0; n < 3; ++n)
      if (mip_slices_[n])
	rebind_slice(*mip_slices_[n]);
  }
  if (do_autoview)
    for_each(&Painter::autoview);
  redraw_all();
  TCLTask::unlock();

  update_state(Module::Completed);
  TCLTask::lock();
  if (executing_) --executing_;
  TCLTask::unlock();

}


int
Painter::set_probe(SliceWindow &window) {
  vector<int> index = current_volume_->world_to_index(mouse_.position_);
  if (current_volume_->index_valid(index)) {
    //    window.slice_num_ = index[window.axis_];
    window.center_(window.axis_) = mouse_.position_(window.axis_);
    extract_window_slices(window);
  }
  return 1;
}


int
Painter::rebind_slice(NrrdSlice &slice) {
  slice.do_lock();
  slice.tex_dirty_ = true;
  slice.do_unlock();
  return 1;
}

  


void
Painter::handle_gui_mouse_enter(GuiArgs &args) {
  ASSERTMSG(layouts_.find(args[2]) != layouts_.end(),
	    ("Cannot handle enter on "+args[2]).c_str());
  ASSERTMSG(current_layout_ == 0, "Haven't left window");
  current_layout_ = layouts_[args[2]];
}


void
Painter::handle_gui_mouse_leave(GuiArgs &args) {
  current_layout_ = 0;
  if (mouse_.window_) {
    SliceWindow &window = *mouse_.window_;
    mouse_.window_ = 0;
    redraw_window(window);

  }
}

void
Painter::update_mouse_state(GuiArgs &args, bool reset) {
  ASSERT(layouts_.find(args[2]) != layouts_.end());
  WindowLayout &layout = *layouts_[args[2]];
  mouse_.window_ = 0;
  for (unsigned int w = 0; w < layout.windows_.size(); ++w) {
    SliceWindow *window = layout.windows_[w];
    if (&layout == current_layout_ && 
        mouse_.x_ >= window->viewport_->x() && 
        mouse_.x_ <  window->viewport_->x() + window->viewport_->width() &&
        mouse_.y_ >= window->viewport_->y() &&
        mouse_.y_ <  window->viewport_->y() + window->viewport_->height())
    {
      mouse_.window_ = window;
      break;
    }
  }

  mouse_.button_ = args.get_int(3);
  mouse_.state_ = args.get_int(4);
  mouse_.X_ = args.get_int(5);
  mouse_.Y_ = args.get_int(6);

  if (reset) {
    mouse_.pick_x_ = mouse_.X_;
    mouse_.pick_y_ = mouse_.Y_;
  }    

  mouse_.x_ = args.get_int(7);
  mouse_.y_ = layout.opengl_->height() - 1 - args.get_int(8);
  mouse_.dx_ = mouse_.X_ - mouse_.pick_x_;
  mouse_.dy_ = mouse_.Y_ - mouse_.pick_y_;
  if (mouse_.window_) 
    mouse_.position_ = screen_to_world(*mouse_.window_, mouse_.x_, mouse_.y_);
}



void
Painter::handle_gui_mouse_button_press(GuiArgs &args) {
  update_mouse_state(args, true);
  ASSERT(mouse_.window_);

  if (!tool_) {
    switch (mouse_.button_) {
    case 1:
      if (!tool_ && mouse_.state_ & MouseState::SHIFT_E == MouseState::SHIFT_E)
        tool_ = scinew PanTool(this);
      else  if (!tool_) 
        tool_ = scinew CLUTLevelsTool(this);
      break;
      
    case 2:
      if (!tool_ && mouse_.state_ & MouseState::SHIFT_E == MouseState::SHIFT_E)
        tool_ = scinew AutoviewTool(this);
      else if (!tool_)
        tool_ = scinew ProbeTool(this);
      break;
    case 3:
      if (!tool_ && mouse_.state_ & MouseState::SHIFT_E == MouseState::SHIFT_E)
        tool_ = scinew ZoomTool(this);
      break;
    case 4:
      if (mouse_.state_ & MouseState::CONTROL_E == MouseState::CONTROL_E) 
        zoom_in(*mouse_.window_);
      else
        mouse_.window_->next_slice();
      break;
      
    case 5:
      if (mouse_.state_ & MouseState::SHIFT_E == MouseState::SHIFT_E) 
        zoom_out(*mouse_.window_);
      else
        mouse_.window_->prev_slice();
      break;
      
    default: 
      break;
    }
  }

  if (tool_) {
    string *err = tool_->mouse_button_press(mouse_);
    if (err) {
      error(*err);
      delete err;
      delete tool_;
      tool_ = 0;
    }
  }
}


void
Painter::handle_gui_mouse_motion(GuiArgs &args) {
  update_mouse_state(args);
  if (tool_)
    tool_->mouse_motion(mouse_);
  if (mouse_.window_)
    redraw_window(*mouse_.window_);
}




void
Painter::handle_gui_mouse_button_release(GuiArgs &args) {
  update_mouse_state(args);

  if (mouse_.button_ == 1 && painting_) { 
    painting_ = 2;
    if (!executing_) {
      ++executing_;
      want_to_execute();
    }
  }

  if (tool_) {
    string *err = tool_->mouse_button_release(mouse_);
    if (err && *err == "Done") {
      cerr << tool_->get_name() << " tool Done\n";
      delete tool_;
      tool_ = 0;
    }

//     if (tool_ && dynamic_cast<CropTool *>(tool_) == 0) {
//       delete tool_;
//       tool_ = 0;
//     }
  }
}

  
  
void
Painter::handle_gui_keypress(GuiArgs &args) {
  if (args.count() != 6)
    SCI_THROW(GuiException(args[0]+" "+args[1]+
			   " expects a win #, keycode, keysym,& time"));

  ASSERT(layouts_.find(args[2]) != layouts_.end());
  WindowLayout &layout = *layouts_[args[2]];

  for (unsigned int w = 0; w < layout.windows_.size(); ++w) {
    SliceWindow &window = *layout.windows_[w];
    if (mouse_.window_ != &window) continue;
    double pan_delta = Round(3.0/(window.zoom_()/100.0));
    if (pan_delta < 1.0) pan_delta = 1.0;

    if (args[4] == "equal" || args[4] == "plus") zoom_in(window);
    else if (args[4] == "minus" || args[4] == "underscore") zoom_out(window);
    else if (args[4] == "less" || args[4] == "comma") window.prev_slice();
    else if (args[4] == "greater" || args[4] == "period") window.next_slice();
    else if (args[4] == "bracketleft") {
      if (current_volume_) {
        current_volume_->colormap_ = Max(0, current_volume_->colormap_-1);
        for_each(&Painter::rebind_slice);
        for_each(&Painter::redraw_window);
      }
    }
    else if (args[4] == "bracketright") {
      if (current_volume_) {
        current_volume_->colormap_ = Min(int(colormap_names_.size()), 
                                         current_volume_->colormap_+1);
        for_each(&Painter::rebind_slice);
        for_each(&Painter::redraw_window);
      }
    }
    else if (args[4] == "0") set_axis(window, 0);
    else if (args[4] == "1") set_axis(window, 1);
    else if (args[4] == "2") set_axis(window, 2);
    else if (args[4] == "f") { 
      if (!tool_) 
        tool_ = scinew FloodfillTool(this);
    }
    else if (args[4] == "g") { 
      if (!tool_) 
        tool_ = scinew PixelPaintTool(this);
    }
    else if (args[4] == "p") { 
      if (!current_volume_) continue;
      current_volume_->opacity_ *= 1.1;
        for_each(&Painter::redraw_window);
      //      cerr << "Op: " << current_volume_->opacity_ << std::endl;
    }
    else if (args[4] == "o") { 
      if (!current_volume_) continue;
      current_volume_->opacity_ /= 1.1;
      for_each(&Painter::redraw_window);
      //      cerr << "Op: " << current_volume_->opacity_ << std::endl;
    }
    else if (args[4] == "d") { 
      create_volume((NrrdVolumes *)1);
      extract_window_slices(window);
    }
    else if (args[4] == "r") {
      if (current_volume_) {
        current_volume_->clut_min_ = current_volume_->data_min_;
        current_volume_->clut_max_ = current_volume_->data_max_;
        for_each(&Painter::rebind_slice);
        for_each(&Painter::redraw_window);
      }
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
//       window.y_ += pan_delta;
//       redraw_window(window);
      if (volumes_.size() < 2 || current_volume_ == volumes_[0]) 
        return;
      for (unsigned int i = 1; i < volumes_.size(); ++i)
        if (current_volume_ == volumes_[i]) {
          current_volume_ = volumes_[i-1];
          cerr << "Current volume: " << current_volume_->name_.get() << std::endl;
          return;
        }
          
    } else if (args[4] == "Up") {
      if (volumes_.size() < 2 || current_volume_ == volumes_.back()) 
        return;
      for (unsigned int i = 0; i < volumes_.size()-1; ++i)
        if (current_volume_ == volumes_[i]) {
          current_volume_ = volumes_[i+1];
          cerr << "Current volume: " << current_volume_->name_.get() << std::endl;
          return;
        }

//       window.y_ -= pan_delta;
//       redraw_window(window);
    }
  }
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
    
  if (args[1] == "enter")         handle_gui_mouse_enter(args);
  else if (args[1] == "leave")    handle_gui_mouse_leave(args);
  else if (args[1] == "motion")	  handle_gui_mouse_motion(args);
  else if (args[1] == "button")   handle_gui_mouse_button_press(args);
  else if (args[1] == "release")  handle_gui_mouse_button_release(args);
  else if (args[1] == "keypress") handle_gui_keypress(args);
  else if (args[1] == "background_thresh") update_background_threshold();
  else if (args[1] == "gradient_thresh")for_each(&Painter::set_paint_dirty);
  else if (args[1] == "undo") undo_paint_stroke();
  else if (args[1] == "set_font_sizes") set_font_sizes(font_size_());
  else if(args[1] == "setgl") {
    TkOpenGLContext *context = \
      scinew TkOpenGLContext(args[2], args.get_int(4), 256, 256);
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
    SliceWindow *window = scinew SliceWindow(*this, ctx->subVar(args[3],0));
    window->layout_ = layout;
    window->name_ = args[2];
    window->viewport_ = scinew OpenGLViewport(layout->opengl_);
    window->axis_ = (layouts_.size())%3;
    window->normal_ = Vector(0,0,0);
    window->normal_[window->axis_] = 1.0;

    layout->windows_.push_back(window);
    autoview(*window);
  } else if(args[1] == "redraw") {
    ASSERT(layouts_.find(args[2]) != layouts_.end());
    for_each(*layouts_[args[2]], &Painter::redraw_window);
  } else if(args[1] == "resize") {
    ASSERT(layouts_.find(args[2]) != layouts_.end());
    //    for_each(*layouts_[args[2]], &Painter::autoview);
  } else if(args[1] == "redrawall") {
    redraw_all();
  } else if(args[1] == "rebind") {
    ASSERT(layouts_.find(args[2]) != layouts_.end());
    for_each(*layouts_[args[2]], &Painter::extract_window_slices);
    for_each(*layouts_[args[2]], &Painter::redraw_window);
  } else if(args[1] == "texture_rebind") {
    if (args.count() == 2) { 
      for_each(&Painter::rebind_slice);
      for_each(&Painter::redraw_window);
    } else {
      for_each(*layouts_[args[2]], &Painter::rebind_slice);
      for_each(*layouts_[args[2]], &Painter::redraw_window);
    }
  } else if(args[1] == "startcrop") {
    tool_ = scinew CropTool(this);
    redraw_all();
  } else if(args[1] == "stopcrop") {
    //    crop_ = 0;
    //    redraw_all();
  } else if(args[1] == "updatecrop") {
    //    update_bbox_from_gui();
    //    redraw_all();
  } else if (args[1] == "setclut") {
    //    const double cache_ww = clut_ww_;
    //    const double cache_wl = clut_wl_;
    //    clut_ww_(); // resets gui context
    //    clut_wl_(); // resets gui context
    //    if (fabs(cache_ww - clut_ww_) < 0.0001 && 
    //	fabs(cache_wl - clut_wl_) < 0.0001) return;
    for_each(&Painter::rebind_slice);
    for_each(&Painter::redraw_window);
    for (int n = 0; n < 3; ++n)
      if (mip_slices_[n])
	rebind_slice(*mip_slices_[n]);
  } else 
    Module::tcl_command(args, userdata);
}



int
Painter::send_slice_textures(NrrdSlice &slice) {
  return 0;
#if 0
  if (!slice.geom_dirty_) return 0;
  slice.do_lock();
  string name = "Slice"+to_string(slice.axis_);
  if (tobjs_[name] == 0) 
    tobjs_[name] = scinew TexSquare();
  //  tobjs_[name]->set_texname(slice.tex_name_);
  set_slice_coords(slice, false);
  tobjs_[name]->set_coords(slice.tex_coords_, slice.pos_coords_);
  int axis = slice.axis_;
  Vector normal(axis==0?1.0:0.0, axis==1?-1.0:0.0, axis==2?1.0:0.0);
  tobjs_[name]->set_normal(normal);
  set_slice_coords(slice, true);
  GeomHandle gobj = tobjs_[name];
  slice.geom_dirty_ = false;
  slice.do_unlock();

  //  if (gobjs_[name]) geom_oport_->delObj(gobjs_[name]);
  //  gobjs_[name] = geom_oport_->addObj(gobj, name, &slice_lock_);
  return 1;
#endif
}

int
Painter::set_paint_dirty(SliceWindow &window) 
{
  window.paint_.nrrd_dirty_ = true;
  return 1;
}

void
Painter::rasterize_widgets_to_cm2(int min, int max, Array3<float> &buffer) 
{
  buffer.initialize(0.0);
  for (int i = min; i <= max; ++i)
    cm2_->widgets()[i]->rasterize(buffer);
}


bool
Painter::rasterize_colormap2() {
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

  for_each(&Painter::set_paint_dirty);

  return true;
}


void
Painter::apply_colormap2_to_slice(Array3<float> &cm2, NrrdSlice &slice)
{
#if 0
  slice.do_lock();
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
#endif
}



int
Painter::extract_window_paint(SliceWindow &window) {
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
Painter::do_paint(SliceWindow &window) {
#if 0
  if (!paint_lock_.tryLock()) return;
  if (!paint_widget_ || !gradient_.get_rep() || !paint_widget_) {
    paint_lock_.unlock();
    return;
  }
  int xyz[3];
  for (int i = 0; i < 3; ++i) {
    xyz[i] = Floor(mouse_.position_(i)/scale_[i]);
    if (xyz[i] < 0 || xyz[i] > max_slice_[i]) return;
  }
  const double gradient = 
    get_value(gradient_->nrrd, xyz[0], xyz[1], xyz[2])/255.0;
  if (gradient < gradient_threshold_) return;
  const double value = 
    get_value(volumes_[0]->nrrd_->nrrd,xyz[0],xyz[1],xyz[2]);
  paint_widget_->add_coordinate(make_pair(value, gradient));
  rasterize_widgets_to_cm2(cm2_->selected(), cm2_->selected(), cm2_buffer_);
  for_each(&Painter::extract_current_paint);
  paint_lock_.unlock();
#endif
}

int
Painter::extract_current_paint(SliceWindow &window)
{
  apply_colormap2_to_slice(cm2_buffer_, window.paint_);
  window.redraw_ = true;
  return 1;
}


void
Painter::update_background_threshold() {
#if 0
  const double min = clut_wl_ - clut_ww_/2.0;
  double alpha = (background_threshold_() - min) / clut_ww_;
  TexSquareMap::iterator iter = tobjs_.begin();
  while (iter != tobjs_.end()) {
    iter->second->set_alpha_cutoff(alpha);
    iter++;
  }  
#endif
  //  if (geom_oport_) geom_oport_->flushViews();
}

void
Painter::undo_paint_stroke() {
  if (!paint_widget_) return;
  if (paint_widget_->pop_stroke()) {
    painting_ = 2;
    want_to_execute();
  }
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
Painter::autoview(SliceWindow &window) {
  if (!current_volume_) return 0;
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

  vector<int> zero(current_volume_->nrrd_->nrrd->dim,0);
  vector<int> index = zero;
  index[xax] = current_volume_->nrrd_->nrrd->axis[xax].size;
  double w_wid = (current_volume_->index_to_world(index) - 
                  current_volume_->index_to_world(zero)).length();
  double w_ratio = wid/w_wid;

  index = zero;
  index[yax] = current_volume_->nrrd_->nrrd->axis[yax].size;
  double w_hei = (current_volume_->index_to_world(index) - 
                  current_volume_->index_to_world(zero)).length();
  double h_ratio = hei/w_hei;

  window.zoom_ = Min(w_ratio*100.0, h_ratio*100.0);
  if (window.zoom_ < 1.0) window.zoom_ = 100.0;
  window.x_ = 0.0;
  window.y_ = 0.0;
  window.center_(xax) = current_volume_->center()(xax);
  window.center_(yax) = current_volume_->center()(yax);

  redraw_window(window);
  return 1;
}
   

int
Painter::create_volume(NrrdVolumes *copies) {
  if (!current_volume_) return -1;
  NrrdVolume *volume = 
    scinew NrrdVolume(ctx->subVar("nrrd"+to_string(volumes_.size()), false));
  volume->nrrd_ = scinew NrrdData();
  nrrdCopy(volume->nrrd_->nrrd, current_volume_->nrrd_->nrrd);
  if (copies) {
    volume->data_min_ = current_volume_->data_min_;
    volume->data_max_ = current_volume_->data_max_;
    volume->clut_min_ = current_volume_->clut_min_;
    volume->clut_max_ = current_volume_->clut_max_;
  }
  //  else
  //    nrrdSetValue(volume->nrrd_->nrrd, 0.0); 
  volumes_.push_back(volume);
  current_volume_ = volume;
  return volumes_.size()-1;
}





  
    
} // end namespace SCIRun
