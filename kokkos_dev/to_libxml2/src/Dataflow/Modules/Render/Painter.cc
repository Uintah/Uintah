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
#include <Core/Geom/NrrdTextureObj.h>
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


Painter::NrrdSlice::NrrdSlice(NrrdVolume *volume, SliceWindow *window) :
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
  tex_name_(0),
  texture_(0)
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


Painter::SliceWindow::SliceWindow(GuiContext *ctx) :  
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
  show_guidelines_(ctx->subVar("show_guidelines"),1),
  fusion_(ctx->subVar("fusion"), 1.0),
  cursor_pixmap_(-1)
{
  paint_under_.name_ = "Paint Under";
  paint_.name_ = "Paint";
  paint_over_.name_ = "Paint Over";

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
  data_max_(1.0)
            
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
  pick_(0),
  show_colormap2_(ctx->subVar("show_colormap2"),0),
  painting_(ctx->subVar("painting"),0),
  texture_filter_(ctx->subVar("texture_filter"),0),
  anatomical_coordinates_(ctx->subVar("anatomical_coordinates"), 1),
  show_text_(ctx->subVar("show_text"), 1),
  font_r_(ctx->subVar("color_font-r"), 1.0),
  font_g_(ctx->subVar("color_font-g"), 1.0),
  font_b_(ctx->subVar("color_font-b"), 1.0),
  font_a_(ctx->subVar("color_font-a"), 1.0),
  dim0_(ctx->subVar("dim0"), 0),
  dim1_(ctx->subVar("dim1"), 0),
  dim2_(ctx->subVar("dim2"), 0),
  geom_flushed_(ctx->subVar("geom_flushed"), 0),
  background_threshold_(ctx->subVar("background_threshold"), 0.0),
  gradient_threshold_(ctx->subVar("gradient_threshold"), 0.0),
  paint_widget_(0),
  paint_lock_("Painter paint lock"),
  temp_tex_data_(0),
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
    max_slice_[a] = -1;
  }

  runner_ = scinew RealDrawer(this);
  runner_thread_ = scinew Thread(runner_, string(id+" OpenGL drawer").c_str());

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
  setup_gl_view(window);
  CHECK_OPENGL_ERROR();
  

  
  for_each(window, &Painter::draw_slice);

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
  //  for_each(&Painter::extract_slice);
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
Painter::draw_slice_lines(SliceWindow &window)
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
  CHECK_OPENGL_ERROR();
}




// renders vertical and horizontal bars that represent
// selected slices in other dimensions
void
Painter::draw_slice_arrows(SliceWindow &window)
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
  CHECK_OPENGL_ERROR();
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
Painter::setup_gl_view(SliceWindow &window)
{
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  CHECK_OPENGL_ERROR();

  int axis = window.axis_;
  CHECK_OPENGL_ERROR();
  if (axis == 0) { // screen +X -> +Y, screen +Y -> +Z
    glRotated(-90,0.,1.,0.);
    glRotated(-90,1.,0.,0.);
  } else if (axis == 1) { // screen +X -> +X, screen +Y -> +Z
    glRotated(-90,1.,0.,0.);
  }
  CHECK_OPENGL_ERROR();
  glTranslated((axis==0)?-double(window.slice_num_)*scale_[0]:0.0,
  	       (axis==1)?-double(window.slice_num_)*scale_[1]:0.0,
  	       (axis==2)?-double(window.slice_num_)*scale_[2]:0.0);
  CHECK_OPENGL_ERROR();
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glGetDoublev(GL_MODELVIEW_MATRIX, window.gl_modelview_matrix_);
  CHECK_OPENGL_ERROR();
  glGetDoublev(GL_PROJECTION_MATRIX, window.gl_projection_matrix_);
  CHECK_OPENGL_ERROR();
  glGetIntegerv(GL_VIEWPORT, window.gl_viewport_);
  CHECK_OPENGL_ERROR();
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
  CHECK_OPENGL_ERROR();
  glOrtho(cx - hwid, cx + hwid, cy - hhei, cy + hhei, minz, maxz);
  CHECK_OPENGL_ERROR();
  glGetDoublev(GL_PROJECTION_MATRIX, window.gl_projection_matrix_);
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


void
Painter::draw_position_label(SliceWindow &window)
{
  FreeTypeFace *font = fonts_["default"];
  if (!font) return;
  
  const string zoom_text = "Zoom: "+to_string(window.zoom_())+"%";
  string position_text;
  if (anatomical_coordinates_()) {
    position_text = ("S: "+to_string(Floor(mouse_.position_.x()/scale_[0]))+
		     " C: "+to_string(Floor(mouse_.position_.y()/scale_[1]))+
		     " A: "+to_string(Floor(mouse_.position_.z()/scale_[2])));
  } else {
    position_text = ("X: "+to_string(Ceil(mouse_.position_.x()))+
		     " Y: "+to_string(Ceil(mouse_.position_.y()))+
		     " Z: "+to_string(Ceil(mouse_.position_.z())));
  }
    
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


template <class T>
void
Painter::apply_colormap_to_raw_data(float *data, T *slicedata, 
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
Painter::apply_colormap(NrrdSlice &slice, float *data) 
{
#if 0
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
#endif 
  return 0;
}


double
Painter::get_value(const Nrrd *nrrd, int x, int y, int z) {
  ASSERT(nrrd->dim >= 3);
  const int position = nrrd->axis[0].size*(z*nrrd->axis[1].size+y)+x;
  switch (nrrd->type) {
  case nrrdTypeChar: {
    char *slicedata = (char *)nrrd->data;
    return (double)slicedata[position];
    BREAK;
  }
  case nrrdTypeUChar: {
    unsigned char *slicedata = (unsigned char *)nrrd->data;
    return (double)slicedata[position];
    BREAK;
  }
  case nrrdTypeShort: {
    short *slicedata = (short *)nrrd->data;
    return (double)slicedata[position];
    BREAK;
  }
  case nrrdTypeUShort: {
    unsigned short *slicedata = (unsigned short *)nrrd->data;
    return (double)slicedata[position];
    BREAK;
  }
  case nrrdTypeInt: {
    int *slicedata = (int *)nrrd->data;
    return (double)slicedata[position];
    BREAK;
  }
  case nrrdTypeUInt: {
    unsigned int *slicedata = (unsigned int *)nrrd->data;
    return (double)slicedata[position];
    BREAK;
  }
  case nrrdTypeLLong: {
    signed long long *slicedata = (signed long long *)nrrd->data;
    return (double)slicedata[position];
    BREAK;
  }
  case nrrdTypeULLong: {
    unsigned long long *slicedata = (unsigned long long *)nrrd->data;
    return (double)slicedata[position];
    BREAK;
  }
  case nrrdTypeFloat: {
    float *slicedata = (float *)nrrd->data;
    return (double)slicedata[position];
    BREAK;
  }
  case nrrdTypeDouble: {
    double *slicedata = (double *)nrrd->data;
    return (double)slicedata[position];
    BREAK;
  }
  default: error("Unsupported data type: "+nrrd->type);
  }
  return 0.0;
}


bool
Painter::bind_nrrd(Nrrd &nrrd) {
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
  default: error("Cant bind nrrd"); return false; BREAK;
  }
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
	       nrrd.axis[prim].size, nrrd.axis[prim+1].size, 
	       0, pixtype, type, nrrd.data);
  return true;
}
  

void
Painter::bind_slice(NrrdSlice &slice, float *tex, bool filter)
{
  if (slice.texture_ && slice.tex_dirty_) { 
    delete slice.texture_; 
    slice.texture_ = 0; 
  }
  if (!slice.texture_) {
    slice.texture_ = scinew NrrdTextureObj(slice.volume_->nrrd_, 
                                           slice.axis_,
                                           slice.slice_num_);
  }

  if (slice.tex_dirty_) {
    float scale=1.0/((slice.volume_->clut_max_-slice.volume_->clut_min_) /
                 (slice.volume_->data_max_-slice.volume_->data_min_));
    float bias = ((slice.volume_->data_min_-slice.volume_->clut_min_) /
                  (slice.volume_->data_max_ - slice.volume_->data_min_));

    scale=1.0/((slice.volume_->clut_max_-slice.volume_->clut_min_) /
           (slice.volume_->data_max_-slice.volume_->data_min_));
    bias = slice.volume_->clut_min_ - double(slice.volume_->data_min_)/scale;
    bias /= (slice.volume_->data_max_-slice.volume_->data_min_);

    //    cerr << "Scale: " << scale << " bias: " << bias << std::endl;
    //    cerr << "min: " << slice.volume_->data_min_*scale+bias 
    //         << "max: " << slice.volume_->data_max_*scale+bias << std::endl;
    GLdouble colormatrix[16] = { scale, 0, 0, 0,
                                 0, scale, 0, 0,
                                 0, 0, scale, 0,
                                 bias, bias, bias,1};
    glMatrixMode(GL_COLOR);
    glLoadMatrixd(colormatrix);
    CHECK_OPENGL_ERROR();  
    
    glEnable(GL_POST_COLOR_MATRIX_COLOR_TABLE);
    if (colormaps_.begin() != colormaps_.end()) {
      ColorMapHandle &cmap = (*colormaps_.begin()).second;
      const float *colortable = cmap->get_rgba();
      glColorTable(GL_POST_COLOR_MATRIX_COLOR_TABLE, GL_RGBA,
                   cmap->resolution(), GL_RGBA, GL_FLOAT, colortable);
    } else {
      unsigned char table[1024];
      for (int i = 0; i < 1024; ++i) 
        table[i] = i % 4 == 3 ? 255 : i/256;
      glColorTable(GL_POST_COLOR_MATRIX_COLOR_TABLE, GL_RGBA, 
                   256, GL_RGBA, GL_UNSIGNED_BYTE, table);
    }

    slice.texture_->set_minmax(slice.volume_->data_min_, 
                               slice.volume_->data_max_);
    //    slice.texture_->set_axis(slice.axis_);
    //    slice.texture_->set_slice(slice.slice_num_);

    slice.texture_->bind();
    glDisable(GL_POST_COLOR_MATRIX_COLOR_TABLE);
    glMatrixMode(GL_COLOR);
    glLoadIdentity();
  }

  slice.tex_dirty_ = false;
  slice.geom_dirty_ = true;
  return;

  const bool bound = glIsTexture(slice.tex_name_);
  if (!bound) {
    glGenTextures(1, &slice.tex_name_);
  }

  glBindTexture(GL_TEXTURE_2D, slice.tex_name_);

  if (!bound || slice.tex_dirty_) {
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);  
#ifndef GL_CLAMP_TO_EDGE
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
#else
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
#endif
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
Painter::draw_slice_quad(NrrdSlice &slice) {
  unsigned int i;
  glBegin( GL_QUADS );
  for (i = 0; i < 4; i++) {
    glTexCoord2fv(&slice.tex_coords_[i*2]);
    glVertex3fv(&slice.pos_coords_[i*3]);
  }
  glEnd();
}
  
int
Painter::draw_slice(NrrdSlice &slice)
{
  extract_slice(slice); // Returns immediately if slice is current
  ASSERT(slice.window_);  
  ASSERT(slice.axis_ == slice.window_->axis_);
  ASSERT(slice.nrrd_.get_rep());

  slice.do_lock();

  // Setup the opacity of the slice to be drawn
  //   GLfloat opacity = slice.opacity_*slice.window_->fusion_();
  //   if (volumes_.size() == 2 && 
  //       slice.volume_->nrrd_.get_rep() == volumes_[1]->nrrd_.get_rep()) {
  //     opacity = slice.opacity_*(1.0 - slice.window_->fusion_); 
  //   }
  
  glColor4f(1.0, 1.0, 1.0, 1.0);//opacity, opacity, opacity, opacity);
  float a = slice.volume_->opacity_;
  glColor4f(a,a,a,a);

  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glEnable(GL_BLEND);
  glEnable(GL_COLOR_MATERIAL);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
 
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glShadeModel(GL_FLAT);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  //  GLfloat ones[4] = {0.2, 0.2, 1.0, 1.0};
  //  glTexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, ones);

  bind_slice(slice, temp_tex_data_);
  slice.texture_->draw_quad(slice.pos_coords_);
  glDisable(GL_BLEND);
  CHECK_OPENGL_ERROR();  

  slice.do_unlock();

  return 1;
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
Painter::set_slice_coords(NrrdSlice &slice, bool origin) {
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
    if (origin)
      slice.pos_coords_[i] *= scale_[i%3];
    //    if (!origin) 
    else
      slice.pos_coords_[i] += slice.volume_->nrrd_->nrrd->axis[i%3].min;
  }

  slice.do_unlock();
}


int
Painter::extract_window_slices(SliceWindow &window) {
  for (unsigned int s = window.slices_.size(); s < volumes_.size(); ++s)
    window.slices_.push_back(scinew NrrdSlice(volumes_[s], &window));
  for_each(&Painter::update_slice_from_window);
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
Painter::extract_slice(NrrdSlice &slice)
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
Painter::extract_mip_slices(NrrdVolume *volume)
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
    
    int max[3];
    for (int i = 0; i < 3; i++) {
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
Painter::send_mip_textures(SliceWindow &window)
{
  return 0;
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
    //    if (gobjs_[name]) geom_oport_->delObj(gobjs_[name]);
    //    gobjs_[name] = geom_oport_->addObj(gobj, name, &slice_lock_);
  }
  return value;
}


void
Painter::set_axis(SliceWindow &window, unsigned int axis) {
  window.axis_ = axis;
  extract_window_slices(window);
  window.redraw_ = true;
}

void
Painter::prev_slice(SliceWindow &window)
{
  if (window.slice_num_() == 0) 
    return;
  window.slice_num_--;
  if (window.slice_num_ < 0)
    window.slice_num_ = 0;
  extract_window_slices(window);
  redraw_all();
}

void
Painter::next_slice(SliceWindow &window)
{
  if (window.slice_num_() == max_slice_[window.axis_()]) 
    return;
  window.slice_num_++;
  if (window.slice_num_ > max_slice_[window.axis_])
    window.slice_num_ = max_slice_[window.axis_];
  extract_window_slices(window);
  redraw_all();
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

  int a;
  vector<NrrdDataHandle> nrrds;

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
      if (cmap.get_rep())
        colormaps_[name] = cmap;
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
  for (a = 0; a < 3; a++) {
    if (max_slice_[a] == -1)
      do_autoview = true;
    max_slice_[a] = -1;
    scale_[a] = airNaN();
  }

  NrrdRange *range = scinew NrrdRange;
  for (unsigned int n = 0; n < nrrds.size(); ++n) {
    NrrdDataHandle nrrdH = nrrds[n];
    nrrdRangeSet(range, nrrdH->nrrd, 0);

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

      current_volume_ = volumes_[n];      
      //      if (n == 0) 
      //	extract_mip_slices(volumes_[n]);	
    }
  }

  delete range;

  // Temporary space to hold the colormapped texture
  if (temp_tex_data_) delete[] temp_tex_data_;
  const int max_dim = 
    Max(Pow2(max_slice_[0]+1), Pow2(max_slice_[1]+1), Pow2(max_slice_[2]+1));
  const int mid_dim = 
    Mid(Pow2(max_slice_[0]+1), Pow2(max_slice_[1]+1), Pow2(max_slice_[2]+1));
  temp_tex_data_ = scinew float[max_dim*mid_dim*4];

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
  const int axis = window.axis_();
  window.slice_num_ = Floor(mouse_.position_(axis)/scale_[axis]);
  if (window.slice_num_ < 0) 
    window.slice_num_ = 0;
  if (window.slice_num_ > max_slice_[axis]) 
    window.slice_num_ = max_slice_[axis];  
  extract_window_slices(window);
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
    redraw_window(*mouse_.window_);
    mouse_.window_ = 0;
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
      next_slice(*mouse_.window_);
    break;
    
  case 5:
    if (mouse_.state_ & MouseState::SHIFT_E == MouseState::SHIFT_E) 
      zoom_out(*mouse_.window_);
    else
      prev_slice(*mouse_.window_);
    break;
    
  default: 
    break;
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
    tool_->mouse_button_release(mouse_);
    if (dynamic_cast<CropTool *>(tool_) == 0) {
      delete tool_;
      tool_ = 0;
    }
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
    else if (args[4] == "less" || args[4] == "comma") prev_slice(window);
    else if (args[4] == "greater" || args[4] == "period") next_slice(window);
    else if (args[4] == "0") set_axis(window, 0);
    else if (args[4] == "1") set_axis(window, 1);
    else if (args[4] == "2") set_axis(window, 2);
    else if (args[4] == "p") { 
      if (!current_volume_) continue;
      current_volume_->opacity_ *= 1.1;
      //      cerr << "Op: " << current_volume_->opacity_ << std::endl;
    }
    else if (args[4] == "o") { 
      if (!current_volume_) continue;
      current_volume_->opacity_ /= 1.1;
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
      }
    } else if (args[4] == "i") {
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
Painter::tcl_command(GuiArgs& args, void* userdata) {
  if(args.count() < 2) {
    args.error("Painter needs a minor command");
    return;
  }

  //  cerr << ":";
  //  for (int a = 0; a < args.count(); ++a)
  //    cerr << args[a] << " ";
  //  cerr << "\n";

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

  //  if (gobjs_[name]) geom_oport_->delObj(gobjs_[name]);
  //  gobjs_[name] = geom_oport_->addObj(gobj, name, &slice_lock_);
  return 1;
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
#endif
}

int
Painter::update_slice_from_window(NrrdSlice &slice) {
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

  slice.do_unlock();
  return 1;
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
    error("Please set SCIRUN_FONT_PATH to a directory with scirun.ttf\n");
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
Painter::setup_slice_nrrd(NrrdSlice &slice)
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
