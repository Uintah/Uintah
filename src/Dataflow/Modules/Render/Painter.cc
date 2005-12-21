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
#include <Dataflow/Network/Scheduler.h>
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
#include <Core/Thread/CleanupManager.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Util/Timer.h>
#include <Core/Util/Environment.h>
#include <Core/Volume/CM2Widget.h>

namespace SCIRun {

  /* Todo:
     X - Persistent volume state when re-executing
     X - Fix Non-origin world_to_index conversion
     X - Use FreeTypeTextTexture class for text
     X - Send Bundles correctly
     X - Ability to render 2D nrrds
     X - Automatic world space grid
     X - Add tool support for ITK filters
       - help mode
       - Better tool mechanism to allow for customization & event fallthrough 
       - Add support for 4, 5-D nrrds (includes time and rgba)
       - Add back in MIP mode
       - Geometry output port
       - Show other windows slice correctly
       - Automatic index space grid
       - Remove TCLTask::lock and replace w/ volume lock
       - View to choose tools
       - Change tools to store error codes
       - Add keybooard to tools
       - Migrate all operations to tools (next_siice, zoom, etc)
       - Faster painting, using current window buffer
       - Use GPU/3DTextures for applying colormap when supported
       - Remove clever offseting for non-power-of-2 suported machines
       - Removal of for_each
       - Support applying CM2
       - Optimize build_index_to_world_matrix out

       Filters:
       confidence connected image filter
       discrete gaussian image filter
       gradient magnitude image filter
       binary dilate/erode filters
       later add greyscale dilate/erode filters
       *aniosotropicimagediffusionfilters* (vector if supported)
       watershed
       

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
  center_(0,0,0),
  normal_(1,0,0),
  slice_num_(ctx->subVar("slice"), 0),
  axis_(ctx->subVar("axis"), 2),
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
  for (int a = 1; a < nrrd->dim; ++a)
    size *= nrrd->axis[a].size;
  return size*nrrd_type_size(nrrd);
}

Painter::NrrdVolume::NrrdVolume(GuiContext *ctx,
                                const string &name,
                                NrrdDataHandle &nrrd) :
  nrrd_(0),
  gui_context_(ctx),
  name_(gui_context_->subVar("name"), name),
  opacity_(gui_context_->subVar("opacity"), 1.0),
  clut_min_(gui_context_->subVar("clut_min"), 0.0),
  clut_max_(gui_context_->subVar("clut_max"), 1.0),
  semaphore_(gui_context_->getfullname().c_str(), 1),
  data_min_(0),
  data_max_(1.0),
  colormap_(ctx->subVar("colormap"), 0),
  stub_axes_()
{
  set_nrrd(nrrd);
}


Painter::NrrdVolume::NrrdVolume(NrrdVolume *copy, 
                                const string &name,
                                bool clear) :
  nrrd_(scinew NrrdData()),
  gui_context_(copy->gui_context_->get_parent()->subVar(name,0)),
  name_(gui_context_->subVar("name"), name),
  opacity_(gui_context_->subVar("opacity"), copy->opacity_.get()),
  clut_min_(gui_context_->subVar("clut_min"), copy->clut_min_.get()),
  clut_max_(gui_context_->subVar("clut_max"), copy->clut_max_.get()),
  semaphore_(gui_context_->getfullname().c_str(), 1),
  data_min_(copy->data_min_),
  data_max_(copy->data_max_),
  colormap_(gui_context_->subVar("colormap"), copy->colormap_.get()),
  stub_axes_(copy->stub_axes_)
{
  nrrdCopy(nrrd_->nrrd, copy->nrrd_->nrrd);
  if (clear) 
    memset(nrrd_->nrrd->data, 0, nrrd_data_size(nrrd_->nrrd));

  set_nrrd(nrrd_);
}

  
void
Painter::NrrdVolume::set_nrrd(NrrdDataHandle &nrrd) 
{
  nrrd_ = nrrd;
  if (nrrd->nrrd->dim == 2) {
    nrrd_ = new NrrdData();
    nrrdAxesInsert(nrrd_->nrrd, nrrd->nrrd, 2);
    nrrd_->nrrd->axis[2].min = 0.0;
    nrrd_->nrrd->axis[2].max = 1.0;
    nrrd_->nrrd->axis[2].spacing = 1.0;
    stub_axes_.push_back(2);
  }

  for (int a = 0; a < nrrd_->nrrd->dim; ++a) {
    if (nrrd_->nrrd->axis[a].min > nrrd_->nrrd->axis[a].max)
      SWAP(nrrd_->nrrd->axis[a].min,nrrd_->nrrd->axis[a].max);
    if (nrrd_->nrrd->axis[a].spacing < 0.0)
      nrrd_->nrrd->axis[a].spacing *= -1.0;
  }

  NrrdRange range;
  nrrdRangeSet(&range, nrrd_->nrrd, 0);
  if (data_min_ != range.min || data_max_ != range.max) {
    data_min_ = range.min;
    data_max_ = range.max;
    clut_min_ = range.min;
    clut_max_ = range.max;
    opacity_ = 1.0;
  }
}


NrrdDataHandle
Painter::NrrdVolume::get_nrrd() 
{
  NrrdDataHandle nrrd = nrrd_;
  for (unsigned int s = 0; s < stub_axes_.size(); ++s) {
    NrrdDataHandle nout = new NrrdData();
    nrrdAxesDelete(nout->nrrd, nrrd->nrrd, stub_axes_[s]);
    nrrd = nout;
  }
  return nrrd;
}


DECLARE_MAKER(Painter)

Painter::Painter(GuiContext* ctx) :
  Module("Painter", ctx, Filter, "Render", "SCIRun"),
  layouts_(),
  volumes_(),
  volume_map_(),
  current_volume_(0),
  colormaps_(),
  tool_(0),
  anatomical_coordinates_(ctx->subVar("anatomical_coordinates"), 1),
  show_text_(ctx->subVar("show_text"), 1),
  font_r_(ctx->subVar("color_font-r"), 1.0),
  font_g_(ctx->subVar("color_font-g"), 1.0),
  font_b_(ctx->subVar("color_font-b"), 1.0),
  font_a_(ctx->subVar("color_font-a"), 1.0),
  bundle_oport_((BundleOPort *)get_oport("Paint Data")),
  freetype_lib_(0),
  fonts_(),
  font_size_(ctx->subVar("font_size"),15.0),
  runner_(0),
  runner_thread_(0),
  filter_(0),
  fps_(0.0),
  current_layout_(0),
  executing_(0)
{
  runner_ = scinew RealDrawer(this);
  runner_thread_ = scinew Thread(runner_, string(id+" OpenGL drawer").c_str());
  mouse_.window_ = 0;
  mouse_.position_ = Point(0,0,0);
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
void
Painter::set_context(Network *net) {
  Module::set_context(net);
  sched->add_callback(this->static_callback, this);
}

int
Painter::render_window(SliceWindow &window) {
  if (!window.redraw_) return 0;
  window.redraw_ = false;
  window.viewport_->make_current();
  window.viewport_->clear();
  window.setup_gl_view();
  if (window.autoview_) {
    window.autoview_ = false;
    autoview(window);
    window.setup_gl_view();
  }
  CHECK_OPENGL_ERROR();
  
  window.render_grid();

  for (unsigned int s = 0; s < window.slices_.size(); ++s)
    window.slices_[s]->draw();

  draw_slice_lines(window);

  draw_guide_lines(window, 
                   mouse_.position_.x(), 
                   mouse_.position_.y(),
                   mouse_.position_.z());
  
  if (tool_) 
    tool_->draw(window);

  window.render_text();
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
  Vector tmp = window.x_dir();
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

  vector<int> max_slice(3,0);
  Vector scale(1,1,1);

  if (current_volume_) {
    scale = current_volume_->scale();
    max_slice = current_volume_->max_index();
  }


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

    
  Point cvll = window.screen_to_world(0, 0);
  Point cvur = window.screen_to_world(window.viewport_->max_width(), 
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
  Vector tmp = window.x_dir();
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


double div_d(double dividend, double divisor) {
  return Floor(dividend/divisor);
}

double mod_d(double dividend, double divisor) {
  return dividend - Floor(dividend/divisor)*divisor;
}



void
Painter::SliceWindow::draw_line(const Point &p1, const Point &p2) {
#if 0
  Point s1 = world_to_screen(p1);
  Point s2 = world_to_screen(p2);
  Point temp1 = screen_to_world(s1.x(), s1.y());
  Point temp2 = screen_to_world(s2.x(), s2.y());

  Vector v1 = s2 - s1;
  Vector v2(0,0,1);
  Vector v3 = Cross(v1,v2);
  if (v3.length() < 0.00001) return;
  v3.normalize();
  s1(0) = Floor(s1.x());
  s1(1) = Floor(s1.y());
  s2(0) = Floor(s2.x());
  s2(1) = Floor(s2.y());
  Point s11 = s1+v3;
  Point s22 = s2+v3;

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glScaled(2.0, 2.0, 2.0);
  glTranslated(-.5, -.5, -.5);
  GLint gl_vp[4];
  glGetIntegerv(GL_VIEWPORT, gl_vp);
  glScaled(1.0/(gl_vp[2] - gl_vp[0]), 
           1.0/(gl_vp[3] - gl_vp[1]), 1.0);
  CHECK_OPENGL_ERROR();
//   s1 = screen_to_world(s1.x(), s1.y());
//   s2 = screen_to_world(s2.x(), s2.y());
//   s11 = screen_to_world(s11.x(), s11.y());
//   s22 = screen_to_world(s22.x(), s22.y());
  glBegin(GL_QUADS);
  glVertex3dv(&s1(0));
  glVertex3dv(&s11(0));
  glVertex3dv(&s22(0));
  glVertex3dv(&s2(0));
  glEnd();

  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  CHECK_OPENGL_ERROR();
#endif
}


// renders vertical and horizontal bars that represent
// selected slices in other dimensions
void
Painter::SliceWindow::render_grid()
{
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

  Point min = screen_to_world(0,0);
  Point max = screen_to_world(viewport_->width()-1, viewport_->height()-1);

  int xax = painter_.x_axis(*this);
  int yax = painter_.y_axis(*this);
  min(xax) = div_d(min(xax), gap)*gap;
  min(yax) = div_d(min(yax), gap)*gap;

  glColor4d(1.0, 1.0, 1.0, 1.0);
  glDisable(GL_TEXTURE_2D);
  glDisable(GL_BLEND);
  CHECK_OPENGL_ERROR();
  FreeTypeTextTexture text("X",painter_.fonts_["default"]);
  int num = 0;
  Point linemin = min;
  Point linemax = min;
  linemax(yax) = max(yax);
  while (linemin(xax) < max(xax)) {
    linemin(xax) = linemax(xax) = min(xax) + gap*num;
    glBegin(GL_LINES);
    glVertex3dv(&linemin(0));
    glVertex3dv(&linemax(0));
    glEnd();

    text.set(to_string(linemin(xax)));

    Point pos = world_to_screen(linemin);
    text.draw(pos.x()+1, pos.y(), FreeTypeTextTexture::sw);

    pos = world_to_screen(linemax);
    text.draw(pos.x()+1, pos.y(), FreeTypeTextTexture::nw);

    num++;
  }

  num = 0;
  linemin = linemax = min;
  linemax(xax) = max(xax);
  while (linemin(yax) < max(yax)) {
    linemin(yax) = linemax(yax) = min(yax) + gap*num;
    glBegin(GL_LINES);
    glVertex3dv(&linemin(0));
    glVertex3dv(&linemax(0));
    glEnd();

    text.set(to_string(linemin(yax)));

    Point pos = world_to_screen(linemin);
    text.draw(pos.x(), pos.y()+1, FreeTypeTextTexture::sw);

    pos = world_to_screen(linemax);
    text.draw(pos.x(), pos.y()+1, FreeTypeTextTexture::se);
    
    CHECK_OPENGL_ERROR();
    num++;
  }
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
  Vector dir = window.x_dir();
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
  Vector dir = window.y_dir();
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

  //  const Point center = painter_.current_volume_->center(axis_, slice_num_);
  double hwid = viewport_->width()*50.0/zoom_();
  double hhei = viewport_->height()*50.0/zoom_();
  double cx = center_(painter_.x_axis(*this));
  double cy = center_(painter_.y_axis(*this));
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
  FreeTypeFace *font = painter_.fonts_["orientation"];
  if (!font) return;

  int prim = painter_.x_axis(*this);
  int sec = painter_.y_axis(*this);
  
  string ltext, rtext, ttext, btext;

  if (painter_.anatomical_coordinates_()) {
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

  FreeTypeTextTexture texture(ltext, font);
  texture.draw(2, viewport_->height()/2,
               FreeTypeTextTexture::w);

  texture.set(rtext);
  texture.draw(viewport_->width()-2, viewport_->height()/2,
               FreeTypeTextTexture::e);

  texture.set(btext);
  texture.draw(viewport_->width()/2, 2,
               FreeTypeTextTexture::s);

  texture.set(ttext);
  texture.draw(viewport_->width()/2, viewport_->height()-2, 
               FreeTypeTextTexture::n);
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
    ColorMapHandle cmap = painter_->get_colormap(volume_->colormap_.get());
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
Painter::SliceWindow::render_text() {
  if (!painter_.show_text_) return;

  FreeTypeFace *font = painter_.fonts_["default"];
  if (!font) return;
  
  FreeTypeTextTexture 
    text("X: "+double_to_string(painter_.mouse_.position_.x())+
         " Y: "+double_to_string(painter_.mouse_.position_.y())+
         " Z: "+double_to_string(painter_.mouse_.position_.z()), font);
  text.draw(0,0);
  const int y_pos = text.height()+2;
  text.set("Zoom: "+to_string(zoom_())+"%");
  text.draw(0, y_pos);
    
  NrrdVolume *vol = painter_.current_volume_;
  if (vol) {
    const float ww = vol->clut_max_ - vol->clut_min_;
    const float wl = vol->clut_min_ + ww/2.0;
    text.set("WL: " + to_string(wl) +  " -- WW: " + to_string(ww));
    text.draw(0, y_pos*2);

    text.set("Min: " + to_string(vol->clut_min_) + 
             " -- Max: " + to_string(vol->clut_max_));
    text.draw(0, y_pos*3);

    if (this == painter_.mouse_.window_) {
      vector<int> index = vol->world_to_index(painter_.mouse_.position_);
      if (vol->index_valid(index)) {
        double val;
        vol->get_value(index, val);
        text.set("Value: " + to_string(val));
        text.draw(0,y_pos*4);
        
        text.set("S: "+to_string(index[0])+
                 " C: "+to_string(index[1])+
                 " A: "+to_string(index[2]));
        text.draw(0, y_pos*5);
      }
    }
  }


  if (string(sci_getenv("USER")) == string("mdavis")) {
    text.set("fps: "+to_string(painter_.fps_));
    text.draw(0, viewport_->height() - 2, FreeTypeTextTexture::nw);
  }

  render_orientation_text();

  font = painter_.fonts_["view"];
  if (!font) return;
  string str;
  if (painter_.anatomical_coordinates_()) { 
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

  text = FreeTypeTextTexture(str, font);
  text.draw(viewport_->width() - 2, 0, FreeTypeTextTexture::se);
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
  for (unsigned int s = 0; s < volumes_.size(); ++s)
    window.slices_[s]->volume_ = volumes_[s];
  for_each(window, &Painter::set_slice_nrrd_dirty);

  return window.slices_.size();
}

int
Painter::set_slice_nrrd_dirty(NrrdSlice &slice) {
  slice.do_lock();
  slice.nrrd_dirty_ = true;
  slice.do_unlock();
  return 0;
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
  gluUnProject(double(x), double(y), 0,
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
      world_coords[i] = p(i)-transform.get(i,transform.ncols()-1);
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
Painter::NrrdVolume::vector_to_index(const Vector &v) {
  DenseMatrix transform = build_index_to_world_matrix();
  ColumnMatrix index_matrix(transform.ncols());
  ColumnMatrix world_coords(transform.nrows());
  for (int i = 0; i < transform.nrows(); ++i)
    if (i < 3) 
      world_coords[i] = v[i];
    else       
      world_coords[i] = 0.0;;
  int tmp, tmp2;
  transform.mult_transpose(world_coords, index_matrix, tmp, tmp2);
  vector<double> return_val(index_matrix.nrows()-1);
  for (unsigned int i = 0; i < return_val.size(); ++i)
    return_val[i] = index_matrix[i];
  return return_val;
}



DenseMatrix
Painter::NrrdVolume::build_index_to_world_matrix() {
  Nrrd *nrrd = nrrd_->nrrd;
  int dim = nrrd->dim+1;
  DenseMatrix matrix(dim, dim);
  matrix.zero();
  for (int i = 0; i < dim-1; ++i)
    if (airExists(nrrd->axis[i].spacing))
      matrix.put(i,i,nrrd->axis[i].spacing);
    else 
      matrix.put(i,i,((nrrd->axis[i].max-nrrd->axis[i].min+1.0)/
                      nrrd->axis[i].size));
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
Painter::send_data()
{
  BundleHandle bundle = new Bundle();
  for (unsigned int v = 0; v < volumes_.size(); ++v) {
    string name = volumes_[v]->name_.get();
    NrrdDataHandle nrrd = volumes_[v]->get_nrrd();
    bundle->setNrrd(name, nrrd);
  }
  BundleOPort *oport = (BundleOPort *)get_oport("Paint Data");
  ASSERT(oport);
  oport->send(bundle);  
}



bool
Painter::receive_filter_data()
{   
  if (!filter_) 
    return false;

  BundleIPort *filter_port = (BundleIPort *)get_iport("Filter Data");
  ASSERT(filter_port);
  BundleHandle bundle = 0;
  filter_port->get(bundle);
  if (bundle.get_rep()) 
    bundles_.push_back(bundle);
  
  return true;
}

bool
Painter::static_callback(void *this_ptr) {
  return ((Painter *)this_ptr)->callback();
}

bool
Painter::callback() {
//   if (filter_) 
//     cerr << "filter = false\n";
  filter_ = 0;
  return true;
}
  
void
Painter::execute()
{
  update_state(Module::JustStarted);
  update_state(Module::NeedData);

  BundleHandle bundle = 0;
  bundles_.clear();

  receive_filter_data();

  port_range_type prange = get_iports("Paint Data");
  port_map_type::iterator pi = prange.first;
  while (pi != prange.second) {
    BundleIPort *iport = (BundleIPort *)get_iport(pi++->second);
    
    if (iport && iport->get(bundle) && bundle.get_rep())
      bundles_.push_back(bundle);
  }
    
  vector<NrrdDataHandle> nrrds;
  vector<string> nrrd_names;
  colormaps_.clear();
  colormap_names_.clear();

  for (unsigned int b = 0; b < bundles_.size(); ++b) {
    int numNrrds = bundles_[b]->numNrrds();
    for (int n = 0; n < numNrrds; n++) {
      string name = bundles_[b]->getNrrdName(n);
      NrrdDataHandle nrrdH = bundles_[b]->getNrrd(name);
      if (!nrrdH.get_rep()) continue;
      if (nrrdH->nrrd->dim < 2)
      {
        warning("Nrrd with dim < 2, skipping.");
        continue;
      }
      nrrds.push_back(nrrdH);
      nrrd_names.push_back(name);
    }
    
    int numColormaps = bundles_[b]->numColorMaps();
    for (int n = 0; n < numColormaps; n++) {
      const string name = bundles_[b]->getColorMapName(n);
      ColorMapHandle cmap = bundles_[b]->getColorMap(name);
      if (cmap.get_rep()) {
        colormaps_[name] = cmap;
        colormap_names_.push_back(name);
      }
    }
  }
  
  update_state(Module::Executing);
  TCLTask::lock();
  
  for (unsigned int n = 0; n < nrrds.size(); ++n) {
    if (volume_map_.find(nrrd_names[n]) == volume_map_.end()) 
      {
        volume_map_[nrrd_names[n]] =  
          new NrrdVolume(ctx->subVar(nrrd_names[n]), nrrd_names[n], nrrds[n]);
      } else {
        volume_map_[nrrd_names[n]]->set_nrrd(nrrds[n]); 
      }
  }

  volumes_.clear();
  NrrdVolumeMap::iterator iter=volume_map_.begin(), last=volume_map_.end();
  while (iter != last) {
    volumes_.push_back(iter->second);
    iter++;
  }

  if (volumes_.size()) 
    current_volume_ = volumes_.back();      

  for_each(&Painter::extract_window_slices);
  for_each(&Painter::mark_autoview);

  TCLTask::unlock();

  send_data();

  update_state(Module::Completed);
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
    mouse_.position_ = mouse_.window_->screen_to_world(mouse_.x_, mouse_.y_);
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
        mouse_.window_->zoom_in();
      else
        mouse_.window_->next_slice();
      break;
      
    case 5:
      if (mouse_.state_ & MouseState::SHIFT_E == MouseState::SHIFT_E) 
        mouse_.window_->zoom_out();
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

  if (tool_) {
    string *err = tool_->mouse_button_release(mouse_);
    if (err && *err == "Done") {
      cerr << tool_->get_name() << " tool Done\n";
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

    if (args[4] == "equal" || args[4] == "plus") window.zoom_in();
    else if (args[4] == "minus" || args[4] == "underscore") window.zoom_out();
    else if (args[4] == "less" || args[4] == "comma") window.prev_slice();
    else if (args[4] == "greater" || args[4] == "period") window.next_slice();
    else if (args[4] == "bracketleft") {
      if (current_volume_) {
        current_volume_->colormap_.set(Max(0, current_volume_->colormap_.get()-1));
        for_each(&Painter::rebind_slice);
        for_each(&Painter::redraw_window);
      }
    }
    else if (args[4] == "bracketright") {
      if (current_volume_) {
        current_volume_->colormap_.set(Min(int(colormap_names_.size()), 
                                           current_volume_->colormap_.get()+1));
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
    else if (args[4] == "h") { 
      if (!tool_) 
        tool_ = scinew ITKThresholdTool(this, false);
    }
    else if (args[4] == "i") { 
      if (!tool_) 
        tool_ = scinew ITKThresholdTool(this, true);
    }
    else if (args[4] == "j") { 
      if (!tool_) 
        tool_ = scinew StatisticsTool(this);
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
//     } else if (args[4] == "Left") {
//       window.x_ += pan_delta;
//       redraw_window(window);
//     } else if (args[4] == "Right") {
//       window.x_ -= pan_delta;
//       redraw_window(window);
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
  else if (args[1] == "set_font_sizes") set_font_sizes(font_size_());
  else if (args[1] == "setgl") {
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
Painter::mark_autoview(SliceWindow &window) {
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
  
//   Vector xdir = window.x_dir();
//   vector<double> x_index = current_volume_->vector_to_index(xdir);
//   for (int i = 0; i < x_index.size(); ++i) 
//     cerr << x_index[i] << ", ";
//   cerr << " <- x_index\n";

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
  window.center_(xax) = current_volume_->center()(xax);
  window.center_(yax) = current_volume_->center()(yax);
  redraw_window(window);
  return 1;
}
   




int
Painter::create_volume(NrrdVolumes *copies) {
  if (!current_volume_) return -1;
  create_volume(current_volume_->name_.get()+"-Copy", 0);
  return volumes_.size()-1;
}


Painter::NrrdVolume *
Painter::create_volume(string name, int nrrdType) {
  if (!current_volume_) return 0;
  volumes_.push_back(new NrrdVolume(current_volume_, name, 1));
  //volume_map_[name] = volumes_.back();
  current_volume_ = volumes_.back();
  for_each(&Painter::extract_window_slices);
  return current_volume_;
}

    
} // end namespace SCIRun
