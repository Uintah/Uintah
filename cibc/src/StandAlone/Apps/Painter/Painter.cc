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

#include <StandAlone/Apps/Painter/Painter.h>
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
#include <Core/Geom/TkOpenGLContext.h>
#include <Core/Geom/OpenGLViewport.h>
#include <Core/Geom/FreeType.h>
#include <Core/GuiInterface/UIvar.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/CleanupManager.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/Environment.h>
#include <Core/Volume/CM2Widget.h>
#include <Core/Geom/TextRenderer.h>
#include <Core/Geom/FontManager.h>
#include <Core/Util/SimpleProfiler.h>
#include <Core/GuiInterface/TCLKeysyms.h>
#include <Core/Skinner/Variables.h>
#include <Core/Events/EventManager.h>
#include <Core/Util/FileUtils.h>

#ifdef HAVE_INSIGHT
#  include <itkImportImageFilter.h>
#include <itkThresholdSegmentationLevelSetImageFilter.h>
#endif

#ifdef _WIN32
#  define snprintf _snprintf
#  define SCISHARE __declspec(dllimport)
#else
#  define SCISHARE
#endif

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
     X - Removal of for_each
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


Painter::SliceWindow::SliceWindow(Skinner::Variables *variables,
                                  Painter *painter) :  
  Skinner::Drawable(variables), 
  painter_(painter),
  name_(variables->get_id()),
  viewport_(0),
  slices_(),
  slice_map_(),
  paint_layer_(0),
  center_(0,0,0),
  normal_(0,0,0),
  slice_num_(0,0), //(!ctx) ? 0 : (!ctx) ? 0 : ctx->subVar("slice"), 0),
  axis_(2),
  zoom_(0,100.0),//!ctx) ? 0 : ctx->subVar("zoom"), 100.0),
  slab_min_(0,0),//(!ctx) ? 0 : ctx->subVar("slab_min"), 0),
  slab_max_(0,0),//(!ctx) ? 0 : ctx->subVar("slab_max"), 0),
  redraw_(true),
  autoview_(true),
  mode_(0,0),//(!ctx) ? 0 : ctx->subVar("mode"),0),
  show_guidelines_(0,1),//(!ctx) ? 0 : ctx->subVar("show_guidelines"),1),
  cursor_pixmap_(-1)
{
  int axis = 2;
  variables->maybe_get_int("axis", axis);
  set_axis(axis);
  //  axis_ = axis;
}



Painter::NrrdVolume::NrrdVolume(GuiContext *ctx,
                                const string &name,
                                NrrdDataHandle &nrrd) :
  nrrd_handle_(0),
  gui_context_(ctx),
  name_((!ctx) ? 0 : ctx->subVar("name"), name),
  name_prefix_(""),
  opacity_((!ctx) ? 0 : ctx->subVar("opacity"), 1.0),
  clut_min_((!ctx) ? 0 : ctx->subVar("clut_min"), 0.0),
  clut_max_((!ctx) ? 0 : ctx->subVar("clut_max"), 1.0),
  mutex_((!ctx) ? 0 : ctx->getfullname().c_str()),
  data_min_(0),
  data_max_(1.0),
  colormap_((!ctx) ? 0 : ctx->subVar("colormap")),
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
  gui_context_(0),
  name_((!gui_context_) ? 0 : gui_context_->subVar("name"), name),
  name_prefix_(copy->name_prefix_),
  opacity_((!gui_context_) ? 0 : gui_context_->subVar("opacity"), copy->opacity_.get()),
  clut_min_((!gui_context_) ? 0 : gui_context_->subVar("clut_min"), copy->clut_min_.get()),
  clut_max_((!gui_context_) ? 0 : gui_context_->subVar("clut_max"), copy->clut_max_.get()),
  mutex_((!gui_context_) ? 0 : gui_context_->getfullname().c_str()),
  data_min_(copy->data_min_),
  data_max_(copy->data_max_),
  colormap_((!gui_context_) ? 0 : gui_context_->subVar("colormap"), copy->colormap_.get()),
  stub_axes_(copy->stub_axes_),
  transform_(),
  keep_(copy->keep_)
{
  copy->mutex_.lock();
  mutex_.lock();

  ASSERT(clear >= 0 && clear <= 2);
  
  switch (clear) {
  case 0: {
    nrrd_handle_ = scinew NrrdData();
    nrrdCopy(nrrd_handle_->nrrd_, copy->nrrd_handle_->nrrd_);
  } break;
  case 1: {
    nrrd_handle_ = scinew NrrdData();
    nrrdCopy(nrrd_handle_->nrrd_, copy->nrrd_handle_->nrrd_);
    memset(nrrd_handle_->nrrd_->data, 0, nrrd_data_size(nrrd_handle_->nrrd_));
  } break;
  default:
  case 2: {
    nrrd_handle_ = copy->nrrd_handle_;
  } break;
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
  //  nrrd_handle_.detach();
  //  nrrdBasicInfoCopy(nrrd_handle_->nrrd_, nrrd->nrrd,0);
  //  nrrdAxisInfoCopy(nrrd_handle_->nrrd_, nrrd->nrrd, 0,0);
  //  nrrdCopy(nrrd_handle_->nrrd_, nrrd->nrrd);
  Nrrd *n = nrrd_handle_->nrrd_;

  stub_axes_.clear();
  if (n->axis[0].size > 4) {
    nrrdAxesInsert(n, n, 0);
    n->axis[0].min = 0.0;
    n->axis[0].max = 1.0;
    n->axis[0].spacing = 1.0;
    stub_axes_.push_back(0);
  }

  if (n->dim == 3) {
    nrrdAxesInsert(n, n, 3);
    n->axis[3].min = 0.0;
    n->axis[3].max = 1.0;
    n->axis[3].spacing = 1.0;
    stub_axes_.push_back(3);
  }


  for (unsigned int a = 0; a < n->dim; ++a) {
    if (n->axis[a].center == nrrdCenterUnknown)
      n->axis[a].center = nrrdCenterNode;

#if 0
    if (airIsNaN(n->axis[a].min) && 
        airIsNaN(n->axis[a].max)) 
    {
      if (airIsNaN(n->axis[a].spacing)) {
        n->axis[0].spacing = 1.0;
      }
#endif

    if (n->axis[a].min > n->axis[a].max)
      SWAP(n->axis[a].min,n->axis[a].max);
    if (n->axis[a].spacing < 0.0)
      n->axis[a].spacing *= -1.0;
  }
#if 0
  NrrdRange range;
  nrrdRangeSet(&range, n, 0);
  if (data_min_ != range.min || data_max_ != range.max) {
    data_min_ = range.min;
    data_max_ = range.max;
    clut_min_ = range.min;
    clut_max_ = range.max;
    opacity_ = 1.0;
  }
#endif
  mutex_.unlock();
  reset_data_range();
  build_index_to_world_matrix();

}



void
Painter::NrrdVolume::reset_data_range() 
{
  mutex_.lock();
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


Painter::Painter(Skinner::Variables *variables, GuiContext* ctx) :
  //  Module("Painter", ctx, Filter, "Render", "SCIRun"),
  Parent(variables),
  cur_window_(0),
  tm_("Painter"),
  pointer_pos_(),
  windows_(),
  volumes_(),
  volume_map_(),
  volume_order_(),
  current_volume_(0),
  undo_volume_(0),
  colormaps_(),
  tools_(),
  anatomical_coordinates_((!ctx) ? 0 : ctx->subVar("anatomical_coordinates"), 1),
  show_grid_((!ctx) ? 0 : ctx->subVar("show_grid"), 1),
  show_text_((!ctx) ? 0 : ctx->subVar("show_text"), 1),
  volume_lock_("Volume"),
  bundles_(),
  filter_volume_(0),
  abort_filter_(false)
{
#ifdef HAVE_INSIGHT
  filter_update_img_ = 0;
#endif

  tm_.add_tool(new PointerToolSelectorTool(this), 50);
  tm_.add_tool(new KeyToolSelectorTool(this), 51);

  InitializeSignalCatcherTargets(0);
  Skinner::Signal *signal = new Skinner::Signal("LoadColorMap1D",
                                                this, "LoadColorMap1D");
  string srcdir = sci_getenv("SCIRUN_SRCDIR")+string("/Core/Skinner/Data/");
  signal->set_signal_data(srcdir+"Rainbow.cmap");
  LoadColorMap1D(signal);
}

Painter::~Painter()
{
//   int count = 1;
//   for (NrrdVolumes::iterator iter = volumes_.begin(); 
//        iter != volumes_.end(); ++iter) {
//     string filename = "/tmp/painter-nrrd"+to_string(count++)+".nrrd";
//     cerr << "saving " << filename << std::endl;
//     nrrdSave(filename.c_str(),
//              (*iter)->nrrd_handle_->nrrd_,
//              0);
//   }


}


static SimpleProfiler profiler("RenderWindow", sci_getenv_p("SCIRUN_PROFILE"));

string
double_to_string(double val)
{
  char s[50];
  snprintf(s, 49, "%1.2f", val);
  return string(s);
}


void
Painter::SliceWindow::render_gl() {
  painter_->volume_lock_.lock();

  NrrdVolume *vol = painter_->current_volume_;

  string clut_ww_wl = "";
  string clut_min_max = "";
  string value = "";
  string xyz_pos = "";
  string sca_pos = "";
  if (vol) {
    const float ww = vol->clut_max_ - vol->clut_min_;
    const float wl = vol->clut_min_ + ww/2.0;
    clut_ww_wl = "WL: " + to_string(wl) +  " -- WW: " + to_string(ww);

    clut_min_max = ("Min: " + to_string(vol->clut_min_) + 
                    " -- Max: " + to_string(vol->clut_max_));
    vector<int> index = vol->world_to_index(painter_->pointer_pos_);
    if (vol->index_valid(index) && painter_->cur_window_ == this) {
      sca_pos = ("S: "+to_string(index[1])+
                 " C: "+to_string(index[2])+
                 " A: "+to_string(index[3]));
      xyz_pos = ("X: "+double_to_string(painter_->pointer_pos_.x())+
                 " Y: "+double_to_string(painter_->pointer_pos_.y())+
                 " Z: "+double_to_string(painter_->pointer_pos_.z()));
      double val = 1.0;
      vol->get_value(index, val);
      value = "Value: " + to_string(val);
    }
  }

  get_vars()->change_parent("value", value, "string", true);
  get_vars()->change_parent("clut_min_max", clut_min_max, "string", true);
  get_vars()->change_parent("clut_ww_wl", clut_ww_wl, "string", true);
  get_vars()->change_parent("xyz_pos", xyz_pos, "string", true);
  get_vars()->change_parent("sca_pos", sca_pos, "string", true);

    

  setup_gl_view();
  if (autoview_) {
    autoview(painter_->current_volume_);
    setup_gl_view();
  }
  CHECK_OPENGL_ERROR();


  for (unsigned int s = 0; s < slices_.size(); ++s) {
    if (paint_layer_ && slices_[s]->volume_ == paint_layer_->volume_)
      paint_layer_->draw();
    else 
      slices_[s]->draw();
  }

  if (painter_->show_grid_()) 
    render_grid();

  painter_->draw_slice_lines(*this);

  event_handle_t redraw_window = new RedrawSliceWindowEvent(*this);
  painter_->tm_.propagate_event(redraw_window);
  render_text();

  //if (painter_->cur_window_ == this) {
  //    Point windowpos(event_.x_, event_.y_, 0);
  //    window.render_guide_lines(windowpos);
  //}


//   for (unsigned int t = 0; t < tools_.size(); ++t) {
//     tools_[t]->draw(window);
//     if (event_.window_ == &window)
//       tools_[t]->draw_mouse_cursor(event_);
//   }

  CHECK_OPENGL_ERROR();
  painter_->volume_lock_.unlock();
}



void
Painter::redraw_all()
{
  for (SliceWindows::iterator i = windows_.begin(); i != windows_.end(); ++i) {
    (*i)->redraw();
  }
}

void
Painter::SliceWindow::redraw() {
  EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
  redraw_ = true;
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
  double vpw = get_region().width();
  double vph = get_region().height();
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
  double vpw = get_region().width();
  double vph = get_region().height();

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
  for (unsigned int win = 0; win < windows_.size(); ++win) {
      SliceWindow &window2 = *(windows_[win]);
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
  const double vw = get_region().width();
  const double vh = get_region().height();
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
  const Skinner::RectRegion &region = get_region();
  const int vw = Ceil(region.width());
  const int vh = Ceil(region.height());

  //  double grey1[4] = { 0.75, 0.75, 0.75, 1.0 };
  //  double grey2[4] = { 0.5, 0.5, 0.5, 1.0 };
  //  double grey3[4] = { 0.25, 0.25, 0.25, 1.0 };
  //  double white[4] = { 1,1,1,1 };
  //  render_frame(0,0, 15, 15, grey1);
  //  render_frame(15,15, 3, 3, white, grey2 );
  //  render_frame(17,17, 2, 2, grey3);
  //  profiler("render_frame");
  double grid_color = 0.25;

  glDisable(GL_TEXTURE_2D);
  CHECK_OPENGL_ERROR();

  Point min = screen_to_world(Floor(region.x1()),Floor(region.y1()));
  Point max = screen_to_world(Ceil(region.x2())-1, Ceil(region.y2())-1);

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
  string str;
  profiler("start");
  TextRenderer *renderer = FontManager::get_renderer(20);
  renderer->set_color(1,1,1,1);
  renderer->set_shadow_color(0,0,0,1);
  renderer->set_shadow_offset(1, -1);
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
    renderer->render(str, pos.x()+1, 2, 
                     TextRenderer::SHADOW | 
                     TextRenderer:: SW | 
                     TextRenderer::REVERSE);
    
    //    pos = world_to_screen(linemax);
    renderer->render(str, pos.x()+1, vh-2, 
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
    renderer->render(str, 2, pos.y(), 
                     TextRenderer::SHADOW | 
                     TextRenderer::NW | 
                     TextRenderer::VERTICAL | 
                     TextRenderer::REVERSE);

    renderer->render(str, vw-2, pos.y(), 
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
  //  glViewport(region_.x1(), region_.y1(), 
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  //  CHECK_OPENGL_ERROR();
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

  double hwid = get_region().width()*50.0/zoom_();
  double hhei = get_region().height()*50.0/zoom_();

  double cx = center_(x_axis());
  double cy = center_(y_axis());
  
  double diagonal = hwid*hwid+hhei*hhei;

  double maxz = center_(axis_) + diagonal*zoom_()/100.0;
  double minz = center_(axis_) - diagonal*zoom_()/100.0;

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(cx - hwid, cx + hwid, 
	  cy - hhei, cy + hhei, 
	  minz, maxz);

  glGetIntegerv(GL_VIEWPORT, gl_viewport_);
  glGetDoublev(GL_MODELVIEW_MATRIX, gl_modelview_matrix_);
  glGetDoublev(GL_PROJECTION_MATRIX, gl_projection_matrix_);
  CHECK_OPENGL_ERROR();
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
#if 0
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
  text->render(ltext, 2, get_region().height()/2,
            TextRenderer::W | TextRenderer::SHADOW | TextRenderer::REVERSE);
  profiler("ltext");

  text->render(rtext,get_region().width()-2, get_region().height()/2,
               TextRenderer::E | TextRenderer::SHADOW | TextRenderer::REVERSE);
  profiler("rtext");

  text->render(btext,get_region().width()/2, 2,
               TextRenderer::S | TextRenderer::SHADOW | TextRenderer::REVERSE);
  profiler("btext");  

  text->render(ttext,get_region().width()/2, get_region().height()-2, 
               TextRenderer::N | TextRenderer::SHADOW | TextRenderer::REVERSE);
  profiler("ttext");

  profiler.leave();
#endif
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

  double vpw = get_region().width();
  double vph = get_region().height();
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



void
Painter::SliceWindow::render_text()
{
  const int yoff = 19;
  const int xoff = 19;
  const double vw = get_region().width();
  const double vh = get_region().height();

  TextRenderer *renderer = FontManager::get_renderer(20);
  NrrdVolume *vol = painter_->current_volume_;
  const int y_pos = renderer->height("X")+2;
  for (unsigned int s = 0; s < slices_.size(); ++s) {
    string str = slices_[s]->volume_->name_prefix_ + 
      slices_[s]->volume_->name_.get();
    if (slices_[s]->volume_ == vol) {
      renderer->set_color(240/255.0, 1.0, 0.0, 1.0);
      str = "->" + str;
    } else {
      renderer->set_color(1.0, 1.0, 1.0, 1.0);
    }

    renderer->render(str,
                     vw-2-xoff, vh-2-yoff-(y_pos*(slices_.size()-1-s)),
                     TextRenderer::NE | TextRenderer::SHADOW);
  }


#if 0
  font1.set_color(1.0, 1.0, 1.0, 1.0);
  const int y_pos = font1.height("X")+2;
  font1.render("Zoom: "+to_string(zoom_())+"%", xoff, yoff, 
               TextRenderer::SHADOW | TextRenderer::SW);


  if (painter_->tools_.size())
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
      font1.render
                   region_.width()-2-xoff, yoff,
                   TextRenderer::SHADOW | TextRenderer::SE);
      profiler("XYZ");
      vector<int> index = vol->world_to_index(painter_->event_.position_);
      if (vol->index_valid(index)) {
        font1.render(,
                     region_.width()-2-xoff, yoff+y_pos,
                     TextRenderer::SHADOW | TextRenderer::SE);
        profiler("SCA");
        double val = 1.0;
        vol->get_value(index, val);
        font1.render("Value: " + to_string(val),
                     region_.width()-2-xoff,y_pos*2+yoff, 
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
  font2.render(str, region_.width() - 2, 2,
               TextRenderer::SHADOW | 
               TextRenderer::SE | 
               TextRenderer::REVERSE);
  profiler("plane");
  profiler.leave();
#endif
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

void
Painter::extract_all_window_slices() {
  for (SliceWindows::iterator i = windows_.begin(); i != windows_.end(); ++i) {
    (*i)->extract_slices();
  }
}

void
Painter::SliceWindow::extract_slices() {

  NrrdVolumes &volumes = painter_->volumes_;

  for (unsigned int s = slices_.size(); s < slices_.size(); ++s) {
    delete slices_[s];
  }

  if (slices_.size() > volumes.size())
    slices_.resize(volumes.size());
  
  for (unsigned int s = slices_.size(); s < volumes.size(); ++s) {
    if (volumes[s] == painter_->current_volume_) {
      center_ = painter_->current_volume_->center();
    }
    slices_.push_back(scinew NrrdSlice(painter_, volumes[s], center_, normal_));
  }

  slice_map_.clear();
  for (unsigned int s = 0; s < volumes.size(); ++s) {
    if (volumes[s] == painter_->current_volume_ &&
        !painter_->current_volume_->inside_p(center_)) {
      int ax = axis_;
      center_(ax) = painter_->current_volume_->center()(ax);
    }

    slice_map_[volumes[s]] = slices_[s];
    slices_[s]->volume_ = volumes[s];
    slices_[s]->plane_ = Plane(center_, normal_);
    slices_[s]->nrrd_dirty_ = true;
  }

  

  if (paint_layer_) {
    delete paint_layer_;
    paint_layer_ = 0;
  }
}



void
Painter::SliceWindow::set_axis(unsigned int axis) {
  axis_ = axis % 3;
  normal_ = Vector(axis == 0 ? 1 : 0,
                   axis == 1 ? 1 : 0,
                   axis == 2 ? 1 : 0);
  extract_slices();
  redraw_ = true;
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
  extract_slices();
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
  extract_slices();
  painter_->redraw_all();
}

void
Painter::move_layer_up()
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
  
  extract_all_window_slices();
  redraw_all();
}

void
Painter::move_layer_down()
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

  extract_all_window_slices();

  redraw_all();
}


void
Painter::opacity_down()
{
  if (current_volume_) {
    current_volume_->opacity_ = 
      Clamp(current_volume_->opacity_-0.05, 0.0, 1.0);
    redraw_all();
  }
}

void
Painter::opacity_up()
{
  if (current_volume_) {
    current_volume_->opacity_ = 
      Clamp(current_volume_->opacity_+0.05, 0.0, 1.0);
    redraw_all();
  }
}



void
Painter::cur_layer_down()
{
  if (volumes_.size() < 2 || current_volume_ == volumes_[0]) 
    return;
  for (unsigned int i = 1; i < volumes_.size(); ++i)
    if (current_volume_ == volumes_[i]) {
      current_volume_ = volumes_[i-1];
      redraw_all();
      return;
    }
}


void
Painter::cur_layer_up()
{
  if (volumes_.size() < 2 || current_volume_ == volumes_.back()) 
    return;
  for (unsigned int i = 0; i < volumes_.size()-1; ++i)
    if (current_volume_ == volumes_[i]) {
      current_volume_ = volumes_[i+1];
      redraw_all();
      return;
    }
}


void
Painter::reset_clut()
{
  if (current_volume_) {
    current_volume_->clut_min_ = current_volume_->data_min_;
    current_volume_->clut_max_ = current_volume_->data_max_;
    set_all_slices_tex_dirty();
    redraw_all();
  }
}




void
Painter::SliceWindow::zoom_in()
{
  zoom_ *= 1.1;
  redraw();
}

void
Painter::SliceWindow::zoom_out()
{
  zoom_ /= 1.1;
  redraw();
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
  xyz[0] -= gl_viewport_[0];
  xyz[1] -= gl_viewport_[1];

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
    if (airExists(nrrd->axis[i].spacing)) {
      if (nrrd->axis[i].spacing == 0.0) {
        nrrd->axis[i].spacing = 1.0;
      }
      matrix.put(i,i,nrrd->axis[i].spacing);
    } else if (airExists(nrrd->axis[i].min) && airExists(nrrd->axis[i].max)) {
      if (nrrd->axis[i].min == nrrd->axis[i].max) {
        nrrd->axis[i].spacing = 1.0;
        matrix.put(i,i,1.0);
      } else {
        matrix.put(i,i,((nrrd->axis[i].max-nrrd->axis[i].min)/
                        nrrd->axis[i].size));
      }
    } else {
      matrix.put(i,i, 1.0);
    }

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

//  BundleOPort *oport = (BundleOPort *)get_oport("Paint Data");
//  ASSERT(oport);
//  oport->send(bundle);  
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
        cerr << "Nrrd with dim < 2, skipping.";
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
  
  // update_state(Module::Executing);

  NrrdVolumeMap::iterator viter = volume_map_.begin();
  NrrdVolumeMap::iterator vend = volume_map_.end();

  for (unsigned int n = 0; n < nrrds.size(); ++n) {
    string name = nrrd_names[n];
    viter = volume_map_.find(name);
    if (viter == vend || viter->second == 0) {
      volume_map_[name] = 
        new NrrdVolume(0, name, nrrds[n]);
      show_volume(name);
    } else {
      if (nrrds[n]->generation > viter->second->nrrd_handle_->generation)
          viter->second->set_nrrd(nrrds[n]);
    }
    volume_map_[name]->keep_ = 1;
  }
}



void
Painter::add_bundle(BundleHandle bundle)
{
  bundles_.push_back(bundle);
  extract_data_from_bundles(bundles_);
  recompute_volume_list();
}


void
Painter::set_probe() {
  for (SliceWindows::iterator i = windows_.begin(); i != windows_.end(); ++i) {
    (*i)->set_probe();
  }
  redraw_all();
}


void
Painter::SliceWindow::set_probe() {
  if (!painter_->current_volume_) return;
  if (painter_->cur_window_ != this &&
      painter_->current_volume_->inside_p(painter_->pointer_pos_)) {
    center_(axis_) = painter_->pointer_pos_(axis_);
    extract_slices();
  }
}


void
Painter::NrrdSlice::set_tex_dirty() {
  tex_dirty_ = true;
}

void
Painter::set_all_slices_tex_dirty() {
  for (SliceWindows::iterator i = windows_.begin(); i != windows_.end(); ++i) {
    for (NrrdSlices::iterator s = (*i)->slices_.begin();
         s != (*i)->slices_.end(); ++s) {
      (*s)->set_tex_dirty();
    }
  }
}


void
Painter::copy_current_layer() {
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


void
Painter::kill_current_layer() {
  if (current_volume_) {
    current_volume_->keep_ = 0;
    recompute_volume_list();
  }    
}

void
Painter::new_current_layer() {
  if (current_volume_) {
    string base = "New Layer";
    int i = 0;
    string name = base + " "+to_string(++i);
    while (volume_map_.find(name) != volume_map_.end())
      name = base + " "+to_string(++i);
      current_volume_ = copy_current_volume(name,1);
  }
}



void
Painter::Event::update_state(GuiArgs &args, Painter &painter) {
  ASSERT(0);
}


void
Painter::SliceWindow::autoview(NrrdVolume *volume, double offset) {
  autoview_ = false;
  double wid = get_region().width() -  2*offset;
  double hei = get_region().height() - 2*offset;

  
//   FreeTypeFace *font = fonts_["orientation"];
//   if (font)
//   {
//     FreeTypeText dummy("X", font);
//     BBox bbox;
//     dummy.get_bounds(bbox);
//     wid -= 2*Ceil(bbox.max().x() - bbox.min().x())+4;
//     hei -= 2*Ceil(bbox.max().y() - bbox.min().y())+4;
//   }
  
  int xax = x_axis();
  int yax = y_axis();

  if (volume) {
    vector<int> zero(volume->nrrd_handle_->nrrd_->dim, 0);
    vector<int> index = zero;
    index[xax+1] = volume->nrrd_handle_->nrrd_->axis[xax+1].size;
    double w_wid = (volume->index_to_world(index) - 
                    volume->index_to_world(zero)).length();
    double w_ratio = wid/w_wid;
    
    index = zero;
    index[yax+1] = volume->nrrd_handle_->nrrd_->axis[yax+1].size;
    double w_hei = (volume->index_to_world(index) - 
                    volume->index_to_world(zero)).length();
    double h_ratio = hei/w_hei;
    
    zoom_ = Min(w_ratio*100.0, h_ratio*100.0);
    if (zoom_ < 1.0) zoom_ = 100.0; // ??
    center_(xax) = volume->center()(xax);
    center_(yax) = volume->center()(yax);
  } else {
    center_ = Point(0,0,0);
    zoom_ = 100;
  }
  redraw();
}
   



void
Painter::create_undo_volume() {
  return;
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
  extract_all_window_slices();
  redraw_all();
  //  delete undo_volume_;
  //  undo_volume_ = 0;
}


void
Painter::recompute_volume_list()
{
  volume_lock_.lock();
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
  
  extract_all_window_slices();
  volume_lock_.unlock();

  redraw_all();  
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

  float min = AIR_POS_INF;
  float max = AIR_NEG_INF;
  
  for (unsigned int i = 0; i < size; ++i)
    if (test[i] > 0.0) {
      //      cerr << test[i] << std::endl;
      mean += src[i];
      squared += src[i]*src[i];
      min = Min(min, src[i]);
      max = Max(max, src[i]);

      ++n;
    }

  mean = mean / n;
  double deviation = sqrt(squared/n-mean*mean);
  //  cerr << "size: " << size << " n: " << n << std::endl;
  //  cerr << "mean: " << mean << " dev: " << deviation << std::endl;
  //  return make_pair(min,max);
  return make_pair(mean, deviation);
}
  
         

Painter::NrrdVolume *
Painter::copy_current_volume(const string &name, int mode) {
  if (!current_volume_) return 0;
  NrrdVolume *vol = new NrrdVolume(current_volume_, name, mode);
  volume_map_[name] = vol;
  volumes_.push_back(vol);
  show_volume(name);
  vol->clut_min_ = vol->data_max_/255.0;
  vol->clut_max_ = vol->data_max_;
  extract_all_window_slices();
  redraw_all();
  return vol;
}
  


#ifdef HAVE_INSIGHT
//ITKDatatypeHandle
//itk::Object::Pointer
ITKDatatypeHandle
Painter::nrrd_to_itk_image(NrrdDataHandle &nrrd) {
  Nrrd *n = nrrd->nrrd_;
  const unsigned int Dim = 3;
  typedef float PixType;
  ASSERT(n->dim == Dim+1);

  if (n->type != nrrdTypeFloat) {
    NrrdDataHandle nrrd2 = new NrrdData();
    if (nrrdConvert(nrrd2->nrrd_, n, nrrdTypeFloat)) {
      char *err = biffGetDone(NRRD);
      string errstr = (err ? err : "");
      free(err);
      throw errstr;
    }
    nrrd = nrrd2;
    n = nrrd->nrrd_;
  }

  ASSERT(n->type == nrrdTypeFloat);
  //  typedef typename itk::Image<float, 3> ImageType;
  //  typedef typename itk::ImageRegionIterator<ImageType> IteratorType;
  
  typedef itk::ImportImageFilter< PixType, Dim > ImportFilterType;
  ImportFilterType::Pointer importFilter = ImportFilterType::New();        
  ImportFilterType::SizeType size;

  double origin[Dim];
  double spacing[Dim];
  unsigned int count = 1;
  for(unsigned int i=0; i < n->dim-1; i++) {
    count *= n->axis[i+1].size;
    size[i] = n->axis[i+1].size;

    if (!AIR_EXISTS(n->axis[i+1].min))
      origin[i] = 0;
    else
    origin[i] = n->axis[i+1].min;

    if (!AIR_EXISTS(n->axis[i+1].spacing))
      spacing[i] = 1.0;
    else
      spacing[i] = n->axis[i+1].spacing;
  }
  ImportFilterType::IndexType start;
  start.Fill(0);
  ImportFilterType::RegionType region;
  region.SetIndex(start);
  region.SetSize(size);
  importFilter->SetRegion(region);
  importFilter->SetOrigin(origin);
  importFilter->SetSpacing(spacing);
  importFilter->SetImportPointer((PixType *)n->data, count, false);
  importFilter->Update();

  Insight::ITKDatatype* result = new Insight::ITKDatatype();  
  result->data_ = importFilter->GetOutput();
  return result;
  //  return importFilter->GetOutput();

  // Insight::ITKDatatype* result = new Insight::ITKDatatype();  
  //  result->data_ = importFilter->GetOutput();
  //  return result;
}




NrrdDataHandle
Painter::itk_image_to_nrrd(ITKDatatypeHandle &img_handle) {
  const unsigned int Dim = 3;
  typedef float PixType;
  typedef itk::Image<PixType, Dim> ImageType;

  ImageType *img = dynamic_cast<ImageType *>(img_handle->data_.GetPointer());
  if (img == 0) {
    return 0;
  }

  //  Insight::ITKDatatype* result = new Insight::ITKDatatype();  
  //  result->data_ = img;//importFilter->GetOutput();
  
  //  LockingHandle<SCIRun::Datatype> *blah = dynamic_cast<LockingHandle<SCIRun::Datatype> *> (&img_handle);

  NrrdData *nrrd_data = new NrrdData(img_handle.get_rep());
  //  nrrd_data->nrrd_ = nrrd;

  Nrrd *nrrd = nrrd_data->nrrd_;///nrrdNew();
  size_t size[NRRD_DIM_MAX];
  size[0] = 1;
  size[1] = img->GetRequestedRegion().GetSize()[0];
  size[2] = img->GetRequestedRegion().GetSize()[1];
  size[3] = img->GetRequestedRegion().GetSize()[2];

  unsigned int centers[NRRD_DIM_MAX];
  centers[0] = nrrdCenterNode; 
  centers[1] = nrrdCenterNode;
  centers[2] = nrrdCenterNode; 
  centers[3] = nrrdCenterNode;

  //  nrrdAlloc_nva(nrrd, nrrdTypeFloat, 4, size);
  nrrdWrap_nva(nrrd, (void *)img->GetBufferPointer(), nrrdTypeFloat, 4, size);
  nrrdAxisInfoSet_nva(nrrd, nrrdAxisInfoCenter, centers);

  nrrd->axis[0].spacing = AIR_NAN;
  nrrd->axis[0].min = 0;
  nrrd->axis[0].max = 1;
  nrrd->axis[0].kind = nrrdKindStub;

  for(unsigned int i = 0; i < Dim; i++) {
    nrrd->axis[i+1].spacing = img->GetSpacing()[i];

    nrrd->axis[i+1].min = img->GetOrigin()[i];
    nrrd->axis[i+1].max = ceil(img->GetOrigin()[i] + 
      ((nrrd->axis[i+1].size-1) * nrrd->axis[i+1].spacing));
    nrrd->axis[i+1].kind = nrrdKindDomain;
  }


  //  nrrd->data = ;
  //  nrrd->ptr = img;
  //  img->Register();
  //  return result;
  
  return nrrd_data;
}



void
Painter::filter_callback(itk::Object *object,
                         const itk::EventObject &event)
{
  itk::ProcessObject::Pointer process = 
    dynamic_cast<itk::ProcessObject *>(object);
  ASSERT(process);
  double value = process->GetProgress();
  if (typeid(itk::ProgressEvent) == typeid(event))
  {
    double total = get_vars()->get_double("Painter::progress_bar_total_width");
    get_vars()->insert("Painter::progress_bar_done_width", 
                       to_string(value * total), "string", true);
    
    get_vars()->insert("Painter::progress_bar_text", 
                       to_string(round(value * 100))+ " %  ", "string", true);

    if (filter_volume_ && filter_update_img_.get_rep()) {
      //      typedef Painter::ITKImageFloat3D ImageType;
      //      typedef itk::ImageToImageFilter<ImageType, ImageType> FilterType;
      typedef itk::ThresholdSegmentationLevelSetImageFilter
        < Painter::ITKImageFloat3D, Painter::ITKImageFloat3D > FilterType;
      
      
      FilterType *filter = dynamic_cast<FilterType *>(object);
      ASSERT(filter);
      volume_lock_.lock();
      filter_update_img_->data_ = filter->GetOutput();
      //filter_update_img_->data_ = filter->GetFeatureImage();
      filter_volume_->nrrd_handle_ = itk_image_to_nrrd(filter_update_img_);
      volume_lock_.unlock();
      set_all_slices_tex_dirty();
      redraw_all();
    }

    redraw_all();

  }



  if (typeid(itk::IterationEvent) == typeid(event))
  {
    //    std::cerr << "Filter Iteration: " << value * 100.0 << "%\n";
  }
  if (abort_filter_) {
    //    abort_filter_ = false;
    process->AbortGenerateDataOn();
  }


}

void
Painter::filter_callback_const(const itk::Object *object,
                               const itk::EventObject &event)
{
  itk::ProcessObject::ConstPointer process = 
    dynamic_cast<const itk::ProcessObject *>(object);
  ASSERT(process);
  double value = process->GetProgress();
  if (typeid(itk::ProgressEvent) == typeid(event))
  {
    std::cerr << "Const Filter Progress: " << value * 100.0 << "%\n";
  }

  if (typeid(itk::IterationEvent) == typeid(event))
  {
    std::cerr << "Const Filter Iteration: " << value * 100.0 << "%\n";
  }
}
#endif // HAVE_INSIGHT

BaseTool::propagation_state_e
Painter::SliceWindow::process_event(event_handle_t event) {
  PointerEvent *pointer = dynamic_cast<PointerEvent *>(event.get_rep());
  const Skinner::RectRegion &region = get_region();
  if (pointer) {
//     Point pointer_point(pointer->get_x(), pointer->get_y(),0);
//     Point min(region_.x1(), region_.y1(),0);
//     Point max(region_.x2(), region_.y2(),0);
    if (pointer->get_x() >= region.x1()  &&
        pointer->get_x() < region.x2()  &&
        pointer->get_y() >= region.y1()  &&
        pointer->get_y() < region.y2()) {
      //      if (region.inside(pointer->get_x(), pointer->get_y())) {
      painter_->cur_window_ = this;
      painter_->pointer_pos_ = screen_to_world(pointer->get_x(), 
                                               pointer->get_y());
    }
  }

  if (painter_->cur_window_ == this) {
    painter_->tm_.propagate_event(event);
  }

  WindowEvent *window = dynamic_cast<WindowEvent *>(event.get_rep());
  if (window && window->get_window_state() == WindowEvent::REDRAW_E) {
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);    
    glViewport(Floor(region.x1()), Floor(region.y1()), 
	       Ceil(region.width()), Ceil(region.height()));
    render_gl();
    glViewport(viewport[0],viewport[1],viewport[2],viewport[3]);
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    
  }
  
  return BaseTool::CONTINUE_E;
}

int
Painter::get_signal_id(const string &signalname) {
  if (signalname == "SliceWindow_Maker") return 1;
  if (signalname == "Painter::start_brush_tool") return 2;
  return 0;
}



Skinner::Drawable *
Painter::maker(Skinner::Variables *vars) 
{
  return new Painter(vars, 0);
}



#if 0

#include <Core/Volume/VolumeRenderer.h>
#include <Core/Events/SceneGraphEvent.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Volume/ColorMap2.h>
#include <Core/Algorithms/Visualization/NrrdTextureBuilderAlgo.h>

void
setup_volume_rendering() {
  event_handle_t scene_event = 0;
  
  CompileInfoHandle ci =
    NrrdTextureBuilderAlgo::get_compile_info(nrrd->type,nrrd->type);
  
  
  const int card_mem = 128;
  cerr << "nrrd texture\n";
  TextureHandle texture = new Texture;
  NrrdTextureBuilderAlgo::build_static(texture, 
				       nrrd_handle, 0, 255,
				       0, 0, 255, card_mem);
  vector<Plane *> *planes = new vector<Plane *>;
  
  
  string fn = string(argv[3]);
  Piostream *stream = auto_istream(fn, 0);
  if (!stream) {
    cerr << "Error reading file '" + fn + "'." << std::endl;
    return -1;
  }  
  // read the file.
  ColorMap2 *cmap2 = new ColorMap2();
  ColorMap2Handle icmap = cmap2;
  try {
    Pio(*stream, icmap);
  } catch (...) {
    cerr << "Error loading "+fn << std::endl;
    icmap = 0;
  }
  delete stream;
  ColorMapHandle cmap;
  vector<ColorMap2Handle> *cmap2v = new vector<ColorMap2Handle>(0);
  cmap2v->push_back(icmap);
  string enabled("111111");
  if (sci_getenv("CMAP_WIDGETS")) 
    enabled = sci_getenv("CMAP_WIDGETS");
  for (unsigned int i = 0; i < icmap->widgets().size(); ++ i) {
    if (i < enabled.size() && enabled[i] == '1') {
      icmap->widgets()[i]->set_onState(1); 
    } else {
      icmap->widgets()[i]->set_onState(0); 
    }
  }

  VolumeRenderer *vol = new VolumeRenderer(texture, 
					   cmap, 
					   *cmap2v, 
					   *planes,
					   Round(card_mem*1024*1024*0.8));
  vol->set_slice_alpha(-0.5);
  vol->set_interactive_rate(4.0);
  vol->set_sampling_rate(4.0);
  vol->set_material(0.322, 0.868, 1.0, 18);
  scene_event = new SceneGraphEvent(vol, "FOO");  
  //  if (!sci_getenv_p("PAINTER_NOSCENE")) 
  //    EventManager::add_event(scene_event);    

}  



#endif






} // end namespace SCIRun
