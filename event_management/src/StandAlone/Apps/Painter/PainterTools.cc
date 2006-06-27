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
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Ports/GeometryPort.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <Core/Bundle/Bundle.h>
#include <Core/Containers/Array3.h>
#include <Core/Datatypes/Field.h> 
#include <Core/Exceptions/GuiException.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/FreeTypeTextTexture.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geom/GeomCull.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/TexSquare.h>
#include <Core/Geom/TkOpenGLContext.h>
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
#include <Core/Events/keysyms.h>

#ifdef HAVE_INSIGHT
#  include <itkGradientMagnitudeImageFilter.h>
#  include <itkConfidenceConnectedImageFilter.h>
#  include <itkCurvatureAnisotropicDiffusionImageFilter.h>
#  include <itkBinaryBallStructuringElement.h>
#  include <itkBinaryDilateImageFilter.h>
#  include <itkBinaryErodeImageFilter.h>
#  include <itkImportImageFilter.h>
#endif


#ifdef _WIN32
#  define SCISHARE __declspec(dllimport)
#else
#  define SCISHARE
#endif


namespace SCIRun {

BaseTool::propagation_state_e
Painter::PainterTool::process_event(event_handle_t)
{
  return CONTINUE_E;
}



Painter::KeyToolSelectorTool::KeyToolSelectorTool(Painter *painter) :
  KeyTool("Painter KeyToolSelectorTool"),
  painter_(painter),
  tm_(painter->tm_)
{
}
  

Painter::KeyToolSelectorTool::~KeyToolSelectorTool()
{
}

BaseTool::propagation_state_e
Painter::KeyToolSelectorTool::key_press(string, int keyval,
                                        unsigned int, unsigned int)
{
  if (!painter_->cur_window_) return STOP_E;
  SliceWindow &window = *painter_->cur_window_;
  if (sci_getenv_p("SCI_DEBUG"))
    cerr << "keyval: " << keyval << std::endl;

  switch (keyval) {
  case SCIRun_equal:    window.zoom_in(); break;
  case SCIRun_minus:    window.zoom_out(); break;
  case SCIRun_comma:    window.prev_slice(); break;
  case SCIRun_period:   window.next_slice(); break;
  case SCIRun_u:        painter_->undo_volume();
  case SCIRun_a:        tm_.add_tool(new CropTool(painter_),100); break;
  case SCIRun_f:        tm_.add_tool(new FloodfillTool(painter_),100); break;
  case SCIRun_b:        tm_.add_tool(new BrushTool(painter_),25); break;
  case SCIRun_m:        tm_.add_tool(new LayerMergeTool(painter_),100); break;
  case SCIRun_l:        tm_.add_tool(new StatisticsTool(painter_),100); break;

  case SCIRun_c:        painter_->copy_current_layer(); break;
  case SCIRun_x:        painter_->kill_current_layer(); break;
  case SCIRun_v:        painter_->new_current_layer(); break;

  case SCIRun_r:        painter_->reset_clut();
  case SCIRun_Left:     painter_->move_layer_down();break;
  case SCIRun_Right:    painter_->move_layer_up();break;
  case SCIRun_Up:       painter_->cur_layer_up();break;
  case SCIRun_Down:     painter_->cur_layer_down();break;

  case SCIRun_p:        painter_->opacity_up();break;
  case SCIRun_o:        painter_->opacity_down();break;
    

  case SCIRun_k:        
    tm_.add_tool(new ITKConfidenceConnectedImageFilterTool(painter_),49); 
    break;

  case SCIRun_g:
    tm_.add_tool(new ITKGradientMagnitudeTool(painter_),100); 
    break;

  case SCIRun_d:
    tm_.add_tool(new ITKCurvatureAnisotropicDiffusionTool(painter_),100); 
    break;
    
  case SCIRun_j: 
    tm_.add_tool(new ITKBinaryDilateErodeTool(painter_),100); 
    break;

  case SCIRun_1: 
    tm_.add_tool(new ITKBinaryDilateTool(painter_),100); 
    break;

  }

  painter_->redraw_all();
  return CONTINUE_E;
}  


#if 0

  if (key == "[") {
    if (current_volume_) {
      current_volume_->colormap_.set(Max(0, current_volume_->colormap_.get()-1));
      set_all_slices_tex_dirty();
      redraw_all();
    }
  }

  if (key == "]") {
    if (current_volume_) {
      current_volume_->colormap_.set(Min(int(colormap_names_.size()), 
                                         current_volume_->colormap_.get()+1));
      set_all_slices_tex_dirty();
      redraw_all();
    }
  }

  if (key == "q") { 
    for (unsigned int t = 0; t < tools_.size(); ++t)
      delete tools_[t];
    tools_.clear();
  }

  if (key == "h") {
    if (tools_.empty()) {
      tools_.push_back(new BrushTool(this));
      tools_.push_back(new ITKThresholdTool(this));
  }

#endif




BaseTool::propagation_state_e
Painter::KeyToolSelectorTool::key_release(string, int, 
                                          unsigned int, unsigned int)
{
  return CONTINUE_E;
}


Painter::PointerToolSelectorTool::PointerToolSelectorTool(Painter *painter) :
  PainterPointerTool(painter, "Painter PointerToolSelectorTool"),
  tm_(painter->tm_)
{
}
  
Painter::PointerToolSelectorTool::~PointerToolSelectorTool()
{
}


BaseTool::propagation_state_e
Painter::PointerToolSelectorTool::pointer_down(int button, int x, int y,
                                               unsigned int modifiers,
                                               int time)
{
  if (!painter_->cur_window_) return STOP_E;
  SliceWindow &window = *painter_->cur_window_;
  switch (button) {
  case 1:
    if (modifiers & EventModifiers::SHIFT_E)
      tm_.add_tool(new PanTool(painter_), 100);
    else
      tm_.add_tool(new CLUTLevelsTool(painter_), 100);
    break;
    
  case 2:
    if (modifiers & EventModifiers::SHIFT_E)
      tm_.add_tool(new AutoviewTool(painter_), 100);
    else
      tm_.add_tool(new ProbeTool(painter_), 100);
    break;

  case 3:
    if (modifiers & EventModifiers::SHIFT_E)
      painter_->tm_.add_tool(new ZoomTool(painter_), 52);
    break;
  case 4:
    if (modifiers & EventModifiers::SHIFT_E)
      window.zoom_in();
    else
      window.next_slice();
    break;
    
  case 5:
    if (modifiers & EventModifiers::SHIFT_E)
      window.zoom_out();
    else
      window.prev_slice();
    break;
  default: 
    break;
  }

  return CONTINUE_E;
}


BaseTool::propagation_state_e
Painter::PointerToolSelectorTool::pointer_up(int button, int x, int y,
                                             unsigned int modifiers,
                                             int time)
{
  return CONTINUE_E;
}

BaseTool::propagation_state_e
Painter::PointerToolSelectorTool::pointer_motion(int button, int x, int y,
                                                 unsigned int modifiers,
                                                 int time)
{
  return CONTINUE_E;
}








Painter::CLUTLevelsTool::CLUTLevelsTool(Painter *painter) : 
  PainterPointerTool(painter, "Color Lookup Table"),
  scale_(1.0), 
  ww_(0), 
  wl_(1.0),
  x_(0),
  y_(0)
{
}


BaseTool::propagation_state_e
Painter::CLUTLevelsTool::pointer_down(int b, int x, int y,
                                      unsigned int m, int t)
{
  NrrdVolume *vol = painter_->current_volume_;
  if (!vol || !painter_->cur_window_) {
    return CONTINUE_E;
  }

  ww_ = vol->clut_max_() - vol->clut_min_();
  wl_ = vol->clut_min_ + ww_ / 2.0;
  x_ = x;
  y_ = y;
  
  const double w = painter_->cur_window_->region().width();
  const double h = painter_->cur_window_->region().width();
  scale_ = (vol->data_max_ - vol->data_min_) / sqrt(w*w+h*h);

  return pointer_motion(b,x,y,m,t);
}


BaseTool::propagation_state_e
Painter::CLUTLevelsTool::pointer_motion(int b, int x, int y, 
                                        unsigned int m, int t)
{
  NrrdVolume *vol = painter_->current_volume_;
  if (!vol) {
    return CONTINUE_E;
  }

  const float ww = ww_+scale_*(y - y_);  
  const float wl = wl_+scale_*(x - x_);
  vol->clut_min_ = wl - ww/2.0;
  vol->clut_max_ = wl + ww/2.0;
  painter_->set_all_slices_tex_dirty();
  painter_->redraw_all();
  return STOP_E;
}



Painter::ZoomTool::ZoomTool(Painter *painter) : 
  PainterPointerTool(painter, "Zoom"),
  window_(0),
  zoom_(0.0),
  x_(0),
  y_(0)
{
}


BaseTool::propagation_state_e
Painter::ZoomTool::pointer_down(int button, int x, int y,
                                unsigned int modifiers,
                                int time)
{
  if (!painter_->cur_window_) {
    return QUIT_AND_CONTINUE_E;
  }
  window_ = painter_->cur_window_;
  zoom_ = window_->zoom_;
  x_ = x;
  y_ = y;
  return STOP_E;
}    

BaseTool::propagation_state_e
Painter::ZoomTool::pointer_motion(int button, int x, int y,
                                unsigned int modifiers,
                                int time)
{
  ASSERT(window_);
  int delta = x + y - x_ - y_;
  window_->zoom_ = Max(0.00001, zoom_ * Pow(1.002,delta));
  window_->redraw();
  return STOP_E;
}    
  


Painter::AutoviewTool::AutoviewTool(Painter *painter) : 
  PainterPointerTool(painter, "Autoview")
{
}


BaseTool::propagation_state_e
Painter::AutoviewTool::pointer_down(int, int, int, unsigned int, int)
{
  if (painter_->cur_window_) {
    painter_->cur_window_->autoview(painter_->current_volume_);
  }
  return QUIT_AND_STOP_E;
}


Painter::ProbeTool::ProbeTool(Painter *painter) : 
  PainterPointerTool(painter, "Probe")
{
}


BaseTool::propagation_state_e
Painter::ProbeTool::pointer_down(int, int, int, unsigned int, int)
{
  painter_->set_probe();
  return STOP_E;
}

BaseTool::propagation_state_e
Painter::ProbeTool::pointer_motion(int, int, int, unsigned int, int)
{
  painter_->set_probe();
  return STOP_E;
}




Painter::PanTool::PanTool(Painter *painter) : 
  PainterPointerTool(painter, "Pan"),
  x_(0),
  y_(0),
  center_(0,0,0),
  window_(0)
{
}


BaseTool::propagation_state_e
Painter::PanTool::pointer_down(int b, int x, int y, unsigned int m, int t)
{
  if (!painter_->cur_window_) {
    return QUIT_AND_CONTINUE_E;
  }
  window_ = painter_->cur_window_;
  x_ = x;
  y_ = y;
  center_ = window_->center_;
  return STOP_E;
}


BaseTool::propagation_state_e
Painter::PanTool::pointer_motion(int b, int x, int y, unsigned int m, int t)
{
  if (!window_) {
    return CONTINUE_E;
  }

  const float scale = 100.0/window_->zoom_;
  int xax = window_->x_axis();
  int yax = window_->y_axis();
  window_->center_(xax) = center_(xax) - scale * (x - x_);
  window_->center_(yax) = center_(yax) - scale * (y - y_);

  painter_->redraw_all();
  return STOP_E;
}



Painter::CropTool::CropTool(Painter *painter) : 
  PainterTool(painter, "Crop"),
  pick_(0)
{
  ASSERT(painter_->current_volume_);
  minmax_[1] = painter_->current_volume_->max_index();
  minmax_[0] = vector<int>(minmax_[1].size(), 0);
  pick_minmax_[0] = minmax_[0];
  pick_minmax_[1] = minmax_[1];
}

Painter::CropTool::~CropTool() {}


int
Painter::CropTool::do_event(Event &event) {

  if (event.type_ == Event::KEY_PRESS_E &&
      event.key_ == " ") {
    //int *minmax[2] = { new int[minmax_[0].size()], new int[minmax_[1].size()] };
    size_t *minmax[2] = { new size_t[minmax_[0].size()], new size_t[minmax_[1].size()] };
    for (int i = 0; i < 2; ++i)
      for (unsigned int a = 0; a < minmax_[0].size(); ++a)
	minmax[i][a] = minmax_[i][a]-(i==1?1:0);
    NrrdDataHandle nout_handle = new NrrdData();
    if (nrrdCrop(nout_handle->nrrd_,
		 painter_->current_volume_->nrrd_handle_->nrrd_,
		 minmax[0], minmax[1])) {
      char *err = biffGetDone(NRRD);
      string str = string("nrrdcrop: ") + err;
      free(err);
      throw str;
    }

    painter_->current_volume_->nrrd_handle_ = nout_handle;
    painter_->current_volume_->build_index_to_world_matrix();
    delete[] minmax[0];
    delete[] minmax[1];
    painter_->extract_all_window_slices();
    painter_->redraw_all();
    return QUIT_E;
  }

  if (event.type_ == Event::KEY_PRESS_E &&
      event.key_ == "q") {
    return QUIT_E;
  }

  if (!event.keys_.empty())
    return FALLTHROUGH_E;

  if (event.type_ == Event::BUTTON_PRESS_E && event.window_) {
    double units = 100.0 / event.window_->zoom_; // world space units per pixel
    pick_ = 1;
    for (int i = 0; i < 2; ++i) {
      pick_minmax_[i] = minmax_[i];
      Point p = painter_->current_volume_->index_to_world(minmax_[i]);
      for (int a = 0; a < 3; ++a) {
        Vector n(a==0 ? 1:0, a==1?1:0, a==2?1:0);
        if (i) n = -1*n;
        Plane plane(p, n);
        pick_dist_[i][a] = plane.eval_point(event.position_)/units;
        if (Abs(pick_dist_[i][a]) < 5.0) pick_ |= 2;
        if (pick_dist_[i][a] < 0.0 && 
	    a != event.window_->axis_) pick_ = pick_ & ~1;
      }
    }
    pick_index_ = painter_->current_volume_->point_to_index(event.position_);

    return HANDLED_E;
  }


  if (pick_ && event.type_ == Event::BUTTON_RELEASE_E && event.window_) {
    for (unsigned int a = 0; a < minmax_[0].size(); ++a)
      if (minmax_[0][a] > minmax_[1][a])
	SWAP(minmax_[0][a],minmax_[1][a]);

    pick_ = 0;
    return HANDLED_E;
  }

  if (pick_ && event.type_ == Event::MOTION_E && event.window_) {
    pick_mouse_motion(event);
    return HANDLED_E;
  }

  return FALLTHROUGH_E;
}


void
Painter::CropTool::pick_mouse_motion(Event &event) {
  ASSERT(pick_ && event.type_ == Event::MOTION_E && event.window_);
  unsigned int axis = event.window_->axis_;
  vector<int> max_index = painter_->current_volume_->max_index();
  vector<double> idx = 
    painter_->current_volume_->point_to_index(event.position_);

  if (pick_ == 1) {  // Clicked inside crop box
    for (unsigned int a = 0; a < idx.size(); ++ a) {    
      for (unsigned int i = 0; i < 2; ++i) {
	double delta = Clamp(idx[a]-Floor(pick_index_[a]),
			     -double(pick_minmax_[0][a]), 
			     double(max_index[a]-pick_minmax_[1][a]));
        minmax_[i][a] = Floor(pick_minmax_[i][a]+delta);
      }
    }
  }
  else { // Clicked on crop box boundary
    for (int i = 0; i < 2; ++i) {
      for (unsigned int a = 0; a < 3; ++a) {
	if (a == axis) continue;
	int newval = Clamp(Round(idx[a+1]), 0, max_index[a+1]);
	if (Abs(pick_dist_[i][a]) < 5.0 && newval != minmax_[(i+1)%2][a+1]) {
	  minmax_[i][a+1] = newval;
	}
      }
    }
  }
  
  painter_->redraw_all();
}



int
Painter::CropTool::draw(SliceWindow &window) {
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  Point ll = painter_->current_volume_->index_to_world(minmax_[0]);
  Point ur = painter_->current_volume_->index_to_world(minmax_[1]);
  Vector dia = ur - ll;

  Vector right = window.x_dir();
  right.normalize();
  right = right*dia;

  Vector up = window.y_dir();
  up.normalize();
  up = up*dia;

  Point lr = ll+right;
  Point ul = ll+up;
  
  //  GLdouble blue[4] = { 0.1, 0.4, 1.0, 0.8 };
  //  GLdouble green[4] = { 0.5, 1.0, 0.1, 0.7 };
  //  GLdouble lt_green[4] = { 0.5, 1.0, 0.1, 0.4 };
  //  GLdouble red[4] = { 0.8, 0.2, 0.4, 0.9 };
  GLdouble grey[4] = { 0.6, 0.6, 0.6, 0.6 }; 
  GLdouble white[4] = { 1.0, 1.0, 1.0, 1.0 }; 
  GLdouble black[4] = { 0.0, 0.0, 0.0, 1.0 }; 
  GLdouble yellow[4] = { 1.0, 0.76, 0.1, 1.0 };
  GLdouble lt_yellow[4] = { 0.8, 0.5, 0.1, 1.0 };  

  GLdouble *colors[5] = { lt_yellow, yellow, black, grey, white };
  GLdouble widths[5] = { 11, 9.0, 7.0, 5.0, 1.0 }; 

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

  widths[0] = 10.0;
  widths[1] = 6.0;
  widths[2] = 2.0;
  
  glEnable(GL_POINT_SMOOTH);
  for (int pass = 2; pass < 5; ++pass) {
    glColor4dv(colors[pass]);
    glPointSize(widths[pass]);
    glBegin(GL_POINTS);
    {
      glVertex3dv(&ll(0));
      glVertex3dv(&lr(0));
      glVertex3dv(&ur(0));
      glVertex3dv(&ul(0));
    }
    glEnd();
  }

  glDisable(GL_POINT_SMOOTH);
  glDisable(GL_BLEND);
  CHECK_OPENGL_ERROR();
  return 0; 
}


void
Painter::CropTool::set_window_cursor(SliceWindow &window, int cursor) 
{
  if (painter_->event_.window_ != &window || 
      window.cursor_pixmap_ == cursor) return;
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
}


Painter::FloodfillTool::FloodfillTool(Painter *painter) :
  PainterTool(painter, "Flood Fill"),
  value_(0.0),
  min_(0.0),
  max_(0.0),
  start_pos_(0,0,0)
{
  painter_->create_undo_volume();
}


Painter::FloodfillTool::~FloodfillTool()
{
}


int
Painter::FloodfillTool::do_event(Event &event) {
  if (!event.window_ || !painter_->current_volume_) 
    return FALLTHROUGH_E;

  if (event.type_ == Event::KEY_PRESS_E &&
      event.key_ == "q")
    return QUIT_E;


  if (event.type_ == Event::KEY_PRESS_E &&
      event.key_ == " ") {
    do_floodfill();
    return QUIT_E;
  }


  NrrdVolume *volume = painter_->current_volume_;
    
  if (event.type_ == Event::BUTTON_PRESS_E) {
    vector<int> index = volume->world_to_index(event.position_);
    if (!volume->index_valid(index))
      return FALLTHROUGH_E;
    if (event.button(1)) {
      min_ = volume->data_max_;
      max_ = volume->data_min_;
      start_pos_ = event.position_;
    }
    return HANDLED_E;
  }
  
  if (event.type_ == Event::MOTION_E) {

    vector<int> index = 
      volume->world_to_index(event.position_);
    if (!volume->index_valid(index)) 
      return FALLTHROUGH_E;
    
    double val;
    volume->get_value(index, val);
    
    if (event.button(1)) {
      min_ = Min(min_, val);
      max_ = Max(max_, val);
      cerr << "Min: " << min_ << "  Max: " << max_ << std::endl;
    }
    
    if (event.button(3)) {
      value_ = val;
      cerr << "value: " << value_ << std::endl;
    }
    painter_->redraw_all();
    return HANDLED_E;
  }
  
  return FALLTHROUGH_E;
}



int
Painter::FloodfillTool::draw(SliceWindow &)
{
  return 0;
}


void
Painter::FloodfillTool::do_floodfill()
{
  NrrdVolume *volume = painter_->current_volume_;
  vector<int> index = volume->world_to_index(start_pos_);
  if (!volume->index_valid(index)) 
    return;

  // Array to hold which indices to visit next
  vector<vector<int> > todo, oldtodo;

  // Push back the first seed point
  todo.push_back(index);

  // Allocated a nrrd to mark where the flood fill has visited
  NrrdDataHandle done_handle = new NrrdData();
  size_t size[NRRD_DIM_MAX];
  size[0] = volume->nrrd_handle_->nrrd_->axis[0].size;
  size[1] = volume->nrrd_handle_->nrrd_->axis[1].size;
  size[2] = volume->nrrd_handle_->nrrd_->axis[2].size;
  size[3] = volume->nrrd_handle_->nrrd_->axis[3].size;
  nrrdAlloc_nva(done_handle->nrrd_, nrrdTypeUChar, 4, size);


  // Set the visited nrrd to empty
  memset(done_handle->nrrd_->data, 0, 
         volume->nrrd_handle_->nrrd_->axis[0].size *
         volume->nrrd_handle_->nrrd_->axis[1].size *
         volume->nrrd_handle_->nrrd_->axis[2].size * 
         volume->nrrd_handle_->nrrd_->axis[3].size);
  int count  = 0;
  unsigned int axes = index.size();
  while (!todo.empty()) {
    ++count;
    if (!(count % 40)) {
      cerr << todo.size() << std::endl;
      painter_->set_all_slices_tex_dirty();
      painter_->redraw_all();
      TCLTask::unlock();
      Thread::yield();
      TCLTask::lock();
    }
      
    oldtodo = todo;
    todo.clear();
    for (unsigned int i = 0; i < oldtodo.size(); ++i)
      volume->set_value(oldtodo[i], value_);
    

    // For each axis
    for (unsigned int i = 0; i < oldtodo.size(); ++i) {
      for (unsigned int a = 1; a < axes; ++a) {
        // Visit the previous and next neighbor indices along this axis
        for (int dir = -1; dir < 2; dir +=2) {
          // Neighbor index starts as current index
          vector<int> neighbor_index = oldtodo[i];
          // Index adjusted in direction we're looking at along axis
          neighbor_index[a] = neighbor_index[a] + dir;
          // Bail if this index is outside the volume
          if (!volume->index_valid(neighbor_index)) continue;
          
          // Check to see if flood fill has already been here
          unsigned char visited;
          nrrd_get_value(done_handle->nrrd_, neighbor_index, visited);
          // Bail if the voxel has been visited
          if (visited) continue;
          
          // Now check to see if this pixel is a candidate to be filled
          double neighborval;
          volume->get_value(neighbor_index, neighborval);
          // Bail if the voxel is outside the flood fill range
          if (neighborval < min_ || neighborval > max_) continue;
          // Mark this voxel as visited
          nrrd_set_value(done_handle->nrrd_, neighbor_index, (unsigned char)1);
          
          todo.push_back(neighbor_index);
        }
      }
    }
  }

  painter_->set_all_slices_tex_dirty();
  painter_->redraw_all();
}


Painter::ITKThresholdTool::ITKThresholdTool(Painter *painter) :
  PainterTool(painter, "ITK Threshold"),
  seed_volume_(0),
  source_volume_(0)
{
}


int
Painter::ITKThresholdTool::do_event(Event &event)
{
  if (event.type_ != Event::KEY_PRESS_E)
    return FALLTHROUGH_E;

  NrrdVolume *cur = painter_->current_volume_;
  NrrdVolume *temp = 0;
  if (event.key_ == "1" && cur && cur != seed_volume_) {
    if (source_volume_) source_volume_->name_prefix_ = "";
    source_volume_ = cur;
    source_volume_->name_prefix_ = "Source - ";
    painter_->redraw_all();
    return HANDLED_E;
  }
  
  if (event.key_ == "2" && cur && cur != source_volume_) {
    if (seed_volume_) seed_volume_->name_prefix_ = "";
    seed_volume_ = cur;
    seed_volume_->name_prefix_ = "Seed - ";
    painter_->redraw_all();
    return HANDLED_E;
  }

  
  if (event.key_ == "q" || event.key_ == "Q") {
    return QUIT_E;
  }
  
  if (event.key_ == " " && seed_volume_ && source_volume_) {
    string name = "ITK Threshold Result";
    temp = painter_->copy_current_volume(name, 1);
    temp->colormap_.set(1);
    temp->data_min_ = -4.0;
    temp->data_max_ = 4.0;
    temp->clut_min_ = 0.0;
    temp->clut_max_ = 0.5;

    name = "ITK Threshold Source";
    temp = new NrrdVolume(source_volume_, name, 2);
    source_volume_->name_prefix_ = "";
    temp->keep_ = 0;
    painter_->volumes_.push_back(temp);
    painter_->volume_map_[name] = temp;

    name = "ITK Threshold Seed";
    pair<double, double> mean = 
      painter_->compute_mean_and_deviation(source_volume_->nrrd_handle_->nrrd_,
                                           seed_volume_->nrrd_handle_->nrrd_);
    double factor = 2.5;
    double min = mean.first - factor*mean.second;
    double max = mean.first + factor*mean.second;
    //    min = mean.first;
    //    max = mean.second;
    nrrdKeyValueAdd(seed_volume_->nrrd_handle_->nrrd_,
                    "lower_threshold", to_string(min).c_str());
    nrrdKeyValueAdd(seed_volume_->nrrd_handle_->nrrd_,
                    "upper_threshold", to_string(max).c_str());
    
    
    temp = new NrrdVolume(seed_volume_, name, 2);
    seed_volume_->name_prefix_ = "";
    temp->keep_ = 0;
    painter_->volume_map_[name] = temp;

    painter_->filter_text_ = get_name() + "\nFilter Running, Please Wait";
    painter_->redraw_all();
    //    painter_->want_to_execute();

    return QUIT_ALL_E;
  }
  
  return FALLTHROUGH_E;
}


Painter::StatisticsTool::StatisticsTool(Painter *painter) :
  PainterTool(painter, "Statistics"),
  standard_deviation_(0.0),
  mean_(0.0),
  sum_(0.0),
  squared_sum_(0.0),
  count_(0)
{
}



int
Painter::StatisticsTool::do_event(Event &event)
{

  NrrdVolume *vol = painter_->current_volume_;
  if (!vol || !event.window_)
    return FALLTHROUGH_E;

  if (event.type_ == Event::BUTTON_PRESS_E && 
      event.button(2)) {

    sum_ = 0;
    squared_sum_ = 0;
    count_ = 0;
    mean_ = 0;
    standard_deviation_ = 0;
    

    vol->clut_min_ = vol->data_min_;
    vol->clut_max_ = vol->data_max_;
    painter_->set_all_slices_tex_dirty();
    painter_->redraw_all();
    return HANDLED_E;
  }

  if ((event.type_ == Event::BUTTON_PRESS_E ||
       event.type_ == Event::MOTION_E)  && 
      event.button(1)) {

    vector<int> index = vol->world_to_index(event.position_);
    if (!vol->index_valid(index)) 
      return FALLTHROUGH_E;

    double value;
    vol->get_value(index, value);

    sum_ += value;
    squared_sum_ += value*value;
    ++count_;

    mean_ = sum_ / count_;
    standard_deviation_ = sqrt(squared_sum_/count_-mean_*mean_);    

    vol->clut_min_ = mean_ - standard_deviation_;
    vol->clut_max_ = mean_ + standard_deviation_;

    painter_->set_all_slices_tex_dirty();
    painter_->redraw_all();
    return HANDLED_E;
  }
  return FALLTHROUGH_E;
}
  

int
Painter::StatisticsTool::draw(SliceWindow &window)
{
#if 0
  if (painter_->font1_ && &window == painter_->event_.window_)
    painter_->font1_->render("Mean: "+to_string(mean_)+
                             "\nStandard Deviation: "+
                             to_string(standard_deviation_), 
                             painter_->event_.x_, painter_->event_.y_,
                             TextRenderer::NW | TextRenderer::SHADOW);
#endif
  return 0;
}






Painter::ITKConfidenceConnectedImageFilterTool::ITKConfidenceConnectedImageFilterTool(Painter *painter) :
  PainterPointerTool(painter, "ITK Confidence Connected\nImage Filter"),
  volume_(0)
{
}


BaseTool::propagation_state_e
Painter::ITKConfidenceConnectedImageFilterTool::pointer_down
(int b, int x, int y, unsigned int m, int t)
{
  return STOP_E;
}

BaseTool::propagation_state_e
Painter::ITKConfidenceConnectedImageFilterTool::pointer_up
(int b, int x, int y, unsigned int m, int t)
{
  if (!volume_) 
    volume_ = painter_->current_volume_;
  if (!volume_->index_valid(seed_))
    seed_ = volume_->world_to_index(painter_->pointer_pos_);
  if (!volume_->index_valid(seed_))
    return QUIT_AND_STOP_E;
    
#ifdef HAVE_INSIGHT    
  typedef itk::ConfidenceConnectedImageFilter
    < Painter::ITKImageFloat3D, Painter::ITKImageFloat3D > FilterType;
  FilterType::Pointer filter = FilterType::New();
  FilterType::IndexType seed_point;
  for(unsigned int i = 0; i < seed_point.GetIndexDimension(); i++) {
    seed_point[i] = seed_[i+1];
  }
  
  filter->SetNumberOfIterations(3);
  filter->SetMultiplier(2.0);
  filter->SetSeed(seed_point);
  filter->SetReplaceValue(255.0);
  filter->SetInitialNeighborhoodRadius(1);

  string name = "Confidence Connected";
  NrrdVolume *temp = new NrrdVolume(volume_, name, 2);
  painter_->volume_map_[name] = temp;
  //temp->colormap_.set(2);
  temp->clut_min_ = temp->data_min_ = 0;
  temp->clut_max_ = temp->data_max_ = 255.0;
  painter_->current_volume_ = temp;

  painter_->show_volume(name);
  painter_->recompute_volume_list();
  cerr << "starting\n";
  painter_->do_itk_filter<Painter::ITKImageFloat3D>(filter, 
                                                    temp->nrrd_handle_);
  cerr << "done\n";
  painter_->set_all_slices_tex_dirty();
  painter_->redraw_all();
#endif
  
  return QUIT_AND_STOP_E;
}

BaseTool::propagation_state_e
Painter::ITKConfidenceConnectedImageFilterTool::pointer_motion
(int b, int x, int y, unsigned int m, int t)
{
//   if (b) {
//     volume_ = painter_->current_volume_;
//     if (volume_)
//       seed_ = volume_->world_to_index(event.position_);
//     painter_->redraw_all();
//     return STOP_E;
//   }
//   return CONTINUE_E;
  return CONTINUE_E;
}



#if 0
int 
Painter::ITKConfidenceConnectedImageFilterTool::do_event(Event &event)
{
  bool finish = (event.type_ == Event::KEY_PRESS_E && event.key_ == " ");
  if (!finish && event.keys_.size()) 
    return FALLTHROUGH_E;

  if (finish ||
      (event.type_ == Event::BUTTON_RELEASE_E && event.button(3))) {
  }

  if (event.state_ & Event::BUTTON_1_E) {
    volume_ = painter_->current_volume_;
    if (volume_)
      seed_ = volume_->world_to_index(event.position_);
    painter_->redraw_all();
    return HANDLED_E;
  }

  return FALLTHROUGH_E;
}

int
Painter::ITKConfidenceConnectedImageFilterTool::draw(Painter::SliceWindow &window)
{
  if (!volume_ || !volume_->index_valid(seed_)) return 0;
  vector<double> index(seed_.size());
  index[0] = seed_[0];
  for (unsigned int s = 1; s < index.size(); ++s)
    index[s] = seed_[s]+0.5;

  Vector left = window.x_dir();
  Vector up = window.y_dir();
  Point center = volume_->index_to_point(index);
  Point p;

  //  double one = 100.0 / window.zoom_; // world space units per one pixel
  double units = window.zoom_ / 100.0;  // Pixels per world space unit
  double s = units/2.0;
  double e = s+Clamp(s, 5.0, Max(units, 5.0));

  for (int pass = 0; pass < 3; ++pass) {
    glLineWidth(5 - pass*2.0);
    if (pass == 0)
      glColor4d(0.0, 0.0, 0.0, 1.0);
    else if (pass == 1)
      glColor4d(1.0, 0.0, 0.0, 1.0);
    else
      glColor4d(1.0, 0.7, 0.6, 1.0);

    glBegin(GL_LINES);    
    p = center + s * up;
    glVertex3dv(&p(0));
    p = center + e * up;
    glVertex3dv(&p(0));
    
    p = center - s * up;
    glVertex3dv(&p(0));
    p = center - e * up;
    glVertex3dv(&p(0));
    
    p = center + s * left;
    glVertex3dv(&p(0));
    p = center + e * left;
    glVertex3dv(&p(0));
    
    p = center - s * left;
    glVertex3dv(&p(0));
    p = center - e * left;
    glVertex3dv(&p(0));
    glEnd();
    CHECK_OPENGL_ERROR();
  }

  glLineWidth(1.0);
  
  return 0;
}
  
#endif
  

Painter::ITKGradientMagnitudeTool::ITKGradientMagnitudeTool(Painter *painter) :
  PainterTool(painter,"ITK Gradient Magnitude")
{
#ifdef HAVE_INSIGHT
  typedef itk::GradientMagnitudeImageFilter
    < Painter::ITKImageFloat3D, Painter::ITKImageFloat3D > FilterType;
  FilterType::Pointer filter = FilterType::New();

  NrrdVolume *vol = new NrrdVolume(painter_->current_volume_, name_, 2);
  painter_->volume_map_[name_] = vol;
  painter_->show_volume(name_);
  painter_->recompute_volume_list();

  painter_->do_itk_filter<Painter::ITKImageFloat3D>(filter, vol->nrrd_handle_);
  vol->reset_data_range();
  painter_->current_volume_ = vol;
  painter_->redraw_all();
#endif
}




Painter::ITKBinaryDilateTool::ITKBinaryDilateTool(Painter *painter) :
  BaseTool("ITK Fill Holes")
{
#ifdef HAVE_INSIGHT
  typedef itk::BinaryBallStructuringElement< float, 3> StructuringElementType;
  typedef itk::BinaryDilateImageFilter
    < Painter::ITKImageFloat3D, Painter::ITKImageFloat3D, StructuringElementType > FilterType;
  FilterType::Pointer filter = FilterType::New();

  StructuringElementType structuringElement;
  structuringElement.SetRadius( 2 );
  structuringElement.CreateStructuringElement();

  filter->SetKernel(structuringElement);
  filter->SetDilateValue(255);

  NrrdVolume *vol = painter->current_volume_;
  painter->do_itk_filter<Painter::ITKImageFloat3D>(filter, vol->nrrd_handle_);
  painter->redraw_all();
#endif
}



Painter::ITKBinaryDilateErodeTool::ITKBinaryDilateErodeTool(Painter *painter) :
  PainterTool(painter,"ITK Fill Holes")
{
#ifdef HAVE_INSIGHT
  typedef itk::BinaryBallStructuringElement< float, 3> StructuringElementType;
  typedef itk::BinaryDilateImageFilter
    < Painter::ITKImageFloat3D, Painter::ITKImageFloat3D, StructuringElementType > FilterType;
  FilterType::Pointer filter = FilterType::New();


  StructuringElementType structuringElement;
  structuringElement.SetRadius( 3 );
  structuringElement.CreateStructuringElement();

  filter->SetKernel(structuringElement);
  filter->SetDilateValue(255);

  typedef itk::BinaryErodeImageFilter
    < Painter::ITKImageFloat3D, Painter::ITKImageFloat3D, StructuringElementType > FilterType2;
  FilterType2::Pointer filter2 = FilterType2::New();

  filter2->SetKernel(structuringElement);
  filter2->SetErodeValue(255);

  NrrdVolume *vol = new NrrdVolume(painter_->current_volume_, name_, 2);
  painter_->volume_map_[name_] = vol;
  painter_->show_volume(name_);
  painter_->recompute_volume_list();
  
  painter_->do_itk_filter<Painter::ITKImageFloat3D>(filter, vol->nrrd_handle_);
  painter_->do_itk_filter<Painter::ITKImageFloat3D>(filter2, vol->nrrd_handle_);

  painter_->current_volume_ = vol;
  painter_->redraw_all();
#endif
}


Painter::ITKCurvatureAnisotropicDiffusionTool::ITKCurvatureAnisotropicDiffusionTool(Painter *painter) :
  PainterTool(painter,"ITK Curvature Anisotropic Diffusion")
{
#ifdef HAVE_INSIGHT 
  typedef itk::CurvatureAnisotropicDiffusionImageFilter
    < Painter::ITKImageFloat3D, Painter::ITKImageFloat3D > FilterType;
  FilterType::Pointer filter = FilterType::New();
  
  filter->SetNumberOfIterations(5);
  filter->SetTimeStep(0.075);
  filter->SetConductanceParameter(1.0);
  
  NrrdVolume *vol = new NrrdVolume(painter_->current_volume_, name_, 2);
  painter_->volume_map_[name_] = vol;
  painter_->show_volume(name_);
  painter_->recompute_volume_list();
  
  painter_->do_itk_filter<Painter::ITKImageFloat3D>(filter, vol->nrrd_handle_);

  painter_->current_volume_ = vol;
  painter_->redraw_all();
#endif
}


Painter::LayerMergeTool::LayerMergeTool(Painter *painter):
  PainterTool(painter, "Layer Merge")
{
  NrrdVolumeOrder::iterator volname = 
    std::find(painter_->volume_order_.begin(), 
              painter_->volume_order_.end(), 
              painter_->current_volume_->name_.get());
  
  if (volname == painter_->volume_order_.begin()) return;
  NrrdVolume *vol1 = painter_->volume_map_[*volname];
  NrrdVolume *vol2 = painter_->volume_map_[*(--volname)];
    

  NrrdData *nout = new NrrdData();
  NrrdIter *ni1 = nrrdIterNew();
  NrrdIter *ni2 = nrrdIterNew();
  
  
  nrrdIterSetNrrd(ni1, vol1->nrrd_handle_->nrrd_);
  nrrdIterSetNrrd(ni2, vol2->nrrd_handle_->nrrd_);
  
  if (nrrdArithIterBinaryOp(nout->nrrd_, nrrdBinaryOpMax, ni1, ni2)) {
    char *err = biffGetDone(NRRD);
    string errstr = (err ? err : "");
    free(err);
    throw errstr;
  }

  nrrdIterNix(ni1);
  nrrdIterNix(ni2);

  nrrdKeyValueCopy(nout->nrrd_,  vol1->nrrd_handle_->nrrd_);
  nrrdKeyValueCopy(nout->nrrd_,  vol2->nrrd_handle_->nrrd_);
  
  vol1->nrrd_handle_->nrrd_ = nout->nrrd_;
  vol2->keep_ = 0;
  
  painter_->recompute_volume_list();
  painter_->current_volume_ = vol1;
}
  

  


} // End namespace SCIRun
