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
#  include <itkConfidenceConnectedImageFilter.h>
#  include <itkThresholdSegmentationLevelSetImageFilter.h>
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
    //  case SCIRun_u:        painter_->undo_volume();
  case SCIRun_a:        tm_.add_tool(new CropTool(painter_),100); break;
  case SCIRun_f:        tm_.add_tool(new FloodfillTool(painter_),100); break;
  case SCIRun_b:        tm_.add_tool(new BrushTool(painter_),25); break;
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

  case SCIRun_u:            
    if (painter_->current_volume_) {
      painter_->current_volume_->colormap_.set(Max(0,painter_->current_volume_->colormap_.get()-1));
      painter_->set_all_slices_tex_dirty();
      painter_->redraw_all();
    } break;
  case SCIRun_i:
    if (painter_->current_volume_) {
      painter_->current_volume_->colormap_.set(Min(int(painter_->colormap_names_.size()), 
                                         painter_->current_volume_->colormap_.get()+1));
      painter_->set_all_slices_tex_dirty();
      painter_->redraw_all();
    } break;    
  }

  painter_->redraw_all();
  return CONTINUE_E;
}  


#if 0

  if (key == "[") {
  }

  if (key == "]") {
  }

  if (key == "q") { 
    for (unsigned int t = 0; t < tools_.size(); ++t)
      delete tools_[t];
    tools_.clear();
  }

  if (key == "h") {
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
  
  const double w = painter_->cur_window_->get_region().width();
  const double h = painter_->cur_window_->get_region().height();
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
    return QUIT_AND_STOP_E;
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
  if (!window_ || (button != 3) || (modifiers != EventModifiers::SHIFT_E)) {
    return QUIT_AND_STOP_E;
  }

  //  if (window_) {
  int delta = x + y - x_ - y_;
  window_->zoom_ = Max(0.00001, zoom_ * Pow(1.002,delta));
  window_->redraw();
    //  }
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
  BaseTool("Crop"),
  PointerTool("Crop"),
  painter_(painter),
  pick_(0)
{
  ASSERT(painter_->current_volume_);
  minmax_[1] = painter_->current_volume_->max_index();
  minmax_[0] = vector<int>(minmax_[1].size(), 0);
  pick_minmax_[0] = minmax_[0];
  pick_minmax_[1] = minmax_[1];
}

Painter::CropTool::~CropTool() {}

BaseTool::propagation_state_e
Painter::CropTool::pointer_motion
(int b, int x, int y, unsigned int m, int t) 
{
  if (!pick_) {
    return CONTINUE_E;
  }

  SliceWindow *window = painter_->cur_window_;
  ASSERT(window);
  unsigned int axis = window->axis_;
  vector<int> max_index = painter_->current_volume_->max_index();
  vector<double> idx = 
    painter_->current_volume_->point_to_index(painter_->pointer_pos_);

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
  return STOP_E;
}

BaseTool::propagation_state_e
Painter::CropTool::pointer_down
(int b, int x, int y, unsigned int m, int t) 
{
  if (b == 1 && !m) {
    SliceWindow *window = painter_->cur_window_;
    ASSERT(window);
    
    double units = 100.0 / window->zoom_; // world space units per pixel
    pick_ = 1;
    for (int i = 0; i < 2; ++i) {
      pick_minmax_[i] = minmax_[i];
      Point p = painter_->current_volume_->index_to_world(minmax_[i]);
      for (int a = 0; a < 3; ++a) {
        Vector n(a==0 ? 1:0, a==1?1:0, a==2?1:0);
        if (i) n = -1*n;
        Plane plane(p, n);
        pick_dist_[i][a] = plane.eval_point(painter_->pointer_pos_)/units;
        if (Abs(pick_dist_[i][a]) < 5.0) pick_ |= 2;
        if (pick_dist_[i][a] < 0.0 && 
	    a != window->axis_) pick_ = pick_ & ~1;
      }
    }
    pick_index_ = 
      painter_->current_volume_->point_to_index(painter_->pointer_pos_);
    return STOP_E;
  }
  return CONTINUE_E;
}



BaseTool::propagation_state_e
Painter::CropTool::pointer_up
(int b, int x, int y, unsigned int m, int t) 
{
  if (pick_) {
    for (unsigned int a = 0; a < minmax_[0].size(); ++a)
      if (minmax_[0][a] > minmax_[1][a])
        SWAP(minmax_[0][a],minmax_[1][a]);
    
    pick_ = 0;
    return STOP_E;
  }
  return CONTINUE_E;
}


BaseTool::propagation_state_e 
Painter::CropTool::process_event(event_handle_t event)
{
  RedrawSliceWindowEvent *redraw = 
    dynamic_cast<RedrawSliceWindowEvent *>(event.get_rep());
  if (redraw) {
    draw_gl(redraw->get_window());
  }

  if (dynamic_cast<FinishEvent *>(event.get_rep())) {
    finish();
  }

  if (dynamic_cast<QuitEvent *>(event.get_rep())) {
    return QUIT_AND_CONTINUE_E;
  }
 
  return CONTINUE_E;
}



int
Painter::CropTool::draw_gl(SliceWindow &window) {
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
  for (int pass = 0; pass < 5; ++pass) {
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
Painter::CropTool::finish()
{
  size_t *minmax[2] = { new size_t[minmax_[0].size()], 
                        new size_t[minmax_[1].size()] };

  for (int i = 0; i < 2; ++i) {
    for (unsigned int a = 0; a < minmax_[0].size(); ++a) {
      minmax[i][a] = minmax_[i][a]-(i==1?1:0);
    }
  }

  NrrdDataHandle nout_handle = new NrrdData();
  if (nrrdCrop(//painter_->current_volume_->nrrd_handle_->nrrd_,
               nout_handle->nrrd_,
               painter_->current_volume_->nrrd_handle_->nrrd_,
               minmax[0], minmax[1])) {
    char *err = biffGetDone(NRRD);
    string str = string("nrrdcrop: ") + err;
    free(err);
    throw str;
  }
  
  painter_->current_volume_->set_nrrd(nout_handle);
  
  //  painter_->current_volume_->build_index_to_world_matrix();

  delete[] minmax[0];
  delete[] minmax[1];
  painter_->extract_all_window_slices();
  painter_->redraw_all();  
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
  BaseTool("ITK Threshold"),
  painter_(painter),
  seed_volume_(0),
  filter_(0)
{
}



BaseTool::propagation_state_e 
Painter::ITKThresholdTool::process_event
(event_handle_t event)
{

  if (dynamic_cast<SetLayerEvent *>(event.get_rep())) {
    if (seed_volume_) seed_volume_->name_prefix_ = "";
    seed_volume_ = painter_->current_volume_;
    seed_volume_->name_prefix_ = "Seed - ";
    painter_->redraw_all();
  }

  if (dynamic_cast<FinishEvent *>(event.get_rep())) {
    if (painter_->current_volume_  == seed_volume_) {
      painter_->get_vars()->insert
        ("Painter::status_text",
         "Cannot use same layers for source and seed", "string", true);
      painter_->redraw_all();
      return STOP_E;
    }

    if (!seed_volume_) {
      painter_->get_vars()->insert
        ("Painter::status_text",
         "No seed layer set", "string", true);
      painter_->redraw_all();
      return STOP_E;
    }

    if (filter_.IsNull()) 
      finish();
    else 
      cont();
    return CONTINUE_E;
  }

  if (dynamic_cast<QuitEvent *>(event.get_rep())) {
    return QUIT_AND_CONTINUE_E;
  }
 
  return CONTINUE_E;
}

void
Painter::ITKThresholdTool::cont()
{
  cerr << "CONT\n";
  //  filter_->ReverseExpansionDirectionOff();
  filter_->ManualReinitializationOn();
  filter_->Modified();
  NrrdDataHandle temp = 0;
  painter_->do_itk_filter<Painter::ITKImageFloat3D>(filter_, temp);
}


void
Painter::ITKThresholdTool::finish()
{
#if HAVE_INSIGHT
  NrrdDataHandle source_nrrdh = painter_->current_volume_->nrrd_handle_;
  filter_ = FilterType::New();
  painter_->get_vars()->insert("ToolDialog::text", "ITK Threshold Segmentation Level Set Running...", "string", true);  
  painter_->get_vars()->unset("ProgressBar::bar_height");
  //  painter_->get_vars()->insert("ToolDialog::button_height", "0", "string", true);
  painter_->get_vars()->insert("Painter::progress_bar_total_width","500","string", true);


  string name = "ITK Threshold Result";
  Painter::NrrdVolume *new_layer = new NrrdVolume(seed_volume_, name, 0);
  new_layer->colormap_.set(1);
  new_layer->data_min_ = -4.0;
  new_layer->data_max_ = 4.0;
  new_layer->clut_min_ = 4.0/255.0;
  new_layer->clut_max_ = 4.0;

  //  new_layer->clut_min_ = -4.0;//0.5/255.0;
  //  new_layer->clut_max_ = 4.0;//0.5;

  painter_->volume_map_[name] = new_layer;
  painter_->volumes_.push_back(new_layer);
  painter_->show_volume(name);
  painter_->recompute_volume_list();
  
  
  
  NrrdDataHandle seed_nrrdh = new_layer->nrrd_handle_;

  name = "ITK Threshold Seed";
  pair<double, double> mean = painter_->compute_mean_and_deviation
    (source_nrrdh->nrrd_, seed_nrrdh->nrrd_);

  double factor = 2.5;
  double min = mean.first - factor*mean.second;
  double max = mean.first + factor*mean.second;


  filter_->SetLowerThreshold(min);
  filter_->SetUpperThreshold(max);


  string minmaxstr = "Threshold min: " + to_string(min) +
    " Threshold max: " + to_string(max);
  
  painter_->get_vars()->insert("Painter::status_text",
                               minmaxstr, "string", true);


  string scope = "ITKThresholdTool::";
  Skinner::Variables *vars = painter_->get_vars();
  filter_->SetCurvatureScaling(vars->get_double(scope+"curvatureScaling"));
  filter_->SetPropagationScaling(vars->get_double(scope+"propagationScaling"));
  filter_->SetEdgeWeight(vars->get_double(scope+"edgeWeight"));
  filter_->SetNumberOfIterations(vars->get_int(scope+"numberOfIterations"));
  filter_->SetMaximumRMSError(vars->get_double(scope+"maximumRMSError"));
  if (vars->get_bool(scope+"reverseExpansionDirection")) 
    filter_->ReverseExpansionDirectionOn();
  else 
    filter_->ReverseExpansionDirectionOff();
  filter_->SetIsoSurfaceValue(vars->get_double(scope+"isoSurfaceValue"));
  filter_->SetSmoothingIterations(vars->get_int(scope+"smoothingIterations"));
  filter_->SetSmoothingTimeStep(vars->get_double(scope+"smoothingTimeStep"));
  filter_->SetSmoothingConductance(vars->get_double(scope+"smoothingConductance"));

  ITKDatatypeHandle img_handle = painter_->nrrd_to_itk_image(source_nrrdh);
  Painter::ITKImageFloat3D *imgp = 
    dynamic_cast<Painter::ITKImageFloat3D *>(img_handle->data_.GetPointer());
  filter_->SetFeatureImage(imgp);


  painter_->filter_volume_ = new_layer;
  painter_->filter_update_img_ = painter_->nrrd_to_itk_image(seed_nrrdh);

  painter_->do_itk_filter<Painter::ITKImageFloat3D>(filter_, seed_nrrdh);
  new_layer->nrrd_handle_ = seed_nrrdh;

  painter_->set_all_slices_tex_dirty();
  painter_->redraw_all();
#endif
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
  BaseTool("ITK Confidence Connected\nImage Filter"),
  PainterPointerTool(painter, "ITK Confidence Connected\nImage Filter"),
  seed_(),
  volume_(0)
{
}


BaseTool::propagation_state_e
Painter::ITKConfidenceConnectedImageFilterTool::pointer_down
(int b, int x, int y, unsigned int m, int t)
{
  BaseTool::propagation_state_e state = pointer_motion(b,x,y,m,t);

  return state;
}

BaseTool::propagation_state_e
Painter::ITKConfidenceConnectedImageFilterTool::pointer_up
(int b, int x, int y, unsigned int m, int t)
{
  return pointer_motion(b,x,y,m,t);
}


void
Painter::ITKConfidenceConnectedImageFilterTool::finish() {
  if (!volume_) 
    return;

  if (!volume_->index_valid(seed_))
    return;

#ifdef HAVE_INSIGHT    
  painter_->get_vars()->insert("ToolDialog::text", 
                     " ITK Confidence Connected Filter Running...",
                     "string", true);

  painter_->get_vars()->unset("ProgressBar::bar_height");
  painter_->get_vars()->insert("ToolDialog::button_height", "0", "string", true);
  painter_->get_vars()->insert("Painter::progress_bar_total_width","500","string", true);
  painter_->redraw_all();


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
  filter->SetReplaceValue(1.0);
  filter->SetInitialNeighborhoodRadius(1);

  string name = "Confidence Connected";
  NrrdVolume *temp = new NrrdVolume(volume_, name, 2);
  painter_->volume_map_[name] = temp;
  temp->colormap_.set(1);
  temp->clut_min_ = temp->data_min_ = 0.5;
  temp->clut_max_ = temp->data_max_ = 1.0;
  painter_->current_volume_ = temp;

  cerr << "starting\n";
  painter_->do_itk_filter<Painter::ITKImageFloat3D>(filter, 
                                                    temp->nrrd_handle_);
  cerr << "done\n";
  painter_->show_volume(name);
  painter_->recompute_volume_list();

  painter_->set_all_slices_tex_dirty();
  painter_->redraw_all();
#endif
}

BaseTool::propagation_state_e
Painter::ITKConfidenceConnectedImageFilterTool::pointer_motion
(int b, int x, int y, unsigned int m, int t)
{
  if (b == 1 && !m) {
    if (!volume_) 
      volume_ = painter_->current_volume_;
    if (volume_) {
      vector<int> newseed = volume_->world_to_index(painter_->pointer_pos_);
      if (volume_->index_valid(newseed)) 
        seed_ = newseed;

      painter_->redraw_all();
      return STOP_E;
    }
  }
  return CONTINUE_E;

}



BaseTool::propagation_state_e 
Painter::ITKConfidenceConnectedImageFilterTool::process_event
(event_handle_t event)
{
  RedrawSliceWindowEvent *redraw = 
    dynamic_cast<RedrawSliceWindowEvent *>(event.get_rep());
  if (redraw) {
    draw_gl(redraw->get_window());
  }

  if (dynamic_cast<FinishEvent *>(event.get_rep())) {
    finish();
  }

  if (dynamic_cast<QuitEvent *>(event.get_rep())) {
    return QUIT_AND_STOP_E;
  }
 
  return CONTINUE_E;
}
  


#if 0
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

#endif



void
Painter::ITKConfidenceConnectedImageFilterTool::draw_gl(Painter::SliceWindow &window)
{
  if (!volume_ || !volume_->index_valid(seed_)) return;

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
}
  


#if 0
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
  

#endif  


} // End namespace SCIRun
