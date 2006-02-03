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


#ifdef _WIN32
#define SCISHARE __declspec(dllimport)
#define sleep(z) Sleep(z*1000)
#else
#define SCISHARE
#endif

extern "C" SCISHARE Tcl_Interp* the_interp;

namespace SCIRun {



Painter::CLUTLevelsTool::CLUTLevelsTool(Painter *painter) : 
  PainterTool(painter, "Color Lookup Table"),
  scale_(1.0), ww_(0), wl_(1.0)
{
}


string *
Painter::CLUTLevelsTool::mouse_button_press(MouseState &mouse) {
  if (!painter_->current_volume_) 
    return scinew string("No layer selected!");
  ww_ = (painter_->current_volume_->clut_max_() - 
         painter_->current_volume_->clut_min_());
  wl_ =  painter_->current_volume_->clut_min_ + ww_ / 2.0;;

  const double w = painter_->current_layout_->opengl_->width();
  const double h = painter_->current_layout_->opengl_->height();
  scale_ = (painter_->current_volume_->data_max_ - 
            painter_->current_volume_->data_min_) / sqrt(w*w+h*h);
  return mouse_motion(mouse); 
}

string *
Painter::CLUTLevelsTool::mouse_button_release(MouseState &mouse) {
  return scinew string("Done");
}


string *
Painter::CLUTLevelsTool::mouse_motion(MouseState &mouse) {
  const float ww = ww_+scale_*mouse.dy_;  
  const float wl = wl_+scale_*mouse.dx_;
  painter_->current_volume_->clut_min_ = wl - ww/2.0;
  painter_->current_volume_->clut_max_ = wl + ww/2.0;
  painter_->for_each(&Painter::rebind_slice);
  painter_->redraw_all();
  return 0; 
}



Painter::ZoomTool::ZoomTool(Painter *painter) : 
  PainterTool(painter, "Zoom"),
  zoom_(0.0), window_(0)
{
}


string *
Painter::ZoomTool::mouse_button_press(MouseState &mouse) {
  if (!mouse.window_)
    return scinew string("Cant Zoom!, mouse not in window");
  window_ = mouse.window_;
  zoom_ = window_->zoom_;
  return 0;
}

string *
Painter::ZoomTool::mouse_button_release(MouseState &mouse) {
  return scinew string("Done");
}


string *
Painter::ZoomTool::mouse_motion(MouseState &mouse) {
  //  double scale = 1.0;//exp(log(window_->zoom_/100.0)/log(10.0));
  //  cerr << scale << std::endl;
  //  window_->zoom_ = Max(1.0,zoom_+scale*(mouse.dx_+mouse.dy_));
  window_->zoom_ = Max(0.00001,zoom_*Pow(1.002,mouse.dx_+mouse.dy_));
  painter_->redraw_window(*window_);
  return 0; 
}


Painter::AutoviewTool::AutoviewTool(Painter *painter) : 
  PainterTool(painter, "Autoview")
{
}


string *
Painter::AutoviewTool::mouse_button_press(MouseState &mouse) {
  if (!mouse.window_)
    return scinew string("Cant Autoview!, mouse not in window");
  painter_->autoview(*(mouse.window_));
  return 0;
}

string *
Painter::AutoviewTool::mouse_button_release(MouseState &mouse) {
  return scinew string("Done");
}

Painter::ProbeTool::ProbeTool(Painter *painter) : 
  PainterTool(painter, "Probe")
{
}


string *
Painter::ProbeTool::mouse_button_press(MouseState &mouse) {
  return mouse_motion(mouse);
}

string *
Painter::ProbeTool::mouse_button_release(MouseState &mouse) {
  return scinew string("Done");
}


string *
Painter::ProbeTool::mouse_motion(MouseState &mouse) {
  painter_->for_each(&Painter::set_probe);
  painter_->redraw_all();
  return 0;
}



Painter::PanTool::PanTool(Painter *painter) : 
  PainterTool(painter, "Pan"),
  center_(0,0,0), window_(0)
{
}


string *
Painter::PanTool::mouse_button_press(MouseState &mouse) {
  if (!mouse.window_)
    return scinew string("Cant Pan, mouse not in window");
  window_ = mouse.window_;
  center_ = window_->center_;
  return 0;
}


string *
Painter::PanTool::mouse_button_release(MouseState &mouse) {
  return scinew string("Done");
}

string *
Painter::PanTool::mouse_motion(MouseState &mouse) {
  const float scale = 100.0/window_->zoom_;
  int xax = window_->x_axis();
  int yax = window_->y_axis();
  window_->center_(xax) = center_(xax) - mouse.dx_ * scale;
  window_->center_(yax) = center_(yax) + mouse.dy_ * scale;
  painter_->redraw_window(*window_);
  return 0; 
}




Painter::CropTool::CropTool(Painter *painter) : 
  PainterTool(painter, "Crop"),
  pick_(0)
{
  //  bbox_ = BBox(painter_->current_volume_->min_, painter_.current_volume_->max_);
  bbox_ = 
    BBox(Point(0,0,0), 
         Point(painter_->current_volume_->nrrd_->nrrd->axis[0].size, 
               painter_->current_volume_->nrrd_->nrrd->axis[1].size, 
               painter_->current_volume_->nrrd_->nrrd->axis[2].size));
  draw_bbox_ = bbox_;
  update_bbox_to_gui();
  
}

Painter::CropTool::~CropTool() {}


string *
Painter::CropTool::mouse_button_press(MouseState &mouse) {
  if (!mouse.window_) 
    return scinew string("No window!");

  compute_crop_pick_boxes(*(mouse.window_));
  pick_ = get_pick_from_mouse(mouse);
  //  bbox_ = BBox(painter_current_volume_->min_, painter_.current_volume_->max_);
  return 0;
}


string *
Painter::CropTool::mouse_motion(MouseState &mouse) {
  SliceWindow &window = *(mouse.window_);
  pair<Vector, Vector> crop_delta = get_crop_vectors(window, pick_);
  Vector crop_delta_x = crop_delta.first*mouse.dx_;
  Vector crop_delta_y = crop_delta.second*mouse.dy_;
  Point min = bbox_.min();
  Point max = bbox_.max();
  const int p = window.x_axis();
  const int s = window.y_axis();

  vector<int> max_slice = painter_->current_volume_->max_index();

  //  UIint *uimin[3] = { &crop_min_x_, &crop_min_y_, &crop_min_z_ };
  // UIint *uimax[3] = { &crop_min_x_, &crop_min_y_, &crop_min_z_ };
  int uiminpad[3] = {0,0,0}; //{crop_min_pad_x_(), crop_min_pad_y_(), crop_min_pad_z_()};
  int uimaxpad[3] = {0,0,0}; //{crop_max_pad_x_(), crop_max_pad_y_(), crop_max_pad_z_()};
  for (int n = 0; n < 3; n++)
    uimaxpad[n] += uiminpad[n];
  switch (pick_) {
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
    if (max(p)+crop_delta_x[p] > (max_slice[p]+uimaxpad[p]+1.0))
      crop_delta_x[p] = (max_slice[p]+uimaxpad[p]+1.0)-max(p);
    if (max(s)+crop_delta_y[s] > (max_slice[s]+uimaxpad[s]+1.0))
      crop_delta_y[s] = (max_slice[s]+uimaxpad[s]+1.0)-max(s);

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
    if (max(i) > (max_slice[i]+uimaxpad[i]+1.0)) 
      max(i) = (max_slice[i]+uimaxpad[i]+1.0);    
  }

  for (i = 0; i < 3; ++i)
    if (fabs(min(i)-max(i)) < 0.0001)  // floating point equal
      if (min(i)+uiminpad[i] > 0.0001) 
	min(i) = max(i)-1.0;
      else
	max(i) = -uiminpad[i]+1.0;

  draw_bbox_ = BBox(min, max);

  compute_crop_pick_boxes(window);
  update_bbox_to_gui();
  painter_->redraw_all();

  return 0; 
}



string *
Painter::CropTool::mouse_button_release(MouseState &mouse) {
  if (pick_) {
    bbox_ = draw_bbox_;
    pick_ = 0;
  }

  return 0;
}



string *
Painter::CropTool::draw(SliceWindow &window) {
  float unscaled_one = 1.0;
  Vector tmp = window.screen_to_world(1, 0) - window.screen_to_world(0, 0);
  tmp[window.axis_] = 0;
  float screen_space_one = Max(fabs(tmp[0]), fabs(tmp[1]), fabs(tmp[2]));
  if (screen_space_one > unscaled_one) 
    unscaled_one = screen_space_one;

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  const int axis = window.axis_;
  int p = window.x_axis();
  int s = window.y_axis();
  
  Vector scale = painter_->current_volume_->scale();

  double ll[3], ur[3], lr[3], ul[3], upper[3], lower[3], left[3], right[3];
  ll[0] = draw_bbox_.min().x()*scale[0];
  ll[1] = draw_bbox_.min().y()*scale[1];
  ll[2] = draw_bbox_.min().z()*scale[2];

  ur[0] = draw_bbox_.max().x()*scale[0];
  ur[1] = draw_bbox_.max().y()*scale[1];
  ur[2] = draw_bbox_.max().z()*scale[2];

  ll[axis] = int(window.slice_num_)*scale[axis];
  ur[axis] = int(window.slice_num_)*scale[axis];
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

  if (double(window.slice_num_) >= draw_bbox_.min()(window.axis_) &&
      double(window.slice_num_) <= (draw_bbox_.max()(window.axis_)-1.0))
    glColor4dv(green);
  else
    glColor4dv(lt_green);
  glColor4d(0.0, 0.0, 0.0, 0.75);


  Point sll = window.screen_to_world(0, 0);
  Point slr = window.screen_to_world(window.viewport_->width(), 0);
  Point sul = window.screen_to_world(0, window.viewport_->height());
  Point sur = window.screen_to_world(window.viewport_->width(), 
                                     window.viewport_->height());

  glBegin(GL_QUADS);
  
  glVertex3dv(&sll(0));
  glVertex3dv(&slr(0));
  glVertex3d(s==0?lr[0]:slr(0), s==1?lr[1]:slr(1), s==2?lr[2]:slr(2));
  glVertex3d(s==0?lr[0]:sll(0), s==1?lr[1]:sll(1), s==2?lr[2]:sll(2));

  glVertex3d(s==0?ur[0]:sul(0), s==1?ur[1]:sul(1), s==2?ur[2]:sul(2));
  glVertex3d(s==0?ur[0]:sur(0), s==1?ur[1]:sur(1), s==2?ur[2]:sur(2));
  glVertex3dv(&sur(0));
  glVertex3dv(&sul(0));

  glVertex3d(s==0?ll[0]:sll(0), s==1?ll[1]:sll(1), s==2?ll[2]:sll(2));
  glVertex3dv(ll);
  glVertex3dv(ul);
  glVertex3d(s==0?ul[0]:sul(0), s==1?ul[1]:sul(1), s==2?ul[2]:sul(2));

  glVertex3dv(lr);
  glVertex3d(s==0?lr[0]:slr(0), s==1?lr[1]:slr(1), s==2?lr[2]:slr(2));
  glVertex3d(s==0?ur[0]:sur(0), s==1?ur[1]:sur(1), s==2?ur[2]:sur(2));
  glVertex3dv(ur);

  //  glVertex3dv(ll);
  //  glVertex3dv(lr);
  //  glVertex3dv(ur);
  //  glVertex3dv(ul);
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

  compute_crop_pick_boxes(window);
  //  set_window_cursor(window, get_pick_from_mouse(mouse));
  return 0; 
}


void
Painter::CropTool::compute_crop_pick_boxes(SliceWindow &window) 
{
  const int axis = window.axis_;
  int p = window.x_axis();
  int s = window.y_axis();
  
  Vector scale = painter_->current_volume_->scale();

  Point ll, ur, lr, ul;
  ll(0) = draw_bbox_.min().x()*scale[0];
  ll(1) = draw_bbox_.min().y()*scale[1];
  ll(2) = draw_bbox_.min().z()*scale[2];

  ur(0) = draw_bbox_.max().x()*scale[0];
  ur(1) = draw_bbox_.max().y()*scale[1];
  ur(2) = draw_bbox_.max().z()*scale[2];

  ll(axis) = int(window.slice_num_)*scale[axis];
  ur(axis) = int(window.slice_num_)*scale[axis];
  int i;
  for (i = 0; i < 3; ++i) {
    lr(i) = p==i?ur(i):ll(i);
    ul(i) = s==i?ur(i):ll(i);
  }
  
  ll = window.world_to_screen(ll);
  lr = window.world_to_screen(lr);
  ur = window.world_to_screen(ur);
  ul = window.world_to_screen(ul);

  Vector delta(3.0, 3.0, 1.0);
  pick_boxes_.reserve(9);
  pick_boxes_.push_back(BBox(ll-delta, ll+delta)); // Lower-Left 1
  pick_boxes_.push_back(BBox(lr-delta, lr+delta)); // Lower-Right 2
  pick_boxes_.push_back(BBox(ur-delta, ur+delta)); // Upper-right 3
  pick_boxes_.push_back(BBox(ul-delta, ul+delta)); // Upper-Left 4
  pick_boxes_.push_back(BBox(ll-delta, ul+delta)); // Left 5
  pick_boxes_.push_back(BBox(ll-delta, lr+delta)); // Lower 6
  pick_boxes_.push_back(BBox(lr-delta, ur+delta)); // Right 7
  pick_boxes_.push_back(BBox(ul-delta, ur+delta)); // Upper 8
  pick_boxes_.push_back(BBox(ll-delta, ur+delta)); // Entire Crop Box
}

void
Painter::CropTool::update_bbox_from_gui() 
{
#if 0 // todo
  draw_bbox_ = 
    BBox(Point(double(Min(crop_min_x_(), crop_max_x_())),
	       double(Min(crop_min_y_(), crop_max_y_())),
	       double(Min(crop_min_z_(), crop_max_z_()))),
	 Point(double(Max(crop_min_x_(), crop_max_x_())+1),
	       double(Max(crop_min_y_(), crop_max_y_())+1),
	       double(Max(crop_min_z_(), crop_max_z_())+1)));
  bbox_ = draw_bbox_;
#endif
}

void
Painter::CropTool::update_bbox_to_gui() 
{
#if 0
  crop_min_x_ = int(draw_bbox_.min().x());
  crop_min_y_ = int(draw_bbox_.min().y());
  crop_min_z_ = int(draw_bbox_.min().z());
  crop_max_x_ = int(draw_bbox_.max().x()-1.0);
  crop_max_y_ = int(draw_bbox_.max().y()-1.0);
  crop_max_z_ = int(draw_bbox_.max().z()-1.0);
#endif
}


int
Painter::CropTool::get_pick_from_mouse(MouseState &mouse)
{
  Point pos(mouse.x_, mouse.y_, 0.0);
  // Optimization: Check the last pick box first, 
  // assuming it encloses all the previous pick boxes
  if (!pick_boxes_.back().inside(pos)) return 0;
  else for (unsigned int i = 0; i < pick_boxes_.size(); ++i)
    if (pick_boxes_[i].inside(pos)) return i+1;
  return 0; //pick_boxes.size();
}

void
Painter::CropTool::set_window_cursor(SliceWindow &window, int cursor) 
{
  if (painter_->mouse_.window_ != &window || 
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
  Tk_Window &tkwin = painter_->layouts_[window.name_]->opengl_->tkwin_;
  Tk_DefineCursor(tkwin, Tk_GetCursor(the_interp, tkwin, 
				      ccast_unsafe(cursor_name)));
}


pair<Vector, Vector>
Painter::CropTool::get_crop_vectors(SliceWindow &window, int pick) 
{

  Vector tmp = window.screen_to_world(1, 0) - window.screen_to_world(0, 0);
  tmp[window.axis_] = 0;
  const float one = Max(fabs(tmp[0]), fabs(tmp[1]), fabs(tmp[2]));
  const int xax = window.x_axis();
  const int yax = window.y_axis();
  Vector x_delta(0.0, 0.0, 0.0), y_delta(0.0, 0.0, 0.0);
  Vector scale = painter_->current_volume_->scale();
  if (pick != 6 && pick != 8)
    x_delta[xax] = one/scale[xax];
  if (pick != 5 && pick != 7)
    y_delta[yax] = -one/scale[yax];
  
  return make_pair(x_delta, y_delta);
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

string *
Painter::FloodfillTool::mouse_button_press(MouseState &mouse)
{
  NrrdVolume *volume = painter_->current_volume_;
  if (!mouse.window_ || !painter_->current_volume_) 
    return scinew string("No window or current layer");
  
  if (mouse.button_ == 1) {
    vector<int> index = volume->world_to_index(mouse.position_);
    if (!volume->index_valid(index)) return 0;

    min_ = painter_->current_volume_->data_max_;
    max_ = painter_->current_volume_->data_min_;
    start_pos_ = mouse.position_;
  }

  return mouse_motion(mouse);
}


string *
Painter::FloodfillTool::mouse_button_release(MouseState &mouse)
{
  if (mouse.button_ != 1) return 0;

  NrrdVolume *volume = painter_->current_volume_;
  vector<int> index = volume->world_to_index(start_pos_);
  if (!volume->index_valid(index)) return 0;

  // Array to hold which indices to visit next
  vector<vector<int> > todo, oldtodo;

  // Push back the first seed point
  todo.push_back(index);

  // Allocated a nrrd to mark where the flood fill has visited
  NrrdDataHandle done = new NrrdData();
  nrrdAlloc(done->nrrd, nrrdTypeUChar, 4, 
            volume->nrrd_->nrrd->axis[0].size, 
            volume->nrrd_->nrrd->axis[1].size, 
            volume->nrrd_->nrrd->axis[2].size, 
            volume->nrrd_->nrrd->axis[3].size);

  // Set the visited nrrd to empty
  memset(done->nrrd->data, 0, 
         volume->nrrd_->nrrd->axis[0].size *
         volume->nrrd_->nrrd->axis[1].size *
         volume->nrrd_->nrrd->axis[2].size * 
         volume->nrrd_->nrrd->axis[3].size);
  int count  = 0;
  unsigned int axes = index.size();
  while (!todo.empty()) {
    ++count;
    if (!(count % 40)) {
      cerr << todo.size() << std::endl;
      painter_->for_each(&Painter::rebind_slice);
      painter_->redraw_all();
      TCLTask::unlock();
      sleep(0);
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
          nrrd_get_value(done->nrrd, neighbor_index, visited);
          // Bail if the voxel has been visited
          if (visited) continue;
          
          // Now check to see if this pixel is a candidate to be filled
          double neighborval;
          volume->get_value(neighbor_index, neighborval);
          // Bail if the voxel is outside the flood fill range
          if (neighborval < min_ || neighborval > max_) continue;
          // Mark this voxel as visited
          nrrd_set_value(done->nrrd, neighbor_index, (unsigned char)1);
          
          todo.push_back(neighbor_index);
        }
      }
    }
  }

  painter_->for_each(&Painter::rebind_slice);
  painter_->redraw_all();
  return scinew string("Done");
}

string *
Painter::FloodfillTool::mouse_motion(MouseState &mouse)
{
  NrrdVolume *volume = painter_->current_volume_;
  if (!volume) return 0;
  vector<int> index = volume->world_to_index(mouse.position_);
  if (!volume->index_valid(index)) return 0;

  double val;
  volume->get_value(index, val);

  if (mouse.button(1)) {
    min_ = Min(min_, val);
    max_ = Max(max_, val);
    cerr << "Min: " << min_ << "  Max: " << max_ << std::endl;
  }
  
  if (mouse.button(3)) {
    value_ = val;
    cerr << "value: " << value_ << std::endl;
  }

  
  return 0;
}

string *
Painter::FloodfillTool::draw(SliceWindow &window)
{
  return 0;
}



Painter::PixelPaintTool::PixelPaintTool(Painter *painter) :
  PainterTool(painter, "Pixel Paint")
{
}


Painter::PixelPaintTool::~PixelPaintTool()
{
}

string *
Painter::PixelPaintTool::mouse_button_press(MouseState &mouse)
{
  return mouse_motion(mouse);
}

string *
Painter::PixelPaintTool::mouse_button_release(MouseState &mouse)
{
  if (mouse.state_ & MouseState::BUTTON_2_E) {
    painter_->filter_ = 20;
    painter_->send_data();
    painter_->want_to_execute();
    return new string ("Done");
  }
  return 0;
}

string *
Painter::PixelPaintTool::mouse_motion(MouseState &mouse)
{
  NrrdVolume *volume = painter_->current_volume_;
  if (!volume) return 0;

  if (mouse.state_ & MouseState::BUTTON_1_E) {
    vector<int> index = 
      volume->world_to_index(mouse.position_);
    if (!volume->index_valid(index)) return 0;
    volume->set_value(index, value_);
    painter_->for_each(&Painter::rebind_slice);
    painter_->redraw_all();
  }

  if (mouse.state_ & MouseState::BUTTON_3_E) {
    vector<int> index = 
      volume->world_to_index(mouse.position_);
    if (!volume->index_valid(index)) return 0;
    volume->get_value(index, value_);
  }

  return 0;
}



Painter::ITKThresholdTool::ITKThresholdTool(Painter *painter, bool test) :
  PainterTool(painter, "ITK Threshold"),
  volume_(0),
  value_(1.0)
{
  if (!painter->current_volume_) 
    throw "No current volume";
  volume_ = painter_->create_volume("ITK Threshold Seed"+(test?string(" TEST"):string("")),0);
}


Painter::ITKThresholdTool::~ITKThresholdTool()
{
}

string *
Painter::ITKThresholdTool::mouse_button_press(MouseState &mouse)
{
  return mouse_motion(mouse);
}

string *
Painter::ITKThresholdTool::mouse_button_release(MouseState &mouse)
{
  if (mouse.state_ & MouseState::BUTTON_2_E) {
    painter_->filter_ = 1;
    painter_->send_data();
    painter_->want_to_execute();
    return new string ("Done");
  }
  return 0;
}

string *
Painter::ITKThresholdTool::mouse_motion(MouseState &mouse)
{
  if (mouse.state_ & MouseState::BUTTON_1_E) {
    vector<int> index = 
      volume_->world_to_index(mouse.position_);
    if (!volume_->index_valid(index)) return 0;
    volume_->set_value(index, value_);
    painter_->for_each(&Painter::rebind_slice);
    painter_->redraw_all();
  }

  if (mouse.state_ & MouseState::BUTTON_3_E) {
    vector<int> index = 
      painter_->current_volume_->world_to_index(mouse.position_);
    if (!painter_->current_volume_->index_valid(index)) return 0;
    painter_->current_volume_->get_value(index, value_);
  }

  return 0;
}

Painter::StatisticsTool::StatisticsTool(Painter *painter) :
  PainterTool(painter, "Statistics"),
  standard_deviation_(0.0),
  mean_(0.0),
  values_()
{
}


Painter::StatisticsTool::~StatisticsTool()
{
}

// string *
// Painter::StatisticsTool::mouse_button_press(MouseState &mouse)
// {
//   return mouse_motion(mouse);
// }

string *
Painter::StatisticsTool::mouse_button_release(MouseState &mouse)
{
  NrrdVolume *vol = painter_->current_volume_;
  if (vol && mouse.state_ & MouseState::BUTTON_1_E) {
    vol->clut_min_ = mean_ - standard_deviation_;
    vol->clut_max_ = mean_ + standard_deviation_;
    painter_->for_each(&Painter::rebind_slice);
    painter_->redraw_all();

  } else if (vol && mouse.state_ & MouseState::BUTTON_2_E) {
    vol->clut_min_ = vol->data_min_;
    vol->clut_max_ = vol->data_max_;
    painter_->for_each(&Painter::rebind_slice);
    painter_->redraw_all();

  } 
    
  if (mouse.state_ & MouseState::BUTTON_3_E)
    return new string ("Done");
    
  return 0;
}

string *
Painter::StatisticsTool::mouse_motion(MouseState &mouse)
{
  NrrdVolume *vol = painter_->current_volume_;
  if (!vol) return 0;
  if (mouse.state_ & MouseState::BUTTON_1_E) {
    vector<int> index = vol->world_to_index(mouse.position_);
    if (!vol->index_valid(index)) return 0;
    double value;
    vol->get_value(index, value);
    values_.push_back(value);
    recompute();
    vol->clut_min_ = mean_ - standard_deviation_;
    vol->clut_max_ = mean_ + standard_deviation_;
    painter_->for_each(&Painter::rebind_slice);
    painter_->redraw_all();

  } else if (mouse.state_ & MouseState::BUTTON_2_E) {
    values_.clear();
    recompute();
  }
  return 0;
}

void
Painter::StatisticsTool::recompute() {
  mean_ = 0.0;
  for (unsigned int n = 0; n < values_.size(); ++n)
    mean_ += values_[n];
  mean_ /= values_.size();
  double variance = 0.0;
  for (unsigned int n = 0; n < values_.size(); ++n)
    variance += (values_[n]-mean_)*(values_[n]-mean_);
  variance /= values_.size();
  standard_deviation_ = sqrt(variance);
#if 0
  cerr << "size: " << values_.size() << std::endl;
  cerr << "standard_deviation: " << standard_deviation_ << std::endl;
  cerr << "variance: " << variance << std::endl;

  double temp = 0.0;
  for (unsigned int n = 0; n < values_.size(); ++n)
    temp += values_[n]*values_[n];
  temp /= values_.size();
  cerr << "temp: " << temp << std::endl;
  double temp2 = -(mean_*mean_) + temp;
  cerr << "temp2: " << temp2 << std::endl << std::endl;
#endif

  
}

string *
Painter::StatisticsTool::draw(SliceWindow &window)
{
  if (&window != painter_->mouse_.window_) return 0;
  string std = "Standard Deviation: "+to_string(standard_deviation_);
  string mean = "Mean: "+to_string(mean_);
  FreeTypeTextTexture text(mean, painter_->fonts_["default"]);
  text.draw(painter_->mouse_.x_, painter_->mouse_.y_, FreeTypeTextTexture::nw);
  text.set(std);
  text.draw(painter_->mouse_.x_, painter_->mouse_.y_+text.height()+2, 
            FreeTypeTextTexture::nw);
                      
  return 0;
}

Painter::PaintTool::PaintTool(Painter *painter) :
  PainterTool(painter, "Paint"),
  value_(airNaN()),
  last_index_(),
  radius_(5.0),
  drawing_(false)
{
  painter_->create_undo_volume();
}


Painter::PaintTool::~PaintTool()
{
}

string *
Painter::PaintTool::mouse_button_press(MouseState &mouse)
{
  if (mouse.button_ == 1) {
    SliceWindow *window = mouse.window_;
    if (!window->paint_layer_) {
      window->paint_layer_ = 
        scinew NrrdSlice(painter_, painter_->current_volume_, 
                         window->center_, window->normal_);
      window->paint_layer_->bind();;
    }
    if (airIsNaN(value_))
      value_ = painter_->current_volume_->clut_max_;

    last_index_ = 
      window->paint_layer_->volume_->world_to_index(mouse.position_);
    
    vector<int> index1(last_index_.size()-1, 0);
    for (unsigned int i = 0, j=0; i < last_index_.size(); ++i) 
      if (int(i) != (window->axis_+1)) {
        index1[j] = last_index_[i];
        ++j;
      }
    
    splat(window->paint_layer_->texture_->nrrd_->nrrd, radius_, 
          index1[1], index1[2]);

    window->paint_layer_->texture_->apply_colormap(index1[1], index1[2],
                                                   index1[1]+1, index1[2]+1,
                                                   Ceil(radius_));

    //    painter_->for_each(&Painter::rebind_slice);
    painter_->redraw_all();    
    drawing_ = true;
  } else if (mouse.button_ == 3) {
    vector<int> index = 
      painter_->current_volume_->world_to_index(mouse.position_);
    if (painter_->current_volume_->index_valid(index))
      painter_->current_volume_->get_value(index, value_);
    radius_ *= 1.1;
    painter_->redraw_all();    
  } else if (mouse.button_ == 4) {
    radius_ *= 1.1;
    painter_->redraw_all();    
  } else if (mouse.button_ == 5) {
    radius_ /= 1.1;
    painter_->redraw_all();
  }

  return 0;
}

string *
Painter::PaintTool::mouse_button_release(MouseState &mouse)

{  
  if (mouse.button_ == 1) {
    drawing_ = false;
    if (!mouse.window_->paint_layer_) return 0;
    NrrdSlice &paint = *mouse.window_->paint_layer_;
    painter_->current_volume_->mutex_.lock();
    //    NrrdData *nout = scinew NrrdData();
    if (nrrdSplice(painter_->current_volume_->nrrd_->nrrd,
                   painter_->current_volume_->nrrd_->nrrd,
                   paint.texture_->nrrd_->nrrd,
                   mouse.window_->axis_+1,
                   last_index_[mouse.window_->axis_+1])) {
      painter_->current_volume_->mutex_.unlock();
      char *err = biffGetDone(NRRD);
      cerr << string("Error on line #")+to_string(__LINE__) +
        string(" executing nrrd command: nrrdSplice \n")+
        string("Message: ")+string(err);
      free(err);
      return 0;
    }
    //    painter_->current_volume_->nrrd_ = nout;
    painter_->current_volume_->mutex_.unlock();
               
    if (mouse.window_->paint_layer_) {
      delete mouse.window_->paint_layer_;
      mouse.window_->paint_layer_ = 0;
    }
    painter_->for_each(&Painter::rebind_slice);
    painter_->redraw_all();
  }
  

  if (mouse.button_ == 2) {
    return new string ("Done");
  }
  return 0;
}

string *
Painter::PaintTool::mouse_motion(MouseState &mouse)
{
  SliceWindow *window = mouse.window_;
  if (!window) {
    drawing_ = false;
    return 0;
  }
  
  if (!window->paint_layer_) return 0;

  NrrdSlice &paint = *window->paint_layer_;
  NrrdVolume *volume = paint.volume_;
  
  vector<int> index = 
    volume->world_to_index(mouse.position_);
  if (!volume->index_valid(index)) return 0;
  
  if (mouse.button(1) && drawing_) {

    vector<int> index1(index.size()-1, 0);
    vector<int> index2 = index1;
    for (unsigned int i = 0, j=0; i < index.size(); ++i) 
      if (int(i) != (window->axis_+1)) {
        index1[j] = last_index_[i];
        index2[j] = index[i];
        ++j;
      }


    line(paint.texture_->nrrd_->nrrd, radius_, 
         index1[1], index1[2],
         index2[1], index2[2], true);

    paint.texture_->apply_colormap(index1[1], index1[2], 
                                   index2[1], index2[2],
                                   Ceil(radius_));
    painter_->redraw_all();
  }

  if (mouse.button(1))
    drawing_ = true;
  
  last_index_ = index;

  return 0;
}

string *
Painter::PaintTool::draw(SliceWindow &window)
{
  return 0;
}


string *
Painter::PaintTool::draw_mouse_cursor(MouseState &mouse)
{
  glColor4f(1.0, 0.0, 0.0, 1.0);
  glLineWidth(2.0);
  glBegin(GL_LINES);


  int x0 = Floor(mouse.position_(mouse.window_->x_axis()));
  int y0 = Floor(mouse.position_(mouse.window_->y_axis()));
  int z0 = Floor(mouse.position_(mouse.window_->axis_));
  //  int wid = int(Ceil(radius_));
  const int wid = Round(radius_);
  for (int y = y0-wid; y <= y0+wid; ++y)
    for (int x = x0-wid; x <= x0+wid; ++x) {
      float dist = sqrt(double(x0-x)*(x0-x)+(y0-y)*(y0-y));
      if (dist <= radius_ && 
          sqrt(double(x0-(x+1))*(x0-(x+1))+(y0-(y+0))*(y0-(y+0))) > radius_) {
        glVertex3d(x+1, y, z0);
        glVertex3d(x+1, y+1, z0);
      }

      if (dist <= radius_ && 
          sqrt(double(x0-(x+0))*(x0-(x+0))+(y0-(y+1))*(y0-(y+1))) > radius_) {
        glVertex3d(x, y+1, z0);
        glVertex3d(x+1, y+1, z0);
      }

      if (dist <= radius_ && 
          sqrt(double(x0-(x-1))*(x0-(x-1))+(y0-(y+0))*(y0-(y+0))) > radius_) {
        glVertex3d(x, y, z0);
        glVertex3d(x, y+1, z0);
      }

      if (dist <= radius_ && 
          sqrt(double(x0-(x+0))*(x0-(x+0))+(y0-(y-1))*(y0-(y-1))) > radius_) {
        glVertex3d(x, y, z0);
        glVertex3d(x+1, y, z0);
      }

    }

                 



//         glVertex3d(mouse.position_.x()+1, mouse.position_.y(), mouse.position_.z());

//       if (x >= 0 && x < nrrd->axis[1].size &&
//           y >= 0 && y < nrrd->axis[2].size) 
//         {
//           index[1] = x;
//           index[2] = y;


//             //            dist = 1.0 - dist;
//             //            dist *= painter_->current_volume_->clut_max_ - painter_->current_volume_->clut_min_;
//             //            dist += painter_->current_volume_->clut_min_;
//             //            float val;
//             //            nrrd_get_value(nrrd, index, val);
//             nrrd_set_value(nrrd, index, Max(dist, val));//nrrd_get_value(nrrd,index);
//           }
//         }


//   glVertex3d(mouse.position_.x(), mouse.position_.y(), mouse.position_.z());
//   glVertex3d(mouse.position_.x()+1, mouse.position_.y(), mouse.position_.z());
  
//   glVertex3d(mouse.position_.x(), mouse.position_.y(), mouse.position_.z());
//   glVertex3d(mouse.position_.x()-1, mouse.position_.y(), mouse.position_.z());

//   glVertex3d(mouse.position_.x(), mouse.position_.y(), mouse.position_.z());
//   glVertex3d(mouse.position_.x(), mouse.position_.y()+1, mouse.position_.z());

//   glVertex3d(mouse.position_.x(), mouse.position_.y(), mouse.position_.z());
  glVertex3d(mouse.position_.x(), mouse.position_.y()-1, mouse.position_.z());
  glEnd();
  glLineWidth(1.0);
  return 0;
}



void
Painter::PaintTool::splat(Nrrd *nrrd, double radius, int x0, int y0)
{
  vector<int> index(3,0);
  const int wid = Round(radius);
  for (int y = y0-wid; y <= y0+wid; ++y)
    for (int x = x0-wid; x <= x0+wid; ++x)
      if (x >= 0 && x < nrrd->axis[1].size &&
          y >= 0 && y < nrrd->axis[2].size) 
        {
          index[1] = x;
          index[2] = y;
          float dist = sqrt(double(x0-x)*(x0-x)+(y0-y)*(y0-y))/radius;
          if (dist <= 1.0) {
            //            dist = 1.0 - dist;
            //            dist *= painter_->current_volume_->clut_max_ - painter_->current_volume_->clut_min_;
            //            dist += painter_->current_volume_->clut_min_;
            //            float val;
            //            nrrd_get_value(nrrd, index, val);
            nrrd_set_value(nrrd, index, Max(dist, value_));//nrrd_get_value(nrrd,index);
          }
        }
}

void 
Painter::PaintTool::line(Nrrd *nrrd, double radius,
                     int x0, int y0, int x1, int y1, bool first)
{
  if (x0 < 0 || x0 >= nrrd->axis[1].size || 
      x1 < 0 || x1 >= nrrd->axis[1].size || 
      y0 < 0 || y0 >= nrrd->axis[2].size || 
      y1 < 0 || y1 >= nrrd->axis[2].size) return;
  int dy = y1 - y0;
  int dx = x1 - x0;
  int sx = 1;
  int sy = 1;
  int frac = 0;
  if (dy < 0) { 
    dy = -dy;
    sy = -1; 
  } 
  if (dx < 0) { 
    dx = -dx;  
    sx = -1;
  } 
  dy <<= 1;
  dx <<= 1;
  if (first) splat(nrrd, radius, x0, y0);
  if (dx > dy) {
    frac = dy - (dx >> 1);
    while (x0 != x1) {
      if (frac >= 0) {
        y0 += sy;
        frac -= dx;
      }
      x0 += sx;
      frac += dy;
      splat(nrrd, radius, x0, y0);
    }
  } else {
    frac = dx - (dy >> 1);
    while (y0 != y1) {
      if (frac >= 0) {
        x0 += sx;
        frac -= dy;
      }
      y0 += sy;
      frac += dx;
      splat(nrrd, radius, x0, y0);
    }
  }
}


/*

Painter::TemplateTool::TemplateTool(Painter *painter) :
  PainterTool(painter, "Template")
{
}


Painter::TemplateTool::~TemplateTool()
{
}

string *
Painter::TemplateTool::mouse_button_press(MouseState &mouse)
{
  return 0;
}

string *
Painter::TemplateTool::mouse_button_release(MouseState &mouse)
{
  return 0;
}

string *
Painter::TemplateTool::mouse_motion(MouseState &mouse)
{
  return 0;
}

string *
Painter::TemplateTool::draw(SliceWindow &window)
{
  return 0;
}

string *
Painter::TemplateTool::draw_mouse_cursor(MouseState &mouse)
{
  return 0;
}
*/




      




} // End namespace SCIRun
