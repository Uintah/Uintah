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


Painter::ITKThresholdTool::ITKThresholdTool(Painter *painter) :
  PainterTool(painter, "ITK Threshold"),
  seed_volume_(0),
  source_volume_(0)
{
}


int
Painter::ITKThresholdTool::do_event(MouseState &mouse)
{
  if (mouse.type_ != MouseState::KEY_PRESS_E)
    return FALLTHROUGH_E;

  NrrdVolume *cur = painter_->current_volume_;
  NrrdVolume *temp = 0;
  if (mouse.key_ == "1" && cur && cur != seed_volume_) {
    if (source_volume_) source_volume_->name_prefix_ = "";
    source_volume_ = cur;
    source_volume_->name_prefix_ = "Source - ";
    painter_->redraw_all();
    return HANDLED_E;
  }
  
  if (mouse.key_ == "2" && cur && cur != source_volume_) {
    if (seed_volume_) seed_volume_->name_prefix_ = "";
    seed_volume_ = cur;
    seed_volume_->name_prefix_ = "Seed - ";
    painter_->redraw_all();
    return HANDLED_E;
  }

  
  if (mouse.key_ == "q" || mouse.key_ == "Q") {
    return QUIT_E;
  }
  
  if (mouse.key_ == " " && seed_volume_ && source_volume_) {
    string name = "ITK Threshold Result";
    temp = painter_->copy_current_volume(name, 2);
    temp->colormap_.set(2);
    temp->data_min_ = -4.0;
    temp->data_max_ = 4.0;
    temp->clut_min_ = -0.5;
    temp->clut_max_ = 0.5;

    name = "ITK Threshold Source";
    temp = new NrrdVolume(source_volume_, name, 2);
    source_volume_->name_prefix_ = "";
    temp->keep_ = 0;
    painter_->volumes_.push_back(temp);
    painter_->volume_map_[name] = temp;

    name = "ITK Threshold Seed";
    pair<double, double> mean = 
      painter_->compute_mean_and_deviation(source_volume_->nrrd_->nrrd,
                                           seed_volume_->nrrd_->nrrd);
    double min = mean.first - mean.second;
    double max = mean.first + mean.second;
    nrrdKeyValueAdd(seed_volume_->nrrd_->nrrd,
                    "lower_threshold", to_string(min).c_str());
    nrrdKeyValueAdd(seed_volume_->nrrd_->nrrd,
                    "upper_threshold", to_string(max).c_str());
    
    
    temp = new NrrdVolume(seed_volume_, name, 2);
    seed_volume_->name_prefix_ = "";
    temp->keep_ = 0;
    painter_->volume_map_[name] = temp;

    painter_->filter_ = 1;
    painter_->filter_text_ = get_name() + "\nFilter Running, Please Wait";
    painter_->redraw_all();
    painter_->want_to_execute();

    return QUIT_E;
  }
  
  return FALLTHROUGH_E;
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






Painter::ITKConfidenceConnectedImageFilterTool::ITKConfidenceConnectedImageFilterTool(Painter *painter) :
  PainterTool(painter, "ITK Confidence Connected\nImage Filter"),
  volume_(0)
{
}


int 
Painter::ITKConfidenceConnectedImageFilterTool::do_event(MouseState &mouse)
{
  bool finish = (mouse.type_ == MouseState::KEY_PRESS_E && mouse.key_ == " ");
  if (!finish && mouse.keys_.size()) 
    return FALLTHROUGH_E;

  if (finish ||
      (mouse.type_ == MouseState::BUTTON_RELEASE_E && mouse.button(3))) {
    if (!volume_) 
      volume_ = painter_->current_volume_;
    if (!volume_->index_valid(seed_))
      seed_ = volume_->world_to_index(mouse.position_);
    if (!volume_->index_valid(seed_))
      return QUIT_E;
    
    nrrdKeyValueAdd(volume_->nrrd_->nrrd, 
                    "seed_point0", to_string(seed_[1]).c_str());
    nrrdKeyValueAdd(volume_->nrrd_->nrrd, 
                    "seed_point1", to_string(seed_[2]).c_str());
    nrrdKeyValueAdd(volume_->nrrd_->nrrd, 
                    "seed_point2", to_string(seed_[3]).c_str());

    NrrdVolume *temp;
    string name = "ITK Confidence Connected";
    temp = new NrrdVolume(volume_, name, 2);
    temp->keep_ = 0;
    painter_->volume_map_[name] = temp;
    
    name = "Connected";
    temp = new NrrdVolume(volume_, name, 2);
    temp->colormap_.set(1);
    painter_->volume_map_[name] = temp;
    painter_->show_volume(name);

    painter_->filter_ = 1;
    painter_->filter_text_ = get_name() + "\nFilter Running, Please Wait";
    painter_->redraw_all();
    painter_->want_to_execute();
    
    return QUIT_E;
  }

  if (mouse.state_ & MouseState::BUTTON_1_E) {
    volume_ = painter_->current_volume_;
    if (volume_)
      seed_ = volume_->world_to_index(mouse.position_);
    painter_->redraw_all();
    return HANDLED_E;
  }

  return FALLTHROUGH_E;
}

string *
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
  
  

Painter::ITKGradientMagnitudeTool::ITKGradientMagnitudeTool(Painter *painter) :
  PainterTool(painter,"ITK Gradient Magnitude")
{
  string name = "ITK Gradient Magnitude";
  NrrdVolume *vol = new NrrdVolume(painter_->current_volume_, name, 2);
  vol->keep_ = 0;
  painter_->volume_map_[name] = vol;

  name = "Gradient";
  vol = new NrrdVolume(painter_->current_volume_, name, 2);
  vol->colormap_.set(0);
  painter_->volume_map_[name] = vol;
  painter_->show_volume(name);

  painter_->filter_ = 1;
  painter_->filter_text_ = get_name() + "\nFilter Running, Please Wait";
  painter_->redraw_all();
  painter_->want_to_execute();
}


Painter::ITKBinaryDilateErodeTool::ITKBinaryDilateErodeTool(Painter *painter) :
  PainterTool(painter,"ITK Fill Holes")
{
  string name = "ITK Binary Dilate Erode";
  NrrdVolume *vol = new NrrdVolume(painter_->current_volume_, name, 2);
  vol->keep_ = 0;
  painter_->volume_map_[name] = vol;

  name = "Filled";
  vol = new NrrdVolume(painter_->current_volume_, name, 2);
  vol->colormap_.set(1);
  painter_->volume_map_[name] = vol;
  painter_->show_volume(name);

  painter_->filter_ = 1;
  painter_->filter_text_ = get_name() + "\nFilter Running, Please Wait";
  painter_->redraw_all();
  painter_->want_to_execute();
}


Painter::ITKCurvatureAnisotropicDiffusionTool::ITKCurvatureAnisotropicDiffusionTool(Painter *painter) :
  PainterTool(painter,"ITK Curvature Anisotropic Diffusion")
{
  string name = "ITK Curvature Anisotropic Diffusion";
  NrrdVolume *vol = new NrrdVolume(painter_->current_volume_, name, 2);
  vol->keep_ = 0;
  painter_->volume_map_[name] = vol;

  name = "Anisotropic Diffusion";
  vol = new NrrdVolume(painter_->current_volume_, name, 2);
  vol->colormap_.set(0);
  painter_->volume_map_[name] = vol;
  painter_->show_volume(name);

  painter_->filter_ = 1;
  painter_->filter_text_ = get_name() + "\nFilter Running, Please Wait";
  painter_->redraw_all();
  painter_->want_to_execute();
}



} // End namespace SCIRun
