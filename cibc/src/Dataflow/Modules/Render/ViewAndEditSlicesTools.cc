/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

#include <Dataflow/Modules/Render/ViewAndEditSlices.h>
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
#include <Dataflow/GuiInterface/TkOpenGLContext.h>
#include <Core/Geom/OpenGLViewport.h>
#include <Core/Geom/FreeType.h>
#include <Dataflow/GuiInterface/TCLTask.h>
#include <Dataflow/GuiInterface/UIvar.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Util/Timer.h>
#include <Core/Util/Environment.h>
#include <Core/Volume/CM2Widget.h>
#include <sgi_stl_warnings_off.h> 
#include <algorithm>
#include <sgi_stl_warnings_on.h> 

#ifdef _WIN32
#define SCISHARE __declspec(dllimport)
#define sleep(z) Sleep(z*1000)
#else
#define SCISHARE
#endif

extern "C" SCISHARE Tcl_Interp* the_interp;

namespace SCIRun {



ViewAndEditSlices::CLUTLevelsTool::CLUTLevelsTool(ViewAndEditSlices *painter) : 
  ViewAndEditSlicesTool(painter, "Color Lookup Table"),
  scale_(1.0), ww_(0), wl_(1.0)
{
}


int
ViewAndEditSlices::CLUTLevelsTool::do_event(Event &event) {
  if (!event.keys_.empty()) 
    return FALLTHROUGH_E;
  
  NrrdVolume *vol = painter_->current_volume_;
  if (!vol) 
    return FALLTHROUGH_E;
  
  //  if (!event.button(1))
  //    return FALLTHROUGH_E;
  
  if (event.type_ == Event::BUTTON_RELEASE_E)
    return QUIT_E;

  
  if (event.type_ == Event::BUTTON_PRESS_E) {
    ww_ = vol->clut_max_() - vol->clut_min_();
    wl_ = vol->clut_min_ + ww_ / 2.0;
    
    press_event_ = event;
    
    const double w = painter_->current_layout_->opengl_->width();
    const double h = painter_->current_layout_->opengl_->height();
    scale_ = (vol->data_max_ - vol->data_min_) / sqrt(w*w+h*h);
  }

  if (event.type_ == Event::BUTTON_PRESS_E ||
      event.type_ == Event::MOUSE_MOTION_E) {
    const float ww = ww_+scale_*(event.Y_ - press_event_.Y_);  
    const float wl = wl_+scale_*(event.X_ - press_event_.X_);
    vol->clut_min_ = wl - ww/2.0;
    vol->clut_max_ = wl + ww/2.0;
    painter_->for_each(&ViewAndEditSlices::rebind_slice);
    painter_->redraw_all();
    return HANDLED_E;
  }

  return FALLTHROUGH_E;
}


ViewAndEditSlices::ZoomTool::ZoomTool(ViewAndEditSlices *painter) : 
  ViewAndEditSlicesTool(painter, "Zoom"),
  zoom_(0.0)
{
}


int
ViewAndEditSlices::ZoomTool::do_event(Event &event) {  
  if (event.type_ == Event::BUTTON_RELEASE_E)
    return QUIT_E;
  
  if (event.type_ == Event::BUTTON_PRESS_E) {
    if (!event.window_)
      return FALLTHROUGH_E;
    press_event_ = event;
    zoom_ = event.window_->zoom_;
    return HANDLED_E;
  }
  
  if (event.type_ == Event::MOUSE_MOTION_E && press_event_.window_) {
    int delta = event.X_+event.Y_-press_event_.X_-press_event_.Y_;
    press_event_.window_->zoom_ = Max(0.00001, zoom_ * Pow(1.002,delta));
    painter_->redraw_window(*press_event_.window_);
    return HANDLED_E;
  }
  
  return FALLTHROUGH_E;
}


ViewAndEditSlices::AutoviewTool::AutoviewTool(ViewAndEditSlices *painter) : 
  ViewAndEditSlicesTool(painter, "Autoview")
{
}


int
ViewAndEditSlices::AutoviewTool::do_event(Event &event) {
  if (event.window_)
    painter_->autoview(*(event.window_));
  return QUIT_E;
}


ViewAndEditSlices::ProbeTool::ProbeTool(ViewAndEditSlices *painter) : 
  ViewAndEditSlicesTool(painter, "Probe")
{
}


int
ViewAndEditSlices::ProbeTool::do_event(Event &event) {
  if (event.type_ == Event::BUTTON_PRESS_E ||
      event.type_ == Event::MOUSE_MOTION_E){
    painter_->for_each(&ViewAndEditSlices::set_probe);
    painter_->redraw_all();
    return HANDLED_E;
  }
  
  if (event.type_ == Event::BUTTON_RELEASE_E) {
    return QUIT_E;
  }
  
  return FALLTHROUGH_E;
}



ViewAndEditSlices::PanTool::PanTool(ViewAndEditSlices *painter) : 
  ViewAndEditSlicesTool(painter, "Pan"),
  center_(0,0,0)
{
}


int
ViewAndEditSlices::PanTool::do_event(Event &event) {
  if (event.type_ == Event::BUTTON_RELEASE_E)
    return QUIT_E;

  if (event.type_ == Event::BUTTON_PRESS_E) {
    if (!event.window_)
      return FALLTHROUGH_E;
    press_event_ = event;
    center_ = event.window_->center_;
    return HANDLED_E;
  }

  if (event.type_ == Event::MOUSE_MOTION_E) {
    SliceWindow &window = *press_event_.window_;
    const float scale = 100.0/window.zoom_;
    int xax = window.x_axis();
    int yax = window.y_axis();
    window.center_(xax) = center_(xax) - scale * (event.X_ - press_event_.X_);
    window.center_(yax) = center_(yax) + scale * (event.Y_ - press_event_.Y_);
    painter_->redraw_window(window);
    return HANDLED_E;
  }

  return FALLTHROUGH_E;
}



ViewAndEditSlices::CropTool::CropTool(ViewAndEditSlices *painter) : 
  ViewAndEditSlicesTool(painter, "Crop"),
  pick_(0)
{
  ASSERT(painter_->current_volume_);
  minmax_[1] = painter_->current_volume_->max_index();
  minmax_[0] = vector<int>(minmax_[1].size(), 0);
  pick_minmax_[0] = minmax_[0];
  pick_minmax_[1] = minmax_[1];
}

ViewAndEditSlices::CropTool::~CropTool() {}


int
ViewAndEditSlices::CropTool::do_event(Event &event) {

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
    painter_->for_each(&ViewAndEditSlices::extract_window_slices);
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

  if (pick_ && event.type_ == Event::MOUSE_MOTION_E && event.window_) {
    pick_mouse_motion(event);
    return HANDLED_E;
  }

  return FALLTHROUGH_E;
}


void
ViewAndEditSlices::CropTool::pick_mouse_motion(Event &event) {
  ASSERT(pick_ && event.type_ == Event::MOUSE_MOTION_E && event.window_);
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
ViewAndEditSlices::CropTool::draw(SliceWindow &window) {
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
ViewAndEditSlices::CropTool::set_window_cursor(SliceWindow &window, int cursor) 
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
  Tk_Window &tkwin = painter_->layouts_[window.name_]->opengl_->tkwin_;
  Tk_DefineCursor(tkwin, Tk_GetCursor(the_interp, tkwin, 
				      ccast_unsafe(cursor_name)));
}


ViewAndEditSlices::FloodfillTool::FloodfillTool(ViewAndEditSlices *painter) :
  ViewAndEditSlicesTool(painter, "Flood Fill"),
  value_(0.0),
  min_(0.0),
  max_(0.0),
  start_pos_(0,0,0)
{
  painter_->create_undo_volume();
}


ViewAndEditSlices::FloodfillTool::~FloodfillTool()
{
}


int
ViewAndEditSlices::FloodfillTool::do_event(Event &event) {
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
  
  if (event.type_ == Event::MOUSE_MOTION_E) {

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
ViewAndEditSlices::FloodfillTool::draw(SliceWindow &)
{
  return 0;
}


void
ViewAndEditSlices::FloodfillTool::do_floodfill()
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
      painter_->for_each(&ViewAndEditSlices::rebind_slice);
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

  painter_->for_each(&ViewAndEditSlices::rebind_slice);
  painter_->redraw_all();
}


ViewAndEditSlices::ITKThresholdTool::ITKThresholdTool(ViewAndEditSlices *painter) :
  ViewAndEditSlicesTool(painter, "ITK Threshold"),
  seed_volume_(0),
  source_volume_(0)
{
}


int
ViewAndEditSlices::ITKThresholdTool::do_event(Event &event)
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

    painter_->filter_ = 1;
    painter_->filter_text_ = get_name() + "\nFilter Running, Please Wait";
    painter_->redraw_all();
    painter_->want_to_execute();

    return QUIT_E;
  }
  
  return FALLTHROUGH_E;
}


ViewAndEditSlices::StatisticsTool::StatisticsTool(ViewAndEditSlices *painter) :
  ViewAndEditSlicesTool(painter, "Statistics"),
  standard_deviation_(0.0),
  mean_(0.0),
  sum_(0.0),
  squared_sum_(0.0),
  count_(0)
{
}



int
ViewAndEditSlices::StatisticsTool::do_event(Event &event)
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
    painter_->for_each(&ViewAndEditSlices::rebind_slice);
    painter_->redraw_all();
    return HANDLED_E;
  }

  if ((event.type_ == Event::BUTTON_PRESS_E ||
       event.type_ == Event::MOUSE_MOTION_E)  && 
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

    painter_->for_each(&ViewAndEditSlices::rebind_slice);
    painter_->redraw_all();
    return HANDLED_E;
  }
  return FALLTHROUGH_E;
}
  

int
ViewAndEditSlices::StatisticsTool::draw(SliceWindow &window)
{
  if (painter_->font1_ && &window == painter_->event_.window_)
    painter_->font1_->render("Mean: "+to_string(mean_)+
                             "\nStandard Deviation: "+
                             to_string(standard_deviation_), 
                             painter_->event_.x_, painter_->event_.y_,
                             TextRenderer::NW | TextRenderer::SHADOW);
  return 0;
}






ViewAndEditSlices::ITKConfidenceConnectedImageFilterTool::ITKConfidenceConnectedImageFilterTool(ViewAndEditSlices *painter) :
  ViewAndEditSlicesTool(painter, "ITK Confidence Connected\nImage Filter"),
  volume_(0)
{
}


int 
ViewAndEditSlices::ITKConfidenceConnectedImageFilterTool::do_event(Event &event)
{
  bool finish = (event.type_ == Event::KEY_PRESS_E && event.key_ == " ");
  if (!finish && event.keys_.size()) 
    return FALLTHROUGH_E;

  if (finish ||
      (event.type_ == Event::BUTTON_RELEASE_E && event.button(3))) {
    if (!volume_) 
      volume_ = painter_->current_volume_;
    if (!volume_->index_valid(seed_))
      seed_ = volume_->world_to_index(event.position_);
    if (!volume_->index_valid(seed_))
      return QUIT_E;
    
    nrrdKeyValueAdd(volume_->nrrd_handle_->nrrd_, 
                    "seed_point0", to_string(seed_[1]).c_str());
    nrrdKeyValueAdd(volume_->nrrd_handle_->nrrd_, 
                    "seed_point1", to_string(seed_[2]).c_str());
    nrrdKeyValueAdd(volume_->nrrd_handle_->nrrd_, 
                    "seed_point2", to_string(seed_[3]).c_str());

    unsigned long ptr = (unsigned long)painter_;
    nrrdKeyValueAdd(volume_->nrrd_handle_->nrrd_, 
                    "progress_ptr", to_string(ptr).c_str());


    NrrdVolume *temp;
    string name = "ITK Confidence Connected";
    temp = new NrrdVolume(volume_, name, 2);
    temp->keep_ = 0;
    painter_->volume_map_[name] = temp;
    
    name = "Connected";
    temp = new NrrdVolume(volume_, name, 2);
    temp->colormap_.set(2);
    painter_->volume_map_[name] = temp;
    painter_->show_volume(name);

    painter_->filter_ = 1;
    painter_->filter_text_ = get_name() + "\nFilter Running, Please Wait";
    painter_->redraw_all();
    painter_->want_to_execute();
    
    return QUIT_E;
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
ViewAndEditSlices::ITKConfidenceConnectedImageFilterTool::draw(ViewAndEditSlices::SliceWindow &window)
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
  
  

ViewAndEditSlices::ITKGradientMagnitudeTool::ITKGradientMagnitudeTool(ViewAndEditSlices *painter) :
  ViewAndEditSlicesTool(painter,"ITK Gradient Magnitude")
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


ViewAndEditSlices::ITKBinaryDilateErodeTool::ITKBinaryDilateErodeTool(ViewAndEditSlices *painter) :
  ViewAndEditSlicesTool(painter,"ITK Fill Holes")
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


ViewAndEditSlices::ITKCurvatureAnisotropicDiffusionTool::ITKCurvatureAnisotropicDiffusionTool(ViewAndEditSlices *painter) :
  ViewAndEditSlicesTool(painter,"ITK Curvature Anisotropic Diffusion")
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

  unsigned long ptr = (unsigned long)painter_;
  nrrdKeyValueAdd(vol->nrrd_handle_->nrrd_, 
                  "progress_ptr", to_string(ptr).c_str());


  painter_->filter_ = 1;
  painter_->filter_text_ = get_name() + "\nFilter Running, Please Wait";
  painter_->redraw_all();
  painter_->want_to_execute();
}


ViewAndEditSlices::LayerMergeTool::LayerMergeTool(ViewAndEditSlices *painter):
  ViewAndEditSlicesTool(painter, "Layer Merge")
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
