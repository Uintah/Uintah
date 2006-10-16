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
 *   Feburary, 2005
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

namespace SCIRun {


Painter::BrushTool::BrushTool(Painter *painter) :
  PainterTool(painter, "Paint Brush"),
  window_(0),
  slice_(0),
  value_(airNaN()),
  last_index_(),
  radius_(5.0)
{
  painter_->create_undo_volume();
}


Painter::BrushTool::~BrushTool()
{
}


int
Painter::BrushTool::do_event(Event &event)
{
  switch (event.type_) {
  case Event::BUTTON_PRESS_E:   return button_press(event); break;
  case Event::BUTTON_RELEASE_E: return button_release(event); break;
  case Event::MOUSE_MOTION_E:   return my_mouse_motion(event); break;
  case Event::KEY_PRESS_E:      return key_press(event); break;
  default:  break;
  }
  return FALLTHROUGH_E;
}

int
Painter::BrushTool::key_press(Event &event)
{
  if (event.key_ == " ") 
    return QUIT_E;
  if (event.key_ == "v") 
    {
      printf("Enter value: ");
      scanf("%f",&value_);
      printf("Value set to: %f\n",value_);
      return HANDLED_E;
    }
      
  return FALLTHROUGH_E;
}

int
Painter::BrushTool::button_press(Event &event)
{
  if (!event.keys_.empty()) 
    return FALLTHROUGH_E;

  NrrdVolume *vol = painter_->current_volume_;
  if (!vol) 
    return FALLTHROUGH_E;

  window_ = event.window_;
  if (!window_) 
    return FALLTHROUGH_E;
  


  if ( event.button(1)) {
    slice_ = window_->slice_map_[vol];
    if (!slice_)
      return FALLTHROUGH_E;

    if (airIsNaN(value_))
      value_ = painter_->current_volume_->clut_max_;

    last_index_ = vol->world_to_index(event.position_);
    last_index_.erase(last_index_.begin()+slice_->axis());

    splat(slice_->texture_->nrrd_handle_->nrrd_, radius_, 
          last_index_[1], last_index_[2]);

    slice_->texture_->apply_colormap(last_index_[1], last_index_[2],
                                     last_index_[1]+1, last_index_[2]+1,
                                     Ceil(radius_));
    
    painter_->redraw_all();    
    return HANDLED_E;
  } else if (event.button(3)) {
    vector<int> index = 
      vol->world_to_index(event.position_);
    if (vol->index_valid(index))
      vol->get_value(index, value_);
    painter_->redraw_all();    
    return HANDLED_E;
  } else if (event.button(4)) {
    radius_ *= 1.1;
    painter_->redraw_all();    
    return HANDLED_E;
  } else if (event.button(5)) {
    radius_ /= 1.1;
    painter_->redraw_all();
    return HANDLED_E;
  }

  return FALLTHROUGH_E;
}

int
Painter::BrushTool::button_release(Event &event)
{  
  if (!event.keys_.empty()) 
    return FALLTHROUGH_E;

  if (!window_ || !slice_) 
    return FALLTHROUGH_E;

  if (event.button(1)) {
    NrrdVolume *vol = slice_->volume_;
    ASSERT(vol);

    vector<int> window_center = vol->world_to_index(window_->center_);
    int axis = slice_->axis();
    vol->mutex_.lock();

    //    nrrdSave("/tmp/vol.nrrd", vol->nrrd_handle_->nrrd_, 0);
    //    nrrdSave("/tmp/slice.nrrd", slice_->texture_->nrrd_handle_->nrrd_, 0);
    vol->nrrd_handle_->nrrd_->content[0] = 0;
    
    if (nrrdSplice(vol->nrrd_handle_->nrrd_,
                   vol->nrrd_handle_->nrrd_,
                   slice_->texture_->nrrd_handle_->nrrd_,
                   axis, window_center[axis])) {
      vol->mutex_.unlock();
      char *err = biffGetDone(NRRD);
      err_msg_ = string("Error on line #")+to_string(__LINE__) +
        string(" executing nrrd command: nrrdSplice \n")+
        string("Message: ")+string(err);
      free(err);
      return ERROR_E;
    }
    //    painter_->current_volume_->nrrd_ = nout;
    vol->mutex_.unlock();
               
    painter_->for_each(&Painter::rebind_slice);
    painter_->redraw_all();
    slice_ = 0;
    return HANDLED_E;
  }
  
  return FALLTHROUGH_E;
}

int
Painter::BrushTool::my_mouse_motion(Event &event)
{
  if (!event.keys_.empty()) 
    return FALLTHROUGH_E;

  if (!window_)
    return FALLTHROUGH_E;
  
  if (!slice_)
    return FALLTHROUGH_E;

  NrrdVolume *vol = slice_->volume_;
  if (!vol)
    return FALLTHROUGH_E;
    
  vector<int> index = vol->world_to_index(event.position_);
  if (!vol->index_valid(index)) 
    return FALLTHROUGH_E;
  
  if (event.button(1)) {
    //    vector<int> index = = vol->world_to_index(event.position_);
    index.erase(index.begin()+slice_->axis());

    line(slice_->texture_->nrrd_handle_->nrrd_, radius_, 
         last_index_[1], last_index_[2],
         index[1], index[2], true);

    slice_->texture_->apply_colormap(last_index_[1], last_index_[2], 
                                     index[1], index[2],
                                     Ceil(radius_));
    last_index_ = index;
    painter_->redraw_all();
    return HANDLED_E;
  }

  return FALLTHROUGH_E;
}



int
Painter::BrushTool::draw_mouse_cursor(Event &event)
{
  NrrdVolume *vol = painter_->current_volume_;
  if (!event.window_ || !vol) return 0;
  glColor4f(1.0, 0.0, 0.0, 1.0);
  glLineWidth(2.0);
  glBegin(GL_LINES);

  //  int x0 = Floor(event.position_(event.window_->x_axis()));
  //  int y0 = Floor(event.position_(event.window_->y_axis()));
  //  int z0 = Floor(event.position_(event.window_->axis_));
  //  int wid = int(Ceil(radius_));


  Vector up = event.window_->y_dir();
  vector<double> upv = vol->vector_to_index(up);
  upv[max_vector_magnitude_index(upv)] /= 
    Abs(upv[max_vector_magnitude_index(upv)]);
  up =vol->index_to_vector(upv);

  Vector right = event.window_->x_dir();
  vector<double> rightv = vol->vector_to_index(right);
  rightv[max_vector_magnitude_index(rightv)] /= 
    Abs(rightv[max_vector_magnitude_index(rightv)]);
  right = vol->index_to_vector(rightv);

  Point center = vol->index_to_world(vol->world_to_index(event.position_));

  double rsquared = radius_*radius_;
  const int wid = Round(radius_);
  for (int y = -wid; y <= wid; ++y)
    for (int x = -wid; x <= wid; ++x) {
      float dist = x*x+y*y;
      if (dist > rsquared) continue;
      Point p = center + x * right + y * up;
      Point p2;      
      // right
      if ((x+1)*(x+1)+(y+0)*(y+0) > rsquared) {
        p2 = p + right;
        glVertex3dv(&p2(0));
        p2 = p + right+up;
        glVertex3dv(&p2(0));
      }

      // top
      if ((x+0)*(x+0)+(y+1)*(y+1) > rsquared) {
        p2 = p + up;
        glVertex3dv(&p2(0));
        p2 = p + right+up;
        glVertex3dv(&p2(0));
      }
      //left 
      if ((x-1)*(x-1)+(y+0)*(y+0) > rsquared) {
        glVertex3dv(&p(0));
        p2 = p + up;
        glVertex3dv(&p2(0));
      }

      // bottom
      if ((x+0)*(x+0)+(y-1)*(y-1) > rsquared) {
        glVertex3dv(&p(0));
        p2 = p + right;
        glVertex3dv(&p2(0));
      }

    }
  glEnd();
  glLineWidth(1.0);
  return 0;
}



void
Painter::BrushTool::splat(Nrrd *nrrd, double radius, int x0, int y0)
{
  vector<int> index(3,0);
  const unsigned int wid = Round(radius);
  for (unsigned int y = y0-wid; y <= y0+wid; ++y)
    for (unsigned int x = x0-wid; x <= x0+wid; ++x)
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
Painter::BrushTool::line(Nrrd *nrrd, double radius,
                     int x0, int y0, int x1, int y1, bool first)
{
  if (x0 < 0 || x0 >= (int) nrrd->axis[1].size || 
      x1 < 0 || x1 >= (int) nrrd->axis[1].size || 
      y0 < 0 || y0 >= (int) nrrd->axis[2].size || 
      y1 < 0 || y1 >= (int) nrrd->axis[2].size) return;
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

}
