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


Painter::BrushTool::BrushTool(Painter *painter) :
  PainterTool(painter, "Paint Brush"),
  window_(0),
  value_(airNaN()),
  last_index_(),
  radius_(5.0),
  drawing_(false)
{
  painter_->create_undo_volume();
}


Painter::BrushTool::~BrushTool()
{
}


int
Painter::BrushTool::do_event(MouseState &event)
{
  switch (event.type_) {
  case MouseState::BUTTON_PRESS_E:   return button_press(event); break;
  case MouseState::BUTTON_RELEASE_E: return button_release(event); break;
  case MouseState::MOUSE_MOTION_E:   return my_mouse_motion(event); break;
  case MouseState::KEY_PRESS_E:      return key_press(event); break;
  default:  break;
  }
  return FALLTHROUGH_E;
}

int
Painter::BrushTool::key_press(MouseState &mouse)
{
  if (mouse.key_ == " ") 
    return QUIT_E;
  return FALLTHROUGH_E;
}

int
Painter::BrushTool::button_press(MouseState &mouse)
{
  if (!mouse.keys_.empty()) 
    return FALLTHROUGH_E;

  if ( mouse.button_ == 1) {
    window_ = mouse.window_;
    if (!window_) 
      return FALLTHROUGH_E;
    if (!window_->paint_layer_) {
      window_->paint_layer_ = 
        scinew NrrdSlice(painter_, painter_->current_volume_, 
                         window_->center_, window_->normal_);
      window_->paint_layer_->bind();;
    }
    if (airIsNaN(value_))
      value_ = painter_->current_volume_->clut_max_;

    last_index_ = 
      window_->paint_layer_->volume_->world_to_index(mouse.position_);
    
    vector<int> index1(last_index_.size()-1, 0);
    for (unsigned int i = 0, j=0; i < last_index_.size(); ++i) 
      if (int(i) != (window_->axis_+1)) {
        index1[j] = last_index_[i];
        ++j;
      }
    
    splat(window_->paint_layer_->texture_->nrrd_->nrrd, radius_, 
          index1[1], index1[2]);

    window_->paint_layer_->texture_->apply_colormap(index1[1], index1[2],
                                                   index1[1]+1, index1[2]+1,
                                                   Ceil(radius_));

    //    painter_->for_each(&Painter::rebind_slice);
    painter_->redraw_all();    
    drawing_ = true;
    return HANDLED_E;
  } else if (mouse.button_ == 3) {
    vector<int> index = 
      painter_->current_volume_->world_to_index(mouse.position_);
    if (painter_->current_volume_->index_valid(index))
      painter_->current_volume_->get_value(index, value_);
    radius_ *= 1.1;
    painter_->redraw_all();    
    return HANDLED_E;
  } else if (mouse.button_ == 4) {
    radius_ *= 1.1;
    painter_->redraw_all();    
    return HANDLED_E;
  } else if (mouse.button_ == 5) {
    radius_ /= 1.1;
    painter_->redraw_all();
    return HANDLED_E;
  }

  return FALLTHROUGH_E;
}

int
Painter::BrushTool::button_release(MouseState &mouse)
{  
  if (!mouse.keys_.empty()) 
    return FALLTHROUGH_E;

  if (mouse.button_ == 1) {
    drawing_ = false;
    if (!window_ || !window_->paint_layer_) 
      return FALLTHROUGH_E;
    NrrdSlice &paint = *window_->paint_layer_;
    painter_->current_volume_->mutex_.lock();
    //    NrrdData *nout = scinew NrrdData();
    if (nrrdSplice(painter_->current_volume_->nrrd_->nrrd,
                   painter_->current_volume_->nrrd_->nrrd,
                   paint.texture_->nrrd_->nrrd,
                   window_->axis_+1,
                   last_index_[window_->axis_+1])) {
      painter_->current_volume_->mutex_.unlock();
      char *err = biffGetDone(NRRD);
      err_msg_ = string("Error on line #")+to_string(__LINE__) +
        string(" executing nrrd command: nrrdSplice \n")+
        string("Message: ")+string(err);
      free(err);
      return ERROR_E;
    }
    //    painter_->current_volume_->nrrd_ = nout;
    painter_->current_volume_->mutex_.unlock();
               
    if (window_->paint_layer_) {
      delete window_->paint_layer_;
      window_->paint_layer_ = 0;
    }
    painter_->for_each(&Painter::rebind_slice);
    painter_->redraw_all();
    return HANDLED_E;
  }
  
  return FALLTHROUGH_E;
}

int
Painter::BrushTool::my_mouse_motion(MouseState &mouse)
{
  if (!mouse.keys_.empty()) 
    return FALLTHROUGH_E;

  if (!window_) {
    drawing_ = false;
    return FALLTHROUGH_E;
  }
  
  if (!window_->paint_layer_) {
    return FALLTHROUGH_E;
  }

  NrrdSlice &paint = *window_->paint_layer_;
  NrrdVolume *volume = paint.volume_;
  
  vector<int> index = 
    volume->world_to_index(mouse.position_);
  if (!volume->index_valid(index)) 
    return FALLTHROUGH_E;
  
  if (mouse.button(1) && drawing_) {

    vector<int> index1(index.size()-1, 0);
    vector<int> index2 = index1;
    for (unsigned int i = 0, j=0; i < index.size(); ++i) 
      if (int(i) != (window_->axis_+1)) {
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
    last_index_ = index;
    return HANDLED_E;
  }

  if (mouse.button(1)) {
    last_index_ = index;
    drawing_ = true;
    return HANDLED_E;
  }
    
    return FALLTHROUGH_E;
}



string *
Painter::BrushTool::draw_mouse_cursor(MouseState &mouse)
{
  if (!mouse.window_) return 0;
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
Painter::BrushTool::splat(Nrrd *nrrd, double radius, int x0, int y0)
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
Painter::BrushTool::line(Nrrd *nrrd, double radius,
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

}
