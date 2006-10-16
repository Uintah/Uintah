//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : ViewTranslateTool.cc
//    Author : Martin Cole
//    Date   : Wed Jun  7 10:37:52 2006

#include <Core/Events/Tools/ViewTranslateTool.h>
#include <Core/Math/MiscMath.h>
#include <sstream>

namespace SCIRun {

ViewTranslateTool::ViewTranslateTool(string name, ViewToolInterface* i) :
  BaseTool(name),
  PointerTool(name),
  scene_interface_(i),
  start_x_(0),
  start_y_(0),
  u_(0.,0.,0.),
  v_(0.,0.,0.)
{
}

ViewTranslateTool::~ViewTranslateTool()
{
}

BaseTool::propagation_state_e
ViewTranslateTool::pointer_down(int which, int x, int y, 
                                unsigned int, int time)
{
  if (which != 1) return CONTINUE_E;
  start_x_ = x;
  start_y_ = y;
  start_view_ = scene_interface_->view_;

  double znear, zfar;
  if(!scene_interface_->compute_depth(start_view_, znear, zfar))
    return STOP_E; // No objects...

  double aspect = (double(scene_interface_->width()) / 
                   double(scene_interface_->height()));

  double zmid = (znear + zfar) / 2.;
  start_view_.get_viewplane(aspect, zmid, u_, v_);

  scene_interface_->update_mode_string("translate: ");
  return STOP_E;
}

BaseTool::propagation_state_e
ViewTranslateTool::pointer_motion(int which, int x, int y, 
                                  unsigned int, int time)
{
  if (which != 1) return CONTINUE_E;

  const double dx = double(start_x_-x) / double(scene_interface_->width());
  const double dy = double(y-start_y_) / double(scene_interface_->height());

  Vector delta = dx * u_ + dy * v_;

  scene_interface_->view_.eyep(start_view_.eyep() + delta);
  scene_interface_->view_.lookat(start_view_.lookat() + delta);

  scene_interface_->need_redraw();
  ostringstream str;
  str << "translate: " << dx << ", " << dy;
  scene_interface_->update_mode_string(str.str());
  return STOP_E;
}

BaseTool::propagation_state_e
ViewTranslateTool::pointer_up(int which, int x, int y, 
                              unsigned int, int time)
{
  if (which != 1) return CONTINUE_E;
  u_ = Vector(0.0,0.0,0.0);
  v_ = Vector(0.0,0.0,0.0);
  scene_interface_->update_mode_string("");
  //  scene_interface_->need_redraw();
  return STOP_E;
}

} // namespace SCIRun
