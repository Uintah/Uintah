//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
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
//    File   : ViewScaleTool.cc
//    Author : Martin Cole
//    Date   : Tue Jun  6 12:31:54 2006

#include <Core/Events/Tools/ViewScaleTool.h>
#include <Core/Math/MiscMath.h>
#include <sstream>

namespace SCIRun {

ViewScaleTool::ViewScaleTool(string name, ViewToolInterface* i) :
  PointerTool(name),
  scene_interface_(i),
  last_x_(0),
  last_y_(0),
  total_scale_(1.0)
{
}

ViewScaleTool::~ViewScaleTool()
{
}

BaseTool::propagation_state_e
ViewScaleTool::pointer_down(int which, int x, int y, unsigned int, int time)
{
  if (which != 3) return CONTINUE_E;
  scene_interface_->update_mode_string("scale: ");
  last_x_ = x;
  last_y_ = y;
  total_scale_ = 1.0;
  return STOP_E;
}

BaseTool::propagation_state_e
ViewScaleTool::pointer_motion(int which, int x, int y, unsigned int, int time)
{
  if (which != 3) return CONTINUE_E;
  double scl;
  const double xmtn = (last_x_ - x) * 6.0 / scene_interface_->width();
  const double ymtn = (last_y_ - y) * 6.0 / scene_interface_->height();
  last_x_ = x;
  last_y_ = y;
  const double len = sqrt(xmtn * xmtn + ymtn * ymtn);
  if (Abs(xmtn) > Abs(ymtn)) {
    scl = xmtn; 
  } else {
    scl = ymtn;
  }
  if (scl<0) {
    scl = 1.0 / (1.0 + len); 
  } else {
    scl = len + 1.0;
  }
  total_scale_ *= scl;

  View tmpview(scene_interface_->view_);
  tmpview.eyep(tmpview.lookat() + (tmpview.eyep() - tmpview.lookat()) * scl);

  scene_interface_->view_ = tmpview;
  scene_interface_->need_redraw();
  ostringstream str;
  str << "scale: " << 100.0 / total_scale_ << "%";
  scene_interface_->update_mode_string(str.str());
  return STOP_E;
}

BaseTool::propagation_state_e
ViewScaleTool::pointer_up(int which, int x, int y, unsigned int, int time)
{
  if (which != 3) return CONTINUE_E;
  scene_interface_->update_mode_string("");
  scene_interface_->need_redraw();
  return STOP_E;
}

} // namespace SCIRun
