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
//    File   : ViewTranslateTool.cc
//    Author : Martin Cole
//    Date   : Wed Jun  7 10:37:52 2006

#include <Core/Events/Tools/ViewTranslateTool.h>
#include <Core/Math/MiscMath.h>
#include <sstream>

namespace SCIRun {

ViewTranslateTool::ViewTranslateTool(string name, ViewToolInterface* i) :
  PointerTool(name),
  scene_interface_(i),
  last_x_(0),
  last_y_(0),
  total_x_(0.0),
  total_y_(0.0)
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
  last_x_ = x;
  last_y_ = y;
  total_x_ = 0;
  total_y_ = 0;
  scene_interface_->update_mode_string("translate: ");
  return STOP_E;
}

BaseTool::propagation_state_e
ViewTranslateTool::pointer_motion(int which, int x, int y, 
                                  unsigned int, int time)
{
  if (which != 1) return CONTINUE_E;
  int xres = scene_interface_->width();
  int yres = scene_interface_->height();
  double xmtn = double(last_x_ - x) / double(xres);
  double ymtn =- double(last_y_ - y) / double(yres);
  last_x_ = x;
  last_y_ = y;
  // Get rid of roundoff error for the display...
  if (Abs(total_x_) < .001) total_x_ = 0;
  if (Abs(total_y_) < .001) total_y_ = 0;

  View tmpview(scene_interface_->view_);
  double aspect = double(xres) / double(yres);
  double znear, zfar;
  if(!scene_interface_->compute_depth(tmpview, znear, zfar))
    return STOP_E; // No objects...
  double zmid = (znear + zfar) / 2.;
  Vector u, v;
  tmpview.get_viewplane(aspect, zmid, u, v);
  double ul = u.length();
  double vl = v.length();
  Vector trans(u * xmtn + v * ymtn);

  total_x_ += ul * xmtn;
  total_y_ += vl * ymtn;

  // Translate the view...
  tmpview.eyep(tmpview.eyep() + trans);
  tmpview.lookat(tmpview.lookat() + trans);

  // Put the view back...
  scene_interface_->view_ = tmpview;

  scene_interface_->need_redraw();
  ostringstream str;
  str << "translate: " << total_x_ << ", " << total_y_;
  scene_interface_->update_mode_string(str.str());
  return STOP_E;
}

BaseTool::propagation_state_e
ViewTranslateTool::pointer_up(int which, int x, int y, 
                              unsigned int, int time)
{
  if (which != 1) return CONTINUE_E;
  scene_interface_->update_mode_string("");
  scene_interface_->need_redraw();
  return STOP_E;
}

} // namespace SCIRun
