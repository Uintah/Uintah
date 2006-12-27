//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
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
//    File   : PanTool.cc
//    Author : McKay Davis
//    Date   : Sat Oct 14 16:09:01 2006

#include <StandAlone/Apps/Seg3D/PanTool.h>
#include <StandAlone/Apps/Seg3D/Painter.h>

namespace SCIRun {

PanTool::PanTool(Painter *painter) : 
  PointerTool("Pan"),
  painter_(painter),
  x_(0),
  y_(0),
  center_(0,0,0),
  window_(0)
{
}


BaseTool::propagation_state_e
PanTool::pointer_down(int b, int x, int y, unsigned int m, int t)
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
PanTool::pointer_motion(int b, int x, int y, unsigned int m, int t)
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



BaseTool::propagation_state_e
PanTool::pointer_up(int, int, int, unsigned int, int)
{
  return QUIT_AND_STOP_E;
}


}  
