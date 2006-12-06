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
//    File   : ZoomTool.cc
//    Author : McKay Davis
//    Date   : Sat Oct 14 15:58:49 2006

#include <StandAlone/Apps/Painter/ZoomTool.h>
#include <StandAlone/Apps/Painter/Painter.h>

namespace SCIRun {

ZoomTool::ZoomTool(Painter *painter) : 
  PointerTool("Zoom"),
  painter_(painter),
  window_(0),
  zoom_(0.0),
  x_(0),
  y_(0)
{
}


BaseTool::propagation_state_e
ZoomTool::pointer_down(int button, int x, int y,
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
ZoomTool::pointer_motion(int button, int x, int y,
                                unsigned int modifiers,
                                int time)
{
  if (window_) {
    const int delta = x + y_ - x_ - y;
    window_->zoom_ = Max(0.00001, zoom_ * Pow(1.008,delta));
    window_->mark_redraw();
  }
  return STOP_E;
}    


BaseTool::propagation_state_e
ZoomTool::pointer_up(int, int, int, unsigned int, int)
{
  return QUIT_AND_STOP_E;
}


}  
