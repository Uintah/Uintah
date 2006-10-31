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
//    File   : CLUTLevelsTool.cc
//    Author : McKay Davis
//    Date   : Sat Oct 14 16:06:12 2006

#include <StandAlone/Apps/Painter/CLUTLevelsTool.h>
#include <StandAlone/Apps/Painter/Painter.h>

namespace SCIRun {

CLUTLevelsTool::CLUTLevelsTool(Painter *painter) : 
  PointerTool("Color Lookup Table"),
  painter_(painter),
  scale_(1.0), 
  ww_(0), 
  wl_(1.0),
  x_(0),
  y_(0)
{
}


BaseTool::propagation_state_e
CLUTLevelsTool::pointer_down(int b, int x, int y,
                                      unsigned int m, int t)
{
  NrrdVolume *vol = painter_->current_volume_;
  if (!vol || !painter_->cur_window_) {
    return CONTINUE_E;
  }

  ww_ = vol->clut_max_ - vol->clut_min_;
  wl_ = vol->clut_min_ + ww_ / 2.0;
  x_ = x;
  y_ = y;
  
  const double w = painter_->cur_window_->get_region().width();
  const double h = painter_->cur_window_->get_region().height();
  scale_ = (vol->data_max_ - vol->data_min_) / sqrt(w*w+h*h);

  return pointer_motion(b,x,y,m,t);
}


BaseTool::propagation_state_e
CLUTLevelsTool::pointer_motion(int b, int x, int y, 
                                        unsigned int m, int t)
{
  NrrdVolume *vol = painter_->current_volume_;
  if (!vol) {
    return CONTINUE_E;
  }

  const float ww = ww_+scale_*(y - y_);  
  const float wl = wl_+scale_*(x - x_);
  vol->clut_min_ = wl - ww/2.0;
  vol->clut_max_ = wl + ww/2.0;
  painter_->set_all_slices_tex_dirty();
  painter_->redraw_all();
  return STOP_E;
}


BaseTool::propagation_state_e
CLUTLevelsTool::pointer_up(int, int, int,unsigned int, int)
{
  return QUIT_AND_STOP_E;
}


}  
