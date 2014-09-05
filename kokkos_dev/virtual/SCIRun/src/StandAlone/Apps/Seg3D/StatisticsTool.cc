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
//    File   : StatisticsTool.cc
//    Author : McKay Davis
//    Date   : Sat Oct 14 16:14:41 2006

#include <StandAlone/Apps/Seg3D/StatisticsTool.h>
#include <StandAlone/Apps/Seg3D/Painter.h>

namespace SCIRun {

StatisticsTool::StatisticsTool(Painter *painter) :
  BaseTool("Statistics"),
  PointerTool("Statistics"),
  painter_(painter),
  standard_deviation_(0.0),
  mean_(0.0),
  sum_(0.0),
  squared_sum_(0.0),
  count_(0)
{
}


BaseTool::propagation_state_e
StatisticsTool::pointer_down(int b, int x, int y, unsigned int m, int t)
{
  NrrdVolumeHandle vol = painter_->current_volume_;
  cerr << "Boo!\n";
  if (vol.get_rep() && painter_->cur_window_)
  {
    sum_ = 0;
    squared_sum_ = 0;
    count_ = 0;
    mean_ = 0;
    standard_deviation_ = 0;
    

    vol->clut_min_ = vol->data_min_;
    vol->clut_max_ = vol->data_max_;
    painter_->set_all_slices_tex_dirty();
    painter_->redraw_all();
    cerr << "Blah!\n";
    return STOP_E;
  }
  return CONTINUE_E;
}



BaseTool::propagation_state_e
StatisticsTool::pointer_motion(int b, int x, int y, unsigned int m, int t)
{
  cerr << "Foo!\n";
  NrrdVolumeHandle vol = painter_->current_volume_;
  if (vol.get_rep() && painter_->cur_window_)
  {
    vector<int> index = vol->world_to_index(painter_->pointer_pos_);
    if (!vol->index_valid(index)) 
      return CONTINUE_E;

    double value;
    vol->get_value(index, value);

    sum_ += value;
    squared_sum_ += value*value;
    ++count_;

    mean_ = sum_ / count_;
    standard_deviation_ = sqrt(squared_sum_/count_-mean_*mean_);    

    vol->clut_min_ = mean_ - standard_deviation_;
    vol->clut_max_ = mean_ + standard_deviation_;

    painter_->set_all_slices_tex_dirty();
    painter_->redraw_all();
    cerr << "mean: " << mean_ << std::endl;
    cerr << "dev: " << standard_deviation_ << std::endl;
    return STOP_E;
  }
  return CONTINUE_E;
}
  

BaseTool::propagation_state_e
StatisticsTool::pointer_up(int, int, int, unsigned int, int)
{
  return QUIT_AND_STOP_E;
}
  
}  
