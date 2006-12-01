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
//    File   : KeyToolSelectorTool.cc
//    Author : McKay Davis
//    Date   : Sun Oct 15 11:43:45 2006

#include <StandAlone/Apps/Painter/KeyToolSelectorTool.h>
#include <StandAlone/Apps/Painter/Painter.h>
#include <StandAlone/Apps/Painter/AutoviewTool.h>
#include <StandAlone/Apps/Painter/BrushTool.h>
#include <StandAlone/Apps/Painter/CLUTLevelsTool.h>
#include <StandAlone/Apps/Painter/CropTool.h>
#include <StandAlone/Apps/Painter/FloodfillTool.h>
#include <StandAlone/Apps/Painter/PanTool.h>
#include <StandAlone/Apps/Painter/ProbeTool.h>
#include <StandAlone/Apps/Painter/StatisticsTool.h>
#include <StandAlone/Apps/Painter/ZoomTool.h>

namespace SCIRun {


KeyToolSelectorTool::KeyToolSelectorTool(Painter *painter) :
  KeyTool("Painter KeyToolSelectorTool"),
  painter_(painter),
  tm_(painter->tm_)
{
}
  

KeyToolSelectorTool::~KeyToolSelectorTool()
{
}

BaseTool::propagation_state_e
KeyToolSelectorTool::key_press(string, int keyval,
                                        unsigned int, unsigned int)
{
  if (!painter_->cur_window_) return STOP_E;
  SliceWindow &window = *painter_->cur_window_;
  if (sci_getenv_p("SCI_DEBUG"))
    cerr << "keyval: " << keyval << std::endl;

  switch (keyval) {
  case SCIRun_equal:    window.zoom_in(0); break;
  case SCIRun_minus:    window.zoom_out(0); break;
  case SCIRun_comma:    window.move_slice(-1); break;
  case SCIRun_period:   window.move_slice(1); break;
    //  case SCIRun_u:        painter_->undo_volume();
    //  case SCIRun_a:        tm_.add_tool(new CropTool(painter_),100); break;
    //  case SCIRun_f:        tm_.add_tool(new FloodfillTool(painter_),100); break;
    //  case SCIRun_b:        tm_.add_tool(new BrushTool(painter_),25); break;
    //  case SCIRun_l:        tm_.add_tool(new StatisticsTool(painter_),100); break;

  case SCIRun_c:        painter_->CopyLayer(0); break;
  case SCIRun_x:        painter_->DeleteLayer(0); break;
  case SCIRun_v:        painter_->NewLayer(0);break;

  case SCIRun_r:        painter_->reset_clut();
  case SCIRun_Left:     painter_->move_layer_down(painter_->current_volume_);break;
  case SCIRun_Right:    painter_->move_layer_up(painter_->current_volume_);break;
  case SCIRun_Up:       painter_->cur_layer_up();break;
  case SCIRun_Down:     painter_->cur_layer_down();break;

  case SCIRun_p:        painter_->opacity_up();break;
  case SCIRun_o:        painter_->opacity_down();break;

  case SCIRun_u:
    if (painter_->current_volume_.get_rep()) {
      painter_->current_volume_->colormap_ = 
        Max(0,painter_->current_volume_->colormap_-1);
      painter_->set_all_slices_tex_dirty();
      painter_->redraw_all();
    } break;
  case SCIRun_i:
    if (painter_->current_volume_.get_rep()) {
      painter_->current_volume_->colormap_ = 
        Min(int(painter_->colormaps_.size()), 
            painter_->current_volume_->colormap_+1);
      painter_->set_all_slices_tex_dirty();
      painter_->redraw_all();
    } break;    
  }

  painter_->redraw_all();
  return CONTINUE_E;
}  



BaseTool::propagation_state_e
KeyToolSelectorTool::key_release(string, int, 
                                          unsigned int, unsigned int)
{
  return CONTINUE_E;
}

} // End namespace SCIRun
