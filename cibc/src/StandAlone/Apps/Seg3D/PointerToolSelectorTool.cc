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
//    File   : PainterTools.cc
//    Author : McKay Davis
//    Date   : Sat Oct 14 16:17:09 2006

#include <StandAlone/Apps/Seg3D/PointerToolSelectorTool.h>
#include <StandAlone/Apps/Seg3D/Painter.h>
#include <StandAlone/Apps/Seg3D/AutoviewTool.h>
#include <StandAlone/Apps/Seg3D/BrushTool.h>
#include <StandAlone/Apps/Seg3D/CLUTLevelsTool.h>
#include <StandAlone/Apps/Seg3D/CropTool.h>
#include <StandAlone/Apps/Seg3D/FloodfillTool.h>
#include <StandAlone/Apps/Seg3D/PanTool.h>
#include <StandAlone/Apps/Seg3D/ProbeTool.h>
#include <StandAlone/Apps/Seg3D/StatisticsTool.h>
#include <StandAlone/Apps/Seg3D/ZoomTool.h>

namespace SCIRun {

PointerToolSelectorTool::PointerToolSelectorTool(Painter *painter) :
  BaseTool("Painter PointerToolSelectorTool"),
  PointerTool("Painter PointerToolSelectorTool"),
  painter_(painter),
  tm_(painter->tm_)
{
}
  
PointerToolSelectorTool::~PointerToolSelectorTool()
{
}


BaseTool::propagation_state_e
PointerToolSelectorTool::pointer_down(int button, int x, int y,
                                               unsigned int modifiers,
                                               int time)
{
  if (!painter_->cur_window_) return STOP_E;
  SliceWindow &window = *painter_->cur_window_;
  switch (button) {
  case 1:
    if (modifiers & EventModifiers::SHIFT_E)
      tm_.add_tool(new CLUTLevelsTool(painter_), 100);      
    else
      tm_.add_tool(new PanTool(painter_), 100);

    break;
    
  case 2:
    if (modifiers & EventModifiers::SHIFT_E)
      tm_.add_tool(new PainterAutoviewTool(painter_), 100);
    else
      tm_.add_tool(new ProbeTool(painter_), 100);    
    break;

  case 3:
    //    if (modifiers & EventModifiers::SHIFT_E)
    painter_->tm_.add_tool(new ZoomTool(painter_), 52);
    break;
  case 4:
    if (modifiers & EventModifiers::SHIFT_E)
      window.zoom_in(0);
    else
      window.move_slice(1);
    break;
    
  case 5:
    if (modifiers & EventModifiers::SHIFT_E)
      window.zoom_out(0);
    else
      window.move_slice(-1);
    break;
  default: 
    break;
  }

  return CONTINUE_E;
}


BaseTool::propagation_state_e
PointerToolSelectorTool::pointer_up(int button, int x, int y,
                                             unsigned int modifiers,
                                             int time)
{
  return CONTINUE_E;
}

BaseTool::propagation_state_e
PointerToolSelectorTool::pointer_motion(int button, int x, int y,
                                                 unsigned int modifiers,
                                                 int time)
{
  return CONTINUE_E;
}





} // End namespace SCIRun
