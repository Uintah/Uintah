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
//    File   : BrushTool.h
//    Author : McKay Davis
//    Date   : Sat Oct 14 14:46:13 2006

#ifndef LEXOV_BrushTool_h
#define LEXOV_BrushTool_h

#include <Core/Events/Tools/BaseTool.h>
#include <Core/Geometry/BBox.h>
#include <Core/Skinner/Variables.h>
#include <vector>


namespace SCIRun {

using std::vector;
class Painter;
class SliceWindow;

class BrushTool : virtual public BaseTool,
                  public PointerTool {
public:
  BrushTool(Painter *painter);
  ~BrushTool();
  propagation_state_e   process_event(event_handle_t);
  propagation_state_e   pointer_motion(int b, int x, int y,
                                       unsigned m, int t);
  
  propagation_state_e   pointer_down(int b, int x, int y,
                                     unsigned m, int t);
  
  propagation_state_e   pointer_up(int b, int x, int y,
                                   unsigned m, int t);
  
  propagation_state_e   key_press(string key, int keyval,
                                  unsigned int modifiers, unsigned int time);
  
private:
  void                  draw_gl(SliceWindow &);
  void                  line(Nrrd *, double, int, int, int, int, bool);
  void                  splat(Nrrd *, double,int,int);
  
  Painter *             painter_;
  SliceWindow *         window_;
  VolumeSlice *         slice_;
  Skinner::Var<double>  value_;
  vector<int>           last_index_;
  Skinner::Var<double>  radius_;
  bool                  draw_cursor_;
  unsigned int          label_mask_;
};
  
}


#endif
