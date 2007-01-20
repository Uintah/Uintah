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
//    File   : CropTool.h
//    Author : McKay Davis
//    Date   : Sat Oct 14 12:06:30 2006

#ifndef SEG3D_CropTool_h
#define SEG3D_CropTool_h

#include <Core/Events/Tools/BaseTool.h>
#include <Core/Geometry/BBox.h>
#include <vector>


namespace SCIRun {

using std::vector;
class Painter;


class CropTool : public virtual BaseTool,
                 public PointerTool {
public:
  CropTool(Painter *painter);
  ~CropTool();
  propagation_state_e process_event(event_handle_t);
  propagation_state_e pointer_motion(int b, int x, int y,
                                     unsigned m, int t);
  propagation_state_e pointer_down(int b, int x, int y,
                                   unsigned m, int t);
  propagation_state_e pointer_up(int b, int x, int y,
                                 unsigned m, int t);
private:
  int                 draw_gl(SliceWindow &window);
  typedef vector<BBox> BBoxes;
  void                finish();
  void                set_window_cursor(SliceWindow &window, int cursor);
  void                update_to_gui();
  void                update_from_gui();
  Painter *           painter_;
  int                 pick_;
  vector<int>         minmax_[2];
  vector<Skinner::Var<double> >         gui_minmax_[2];
  vector<int>         pick_minmax_[2];
  vector<double>	pick_index_;
  double              pick_dist_[2][3];
};
  

}

#endif
