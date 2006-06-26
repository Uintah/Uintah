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
//    File   : QuitMainWindow.h
//    Author : McKay Davis
//    Date   : Thu Jun  8 15:34:34 MDT 2006



#if !defined(ViewRotateTool_h)
#define ViewRotateTool_h

#include <Core/Events/Tools/ViewToolInterface.h>
#include <Core/Events/Tools/BaseTool.h>
#include <Core/Geom/View.h>
#include <Core/Events/Tools/Ball.h>
#include <Core/Events/Tools/BallMath.h>
#include <Core/Geometry/Transform.h>

namespace SCIRun {

class ViewRotateTool : public PointerTool
{
public:
  ViewRotateTool(string name, ViewToolInterface* i);
  virtual ~ViewRotateTool();

  //! which == button number, x,y in window at event time 
  virtual propagation_state_e pointer_down(int which, int x, int y, 
                                           unsigned int modifiers,
                                           int time);
  //! which == button number, x,y in window at event time 
  virtual propagation_state_e pointer_motion(int which, int x, int y, 
                                             unsigned int modifiers,
                                             int time);
  //! which == button number, x,y in window at event time 
  virtual propagation_state_e pointer_up(int which, int x, int y,
                                         unsigned int modifiers,
                                         int time);

private:
  ViewToolInterface                 *scene_interface_;
  BallData                          *ball_;
  View                               rot_view_;
  int                                last_x_;
  int                                last_y_;
  bool                               rotate_valid_p_;
  double                             eye_dist_;
  Transform                          prev_trans_;
  int			             prev_time_[3]; 
  HVect			             prev_quat_[3];
  int			             last_time_;
};

} // namespace SCIRun

#endif //ViewRotateTool_h
