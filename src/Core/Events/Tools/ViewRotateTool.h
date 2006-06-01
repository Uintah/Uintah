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
//    File   : ViewRotateTool.h
//    Author : Martin Cole
//    Date   : Thu Jun  1 09:27:10 2006


#if !defined(ViewRotateTool_h)
#define ViewRotateTool_h

#include <Core/Events/Tools/BaseTool.h>
#include <Core/Geom/View.h>
#include <Core/Events/Tools/Ball.h>
#include <Core/Events/Tools/BallMath.h>
#include <Core/Geometry/Transform.h>

namespace SCIRun {

//! functionality from the scene required for the ViewRotateTool,
//! as well as the destination view object to modify.
class ViewToolInterface
{
public:
  ViewToolInterface(View &v);
  virtual ~ViewToolInterface();

  virtual int width() const = 0;
  virtual int height() const = 0;
  virtual bool compute_depth(const View& view, double& near, double& far) = 0;
  virtual void update_mode_string(string) const = 0;
  virtual void need_redraw() const = 0;
  View &view_;
};


class ViewRotateTool : public PointerTool
{
public:
  ViewRotateTool(string name, ViewToolInterface* i);
  virtual ~ViewRotateTool();

  //! which == button number, x,y in window at event time 
  //! return event is 0 if consumed, otherwise a valid PointerEvent.
  virtual event_handle_t pointer_down(int which, 
				      int x, int y, int time);
  //! which == button number, x,y in window at event time 
  //! return event is 0 if consumed, otherwise a valid PointerEvent.
  virtual event_handle_t pointer_motion(int which, 
					int x, int y, int time);
  //! which == button number, x,y in window at event time 
  //! return event is 0 if consumed, otherwise a valid PointerEvent.
  virtual event_handle_t pointer_up(int which, 
				    int x, int y, int time);

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
