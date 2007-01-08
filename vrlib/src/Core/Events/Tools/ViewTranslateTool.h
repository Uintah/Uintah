//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
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
//    File   : ViewTranslateTool.h
//    Author : Martin Cole
//    Date   : Wed Jun  7 10:44:24 2006


#if !defined(ViewTranslateTool_h)
#define ViewTranslateTool_h

#include <Core/Events/Tools/ViewToolInterface.h>
#include <Core/Events/Tools/BaseTool.h>
#include <Core/Geometry/Transform.h>

namespace SCIRun {

class ViewTranslateTool : public PointerTool
{
public:
  ViewTranslateTool(string name, ViewToolInterface* i);
  virtual ~ViewTranslateTool();

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
  int                                start_x_;
  int                                start_y_;
  Vector                             u_;
  Vector                             v_;
  View                               start_view_;
};

} // namespace SCIRun

#endif //ViewTranslateTool_h
