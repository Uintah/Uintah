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
//    File   : AutoviewTool.h
//    Author : Martin Cole
//    Date   : Mon Jun  5 10:37:46 2006

#if !defined(AutoviewTool_h)
#define AutoviewTool_h

#include <Core/Events/Tools/BaseTool.h>
#include <Core/Geom/View.h>
#include <Core/Geometry/BBox.h>

namespace SCIRun {

//! functionality from the scene required for the AutoviewTool,
//! as well as the destination view object to modify.
class AutoviewToolInterface
{
public:
  AutoviewToolInterface(View &v);
  virtual ~AutoviewToolInterface();

  virtual bool get_bounds(BBox &bbox) const = 0;
  View &view_;
};


class AutoviewTool : public KeyTool
{
public:
  AutoviewTool(string name, AutoviewToolInterface* i);
  virtual ~AutoviewTool();

  virtual propagation_state_e  key_press(string key, int keyval, 
					 unsigned int modifiers, 
					 unsigned int time);

  virtual propagation_state_e key_release(string key, int keyval, 
                                          unsigned int modifiers, 
                                          unsigned int time)
  {
    return CONTINUE_E;
  }

  virtual propagation_state_e process_event(event_handle_t event);
    

protected:
  bool                         calc_view();

  AutoviewToolInterface                    *interface_;
};

} // namespace SCIRun

#endif //AutoviewTool_h
