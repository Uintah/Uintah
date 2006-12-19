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
//    File   : FilterRedrawEvents.h
//    Author : McKay Davis
//    Date   : Thu Jun  8 15:34:34 MDT 2006



#ifndef FilterRedrawEventsTool_h
#define FilterRedrawEventsTool_h

#include <Core/Events/Tools/BaseTool.h>

namespace SCIRun {

class FilterRedrawEventsTool : public BaseTool
{
public:
  FilterRedrawEventsTool(string name, bool invert=false) : 
    BaseTool(name),
    invert_(invert) {};
  virtual ~FilterRedrawEventsTool() {}

  virtual propagation_state_e process_event(event_handle_t event) {
    WindowEvent *window = dynamic_cast<WindowEvent *>(event.get_rep());
    if (window && window->get_window_state() && WindowEvent::REDRAW_E) {
      return invert_ ? BaseTool::STOP_E : BaseTool::CONTINUE_E;
    }
    return invert_ ?  BaseTool::CONTINUE_E : BaseTool::STOP_E;
  }
private:
  bool          invert_;

};

} // namespace SCIRun

#endif //ViewRotateTool_h
