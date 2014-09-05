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
//    File   : QuitMainWindowTool.h
//    Author : McKay Davis
//    Date   : Thu Jun  8 15:34:34 MDT 2006



#ifndef QuitMainWindowTool_h
#define QuitMainWindowTool_h

#include <Core/Events/Tools/BaseTool.h>

namespace SCIRun {

class QuitMainWindowTool : public BaseTool
{
public:
  QuitMainWindowTool(const string &target) : 
    BaseTool("Quit All on "+target+" quit"),
    target_(target)
  {};
  virtual ~QuitMainWindowTool() {}

  virtual propagation_state_e process_event(event_handle_t event) {
    QuitEvent *quit = dynamic_cast<QuitEvent *>(event.get_rep());
    if (quit && quit->get_target() == target_) {
      return BaseTool::MODIFIED_E;
    }
    return BaseTool::CONTINUE_E;
  }

  virtual void get_modified_event(event_handle_t &event) {
    event->set_target("");
  }
private:
  string        target_;
};

} // namespace SCIRun

#endif //ViewRotateTool_h
