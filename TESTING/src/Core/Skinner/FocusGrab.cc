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
//    File   : FocusGrab.cc
//    Author : McKay Davis
//    Date   : Thu Oct  5 11:50:41 2006

#include <Core/Skinner/Variables.h>
#include <Core/Skinner/FocusGrab.h>
#include <Core/Util/Assert.h>
#include <Core/Containers/StringUtil.h>

namespace SCIRun {
  namespace Skinner {
    FocusGrab::FocusGrab(Variables *variables) :
      Parent(variables),
      grab_(0)
    {
      REGISTER_CATCHER_TARGET(FocusGrab::Grab);
      REGISTER_CATCHER_TARGET(FocusGrab::Ungrab);
    }

    FocusGrab::~FocusGrab()
    {
    }

    MinMax
    FocusGrab::get_minmax(unsigned int ltype)
    {
      return SPRING_MINMAX;
    }

    BaseTool::propagation_state_e
    FocusGrab::process_event(event_handle_t event) {
      WindowEvent *window = dynamic_cast<WindowEvent *>(event.get_rep());
      if (window && window->get_window_state() == WindowEvent::REDRAW_E) {
        return Parent::process_event(event);
      }

      if (grab_) {
        return grab_->process_event(event);
      }
          
      return Parent::process_event(event);
    }


    BaseTool::propagation_state_e
    FocusGrab::Grab(event_handle_t event) {
      Signal *signal = dynamic_cast<Signal *>(event.get_rep());
      ASSERT(signal);
      Drawable *obj = dynamic_cast<Drawable *>(signal->get_signal_thrower());
      ASSERT(obj);
      grab_ = obj;
      //      cerr << "grab: " << grab_->get_id() << std::endl;
      return CONTINUE_E;
    }        

    BaseTool::propagation_state_e
    FocusGrab::Ungrab(event_handle_t event) {
      //      if (grab_) cerr << "Ungrab: " << grab_->get_id() << std::endl;
      grab_ = 0;
      return CONTINUE_E;
    }    
  }
}
