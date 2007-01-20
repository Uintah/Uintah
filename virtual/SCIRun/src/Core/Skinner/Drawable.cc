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
//    File   : Drawable.cc
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:04:57 2006

#include <Core/Skinner/Drawable.h>
#include <Core/Skinner/Variables.h>

namespace SCIRun {
  namespace Skinner {
    Drawable::Drawable(Variables *variables) :
      BaseTool(variables ? variables->get_id() : ""),
      SignalCatcher(),
      SignalThrower(),
      visible_(variables, "visible",1),
      class_(variables, "class", ""),
      redraw_callback_(this, class_()+"::redraw"),
      region_(),
      variables_(variables)
    {
    }
    
    Drawable::~Drawable() 
    {
      if (variables_) delete variables_;
    }

    MinMax
    Drawable::get_minmax(unsigned int)
    {
      return SPRING_MINMAX;
    }


    BaseTool::propagation_state_e
    Drawable::process_event(event_handle_t event)
    {
      PointerEvent *pointer = dynamic_cast<PointerEvent *>(event.get_rep());
      KeyEvent *key = dynamic_cast<KeyEvent *>(event.get_rep());
      WindowEvent *win = dynamic_cast<WindowEvent *>(event.get_rep());

      if (visible_.exists() && !visible_()) {
        return (pointer || key || win) ? STOP_E : CONTINUE_E;
      } 

      event_handle_t result;
      if (win && win->get_window_state() == WindowEvent::REDRAW_E) {
        result = redraw_callback_();
      } else if (pointer) {
        result = new PointerSignal(class_()+"::do_PointerEvent", pointer);
        result = SignalThrower::throw_signal(result);
      } else if (key) {
        result = new KeySignal(class_()+"::do_KeyEvent", key);
        result = SignalThrower::throw_signal(result);
      }

      if (result.get_rep()) {
        Signal *signal = dynamic_cast<Signal *>(result.get_rep());
        ASSERT(signal);
        return signal->get_signal_result();
      }

      return CONTINUE_E;
    }


    void
    Drawable::get_modified_event(event_handle_t &)
    {
    }

    string
    Drawable::get_id() const {
      return variables_->get_id();
    }

    int
    Drawable::get_signal_id(const string &name) const {
      if (name == "redraw") return 1;
      return 0;
    }


    const RectRegion &
    Drawable::get_region() const {
      return region_;
    }

    void
    Drawable::set_region(const RectRegion &region) {
      region_ = region;
    }


    Variables *
    Drawable::get_vars()
    {
      return variables_;
    }

    event_handle_t
    Drawable::throw_signal(const string &signal) {
      SignalThrower *thrower = dynamic_cast<SignalThrower *>(this);
      ASSERT(thrower);
      return thrower->throw_signal(signal, get_vars());
    }




  }
}
    
