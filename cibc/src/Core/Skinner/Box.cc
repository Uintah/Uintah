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
//    File   : Box.cc
//    Author : McKay Davis
//    Date   : Tue Jun 27 12:59:23 2006

#include <Core/Skinner/Variables.h>
#include <Core/Skinner/Box.h>
#include <Core/Events/EventManager.h>
#include <Core/Containers/StringUtil.h>
#include <sci_gl.h>


namespace SCIRun {
  namespace Skinner {

    Drawable *
    Box::maker(Variables *vars) 
    {
      return new Box(vars);
    }

    Box::Box(Variables *variables) :
      Parent(variables),
      color_(variables, "color", Color(1.0, 0.0, 0.0, 1.0)),
      focus_(true)
    {
      REGISTER_CATCHER_TARGET(Box::redraw);
      REGISTER_CATCHER_TARGET(Box::do_PointerEvent);
    }

    Box::~Box()
    {
    }

    MinMax
    Box::minmax(unsigned int ltype)
    {
      return SPRING_MINMAX;
    }

    BaseTool::propagation_state_e
    Box::redraw(event_handle_t)
    {
      glColor4dv(&(color_().r));
      glBegin(GL_QUADS);

      const double *coords = get_region().coords2D();
      glVertex2dv(coords+0);
      glVertex2dv(coords+2);
      glVertex2dv(coords+4);
      glVertex2dv(coords+6);
      glEnd();
      return CONTINUE_E;
    }

    
    BaseTool::propagation_state_e
    Box::do_PointerEvent(event_handle_t event)
    {
      PointerSignal *ps = dynamic_cast<PointerSignal *>(event.get_rep());
      PointerEvent *pointer = ps->get_pointer_event();
      Signal *signal = 0;
      ASSERT(pointer);

      bool inside = get_region().inside(pointer->get_x(), pointer->get_y());
      bool pressed = 
        pointer->get_pointer_state() & PointerEvent::BUTTON_PRESS_E;
      

      if (inside && pressed &&
          (pointer->get_pointer_state() & PointerEvent::BUTTON_1_E)) 
        {
          get_vars()->insert("mousex", to_string(pointer->get_x()), "int", 1);
          get_vars()->insert("mousey", to_string(pointer->get_y()), "int", 1);

          signal = 
            dynamic_cast<Signal *>(throw_signal("button_1_clicked").get_rep());
        }
      
//       if (!inside && pressed) {
//         cerr << "Button clicked outside box\n";
//         signal = dynamic_cast<Signal *>
//           (throw_signal("button_clicked_outside_box").get_rep());
//       }


      if (inside && pressed &&
          (pointer->get_pointer_state() & PointerEvent::BUTTON_3_E)) 
        {
          get_vars()->insert("mousex", to_string(pointer->get_x()), "int", 1);
          get_vars()->insert("mousey", to_string(pointer->get_y()), "int", 1);

          signal = 
            dynamic_cast<Signal *>(throw_signal("button_2_clicked").get_rep());
        } 
        
      if ((pointer->get_pointer_state() & PointerEvent::BUTTON_RELEASE_E) &&
          (pointer->get_pointer_state() & PointerEvent::BUTTON_1_E)) {
        signal = 
          dynamic_cast<Signal *>(throw_signal("button_1_released").get_rep());
      }

      return signal ? signal->get_signal_result() : CONTINUE_E;
    }

    int
    Box::get_signal_id(const string &signalname) const {
      if (signalname == "button_1_clicked") return 1;
      if (signalname == "button_1_released") return 2;
      if (signalname == "button_2_clicked") return 3;
      //      if (signalname == "button_clicked_outside_box") return 4;
      return 0;
    }
  }

}
