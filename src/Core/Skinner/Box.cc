//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
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
//    File   : Box.cc
//    Author : McKay Davis
//    Date   : Tue Jun 27 12:59:23 2006

#include <Core/Skinner/Variables.h>
#include <Core/Skinner/Box.h>
#include <Core/Events/EventManager.h>
#include <sci_gl.h>


namespace SCIRun {
  namespace Skinner {
    Box::Box(Variables *variables, const Color &color) :
      Parent(variables),
      color_(color),
      focus_mode_(false),
      focus_(true)
    {

      REGISTER_CATCHER_TARGET(Box::make_red);
      REGISTER_CATCHER_TARGET(Box::make_blue);
      REGISTER_CATCHER_TARGET(Box::make_green);

      variables->maybe_get_bool("focus_mode", focus_mode_);

    }

    Box::~Box()
    {
    }

    MinMax
    Box::minmax(unsigned int ltype)
    {
      return SPRING_MINMAX;
    }

    void
    Box::draw_gl()
    {
      glColor4dv(&color_.r);
      glBegin(GL_QUADS);

      const double *coords = get_region().coords2D();
      glVertex2dv(coords+0);
      glVertex2dv(coords+2);
      glVertex2dv(coords+4);
      glVertex2dv(coords+6);
      glEnd();
    }

    
    BaseTool::propagation_state_e
    Box::process_event(event_handle_t event)
    {
      WindowEvent *window = dynamic_cast<WindowEvent *>(event.get_rep());
      bool draw = (window && window->get_window_state() == WindowEvent::REDRAW_E);

      if (draw) {
        draw_gl();
      }

      PointerEvent *pointer = dynamic_cast<PointerEvent *>(event.get_rep());
      KeyEvent *key = dynamic_cast<KeyEvent *>(event.get_rep());

      if (pointer) {
        focus_ = get_region().inside(pointer->get_x(), pointer->get_y());
      }

        
      if (pointer && get_region().inside(pointer->get_x(), pointer->get_y()) &&
          (pointer->get_pointer_state() & PointerEvent::BUTTON_PRESS_E) &&
          (pointer->get_pointer_state() & PointerEvent::BUTTON_1_E)) 
      {
        throw_signal("button_1_clicked", get_vars());
      } 

      if (pointer && 
          (pointer->get_pointer_state() & PointerEvent::BUTTON_RELEASE_E) &&
          (pointer->get_pointer_state() & PointerEvent::BUTTON_1_E)) {
        throw_signal("button_1_released", get_vars());
      }


      bool propagate = false;
      if (!pointer && !key) {
        propagate = true;
      }
      
      propagate = (propagate || !focus_mode_ || draw || focus_) ;
        
      if (propagate) {
        return Parent::process_event(event);
      } 

      return STOP_E;
    }


    Drawable *
    Box::maker(Variables *vars) 
    {
      Color color(0,0,0,0);
      vars->maybe_get_color("color", color);
      return new Box(vars, color);
    }

    int
    Box::get_signal_id(const string &signalname) const {
      if (signalname == "button_1_clicked") return 1;
      if (signalname == "button_1_released") return 2;
      return 0;
    }

    BaseTool::propagation_state_e
    Box::make_red(event_handle_t event) {
      color_.r = 1.0;
      color_.g = 0.0;
      color_.b = 0.0;
      color_.a = 1.0;
      EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
      return BaseTool::CONTINUE_E;
    }

    BaseTool::propagation_state_e
    Box::make_blue(event_handle_t event) {
      color_.r = 0.0;
      color_.g = 0.0;
      color_.b = 1.0;
      color_.a = 1.0;
      EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
      return BaseTool::CONTINUE_E;
    }


    BaseTool::propagation_state_e
    Box::make_green(event_handle_t event) {
      color_.r = 0.0;
      color_.g = 1.0;
      color_.b = 0.0;
      color_.a = 1.0;
      EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
      return BaseTool::CONTINUE_E;
    }
  }

}
