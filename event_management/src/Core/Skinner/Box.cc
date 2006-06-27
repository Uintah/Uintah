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
#include <sci_gl.h>


namespace SCIRun {
  namespace Skinner {
    Box::Box(Variables *variables, const Color &color) :
      Drawable(variables),
      color_(color),
      backup_(1.,1.,1.,1.)
    {
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

      const double *coords = region_.coords2D();
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
      if (window && window->get_window_state() == WindowEvent::REDRAW_E)
        draw_gl();
      return CONTINUE_E;
    }


    Drawable *
    Box::maker(Variables *vars, const Drawables_t &, void *) 
    {
      Color color(0,0,0,0);
      vars->maybe_get_color("color", color);
      return new Box(vars, color);
    }

  }

}
