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
//    File   : ViewSubRegion.cc
//    Author : McKay Davis
//    Date   : Sat Aug 12 12:57:50 2006


#include <Core/Skinner/ViewSubRegion.h>
#include <Core/Skinner/Variables.h>
#include <sci_gl.h>

namespace SCIRun {
  namespace Skinner {
    ViewSubRegion::ViewSubRegion(Variables *vars) :
      Parent(vars)
    {
      //      REGISTER_CATCHER_TARGET(ViewSubRegion::redraw);
    }

    ViewSubRegion::~ViewSubRegion() {}
  
    BaseTool::propagation_state_e
    ViewSubRegion::process_event(event_handle_t event)
    {
      WindowEvent *window = dynamic_cast<WindowEvent *>(event.get_rep());
      if (window && window->get_window_state() == WindowEvent::REDRAW_E) {
        RectRegion rect = get_region();
        glEnable(GL_SCISSOR_TEST);
        glScissor((GLint)rect.x1(), (GLint)rect.y1(),
                  (GLint)rect.width(), (GLint)rect.height());        
        Parent::process_event(event);
        glDisable(GL_SCISSOR_TEST);
      } else {
        Parent::process_event(event);
      }
      return CONTINUE_E;
    }
  }
}
