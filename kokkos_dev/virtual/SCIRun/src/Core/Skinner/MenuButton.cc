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
//    File   : MenuButton.cc
//    Author : McKay Davis
//    Date   : Mon Oct  2 18:22:42 2006

#include <Core/Skinner/MenuButton.h>
#include <Core/Skinner/Variables.h>

namespace SCIRun {
  namespace Skinner {
    MenuButton::MenuButton(Variables *vars) :
      Parent(vars)
    {
      REGISTER_CATCHER_TARGET(MenuButton::redraw);
    }

    MenuButton::~MenuButton() {}

    BaseTool::propagation_state_e
    MenuButton::redraw(event_handle_t event) {
      Var<double> x1(get_vars(), "MenuButton::x1");
      Var<double> x2(get_vars(), "MenuButton::x2");
      Var<double> y1(get_vars(), "MenuButton::y1");
      Var<double> y2(get_vars(), "MenuButton::y2");

      const RectRegion &rect = get_region();
      x1 = rect.x1();
      x2 = rect.x2();
      y1 = rect.y1();
      y2 = rect.x2();
      return CONTINUE_E;
    }
  }
}
