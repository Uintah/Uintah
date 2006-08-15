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
//    File   : Arithmetic.cc
//    Author : McKay Davis
//    Date   : Tue Aug  8 20:02:54 2006

#include <Core/Skinner/Variables.h>
#include <Core/Skinner/Arithmetic.h>
#include <Core/Events/EventManager.h>
#include <Core/Containers/StringUtil.h>


namespace SCIRun {
  namespace Skinner {

    Arithmetic::Arithmetic(Variables *variables) :
      Parent(variables)
    {
      REGISTER_CATCHER_TARGET(Arithmetic::increment);
      REGISTER_CATCHER_TARGET(Arithmetic::decrement);
    }

    Arithmetic::~Arithmetic()
    {
    }

    BaseTool::propagation_state_e
    Arithmetic::increment(event_handle_t event) {
      Skinner::Signal *signal = 
        dynamic_cast<Skinner::Signal *>(event.get_rep());
      ASSERT(signal);
      const string &varname = signal->get_vars()->get_string("variable");
      const double val = signal->get_vars()->get_double(varname) + 1.0;
      signal->get_vars()->change_parent(varname, to_string(val), "string", true);
      return BaseTool::CONTINUE_E;
    }

    BaseTool::propagation_state_e
    Arithmetic::decrement(event_handle_t event) {
      Skinner::Signal *signal = 
        dynamic_cast<Skinner::Signal *>(event.get_rep());
      ASSERT(signal);
      const string &varname = signal->get_vars()->get_string("variable");
      const double val = signal->get_vars()->get_double(varname) - 1.0;
      signal->get_vars()->change_parent(varname, to_string(val),"string",true);
      return BaseTool::CONTINUE_E;
    }

  }

}
