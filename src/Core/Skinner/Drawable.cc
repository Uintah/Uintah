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
//    File   : Drawable.cc
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:04:57 2006

#include <Core/Skinner/Drawable.h>
#include <Core/Skinner/Variables.h>

namespace SCIRun {
  namespace Skinner {
    Drawable::Drawable(Variables *variables) :
      BaseTool(variables->get_id()),
      region_(),
      variables_(variables)
    {
    }
    
    Drawable::~Drawable() 
    {
      delete variables_;
    }

    MinMax
    Drawable::minmax(unsigned int)
    {
      return SPRING_MINMAX;
    }

    BaseTool::propagation_state_e
    Drawable::process_event(event_handle_t)
    {
      return STOP_E;
    }

    string
    Drawable::get_id() const {
      return variables_->get_id();
    }

    RectRegion &
    Drawable::region() {
      return region_;
    }

//     void
//     Drawable::set_vars(Variables *variables) {
//       variables_ = variables;
//     }

    Variables *
    Drawable::get_vars()
    {
      return variables_;
    }

    
    



  }
}
    
