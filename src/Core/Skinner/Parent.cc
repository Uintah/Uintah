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
//    File   : Parent.cc
//    Author : McKay Davis
//    Date   : Thu Jun 29 19:23:07 2006

#include <Core/Skinner/Parent.h>


namespace SCIRun {
  namespace Skinner {
    Parent::Parent(Variables *vars) :
      Drawable(vars),
      children_()
    {
    }
    
    Parent::~Parent() {
      for (Drawables_t::iterator citer = children_.begin(); 
           citer != children_.end(); ++citer)
     {
       delete *citer;
     }
    }

    BaseTool::propagation_state_e
    Parent::process_event(event_handle_t e) {
      BaseTool::propagation_state_e state = Drawable::process_event(e);
      if (state != STOP_E) {
        for (Drawables_t::iterator citer = children_.begin(); 
             citer != children_.end(); ++citer)
        {
          (*citer)->set_region(get_region());
          (*citer)->process_event(e);
        }
      }
      return state;
    }

    int
    Parent::get_signal_id(const string &str) {
      return Drawable::get_signal_id(str);
    }


    MinMax
    Parent::get_minmax(unsigned int dir) {
      return Drawable::get_minmax(dir);
    }

    void
    Parent::set_children(const Drawables_t &children) {
      children_ = children;
    }
  }
}
    
