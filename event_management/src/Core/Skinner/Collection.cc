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
//    File   : Collection.cc
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:03:13 2006

#include <Core/Skinner/Variables.h>
#include <Core/Skinner/Collection.h>
#include <Core/Util/Assert.h>

namespace SCIRun {
  namespace Skinner {
    Collection::Collection(Variables *variables, 
                           const Drawables_t &children) :
      Drawable(variables),
      children_(children)
    {
    }

    Collection::~Collection()
    {
      for (Drawables_t::iterator citer = children_.begin(); 
           citer != children_.end(); ++citer)
     {
       delete *citer;
     }
    }

    MinMax
    Collection::minmax(unsigned int ltype)
    {
      return SPRING_MINMAX;
    }

    BaseTool::propagation_state_e
    Collection::process_event(event_handle_t event)
    {
      for (unsigned int i = 0; i < children_.size(); ++i) {
        ASSERT(children_[i]);
        children_[i]->region() = region_;
        children_[i]->process_event(event);
      }
      return CONTINUE_E;
    }

    Drawable *
    Collection::maker(Variables *variables,
                      const Drawables_t &children,
                      void *)
    {
      return new Collection(variables, children);
    }


  }
}
