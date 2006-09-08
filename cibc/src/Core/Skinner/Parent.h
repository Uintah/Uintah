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
//    File   : Parent.h
//    Author : McKay Davis
//    Date   : Thu Jun 29 19:23:31 2006

#ifndef SKINNER_CONTAINER_H
#define SKINNER_CONTAINER_H

#include <Core/Skinner/Drawable.h>

#include <Core/Skinner/share.h>

namespace SCIRun {
  namespace Skinner {
    class SCISHARE Parent : public Drawable {
    public:
      Parent(Variables *);
      virtual ~Parent();
      virtual propagation_state_e       process_event(event_handle_t e);
      virtual int                       get_signal_id(const string &);
      virtual MinMax                    get_minmax(unsigned int);
      virtual void                      set_children(const Drawables_t &);
    protected:
      Drawables_t                       children_;
    };
  }
}
#endif

