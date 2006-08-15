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
//    File   : Window.h
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:04:12 2006

#ifndef SKINNER_WINDOW_H
#define SKINNER_WINDOW_H

#include <Core/Skinner/Parent.h>

namespace SCIRun {
  class OpenGLContext;
  class ThrottledRunnable;
  class Thread;
  namespace Skinner {
    class GLWindow : public Parent {
    public:
      GLWindow (Variables *);
      virtual ~GLWindow();
      virtual propagation_state_e       process_event(event_handle_t);
      virtual MinMax                    get_minmax(unsigned int);
      virtual int                       get_signal_id(const string &) const;
    private:
      CatcherFunction_t                 close;
      int                               width_;
      int                               height_;
      int                               posx_;
      int                               posy_;
      bool                              border_;
      OpenGLContext *                   context_;
      ThrottledRunnable *               spawner_runnable_;
      Thread *                          spawner_thread_;
      ThrottledRunnable *               draw_runnable_;
      Thread *                          draw_thread_;
    };

  }
}

#endif
