//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
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
//    File   : X11EventSpawner.h
//    Author : McKay Davis
//    Date   : Thu Jun  1 19:28 MDT 2006

#ifndef CORE_EVENTS_X11EVENTSPAWNER_H
#define CORE_EVENTS_X11EVENTSPAWNER_H

#include <Core/Thread/Runnable.h>
#include <Core/Events/BaseEvent.h>
#include <X11/Xlib.h>

namespace SCIRun {
   
  class X11EventSpawner : public Runnable {
  public:
    X11EventSpawner(Display *, Window);
    ~X11EventSpawner();
    virtual void                run();
  private:
    typedef event_handle_t (translate_func_t)(XEvent *);
    typedef map<unsigned int, translate_func_t *> translate_map_t;

    static translate_func_t     do_KeyEvent;
    static translate_func_t     do_ButtonEvent;
    static translate_func_t     do_PointerMotion;
    static translate_func_t     do_Expose;
    static translate_func_t     do_Enter;
    static translate_func_t     do_Leave;
    static translate_func_t     do_Configure;

    translate_map_t             translate_event_;
    Display *                   display_;
    Window                      window_;
    long                        mask_;
  };
}

#endif
