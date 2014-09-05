//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
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
//    File   : X11EventSpawner.h
//    Author : McKay Davis
//    Date   : Thu Jun  1 19:28 MDT 2006

#include <sci_defs/x11_defs.h>

#if defined(HAVE_X11)

#ifndef CORE_EVENTS_X11EVENTSPAWNER_H
#define CORE_EVENTS_X11EVENTSPAWNER_H

#include <Core/Thread/Runnable.h>
#include <Core/Events/BaseEvent.h>
#include <Core/Events/EventSpawner.h>
#include <X11/Xlib.h>

namespace SCIRun {
  
  class X11EventSpawner : public EventSpawner {
  public:
    X11EventSpawner(const string &target, Display *, Window);
    ~X11EventSpawner();
    virtual bool                iterate();
  private:
    event_handle_t              xlat_KeyEvent(XEvent *);
    event_handle_t              xlat_ButtonReleaseEvent(XEvent *);
    event_handle_t              xlat_ButtonPressEvent(XEvent *);
    event_handle_t              xlat_PointerMotion(XEvent *);
    event_handle_t              xlat_Expose(XEvent *);
    event_handle_t              xlat_Enter(XEvent *);
    event_handle_t              xlat_Leave(XEvent *);
    event_handle_t              xlat_Configure(XEvent *);
    event_handle_t              xlat_Client(XEvent *);
    event_handle_t              xlat_FocusIn(XEvent *);
    event_handle_t              xlat_FocusOut(XEvent *);
    event_handle_t              xlat_Visibilty(XEvent *);
    event_handle_t              xlat_Map(XEvent *);

    unsigned int                xlat_ModifierState(unsigned int);

    Display *                   display_;
    Window                      window_;
    long                        mask_;
  };
}

#endif 

#endif // defined(HAVE_X11)
