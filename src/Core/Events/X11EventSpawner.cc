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
//    File   : X11EventSpawner.cc
//    Author : McKay Davis
//    Date   : Thu Jun  1 19:28 MDT 2006

#include <Core/Geom/X11Lock.h>
#include <Core/Events/X11EventSpawner.h>
#include <Core/Events/EventManager.h>
#include <Core/Events/BaseEvent.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Util/Timer.h>
#include <Core/Util/Assert.h>
#include <sci_glx.h>
#include <iostream>


namespace SCIRun {


  X11EventSpawner::X11EventSpawner(Display *display, Window window) :
    translate_event_(),
    display_(display),
    window_(window)
  {
    mask_ = (KeyPressMask | 
             KeyReleaseMask |
             ButtonPressMask |
             ButtonReleaseMask |
             EnterWindowMask |
             LeaveWindowMask |
             PointerMotionMask |
             ExposureMask |
             StructureNotifyMask);

    XSelectInput (display_, window_, mask_);

    translate_event_[KeyPress] = &X11EventSpawner::do_KeyEvent;
    translate_event_[KeyRelease] = &X11EventSpawner::do_KeyEvent;
    translate_event_[ButtonPress] = &X11EventSpawner::do_ButtonEvent;
    translate_event_[ButtonRelease] = &X11EventSpawner::do_ButtonEvent;
    translate_event_[MotionNotify] = &X11EventSpawner::do_PointerMotion;
    translate_event_[Expose] = &X11EventSpawner::do_Expose;
    translate_event_[EnterNotify] = &X11EventSpawner::do_Enter;
    translate_event_[LeaveNotify] = &X11EventSpawner::do_Leave;
    translate_event_[ConfigureRequest] = &X11EventSpawner::do_Configure;

  }

  X11EventSpawner::~X11EventSpawner() 
  {
  }


  event_handle_t
  X11EventSpawner::do_KeyEvent(XEvent *xevent)
  {
    XKeyEvent *xeventp = (XKeyEvent*)&xevent;
    KeyEvent *sci_event = new KeyEvent();

    if (xevent->type == KeyPress)
      sci_event->set_key_state(KeyEvent::KEY_PRESS_E);
    else if (xevent->type == KeyRelease)
      sci_event->set_key_state(KeyEvent::KEY_RELEASE_E);

    unsigned int state = 0;
    if (xeventp->state & ShiftMask)    state |= KeyEvent::SHIFT_E;
    if (xeventp->state & LockMask)      state |= KeyEvent::CAPS_LOCK_E;
    if (xeventp->state & ControlMask)   state |= KeyEvent::CONTROL_E;
    if (xeventp->state & Mod1Mask)      state |= KeyEvent::ALT_E;

    sci_event->set_modifiers(state);
    sci_event->set_time(xeventp->time);

    return sci_event;
  }


  event_handle_t
  X11EventSpawner::do_ButtonEvent(XEvent *xevent)
  {
    XButtonEvent *xeventp = (XButtonEvent*)&xevent;
    PointerEvent *sci_event = new PointerEvent();
    unsigned int state = 0;
    if (xeventp->type == ButtonPress)
      state = PointerEvent::BUTTON_PRESS_E;
    else if (xeventp->type == ButtonRelease)
      state = PointerEvent::BUTTON_RELEASE_E;
    
    switch (xeventp->button) {
    case 1 : state |= PointerEvent::BUTTON_1_E; break;
    case 2 : state |= PointerEvent::BUTTON_2_E; break;
    case 3 : state |= PointerEvent::BUTTON_3_E; break;
    case 4 : state |= PointerEvent::BUTTON_4_E; break;
    case 5 : state |= PointerEvent::BUTTON_5_E; break;
    default: break;
    }
    
    sci_event->set_pointer_state(state);
    sci_event->set_x(xeventp->x);
    sci_event->set_y(xeventp->y);
    sci_event->set_time(xeventp->time);
    return sci_event;
  }


  event_handle_t
  X11EventSpawner::do_PointerMotion(XEvent *xevent)
  {
    XMotionEvent *xeventp = (XMotionEvent*)&xevent;
    PointerEvent *sci_event = new PointerEvent();
    unsigned int state = PointerEvent::MOTION_E;
    
    if (xeventp->state & Button1Mask) state |= PointerEvent::BUTTON_1_E;
    if (xeventp->state & Button2Mask) state |= PointerEvent::BUTTON_2_E;
    if (xeventp->state & Button3Mask) state |= PointerEvent::BUTTON_3_E;
    if (xeventp->state & Button4Mask) state |= PointerEvent::BUTTON_4_E;
    if (xeventp->state & Button5Mask) state |= PointerEvent::BUTTON_5_E;
    
    sci_event->set_pointer_state(state);
    sci_event->set_x(xeventp->x);
    sci_event->set_y(xeventp->y);
    sci_event->set_time(xeventp->time);
    return sci_event;
  }

  event_handle_t
  X11EventSpawner::do_Expose(XEvent *)
  {
    cerr << "Expose\n";
    return new WindowEvent(WindowEvent::REDRAW_E);
  }


  event_handle_t
  X11EventSpawner::do_Leave(XEvent *)
  {
    cerr << "Leave\n";
    return new WindowEvent(WindowEvent::LEAVE_E);
  }

  event_handle_t
  X11EventSpawner::do_Enter(XEvent *)
  {
    cerr << "Enter\n";
    return new WindowEvent(WindowEvent::ENTER_E);
  }

  event_handle_t
  X11EventSpawner::do_Configure(XEvent *)
  {
    cerr << "Configure\n";
    return new WindowEvent(WindowEvent::REDRAW_E);
  }
  
  void
  X11EventSpawner::run()
  {
    TimeThrottle throttle;
    throttle.start();

    XEvent xevent;
    for(;;)
    {
      double time = throttle.time();
     
      X11Lock::lock();
      bool found = XCheckMaskEvent(display_, mask_, &xevent);
      X11Lock::unlock();
    
      if (!found) {
        throttle.wait_for_time(time + 1/200.0); // 200 Hz X11 event rate
        continue;
      }

      translate_func_t *translator = translate_event_[xevent.type];
      if (translator) {
        event_handle_t sci_event = (*translator)(&xevent);
        ASSERT(sci_event.get_rep());
        EventManager::add_event(sci_event);
      }
    }
  }
}
