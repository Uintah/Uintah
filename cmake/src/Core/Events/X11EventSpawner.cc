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
#include <Core/Util/Environment.h>
#include <Core/Thread/Semaphore.h>
#include <sci_glx.h>
#include <iostream>

#include <X11/keysym.h>


namespace SCIRun {



  X11EventSpawner::X11EventSpawner(const string &target, 
                                   Display *display, 
                                   Window window) :
    EventSpawner(target),
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
             StructureNotifyMask |
             VisibilityChangeMask | 
             FocusChangeMask);

    X11Lock::lock();
    XSelectInput (display_, window_, mask_);
    Atom wm_client_delete = XInternAtom(display_,"WM_DELETE_WINDOW",False);
    XSetWMProtocols(display_, window_, &wm_client_delete, 1);
    XStoreName(display, window, target.c_str());
    X11Lock::unlock();

  }

  X11EventSpawner::~X11EventSpawner() 
  {
  }


  event_handle_t
  X11EventSpawner::xlat_KeyEvent(XEvent *xevent)
  {
    XKeyEvent *xeventp = &(xevent->xkey);
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
    sci_event->set_keyval(XLookupKeysym(xeventp,0));
    if (sci_getenv_p("SCI_DEBUG")) {
      cerr << "Keyval: " << sci_event->get_keyval() << std::endl;
    }

    return sci_event;
  }

  unsigned int xlate_ModifierState(unsigned int xmod) {
    unsigned int mods = 0;

    if (xmod & ShiftMask)   mods |= EventModifiers::SHIFT_E;
    if (xmod & LockMask)    mods |= EventModifiers::CAPS_LOCK_E;
    if (xmod & ControlMask) mods |= EventModifiers::CONTROL_E;
    if (xmod & Mod1Mask)    mods |= EventModifiers::ALT_E;
    
    return mods;
  }


  event_handle_t
  X11EventSpawner::xlat_ButtonPressEvent(XEvent *xevent)
  {
    XButtonEvent *xeventp = &(xevent->xbutton);
    PointerEvent *sci_event = new PointerEvent();
    unsigned int state = PointerEvent::BUTTON_PRESS_E;
    ASSERT(xeventp->type == ButtonPress);

    sci_event->set_modifiers(xlate_ModifierState(xeventp->state));

    if (xeventp->button == 1) state |= PointerEvent::BUTTON_1_E;
    if (xeventp->button == 2) state |= PointerEvent::BUTTON_2_E;
    if (xeventp->button == 3) state |= PointerEvent::BUTTON_3_E;
    if (xeventp->button == 4) state |= PointerEvent::BUTTON_4_E;
    if (xeventp->button == 5) state |= PointerEvent::BUTTON_5_E;

    sci_event->set_pointer_state(state);
    sci_event->set_x(xeventp->x);
    sci_event->set_y(xeventp->y);
    sci_event->set_time(xeventp->time);
    return sci_event;
  }

  event_handle_t
  X11EventSpawner::xlat_ButtonReleaseEvent(XEvent *xevent)
  {
    XButtonEvent *xeventp = &(xevent->xbutton);
    PointerEvent *sci_event = new PointerEvent();
    unsigned int state = PointerEvent::BUTTON_RELEASE_E;
    ASSERT(xeventp->type == ButtonRelease);
    
    sci_event->set_modifiers(xlate_ModifierState(xeventp->state));

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
  X11EventSpawner::xlat_PointerMotion(XEvent *xevent)
  {
    XMotionEvent *xeventp = &(xevent->xmotion);
    PointerEvent *sci_event = new PointerEvent();
    unsigned int state = PointerEvent::MOTION_E;

    sci_event->set_modifiers(xlate_ModifierState(xeventp->state));
    
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
  X11EventSpawner::xlat_Expose(XEvent *)
  {
    if (sci_getenv_p("SCI_DEBUG")) {
      cerr << target_ << ": Expose\n";
    }
    return new WindowEvent(WindowEvent::REDRAW_E);
  }


  event_handle_t
  X11EventSpawner::xlat_Leave(XEvent *xevent)
  {
    if (sci_getenv_p("SCI_DEBUG")) {
      cerr << target_ << ": Leave\n";
    }
    return new WindowEvent(WindowEvent::LEAVE_E, target_, 
                           xevent->xcrossing.time);
  }

  event_handle_t
  X11EventSpawner::xlat_Enter(XEvent *xevent)
  {
    if (sci_getenv_p("SCI_DEBUG")) {
      cerr << target_ << ": Enter\n";
    }
    return new WindowEvent(WindowEvent::ENTER_E, target_, 
                           xevent->xcrossing.time);
  }

  event_handle_t
  X11EventSpawner::xlat_Configure(XEvent *)
  {
    if (sci_getenv_p("SCI_DEBUG")) {
      cerr << target_ << ": Configure\n";
    }
    return new WindowEvent(WindowEvent::CONFIGURE_E);
  }

  event_handle_t
  X11EventSpawner::xlat_Client(XEvent *)
  {
    if (sci_getenv_p("SCI_DEBUG")) {
      cerr << target_ << ": Aussimg Quit\n";
    }
    //    return new WindowEvent(WindowEvent::DESTROY_E);
    return new QuitEvent();
  }


  event_handle_t
  X11EventSpawner::xlat_FocusIn(XEvent *)
  {
    if (sci_getenv_p("SCI_DEBUG")) {
      cerr << target_ << ": FocusIn\n";
    }
    return new WindowEvent(WindowEvent::FOCUSIN_E);
  }

  event_handle_t
  X11EventSpawner::xlat_FocusOut(XEvent *)
  {
    if (sci_getenv_p("SCI_DEBUG")) {
      cerr << target_ << ": FocusOut\n";
    }
    return new WindowEvent(WindowEvent::DESTROY_E);
  }

  event_handle_t
  X11EventSpawner::xlat_Visibilty(XEvent *xevent)
  {
    if (sci_getenv_p("SCI_DEBUG")) {
      cerr << target_ << ": Visibilty: " << xevent->xvisibility.state << std::endl;
    }
    return new WindowEvent(WindowEvent::REDRAW_E);
  }

  event_handle_t
  X11EventSpawner::xlat_Map(XEvent *xevent)
  {
    if (sci_getenv_p("SCI_DEBUG")) {
      cerr << target_ << ": Map" << std::endl;
    }
    return new WindowEvent(WindowEvent::CREATE_E);
  }
  
  bool
  X11EventSpawner::iterate()
  {
    XEvent *xevent = new XEvent;
    event_handle_t event = 0;
    X11Lock::lock();
    int pending = XPending(display_);
    X11Lock::unlock();
    for (int i = 0; i < pending; ++i) {
      X11Lock::lock();
      XNextEvent(display_, xevent);
      X11Lock::unlock();
      switch (xevent->type) {
      case KeyPress:		event = xlat_KeyEvent(xevent); break;
      case KeyRelease:          event = xlat_KeyEvent(xevent); break;
      case ButtonPress:         event = xlat_ButtonPressEvent(xevent); break;
      case ButtonRelease:	event = xlat_ButtonReleaseEvent(xevent); break;
      case MotionNotify:	event = xlat_PointerMotion(xevent); break;
      case Expose:		event = xlat_Expose(xevent); break;
      case EnterNotify:         event = xlat_Enter(xevent); break;
      case LeaveNotify:         event = xlat_Leave(xevent); break;
      case ConfigureNotify:	event = xlat_Configure(xevent); break;
      case ClientMessage:	event = xlat_Client(xevent); break;
      case FocusIn:             event = xlat_FocusIn(xevent); break;
      case FocusOut:            event = xlat_FocusOut(xevent); break;
      case VisibilityNotify:    event = xlat_Visibilty(xevent); break;
      case MapNotify:           event = xlat_Map(xevent); break;

      default:                  
        if (sci_getenv_p("SCI_DEBUG")) 
          cerr << target_ << ": Unknown X event type: " 
               << xevent->type << "\n"; 
        break;
      }

      if (event.get_rep()) {
        event->set_target(target_);
        EventManager::add_event(event);
      }
      event = 0;
    }
    return true;
  }
}
