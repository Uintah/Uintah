//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
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
//    File   : OSXEventSpawner.cc
//    Author : McKay Davis
//    Date   : Thu Sep  7 20:34:32 2006

#if defined(__APPLE__)

#include <Core/Geom/X11Lock.h>
#include <Core/Events/OSXEventSpawner.h>
#include <Core/Events/EventManager.h>
#include <Core/Events/BaseEvent.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Util/Timer.h>
#include <Core/Util/Assert.h>
#include <Core/Util/Environment.h>
#include <Core/Thread/Semaphore.h>
#include <iostream>

#include <Carbon/Carbon.h>
#include <Core/Events/keysyms.h>

namespace SCIRun {

static pascal OSStatus
window_event_callback(EventHandlerCallRef nextHandler,
               EventRef theEvent,
               void *userData)
{
  ASSERT (GetEventClass(theEvent) == kEventClassWindow);
  WindowEvent *event = new WindowEvent(WindowEvent::REDRAW_E);
  event->set_target(*((string *)userData));
  EventManager::add_event(event);

#if 0  
  switch (GetEventKind(theEvent)) {
  {
  case kEventWindowDrawContent :
  case kEventWindowBoundsChanged :
  case kEventWindowShown :
  case kEventWindowHidden :
  case kEventWindowClose : ExitToShell();
  }
#endif
  
  return (CallNextEventHandler(nextHandler, theEvent));
}


static pascal OSStatus
window_close_event_callback(EventHandlerCallRef nextHandler,
                            EventRef theEvent,
                            void *userData)
{
  ASSERT (GetEventClass(theEvent) == kEventClassWindow);
  QuitEvent *event = new QuitEvent();
  event->set_target(*((string *)userData));
  EventManager::add_event(event);
  
  return (CallNextEventHandler(nextHandler, theEvent));
}



static pascal OSStatus
quit_event_callback(EventHandlerCallRef nextHandler,
                    EventRef theEvent,
                    void *userData)
{
  cerr << "quit\n";
  QuitEvent *event = new QuitEvent();
  EventManager::add_event(event);
  
  return (CallNextEventHandler(nextHandler, theEvent));
}


static pascal OSStatus
mouse_event_callback(EventHandlerCallRef nextHandler,
                     EventRef theEvent,
                     void *userData)
{
  // Type of mouse event
  unsigned int state = 0;
  switch (GetEventKind(theEvent)) {
  case kEventMouseDown    : state = PointerEvent::BUTTON_PRESS_E; break;
  case kEventMouseUp      : state = PointerEvent::BUTTON_RELEASE_E; break;
  case kEventMouseDragged : state = PointerEvent::MOTION_E; break;
  case kEventMouseMoved   : state = PointerEvent::MOTION_E; break;
  case kEventMouseWheelMoved: {
    EventMouseWheelAxis axis;
    GetEventParameter(theEvent, kEventParamMouseWheelAxis, typeMouseWheelAxis,
                      NULL, sizeof(axis), NULL, &axis);
    // We only handle the Y for now
    if (axis != 1) {
      return (CallNextEventHandler(nextHandler, theEvent));
    }

    static bool down = true;
    // For some reason, kEventMouseWheelMoved events come in pairs, so we'll
    // just assume one is for button down, and the next is for button up
    state = (down ? PointerEvent::BUTTON_PRESS_E : 
             PointerEvent::BUTTON_RELEASE_E);
    down = !down;
    
    SInt32 delta;
    GetEventParameter(theEvent, kEventParamMouseWheelDelta, typeSInt32,
                      NULL, sizeof(delta), NULL, &delta);

    if (delta > 0) {
      state |= PointerEvent::BUTTON_4_E;
    } else if (delta < 0) {
      state |= PointerEvent::BUTTON_5_E;
    }
  } break;
    
  default: break;
  }


  // Get Buttons
  // TODO, change to mouse chord for multiple buttons
  EventMouseButton button;
  GetEventParameter(theEvent, kEventParamMouseButton, typeMouseButton,
                    NULL, sizeof(EventMouseButton), NULL, &button);

  switch (button) {
  case 1: state |= PointerEvent::BUTTON_1_E; break;
    // Swap buttons 2 and 3 for mac
  case 3: state |= PointerEvent::BUTTON_2_E; break;
  case 2: state |= PointerEvent::BUTTON_3_E; break;
  case 4: state |= PointerEvent::BUTTON_4_E; break;
  case 5: state |= PointerEvent::BUTTON_5_E; break;
  default: break;
  }


  // Get mouse cursor location
  Point	point;
  GetEventParameter(theEvent, kEventParamMouseLocation, typeQDPoint,
                    NULL, sizeof(Point), NULL, &point);

  WindowPtr window;  
  GetEventParameter(theEvent, kEventParamWindowRef, typeWindowRef,
                    NULL, sizeof(Point), NULL, &window);
  
  
  RgnHandle win_reg = NewRgn();
  GetWindowRegion(window, kWindowGlobalPortRgn, win_reg);
  Rect rect;
  GetRegionBounds(win_reg, &rect);
  DisposeRgn(win_reg);

  //Create the event
  PointerEvent *sci_event = new PointerEvent();  
  sci_event->set_pointer_state(state);
  sci_event->set_x(point.h - rect.left);
  sci_event->set_y(rect.bottom - point.v);
  
  // Send it to our own event manager
  sci_event->set_target(*((string *)userData));
  EventManager::add_event(sci_event);

  return (CallNextEventHandler(nextHandler, theEvent));
}

static pascal OSStatus
key_event_callback(EventHandlerCallRef nextHandler,
                     EventRef theEvent,
                     void *userData)
{
  KeyEvent *sci_event = new KeyEvent();

  switch (GetEventKind(theEvent)) {
  case kEventRawKeyDown: sci_event->set_key_state(KeyEvent::KEY_PRESS_E);break;
  case kEventRawKeyUp: sci_event->set_key_state(KeyEvent::KEY_RELEASE_E);break;
  default: break;
  }

  UInt32 keycode;
  GetEventParameter(theEvent, kEventParamKeyCode, typeUInt32,
                    NULL, sizeof(keycode), NULL, &keycode);

  char keychar;
  GetEventParameter(theEvent, kEventParamKeyMacCharCodes, typeChar,
                    NULL, sizeof(keychar), NULL, &keychar);


  int keyval = keychar;
  switch (keychar) {
  case 0x0D: keyval = SCIRun_Return; break;
  case 0x09: keyval = SCIRun_Tab; break;
  case 0x08: keyval = SCIRun_BackSpace; break;
  case 0x1C: keyval = SCIRun_Left; break;
  case 0x1D: keyval = SCIRun_Right; break;
  case 0x1E: keyval = SCIRun_Up; break;
  case 0x1F: keyval = SCIRun_Down; break;

  default: break;
  }

  sci_event->set_keyval(keyval);

  // Send it to our own event manager
  sci_event->set_target(*((string *)userData));
  EventManager::add_event(sci_event);
  
  return (CallNextEventHandler(nextHandler, theEvent));
}

#if 0

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
  // Type of mouse event
  unsigned int state = 0;
  switch (GetEventKind(theEvent)) {
  case kEventMouseDown    : state = PointerEvent::BUTTON_PRESS_E; break;
  case kEventMouseUp      : state = PointerEvent::BUTTON_RELEASE_E; break;
  case kEventMouseDragged : state = PointerEvent::MOTION_E; break;
  case kEventMouseMoved   : state = PointerEvent::MOTION_E; break;
  default: break;
  }


  // Get Buttons
  EventMouseButton button;
  // TODO, change to mouse chord for multiple buttons
  GetEventParameter(theEvent, kEventParamMouseButton, typeMouseButton,
                    NULL, sizeof(EventMouseButton), NULL, &button);

  switch (button) {
  case 1: state |= PointerEvent::BUTTON_1_E; break;
  case 2: state |= PointerEvent::BUTTON_2_E; break;
  case 3: state |= PointerEvent::BUTTON_3_E; break;
  case 4: state |= PointerEvent::BUTTON_4_E; break;
  case 5: state |= PointerEvent::BUTTON_5_E; break;
  default: break;
  }


  // Get mouse cursor location
  Point	point;

  WindowPtr window;  
  GetEventParameter(theEvent, kEventParamWindowRef, typeWindowRef,
                    NULL, sizeof(Point), NULL, &window);
  
  
  RgnHandle win_reg = NewRgn();
  GetWindowRegion(window, kWindowGlobalPortRgn, win_reg);
  Rect rect;
  GetRegionBounds(win_reg, &rect);
  DisposeRgn(win_reg);

  //Create the event
  PointerEvent *sci_event = new PointerEvent();  
  sci_event->set_pointer_state(state);
  sci_event->set_x(point.h - rect.left);
  sci_event->set_y(rect.bottom - point.v);
  
}
#endif


OSXEventSpawner::OSXEventSpawner(const string &target, 
                                 WindowPtr window) :
  EventSpawner(target)
{
  X11Lock::lock();
  
  InstallStandardEventHandler(GetWindowEventTarget(window));    
  
  static EventTypeSpec window_flags[] = {
    { kEventClassWindow, kEventWindowUpdate },
    { kEventClassWindow, kEventWindowBoundsChanged }
    /*
      { kEventClassWindow, kEventWindowDrawContent },
      { kEventClassWindow, kEventWindowShown },
      { kEventClassWindow, kEventWindowHidden },
      { kEventClassWindow, kEventWindowActivated },
      { kEventClassWindow, kEventWindowDeactivated },
      { kEventClassWindow, kEventWindowClose },
      
    */
  };
  
  EventHandlerUPP window_callback = NewEventHandlerUPP(window_event_callback);
  InstallWindowEventHandler(window, 
                            window_callback,
                            GetEventTypeCount(window_flags),
                            window_flags, 
                            &target_, 0L);
  DisposeEventHandlerUPP(window_callback);

  
  static EventTypeSpec mouse_flags[] = {
    { kEventClassMouse, kEventMouseDown },
    { kEventClassMouse, kEventMouseUp },
    { kEventClassMouse, kEventMouseMoved },
    { kEventClassMouse, kEventMouseDragged },
    { kEventClassMouse, kEventMouseWheelMoved } 
  };

  
  EventHandlerUPP mouse_callback = NewEventHandlerUPP(mouse_event_callback);
  InstallWindowEventHandler(window, 
                            mouse_callback, 
                            GetEventTypeCount(mouse_flags),
                            mouse_flags, 
                            &target_, 0L);
  DisposeEventHandlerUPP(mouse_callback);
  
  static EventTypeSpec key_flags[] = {
    { kEventClassKeyboard, kEventRawKeyDown },
    { kEventClassKeyboard, kEventRawKeyUp }
  };
  
  EventHandlerUPP key_callback = NewEventHandlerUPP(key_event_callback);
  InstallWindowEventHandler(window, 
                            key_callback, 
                            GetEventTypeCount(key_flags),
                            key_flags, 
                            &target_, 0L);
  DisposeEventHandlerUPP(key_callback);


  static EventTypeSpec close_flags[] = {
    { kEventClassWindow, kEventWindowClose }
  };
  
  EventHandlerUPP window_close_callback = 
    NewEventHandlerUPP(window_close_event_callback);
  InstallWindowEventHandler(window, 
                            window_close_callback, 
                            GetEventTypeCount(close_flags),
                            close_flags, 
                            &target_, 0L);
  DisposeEventHandlerUPP(window_close_callback);


  // OSX sucks, it quits the applicaiton event loop before sending 
  // these quit events!  ARGH!
#if 0
  static EventTypeSpec quit_flags[] = {
    { kEventClassApplication, kEventAppQuit },
    { kEventClassCommand, kHICommandQuit }
  };
  
  EventHandlerUPP quit_callback = 
    NewEventHandlerUPP(quit_event_callback);
  InstallWindowEventHandler(window, 
                            quit_callback, 
                            GetEventTypeCount(quit_flags),
                            quit_flags, 
                            &target_, 0L);
  DisposeEventHandlerUPP(quit_callback);
#endif


  
  X11Lock::unlock();
}

OSXEventSpawner::~OSXEventSpawner() 
{
}

bool
OSXEventSpawner::iterate()
{
  EventRef theEvent;
  EventTargetRef theTarget = GetEventDispatcherTarget();
  while (ReceiveNextEvent(0,NULL,kEventDurationForever, 
                          true, &theEvent) == noErr) {
    SendEventToEventTarget(theEvent,theTarget);
    ReleaseEvent(theEvent);
  }
  
  return true;
}
}

#endif
