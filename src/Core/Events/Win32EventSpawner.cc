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
//    File   : Win32EventSpawner.cc
//    Author : McKay Davis
//    Date   : Thu Jun  1 19:28 MDT 2006

#include <Core/Geom/X11Lock.h>
#include <Core/Events/Win32EventSpawner.h>
#include <Core/Events/EventManager.h>
#include <Core/Events/BaseEvent.h>
#include <Core/Events/keysyms.h>
#include <Core/Geom/Win32OpenGLContext.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Timer.h>
#include <Core/Util/Assert.h>
#include <Core/Util/Environment.h>
#include <Core/Thread/Semaphore.h>
#include <sci_glx.h>
#include <iostream>

#define MOUSE_X(lp) ((int)(short)LOWORD(lp))
#define MOUSE_Y(lp) ((int)(short)HIWORD(lp))

#ifndef BUILD_CORE_STATIC
#  define SCISHARE __declspec(dllimport)
#else
#  define SCISHARE
#endif

extern SCISHARE LRESULT CALLBACK WindowEventProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace SCIRun {

unsigned int win_ModifierState();

unsigned int win_ModifierState() {
  unsigned int mods = 0;

  BYTE keyvals[256];
  GetKeyboardState(keyvals);

  // 0x80 is the the higher bit, signifying the key is pressed
  if ( keyvals[VK_SHIFT] & 0x80) mods |= EventModifiers::SHIFT_E;
  if ( keyvals[VK_CONTROL] & 0x80) mods |= EventModifiers::CONTROL_E;
  if ( keyvals[VK_MENU] & 0x80) mods |= EventModifiers::ALT_E;   // MENU is alt...

  // 0x01 - lower bit, signifying the key is 'toggled' (caps lock, num lock, etc.)
  if ( keyvals[VK_CAPITAL] & 0x80) mods |= EventModifiers::CAPS_LOCK_E;

  return mods;
}

event_handle_t
Win32GLContextRunnable::win_KeyEvent(MSG winevent, bool pressed)
{
  // translate the message to keysym to sci logo.  Build keysym vector[256] and manually initialize it.
  MSG keymsg;
  unsigned long keyval = 0;

  switch (winevent.wParam) {
  // non ASCII keys
  case VK_CLEAR:   keyval = SCIRun_Clear; break;
  case VK_LEFT:    keyval = SCIRun_Left; break;
  case VK_RIGHT:   keyval = SCIRun_Right; break;
  case VK_UP:      keyval = SCIRun_Up; break;
  case VK_DOWN:    keyval = SCIRun_Down; break;
  case VK_HOME:    keyval = SCIRun_Home; break;
  case VK_END:     keyval = SCIRun_End; break;
  case VK_INSERT:  keyval = SCIRun_Insert; break;
  case VK_DELETE:  keyval = SCIRun_Delete; break;
  case VK_NEXT:    keyval = SCIRun_Next; break;
  
    // don't handle shift/control/alt - let the state handle that
  case VK_SHIFT:   keyval = 0; break;
  case VK_CONTROL: keyval = 0; break;
  case VK_MENU:    keyval = 0; break; // menu is ALT
  case VK_CAPITAL: keyval = 0; break;
  case VK_PAUSE:   keyval = SCIRun_Pause; break;
  case VK_PRINT:   keyval = SCIRun_Print; break;
  case VK_SELECT:  keyval = SCIRun_Select; break;
  case VK_EXECUTE: keyval = SCIRun_Execute; break;
  case VK_HELP:    keyval = SCIRun_Help; break;
  case VK_NUMLOCK: keyval = SCIRun_Num_Lock; break;
  case VK_SCROLL:  keyval = SCIRun_Scroll_Lock; break;
  case VK_BACK:    keyval = SCIRun_BackSpace; break;
  case VK_RETURN:  keyval = SCIRun_Return; break;
  case VK_ESCAPE:  keyval = SCIRun_Escape; break;
  case VK_TAB:     keyval = SCIRun_Tab; break;
  default:
    if (winevent.wParam >= VK_F1 && winevent.wParam <= VK_F24) {
      keyval = SCIRun_F1 + (winevent.wParam-VK_F1); break;
    }
    //keyval = winevent.wParam;
    //break;
    if ( TranslateMessage(&winevent)&& PeekMessage(&keymsg, context_->window_, 0, WM_USER, PM_NOREMOVE) && keymsg.message == WM_CHAR) {
      GetMessage(&keymsg, context_->window_, 0, WM_USER);
      keyval = keymsg.wParam;
      break;
    }
  }
  if (keyval != 0) {
    KeyEvent *sci_event = new KeyEvent();

#if 0
    // TranslateEvent converts shift-keys, but SCIRun events (see TextEntry.cc) want un-shifted
    if (keyval >= 'A' && keyval <= 'Z')
      keyval += 32;
    else if (keyval >= '!' && keyval <= '(')
      keyval += 16;
    else if (keyval == ')')
      keyval = '0';
#endif

    if (pressed)
      sci_event->set_key_state(KeyEvent::KEY_PRESS_E);
    else
      sci_event->set_key_state(KeyEvent::KEY_RELEASE_E);

    unsigned int state = win_ModifierState();
    sci_event->set_modifiers(state);
    sci_event->set_time(winevent.time);
    sci_event->set_keyval(keyval);
    if (sci_getenv_p("SCI_DEBUG")) {
      cerr << "Keyval: " << sci_event->get_keyval() << std::endl;
    }
    return sci_event;
  }
  return 0;
}

event_handle_t
Win32GLContextRunnable::win_ButtonPressEvent(MSG winevent, int button)
{
  // get x,y coords by GET_{X,Y}_LPARAM(lparam)
  PointerEvent *sci_event = new PointerEvent();
  unsigned int state = PointerEvent::BUTTON_PRESS_E;

  sci_event->set_modifiers(win_ModifierState());

  if (button == 1) state |= PointerEvent::BUTTON_1_E;
  if (button == 2) state |= PointerEvent::BUTTON_2_E;
  if (button == 3) state |= PointerEvent::BUTTON_3_E;
  //if (button == 4) state |= PointerEvent::BUTTON_4_E;
  //if (button == 5) state |= PointerEvent::BUTTON_5_E;

  sci_event->set_pointer_state(state);
  sci_event->set_x(MOUSE_X(winevent.lParam));
  sci_event->set_y(context_->height_ - 1 - MOUSE_Y(winevent.lParam));
  sci_event->set_time(winevent.time);
  return sci_event;
}

event_handle_t
Win32GLContextRunnable::win_ButtonReleaseEvent(MSG winevent, int button)
{
  PointerEvent *sci_event = new PointerEvent();
  unsigned int state = PointerEvent::BUTTON_RELEASE_E;
  
  sci_event->set_modifiers(win_ModifierState());

  if (button == 1) state |= PointerEvent::BUTTON_1_E;
  if (button == 2) state |= PointerEvent::BUTTON_2_E;
  if (button == 3) state |= PointerEvent::BUTTON_3_E;
  //if (button == 4) state |= PointerEvent::BUTTON_4_E;
  //if (button == 5) state |= PointerEvent::BUTTON_5_E;

  sci_event->set_pointer_state(state);
  sci_event->set_x(MOUSE_X(winevent.lParam));
  sci_event->set_y(context_->height_ - 1 - MOUSE_Y(winevent.lParam));
  sci_event->set_time(winevent.time);
  return sci_event;
}


event_handle_t
Win32GLContextRunnable::win_PointerMotion(MSG winevent)
{
  PointerEvent *sci_event = new PointerEvent();
  unsigned int state = PointerEvent::MOTION_E;

  sci_event->set_modifiers(win_ModifierState());
  
  if (winevent.wParam & MK_LBUTTON) state |= PointerEvent::BUTTON_1_E;
  if (winevent.wParam & MK_MBUTTON) state |= PointerEvent::BUTTON_2_E;
  if (winevent.wParam & MK_RBUTTON) state |= PointerEvent::BUTTON_3_E;
  //if (wineventp->state & Button4Mask) state |= PointerEvent::BUTTON_4_E;
  //if (wineventp->state & Button5Mask) state |= PointerEvent::BUTTON_5_E;
  
  sci_event->set_pointer_state(state);
  sci_event->set_x(MOUSE_X(winevent.lParam));
  sci_event->set_y(context_->height_ - 1 - MOUSE_Y(winevent.lParam));
  sci_event->set_time(winevent.time);
  return sci_event;
}

event_handle_t
Win32GLContextRunnable::win_Wheel(MSG winevent)
{
  PointerEvent *sci_event = new PointerEvent();
  unsigned int state = PointerEvent::BUTTON_PRESS_E;

  sci_event->set_modifiers(win_ModifierState());
  
  int delta = GET_WHEEL_DELTA_WPARAM(winevent.wParam);
  if (delta < 0) 
    state |= PointerEvent::BUTTON_4_E;
  else
    state |= PointerEvent::BUTTON_5_E;
  //if (winevent.wParam & MK_LBUTTON) state |= PointerEvent::BUTTON_1_E;
  //if (winevent.wParam & MK_MBUTTON) state |= PointerEvent::BUTTON_2_E;
  //if (winevent.wParam & MK_RBUTTON) state |= PointerEvent::BUTTON_3_E;
  //if (wineventp->state & Button4Mask) state |= PointerEvent::BUTTON_4_E;
  //if (wineventp->state & Button5Mask) state |= PointerEvent::BUTTON_5_E;
  
  sci_event->set_pointer_state(state);
  sci_event->set_x(MOUSE_X(winevent.lParam));
  sci_event->set_y(context_->height_ - 1 - MOUSE_Y(winevent.lParam));
  cout << "Wheel: " << MOUSE_X(winevent.lParam) << " " << context_->height_ - 1 - MOUSE_Y(winevent.lParam) << endl;
  sci_event->set_time(winevent.time);
  return sci_event;
}

event_handle_t
Win32GLContextRunnable::win_Expose(MSG)
{
  if (sci_getenv_p("SCI_DEBUG")) {
    cerr << target_ << ": Expose\n";
  }
  return new WindowEvent(WindowEvent::REDRAW_E);
}


event_handle_t
Win32GLContextRunnable::win_Leave(MSG winevent)
{
  if (sci_getenv_p("SCI_DEBUG")) {
    cerr << target_ << ": Leave\n";
  }
  return new WindowEvent(WindowEvent::LEAVE_E, target_, 
                          0/*winevent->xcrossing.time*/);
}

event_handle_t
Win32GLContextRunnable::win_Enter(MSG winevent)
{
  if (sci_getenv_p("SCI_DEBUG")) {
    cerr << target_ << ": Enter\n";
  }
  return new WindowEvent(WindowEvent::ENTER_E, target_, 
                          0/*winevent->xcrossing.time*/);
}

event_handle_t
Win32GLContextRunnable::win_Configure(MSG)
{
  if (sci_getenv_p("SCI_DEBUG")) {
    cerr << target_ << ": Configure\n";
  }
  return new WindowEvent(WindowEvent::CONFIGURE_E);
}

event_handle_t
Win32GLContextRunnable::win_Client(MSG)
{
  if (sci_getenv_p("SCI_DEBUG")) {
    cerr << target_ << ": Aussimg Quit\n";
  }
  //    return new WindowEvent(WindowEvent::DESTROY_E);
  return new QuitEvent();
}


event_handle_t
Win32GLContextRunnable::win_FocusIn(MSG)
{
  if (sci_getenv_p("SCI_DEBUG")) {
    cerr << target_ << ": FocusIn\n";
  }
  return new WindowEvent(WindowEvent::FOCUSIN_E);
}

event_handle_t
Win32GLContextRunnable::win_FocusOut(MSG)
{
  if (sci_getenv_p("SCI_DEBUG")) {
    cerr << target_ << ": FocusOut\n";
  }
  return new WindowEvent(WindowEvent::DESTROY_E);
}

event_handle_t
Win32GLContextRunnable::win_Visibilty(MSG winevent)
{
  if (sci_getenv_p("SCI_DEBUG")) {
    //cerr << target_ << ": Visibilty: " << winevent->xvisibility.state << std::endl;
  }
  return new WindowEvent(WindowEvent::REDRAW_E);
}

event_handle_t
Win32GLContextRunnable::win_Map(MSG winevent)
{
  if (sci_getenv_p("SCI_DEBUG")) {
    cerr << target_ << ": Map" << std::endl;
  }
  return new WindowEvent(WindowEvent::CREATE_E);
}
  
bool Win32GLContextRunnable::iterate()
{
  MSG msg;
  while (PeekMessage(&msg, context_->window_, 0, WM_USER, PM_NOREMOVE)) {
    GetMessage(&msg, context_->window_, 0, WM_MOUSELEAVE);
    event_handle_t event = 0;
    switch (msg.message) {
    case WM_MOVE: break;
    case WM_SIZE:  
      context_->width_ = LOWORD(msg.lParam); context_->height_ = HIWORD(msg.lParam); break;
    case WM_KEYDOWN:
      event = win_KeyEvent(msg, true); break;
    case WM_KEYUP:
      event = win_KeyEvent(msg, false); break;
    case WM_LBUTTONDOWN:
        event = win_ButtonPressEvent(msg, 1); break;
    case WM_MBUTTONDOWN:
        event = win_ButtonPressEvent(msg, 2); break;
    case WM_RBUTTONDOWN:
        event = win_ButtonPressEvent(msg, 3); break;
    case WM_LBUTTONUP:
      event = win_ButtonReleaseEvent(msg, 1); break;
    case WM_MBUTTONUP:
      event = win_ButtonReleaseEvent(msg, 2); break;
    case WM_RBUTTONUP:
      event = win_ButtonReleaseEvent(msg, 3); break;
    case WM_MOUSEWHEEL:
      event = win_Wheel(msg); break;
    case WM_MOUSEMOVE:
      // need to call 'TrackMouseEvent' so we can get a 'mouseLeave' event.  A 'mouseEnter' event will be 
      // the first motion event after the pointer has been outside
      if (!mouse_in_window) {
        mouse_in_window = true;
        event_handle_t subevent = win_Enter(msg); 
        if (subevent.get_rep()) {
          subevent->set_target(target_);
          EventManager::add_event(subevent);
        }
	TRACKMOUSEEVENT tme;
	tme.cbSize = sizeof(TRACKMOUSEEVENT);
	tme.dwFlags = TME_LEAVE;
	tme.hwndTrack = context_->window_;
	TrackMouseEvent(&tme);
      }
      event = win_PointerMotion(msg); break;
    //case WM_NCMOUSELEAVE:
    case WM_MOUSELEAVE:       event = win_Leave(msg); mouse_in_window = false; break;

    case WM_PAINT:  
    {
      PAINTSTRUCT ps;
      // must call Begin/EndPaint or we will continue to get these
      BeginPaint(context_->window_, &ps);
      event = win_Expose(msg); 
      EndPaint(context_->window_, &ps);
      break;
    }
    case WM_CLOSE: 
      event = scinew QuitEvent(); break;
    default:
      CallWindowProc(WindowEventProc, msg.hwnd, msg.message, msg.wParam, msg.lParam);
      if (sci_getenv_p("SCI_DEBUG")) 
        cerr << target_ << ": Unknown Win32 event type: " 
              << msg.message << "\n"; 
      break;
    }

    if (event.get_rep()) {
      event->set_target(target_);
      //cout << "  ES: Sending event: " << msg.message << "\n";
      EventManager::add_event(event);
    }
      event = 0;
  }
  return true;
}

void Win32GLContextRunnable::run()
{
  context_ = scinew Win32OpenGLContext(visual_, x_, y_, width_, height_, border_);
  created_.up();
  ThrottledRunnable::run();
}

Win32GLContextRunnable::Win32GLContextRunnable(const string& target, int visual, int x, int y, int width, int height, bool border) :
  visual_(visual), x_(x), y_(y), width_(width), height_(height), border_(border), created_("OpenGLWindow", 0),
  EventSpawner(target)
{
  mouse_in_window = false;
}

Win32OpenGLContext* Win32GLContextRunnable::getContext()
{
  // just wait for context to be created, and then release
  created_.down();
  created_.up();
  return context_;
}

}
