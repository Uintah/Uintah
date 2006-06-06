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
//    File   : BaseEvent.h
//    Author : McKay Davis, Martin Cole
//    Date   : Wed May 24 07:58:40 2006


// Note: this file gets swig'd into python, 
// so please add new event types in separate files.

#if !defined(BaseEvent_h)
#define BaseEvent_h

#include <Core/Containers/LockingHandle.h>
#include <Core/Thread/Mutex.h>
#include <string>


namespace SCIRun {

using namespace std;

class BaseEvent 
{
public:
  BaseEvent(const string &target = "", 
            unsigned int time = 0);

  virtual ~BaseEvent();

  BaseEvent& operator=(const BaseEvent&);

  //! Accessors 
  //! time the event took place
  unsigned int          get_time() const { return time_; }
  string                get_target() const { return target_; }

  //! Mutators
  void                  set_time(unsigned t) { time_ = t; }
  void                  set_target(const string &t) { target_ = t; }

  virtual bool          is_pointer_event() { return false; }
  virtual bool          is_key_event() { return false; }
  virtual bool          is_window_event() { return false; }
  virtual bool          is_scene_graph_event() { return false; }

  //! The ref_cnt var so that we can have handles to this type of object.
  int ref_cnt;
  Mutex lock;
private:
  //! The event timestamp
  unsigned int          time_;
  string                target_;
};


class PointerEvent : public BaseEvent 
{
public:
  enum {
    MOTION_E          = 1 << 1,
    BUTTON_PRESS_E    = 1 << 2,
    BUTTON_RELEASE_E  = 1 << 3,
    BUTTON_1_E        = 1 << 4,
    BUTTON_2_E        = 1 << 5,
    BUTTON_3_E        = 1 << 6,
    BUTTON_4_E        = 1 << 7,
    BUTTON_5_E        = 1 << 8
  };

  PointerEvent(unsigned int state = 0, 
               int x = 0, 
               int y = 0,
               const string &target = "", 
               unsigned int time = 0);

  virtual ~PointerEvent();

  virtual bool          is_pointer_event() { return true; }

  //! Accessors
  unsigned int          get_pointer_state() const { return p_state_; }
  int                   get_x() const { return x_; }
  int                   get_y() const { return y_; }

  //! Mutators
  void                  set_pointer_state(unsigned int s) { p_state_ = s; }
  void                  set_x(int x) { x_ = x; }
  void                  set_y(int y) { y_ = y; }
private:
  unsigned int          p_state_;
  
  //! The event's window x and y  coordinates
  int                   x_;
  int                   y_;
};


class KeyEvent : public BaseEvent 
{
public:
  //! Key state
  enum {
    KEY_PRESS_E       = 1 << 1,
    KEY_RELEASE_E     = 1 << 2,
  };

  //! Modifiers
  enum {
    SHIFT_E           = 1 << 0,
    CAPS_LOCK_E       = 1 << 1,
    CONTROL_E         = 1 << 2,
    ALT_E             = 1 << 3,
    M1_E              = 1 << 4,
    M2_E              = 1 << 5,
    M3_E              = 1 << 6,
    M4_E              = 1 << 7,
  };

  KeyEvent(unsigned int key_state = 0,
           unsigned int modifiers = 0,
           int keyval = 0,
           const string &key_string = "",
           const string &target = "",
           unsigned int time = 0);
  virtual ~KeyEvent();

  virtual bool          is_key_event() { return true; }

  //! Accessors
  unsigned int          get_key_state() const { return k_state_; }
  unsigned int          get_modifiers() const { return modifiers_; }
  int                   get_keyval() const { return keyval_; }
  string                get_key_string() const { return key_str_; }

  //! Mutators
  void                  set_key_state(unsigned int s) { k_state_ = s; }
  void                  set_modifiers(unsigned int m) { modifiers_ = m; }
  void                  set_keyval(int kv) { keyval_ = kv; }
  void                  set_key_string(string ks) { key_str_ = ks; }

private:
  unsigned int         k_state_;
  unsigned int         modifiers_;
  int                  keyval_;
  string               key_str_;
};

class WindowEvent : public BaseEvent 
{
public:
  enum {
    CREATE_E          = 1 << 0,
    DESTROY_E         = 1 << 1,
    ENTER_E           = 1 << 2,
    LEAVE_E           = 1 << 3,
    EXPOSE_E          = 1 << 4,
    CONFIGURE_E       = 1 << 5,
    REDRAW_E          = 1 << 6
  };

  WindowEvent(unsigned int state = 0, 
              const string &target = "",
              unsigned int time = 0);
  virtual ~WindowEvent();

  virtual bool          is_window_event() { return true; }
  
  //! Accessors
  unsigned int          get_window_state() const { return w_state_; }

  //! Mutators
  void                  set_window_state(unsigned int ws) { w_state_ = ws; }

private:
  unsigned int          w_state_;
};

class QuitEvent : public BaseEvent {
public:
  QuitEvent() {}
  virtual ~QuitEvent() {}
};

typedef LockingHandle<BaseEvent> event_handle_t;


} // namespace SCIRun

#endif // BaseEvent_h
