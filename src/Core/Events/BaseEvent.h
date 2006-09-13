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

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Thread/Mutex.h>
#include <string>

#include <Core/Events/share.h>

namespace SCIRun {

using namespace std;


class SCISHARE BaseEvent : public Datatype
{
public:
  //! target is the id of the mailbox given by EventManager, 
  //! if it is an empty string then all Event Mailboxes get the event.
  BaseEvent(const string &target = "", 
            long int time = 0);

  virtual ~BaseEvent();

  virtual BaseEvent *clone() = 0;
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  //! Accessors 
  //! time the event took place
  long int              get_time() const { return time_; }
  string                get_target() const { return target_; }

  //! Mutators
  void                  set_time(long int t) { time_ = t; }
  void                  set_target(const string &t) { target_ = t; }

  virtual bool          is_pointer_event() { return false; }
  virtual bool          is_key_event() { return false; }
  virtual bool          is_window_event() { return false; }
  virtual bool          is_scene_graph_event() { return false; }
  virtual bool          is_tm_notify_event() { return false; }
  virtual bool          is_command_event() { return false; }
private:
  //! The event timestamp
  long int              time_;
  string                target_;
};

class SCISHARE EventModifiers 
{
public: 
  EventModifiers();
  virtual ~EventModifiers();
  virtual void          io(Piostream&);
  static PersistentTypeID type_id;
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

  //! Accessors
  unsigned int          get_modifiers() const { return modifiers_; }
  //! Mutators
  void                  set_modifiers(unsigned int m) { modifiers_ = m; }
protected:
  unsigned int         modifiers_;
};

class SCISHARE PointerEvent : public BaseEvent, public EventModifiers
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
               unsigned long time = 0);

  virtual ~PointerEvent();
  virtual void          io(Piostream&);
  virtual PointerEvent *clone() { return new PointerEvent(*this); }
  static PersistentTypeID type_id;

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


class SCISHARE KeyEvent : public BaseEvent, public EventModifiers
{
public:
  //! Key state
  enum {
    KEY_PRESS_E       = 1 << 1,
    KEY_RELEASE_E     = 1 << 2,
  };

  KeyEvent(unsigned int key_state = 0,
           unsigned int modifiers = 0,
           int keyval = 0,
           const string &key_string = "",
           const string &target = "",
           unsigned long time = 0);
  virtual ~KeyEvent();

  virtual KeyEvent *    clone() { return new KeyEvent(*this); }
  virtual void          io(Piostream&);
  static PersistentTypeID type_id;

  virtual bool          is_key_event() { return true; }

  //! Accessors
  unsigned int          get_key_state() const { return k_state_; }
  int                   get_keyval() const { return keyval_; }
  string                get_key_string() const { return key_str_; }

  //! Mutators
  void                  set_key_state(unsigned int s) { k_state_ = s; }
  void                  set_keyval(int kv) { keyval_ = kv; }
  void                  set_key_string(string ks) { key_str_ = ks; }

private:
  unsigned int         k_state_;
  int                  keyval_;
  string               key_str_;
};

class SCISHARE WindowEvent : public BaseEvent 
{
public:
  enum {
    CREATE_E          = 1 << 0,
    DESTROY_E         = 1 << 1,
    ENTER_E           = 1 << 2,
    LEAVE_E           = 1 << 3,
    EXPOSE_E          = 1 << 4,
    CONFIGURE_E       = 1 << 5,
    FOCUSIN_E         = 1 << 6,
    FOCUSOUT_E        = 1 << 7,
    REDRAW_E          = 1 << 8
  };

  WindowEvent(unsigned int state = 0, 
              const string &target = "",
              unsigned long time = 0);

  virtual ~WindowEvent();
  virtual WindowEvent * clone() { return new WindowEvent(*this); }
  virtual void          io(Piostream&);
  static PersistentTypeID type_id;

  virtual bool          is_window_event() { return true; }
  
  //! Accessors
  unsigned int          get_window_state() const { return w_state_; }

  //! Mutators
  void                  set_window_state(unsigned int ws) { w_state_ = ws; }

private:
  unsigned int          w_state_;
};

class SCISHARE QuitEvent : public BaseEvent {
public:
  QuitEvent();
  virtual ~QuitEvent();
  virtual void          io(Piostream&);
  static PersistentTypeID type_id;

  virtual QuitEvent *   clone() { return new QuitEvent(*this); }
};


class SCISHARE RedrawEvent : public BaseEvent {
public:
  RedrawEvent();
  virtual ~RedrawEvent();
  virtual void          io(Piostream&);
  virtual RedrawEvent * clone() { return new RedrawEvent(*this); }
};

class SCISHARE TMNotifyEvent : public BaseEvent {
public:
  enum {
    START_E,
    STOP_E,
    SUSPEND_E,
    RESUME_E
  };

  TMNotifyEvent(const string &id = "", unsigned int s = START_E, 
		const string &target = "", unsigned long time = 0);
  virtual ~TMNotifyEvent();

  virtual void          io(Piostream&);
  virtual TMNotifyEvent * clone() { return new TMNotifyEvent(*this); }

  virtual bool          is_tm_notify_event() { return true; }

  string       get_tool_id()      const          { return tool_id_; }
  unsigned int get_notify_state() const          { return notify_state_; }

  void         set_tool_id(string id)            { tool_id_ = id; } 
  void         set_notify_state(unsigned int s)  { notify_state_ = s; }

private:
  string                    tool_id_;
  unsigned int              notify_state_;
};


//! a command event is a one time command. for example a command issued
//! from a toolbar, and has no other interaction than to do it.
class CommandEvent : public BaseEvent {
public:
  CommandEvent(const string& target = "", long int time = 0);
  virtual ~CommandEvent();
  virtual void          io(Piostream&);
  virtual CommandEvent * clone() { return new CommandEvent(*this); }

  virtual  bool is_command_event()  { return true; }

  string        get_command() const { return command_; }
  void          set_command(const string &cmmd)   { command_ = cmmd; }

private:
  string              command_;
};

typedef LockingHandle<BaseEvent> event_handle_t;


} // namespace SCIRun

#endif // BaseEvent_h
