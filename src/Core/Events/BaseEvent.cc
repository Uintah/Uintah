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


#include <Core/Events/BaseEvent.h>
#include <string>

namespace SCIRun {
    
using namespace std;

Persistent *make_PointerEvent() { return new PointerEvent(); }
Persistent *make_WindowEvent() { return new WindowEvent(); }
Persistent *make_KeyEvent() { return new KeyEvent(); }
Persistent *make_QuitEvent() { return new QuitEvent(); }


PersistentTypeID BaseEvent::type_id("BaseEvent", "", 0);
PersistentTypeID PointerEvent::type_id("PointerEvent", "BaseEvent", 
                                       &make_PointerEvent);
PersistentTypeID WindowEvent::type_id("WindowEvent", "BaseEvent", 
                                      &make_WindowEvent);
PersistentTypeID QuitEvent::type_id("QuitEvent", "BaseEvent",
                                    make_QuitEvent);
PersistentTypeID KeyEvent::type_id("KeyEvent", "BaseEvent",
                                   &make_KeyEvent);
PersistentTypeID EventModifiers::type_id("EventModifiers", "", 0);

BaseEvent::BaseEvent(const string &target, 
                     long int time) :
  Datatype(),
  time_(time),
  target_(target)
{
}

BaseEvent::~BaseEvent()
{
}

const int BASEEVENT_VERSION = 1;
void
BaseEvent::io(Piostream &stream) {
  stream.begin_class(type_id.type, BASEEVENT_VERSION);
  SCIRun::Pio(stream, target_);
  SCIRun::Pio(stream, time_);
  stream.end_class();
}
  
  


EventModifiers::EventModifiers() :
  modifiers_(0)
{
}

EventModifiers::~EventModifiers()
{
}

void
EventModifiers::io(Piostream &stream) {
  stream.begin_class(type_id.type, 1);
  SCIRun::Pio(stream, modifiers_);
  stream.end_class();
}


PointerEvent::PointerEvent(unsigned int state,
                           int x,
                           int y,
                           const string &target,
                           unsigned long time) :
  BaseEvent(target, time),
  p_state_(state),
  x_(x),
  y_(y)
{
}

PointerEvent::~PointerEvent()
{
}

const int POINTEREVENT_VERSION = 1;
void
PointerEvent::io(Piostream &stream) {
  stream.begin_class(type_id.type, POINTEREVENT_VERSION);
  BaseEvent::io(stream);
  EventModifiers::io(stream);
  SCIRun::Pio(stream, p_state_);
  SCIRun::Pio(stream, x_);
  SCIRun::Pio(stream, y_);
  stream.end_class();
}


KeyEvent::KeyEvent(unsigned int key_state,
                   unsigned int modifiers,
                   int keyval,
                   const string &key_string,
                   const string &target,
                   unsigned long time) :
  BaseEvent(target, time),
  k_state_(key_state),
  keyval_(keyval),
  key_str_(key_string)
{
}

KeyEvent::~KeyEvent()
{
}

const int KEYEVENT_VERSION = 1;
void
KeyEvent::io(Piostream &stream) {
  stream.begin_class(type_id.type, KEYEVENT_VERSION);
  BaseEvent::io(stream);
  EventModifiers::io(stream);
  SCIRun::Pio(stream, k_state_);
  SCIRun::Pio(stream, keyval_);
  SCIRun::Pio(stream, key_str_);
  stream.end_class();
}


WindowEvent::WindowEvent(unsigned int state,
                         const string &target,
                         unsigned long time) :
  BaseEvent(target, time),
  w_state_(state)
{
}

WindowEvent::~WindowEvent()
{
}   

const int WINDOWEVENT_VERSION = 1;
void
WindowEvent::io(Piostream &stream) {
  stream.begin_class(type_id.type, WINDOWEVENT_VERSION);
  BaseEvent::io(stream);
  SCIRun::Pio(stream, w_state_);
  stream.end_class();
}

RedrawEvent::RedrawEvent()
{

}

RedrawEvent::~RedrawEvent()
{

}

const int REDRAWEVENT_VERSION = 1;
void
RedrawEvent::io(Piostream &stream) {
  stream.begin_class(type_id.type, REDRAWEVENT_VERSION);
  BaseEvent::io(stream);
  stream.end_class();
}



AutoviewEvent::AutoviewEvent()
{

}

AutoviewEvent::~AutoviewEvent()
{

}
void
AutoviewEvent::io(Piostream &stream) {
  stream.begin_class(type_id.type, 1);
  BaseEvent::io(stream);
  stream.end_class();
}


QuitEvent::QuitEvent()
{

}

QuitEvent::~QuitEvent() 
{

}
const int QUITEVENT_VERSION = 1;
void
QuitEvent::io(Piostream &stream) {
  stream.begin_class(type_id.type, QUITEVENT_VERSION);
  BaseEvent::io(stream);
  stream.end_class();
}


TMNotifyEvent::TMNotifyEvent(const string &id,
			     unsigned int s,
			     const string &target,
			     unsigned long time) :
  BaseEvent(target, time),
  tool_id_(id),
  tool_mode_(""), // optional 
  notify_state_(s)
{
}

TMNotifyEvent::TMNotifyEvent(const TMNotifyEvent &rhs) :
  BaseEvent(rhs),
  tool_id_(rhs.tool_id_),
  tool_mode_(rhs.tool_mode_),
  notify_state_(rhs.notify_state_)
{}

TMNotifyEvent::~TMNotifyEvent()
{
}

const int TMNOTIFYEVENT_VERSION = 1;
void
TMNotifyEvent::io(Piostream &stream) {
  stream.begin_class(type_id.type, TMNOTIFYEVENT_VERSION);
  BaseEvent::io(stream);
  SCIRun::Pio(stream, tool_id_);
  SCIRun::Pio(stream, notify_state_);
  stream.end_class();
}

CommandEvent::CommandEvent(const string &target,
			   long int time) :
  BaseEvent(target, time),
  command_("")
{
}

CommandEvent::~CommandEvent()
{
}

const int COMMANDEVENT_VERSION = 1;
void
CommandEvent::io(Piostream &stream) {
  stream.begin_class(type_id.type, COMMANDEVENT_VERSION);
  BaseEvent::io(stream);
  SCIRun::Pio(stream, command_);
  stream.end_class();
}

} // namespace SCIRun

