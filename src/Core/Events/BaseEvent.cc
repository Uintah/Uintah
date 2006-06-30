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


void
BaseEvent::io(Piostream &stream) {
  stream.begin_class(type_id.type, 1);
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

void
PointerEvent::io(Piostream &stream) {
  stream.begin_class(type_id.type, 1);
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


void
KeyEvent::io(Piostream &stream) {
  stream.begin_class(type_id.type, 1);
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

void
WindowEvent::io(Piostream &stream) {
  stream.begin_class(type_id.type, 1);
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

void
RedrawEvent::io(Piostream &stream) {
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

void
QuitEvent::io(Piostream &stream) {
  stream.begin_class(type_id.type, 1);
  BaseEvent::io(stream);
  stream.end_class();
}

} // namespace SCIRun

