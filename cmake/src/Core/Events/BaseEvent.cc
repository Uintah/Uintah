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

static int count = 0;

BaseEvent::BaseEvent(const string &target, 
                     unsigned int time) :
  Datatype(),
  //  ref_cnt(0),
  //  lock("BaseEvent lock"),
  time_(time ? time : count++),
  target_(target)
{
}

BaseEvent::~BaseEvent()
{
}

BaseEvent& 
BaseEvent::operator=(const BaseEvent& rhs)
{
  time_ = rhs.time_;
  target_ = rhs.target_;
  return *this;
}

EventModifiers::EventModifiers() :
  modifiers_(0)
{
}

EventModifiers::~EventModifiers()
{
}


PointerEvent::PointerEvent(unsigned int state,
                           int x,
                           int y,
                           const string &target,
                           unsigned int time) :
  BaseEvent(target, time),
  p_state_(state),
  x_(x),
  y_(y)
{
}

PointerEvent::~PointerEvent()
{
}

KeyEvent::KeyEvent(unsigned int key_state,
                   unsigned int modifiers,
                   int keyval,
                   const string &key_string,
                   const string &target,
                   unsigned int time) :
  BaseEvent(target, time),
  k_state_(key_state),
  keyval_(keyval),
  key_str_(key_string)
{
}

KeyEvent::~KeyEvent()
{
}

WindowEvent::WindowEvent(unsigned int state,
                         const string &target,
                         unsigned int time) :
  BaseEvent(target, time),
  w_state_(state)
{
}

WindowEvent::~WindowEvent()
{
}   

} // namespace SCIRun

