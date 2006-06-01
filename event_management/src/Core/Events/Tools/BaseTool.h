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
//    File   : BaseTool.h
//    Author : Martin Cole
//    Date   : Thu May 25 21:03:56 2006


#if !defined(BaseTool_h)
#define BaseTool_h

#include <Core/Containers/Handle.h>
#include <Core/Events/BaseEvent.h>
#include <string>

namespace SCIRun {

using namespace std;

class BaseTool
{
public:
  BaseTool(string name);
  virtual ~BaseTool();
  
  string name() const { return name_; }

  virtual event_handle_t process_event(event_handle_t event) = 0;

  //! The ref_cnt var so that we can have handles to this type of object.
  int ref_cnt;
private:
  string name_;
};

typedef Handle<BaseTool> tool_handle_t;

class PointerTool : public BaseTool
{
public:
  PointerTool(string name);
  virtual ~PointerTool();
  
  //! which == button number, x,y in window at event time 
  //! return event is 0 if consumed, otherwise a valid PointerEvent.
  virtual event_handle_t pointer_down(int which, 
				      int x, int y, int time) = 0;
  //! which == button number, x,y in window at event time 
  //! return event is 0 if consumed, otherwise a valid PointerEvent.
  virtual event_handle_t pointer_motion(int which, 
					int x, int y, int time) = 0;
  //! which == button number, x,y in window at event time 
  //! return event is 0 if consumed, otherwise a valid PointerEvent.
  virtual event_handle_t pointer_up(int which, 
				    int x, int y, int time) = 0;
private:
};

class KeyTool : public BaseTool
{
public:
  KeyTool(string name);
  virtual ~KeyTool();
  
  virtual event_handle_t key_press(string key, int keyval, 
				   unsigned int modifiers, 
				   unsigned int time) = 0;
  virtual event_handle_t key_release(string key, int keyval, 
				     unsigned int modifiers, 
				     unsigned int time) = 0;
private:
};

class WindowTool : public BaseTool
{
public:
  WindowTool(string name);
  virtual ~WindowTool();
  
  virtual event_handle_t create_notify(unsigned int time) = 0;
  virtual event_handle_t destroy_notify(unsigned int time) = 0;
  virtual event_handle_t enter_notify(unsigned int time) = 0;
  virtual event_handle_t leave_notify(unsigned int time) = 0;
  virtual event_handle_t expose_notify(unsigned int time) = 0;
  virtual event_handle_t configure_notify(unsigned int time) = 0;
  virtual event_handle_t redraw_notify(unsigned int time) = 0;
private:
};


} // namespace SCIRun

#endif // BaseTool_h
