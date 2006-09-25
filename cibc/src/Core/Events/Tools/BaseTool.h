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

#include <Core/Events/share.h>
namespace SCIRun {

using namespace std;

class SCISHARE BaseTool
{
public:
  enum propagation_state_e {
    STOP_E,
    MODIFIED_E,
    CONTINUE_E,
    QUIT_AND_STOP_E,
    QUIT_AND_CONTINUE_E,
  };


  BaseTool(string name);
  virtual ~BaseTool();
  
  string get_name() const { return name_; }

  //! tools can get restarted and should reset to thier 'starting' state.
  virtual void reset() {}

  virtual propagation_state_e process_event(event_handle_t event) 
  { 
    return CONTINUE_E;
  }

  virtual void get_modified_event(event_handle_t &event)
  {
    ASSERTFAIL("BaseTool get_modified_event called");
  }

  //! The ref_cnt var so that we can have handles to this type of object.
  int ref_cnt;
protected:
  BaseTool() : ref_cnt(0), name_("") {}
private:
  string name_;
};

typedef Handle<BaseTool> tool_handle_t;

class SCISHARE PointerTool : public virtual BaseTool
{
public:
  PointerTool(string name);
  virtual ~PointerTool();
  
  //! which == button number, x,y in window at event time 
  virtual propagation_state_e pointer_down(int which, int x, int y, 
                                           unsigned int modifiers,
                                           int time)
  {
    return CONTINUE_E;
  }

  //! which == button number, x,y in window at event time 
  virtual propagation_state_e pointer_motion(int which, int x, int y, 
                                             unsigned int modifiers,
                                             int time)
  {
    return CONTINUE_E;
  }

  //! which == button number, x,y in window at event time 
  virtual propagation_state_e pointer_up(int which, int x, int y, 
                                         unsigned int modifiers,
                                         int time)
  {
    return CONTINUE_E;
  }

private:
};

class SCISHARE KeyTool :  public virtual BaseTool
{
public:
  KeyTool(string name);
  virtual ~KeyTool();
  
  virtual propagation_state_e key_press(string key, int keyval, 
                                        unsigned int modifiers, 
                                        unsigned int time)
  {
    return CONTINUE_E;
  }

  virtual propagation_state_e key_release(string key, int keyval, 
                                          unsigned int modifiers,
                                          unsigned int time)
  {
    return CONTINUE_E;
  }

private:
};

class WindowTool : public virtual BaseTool
{
public:
  WindowTool(string name);
  virtual ~WindowTool();
  
  virtual propagation_state_e create_notify(unsigned int time)
  {
    return CONTINUE_E;
  }

  virtual propagation_state_e destroy_notify(unsigned int time)
  {
    return CONTINUE_E;
  }

  virtual propagation_state_e enter_notify(unsigned int time)
  {
    return CONTINUE_E;
  }

  virtual propagation_state_e leave_notify(unsigned int time)
  {
    return CONTINUE_E;
  }

  virtual propagation_state_e expose_notify(unsigned int time)
  {
    return CONTINUE_E;
  }

  virtual propagation_state_e configure_notify(unsigned int time)
  {
    return CONTINUE_E;
  }

  virtual propagation_state_e redraw_notify(unsigned int time)
  {
    return CONTINUE_E;
  }

private:
};


//! A TMNotificationTool is used by a tool manager and pushes and pops tools
//! on its stacks based on the notifications it recieves. 
class TMNotificationTool :  public virtual BaseTool
{
public:
  TMNotificationTool(string name);
  virtual ~TMNotificationTool();


  virtual propagation_state_e start_tool(string id, unsigned int time, 
					 string mode)
  {
    return CONTINUE_E;
  }

  virtual propagation_state_e stop_tool(string id, unsigned int time)
  {
    return CONTINUE_E;
  }

  virtual propagation_state_e suspend_tool(string id, unsigned int time)
  {
    return CONTINUE_E;
  }

  virtual propagation_state_e resume_tool(string id, unsigned int time,
					  string mode)
  {
    return CONTINUE_E;
  }
  
private:
};

//! A CommandTool waits for various user defined commands, and executes
//! the command when the command is issued. 
class CommandTool :  public virtual BaseTool
{
public:
  CommandTool(string name);
  virtual ~CommandTool();

  virtual propagation_state_e issue_command(const string &cmmd, 
					    unsigned int time)
  {
    return CONTINUE_E;
  }

private:
};



} // namespace SCIRun

#endif // BaseTool_h
