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
//    File   : ToolManager.h
//    Author : Martin Cole
//    Date   : Thu May 25 08:01:56 2006

#if !defined(ToolManager_h)
#define ToolManager_h

#include <Core/Events/Tools/BaseTool.h>
#include <string>
#include <stack>
#include <map>

namespace SCIRun {

class BaseEvent;

using namespace std;

//! The ToolManager holds a prioritized set of stacks of tools.  Each of the 
//! top() tools on the stacks recieve the events in priority order.  Tools 
//! deeper down the stack do not recieve events.  

class ToolManager 
{
public:  

  ToolManager(string name);
  ~ToolManager();

  //! add the tool t on the stack with the specified priority.
  //! the ToolManager takes ownership of the added, tool, calling delete
  void add_tool(tool_handle_t t, unsigned priority);

  //! propagate the event to each top() tool on all the stacks in 
  //! priority order, returning the possibly modified event.
  event_handle_t propagate_event(event_handle_t event);
  
private:

  typedef stack<tool_handle_t>                                      ts_stack_t;
  typedef map<unsigned, ts_stack_t, less<unsigned> >                ps_map_t;
  typedef map<string, pair<tool_handle_t, unsigned>, less<string> > nt_map_t;

  ps_map_t            stacks_;
  nt_map_t            tools_;

  string              name_;
};

} // namespace SCIRun
#endif // ToolManager_h
