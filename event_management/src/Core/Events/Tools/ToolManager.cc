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
//    File   : ToolManager.cc
//    Author : Martin Cole
//    Date   : Thu May 25 18:52:10 2006

#include <Core/Events/Tools/ToolManager.h>
#include <Core/Events/Tools/BaseTool.h>

namespace SCIRun {


ToolManager::ToolManager(string name) :
  name_(name)
{
}

ToolManager::~ToolManager()
{
}


void 
ToolManager::add_tool(tool_handle_t t, unsigned priority) 
{
  //! get the stack for the priority, adding it if it does not yet exist.
  ts_stack_t &s  = stacks_[priority];
  s.push(t);
  //! also store off the tool by name.
  pair<tool_handle_t, unsigned> tp(t, priority);
  tools_[t->name()] = tp;
}

event_handle_t 
ToolManager::propagate_event(event_handle_t event)
{
  ps_map_t::iterator iter = stacks_.begin();
  while (iter != stacks_.end()) {
    ts_stack_t &s = (*iter).second; ++iter;
    tool_handle_t t = s.top();
    cerr << "propagating event to tool: " << t->name() << endl;
  }
  return event;
}

} // namespace SCIRun
