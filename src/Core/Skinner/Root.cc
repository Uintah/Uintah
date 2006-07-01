//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
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
//    File   : Root.cc
//    Author : McKay Davis
//    Date   : Fri Jun 30 22:10:07 2006
#include <Core/Skinner/Skinner.h>
#include <Core/Skinner/Root.h>
#include <Core/Skinner/Variables.h>
#include <Core/Skinner/Window.h>
#include <Core/Events/Tools/FilterRedrawEventsTool.h>
#include <iostream>

using std::cerr;
using std::endl;

namespace SCIRun {
  namespace Skinner {
    Root::Root(Variables *variables) : 
      Parent(variables),
      windows_()
    {
      REGISTER_CATCHER_TARGET(Root::GLWindow_Maker);
    }

    Root::~Root() {
    }

    BaseTool::propagation_state_e
    Root::GLWindow_Maker(event_handle_t event) {
      MakerSignal *maker_signal = 
        dynamic_cast<Skinner::MakerSignal *>(event.get_rep());
      ASSERT(maker_signal);
      
      windows_.push_back(new GLWindow(maker_signal->get_vars()));

      maker_signal->set_signal_thrower(windows_.back());
      maker_signal->set_signal_name(maker_signal->get_signal_name()+"_Done");
      return MODIFIED_E;
    }

    void
    Root::spawn_redraw_threads() {
      BaseTool *event_tool = new FilterRedrawEventsTool("Redraw Filter", 1);
      for (unsigned int w = 0; w < windows_.size(); ++w) {
        GLWindow *window = windows_[w];
        string id = window->get_id();
        ThrottledRunnableToolManager *runner = 
          new ThrottledRunnableToolManager(id, 120.0);
        runner->add_tool(event_tool,1);
        runner->add_tool(window, 2);
        id = id + " Throttled Tool Manager";
        Thread *thread = new Thread(runner, id.c_str());
        thread->detach();
      }
    }
  }
}
    
