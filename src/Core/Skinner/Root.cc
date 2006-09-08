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
#include <Core/Skinner/FocusRegion.h>
#include <Core/Skinner/Arrow.h>
#include <Core/Skinner/Arc.h>
#include <Core/Skinner/Graph2D.h>
#include <Core/Skinner/Text.h>
#include <Core/Skinner/Arithmetic.h>
#include <Core/Skinner/MenuManager.h>
#include <Core/Skinner/ViewSubRegion.h>
#include <Core/Events/Tools/FilterRedrawEventsTool.h>
#include <iostream>

#include <algorithm>

using std::cerr;
using std::endl;

namespace SCIRun {
  namespace Skinner {
    Root::Root(Variables *variables) : 
      Parent(variables),
      windows_()
    {
      REGISTER_CATCHER_TARGET(Root::Redraw);
      REGISTER_CATCHER_TARGET(Root::Arc_Maker);
      REGISTER_CATCHER_TARGET(Root::Arrow_Maker);
      REGISTER_CATCHER_TARGET(Root::Arithmetic_Maker);
      REGISTER_CATCHER_TARGET(Root::GLWindow_Maker);
      REGISTER_CATCHER_TARGET(Root::GLWindow_Destructor);
      REGISTER_CATCHER_TARGET(Root::FocusRegion_Maker);
      REGISTER_CATCHER_TARGET(Root::Graph2D_Maker);
      //      REGISTER_CATCHER_TARGET(Root::ColorMap2D_Maker);
      REGISTER_CATCHER_TARGET(Root::MenuManager_Maker);
      REGISTER_CATCHER_TARGET(Root::Text_Maker);
      REGISTER_CATCHER_TARGET(Root::ViewSubRegion_Maker);
      REGISTER_CATCHER_TARGET_BY_NAME(Quit, Root::Quit);
    }


    Root::~Root() {
    }

    DECLARE_SKINNER_MAKER(Root, Arc);
    DECLARE_SKINNER_MAKER(Root, Arithmetic);
    DECLARE_SKINNER_MAKER(Root, Arrow);
    DECLARE_SKINNER_MAKER(Root, Text);
    DECLARE_SKINNER_MAKER(Root, Graph2D);
    DECLARE_SKINNER_MAKER(Root, ViewSubRegion);
    DECLARE_SKINNER_MAKER(Root, MenuManager);
    DECLARE_SKINNER_MAKER(Root, FocusRegion);
    //    DECLARE_SKINNER_MAKER(Root, ColorMap2D);

    BaseTool::propagation_state_e
    Root::GLWindow_Maker(event_handle_t event) {
      GLWindow *window = dynamic_cast<GLWindow *>
        (construct_class_from_maker_signal<GLWindow>(event));
      ASSERT(window);
      windows_.push_back(window);
      return STOP_E;
    }

    BaseTool::propagation_state_e
    Root::GLWindow_Destructor(event_handle_t event) {
      Signal *signal = dynamic_cast<Skinner::Signal *>(event.get_rep());
      ASSERT(signal);
      
      GLWindow *window = 
        dynamic_cast<Skinner::GLWindow *>(signal->get_signal_thrower());
      ASSERT(window);
      
      GLWindows_t::iterator iter =
        find(windows_.begin(), windows_.end(), window);
      ASSERT(iter != windows_.end());

      windows_.erase(iter);

      Drawables_t::iterator citer = 
        find(children_.begin(), children_.end(), window);
      if (citer != children_.end()) {
        children_.erase(citer);
      }

      return MODIFIED_E;
    }


    BaseTool::propagation_state_e 
    Root::Quit(event_handle_t event) {
      EventManager::add_event(new QuitEvent());
      return STOP_E;
    }

    BaseTool::propagation_state_e 
    Root::Redraw(event_handle_t event) {
      EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
      return CONTINUE_E;
    }
    

    void
    Root::spawn_redraw_threads() {
      BaseTool *event_tool = new FilterRedrawEventsTool("Redraw Filter", 1);
#if 0
      string id = get_id();
      ThrottledRunnableToolManager *runner = 
        new ThrottledRunnableToolManager(id, 120.0);

      runner->add_tool(event_tool,1);
      runner->add_tool(this, 2);
      
      id = id + " Throttled Tool Manager";
      Thread *thread = new Thread(runner, id.c_str());
      thread->detach();
#else 

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
#endif
    }
  }
}
    
