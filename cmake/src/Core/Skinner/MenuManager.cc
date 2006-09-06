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
//    File   : MenuManager.cc
//    Author : McKay Davis
//    Date   : Fri Aug 11 20:45:24 2006

#include <Core/Skinner/MenuList.h>
#include <Core/Skinner/MenuManager.h>
#include <Core/Skinner/Variables.h>
#include <Core/Events/EventManager.h>
#include <Core/Geom/FontManager.h>
#include <Core/Containers/StringUtil.h>

namespace SCIRun {
  namespace Skinner {
    MenuManager::MenuManager(Variables *vars) :
      Parent(vars)
    {
      REGISTER_CATCHER_TARGET(MenuManager::MenuList_Maker);
      REGISTER_CATCHER_TARGET(MenuManager::show_MenuList);
      REGISTER_CATCHER_TARGET(MenuManager::hide_MenuList);
    }

    MenuManager::~MenuManager() {}


    BaseTool::propagation_state_e
    MenuManager::process_event(event_handle_t event)
    {
      Parent::process_event(event);

      VisibleMenuLists_t::iterator viter = visible_.begin();
      for(;viter != visible_.end(); ++viter) {
        Drawable *obj = viter->second;
        obj->get_vars()->insert("visible", "1", "string", true);
        obj->process_event(event);
        obj->get_vars()->insert("visible", "0", "string", true);
      }

      return CONTINUE_E;
    }

    BaseTool::propagation_state_e
    MenuManager::MenuList_Maker(event_handle_t maker_signal)
    {
      MenuList *menu_list = 
        construct_child_from_maker_signal<MenuList>(maker_signal);
      menu_list->get_vars()->insert("visible", "0", "string", true);
      menu_lists_.push_back(menu_list);
      return CONTINUE_E;
    }


    BaseTool::propagation_state_e
    MenuManager::show_MenuList(event_handle_t event) {
      
      Signal *signal = dynamic_cast<Signal *>(event.get_rep());
      const string id = signal->get_vars()->get_id();
      
      MenuLists_t::iterator citer = menu_lists_.begin();
      for (; citer != menu_lists_.end(); ++citer) {
        string full_id = (*citer)->get_id();
        if (full_id.find_last_of(id) != string::npos) {
          visible_.insert(make_pair(id,*citer));
          EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
          return CONTINUE_E;
        }
      }

      return STOP_E;
    }

    BaseTool::propagation_state_e
    MenuManager::hide_MenuList(event_handle_t event) {
      Signal *signal = dynamic_cast<Signal *>(event.get_rep());
      const string id = signal->get_vars()->get_id();
      if (visible_.find(id) != visible_.end()) {
        visible_.erase(id);
      }

      EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
      return CONTINUE_E;
    }
  }
}
