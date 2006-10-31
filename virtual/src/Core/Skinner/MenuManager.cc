//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
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
#include <Core/Skinner/MenuButton.h>
#include <Core/Skinner/MenuManager.h>
#include <Core/Skinner/Variables.h>
#include <Core/Events/EventManager.h>
#include <Core/Geom/FontManager.h>
#include <Core/Containers/StringUtil.h>

namespace SCIRun {
  namespace Skinner {
    MenuManager::MenuManager(Variables *vars) :
      Parent(vars),
      mutex_("MenuManager"),
      menu_lists_(),
      menu_buttons_(),
      visible_menulists_()
    {
      REGISTER_CATCHER_TARGET(MenuManager::MenuList_Maker);
      REGISTER_CATCHER_TARGET(MenuManager::MenuButton_Maker);
      REGISTER_CATCHER_TARGET(MenuManager::show_MenuList);
      REGISTER_CATCHER_TARGET(MenuManager::hide_MenuList);
    }

    MenuManager::~MenuManager() {}


    BaseTool::propagation_state_e
    MenuManager::process_event(event_handle_t event)
    {
      WindowEvent *window = dynamic_cast<WindowEvent *>(event.get_rep());
      bool redraw = window && window->get_window_state() == WindowEvent::REDRAW_E;

      //      PointerEvent *pointer = dynamic_cast<PointerEvent *>(event.get_rep());
      //      bool click = pointer && (pointer->get_pointer_state() & PointerEvent::BUTTON_PRESS_E) ;

      if (redraw) {
        Parent::process_event(event);
      }

      VisibleMenuLists_t::iterator viter = visible_menulists_.begin();
      VisibleMenuLists_t::iterator vend = visible_menulists_.end();
      for(;viter != vend; ++viter) {          

        Drawable *obj = viter->second;
        Var<bool> visible(obj->get_vars(), "visible");
        visible = true;
        if (redraw) {
          Var<double> x1(obj->get_vars(), "x1",0);
          Var<double> y1(obj->get_vars(), "y1",0);
          Var<double> x2(obj->get_vars(), "x2",0);
          Var<double> y2(obj->get_vars(), "y2",0);
          obj->set_region(RectRegion(x1(),y1(),x2(), y2()));
        }
        
        obj->process_event(event);
        visible = false;
        if (vend != visible_menulists_.end()) break;
        if (visible_menulists_.find(viter->first) != viter) break;
      }


      if (!redraw) {
        Parent::process_event(event);
      }

#if 0
      if (click && visible_menulists_.size()) {
        visible_menulists_.clear();
      }
#endif


      return CONTINUE_E;
    }

    BaseTool::propagation_state_e
    MenuManager::MenuList_Maker(event_handle_t maker_signal)
    {
      MenuList *menu_list = 
        construct_child_from_maker_signal<MenuList>(maker_signal);
      Var<bool> visible(menu_list->get_vars(), "visible");
      visible = false;
      menu_lists_.push_back(menu_list);
      return STOP_E;
    }

    BaseTool::propagation_state_e
    MenuManager::MenuButton_Maker(event_handle_t maker_signal)
    {
      MenuButton *menu_button = 
        construct_child_from_maker_signal<MenuButton>(maker_signal);
      menu_buttons_.push_back(menu_button);
      return STOP_E;
    }


    BaseTool::propagation_state_e
    MenuManager::show_MenuList(event_handle_t event) {
      
      Signal *signal = dynamic_cast<Signal *>(event.get_rep());
      Variables *vars = signal->get_vars();
      const string id = vars->get_id();
            
      MenuLists_t::iterator citer = menu_lists_.begin();
      for (; citer != menu_lists_.end(); ++citer) {
        string full_id = (*citer)->get_id();
        if (ends_with(full_id,id)) {
          if (vars->exists("MenuButton::x1")) {
            Var<int> width((*citer)->get_vars(), "width",100);
            Var<int> height((*citer)->get_vars(), "height",100);

            Var<double> x1((*citer)->get_vars(), "x1");
            x1 = vars->get_double("MenuButton::x1");

            Var<double> y1((*citer)->get_vars(), "y1");
            y1 = vars->get_double("MenuButton::y1") - height();

            Var<double> x2((*citer)->get_vars(), "x2");
            x2 = x1() + width();

            Var<double> y2((*citer)->get_vars(), "y2");
            y2 = vars->get_double("MenuButton::y1");

            (*citer)->set_region(RectRegion(x1(),y1(),x2(), y2()));
          }
            
          mutex_.lock();
          visible_menulists_.insert(make_pair(full_id,*citer));
          mutex_.unlock();
          EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
          return CONTINUE_E;
        }
      }

      return STOP_E;
    }

    BaseTool::propagation_state_e
    MenuManager::hide_MenuList(event_handle_t event) {
      Signal *signal = dynamic_cast<Signal *>(event.get_rep());
      if (!signal->get_vars()->exists("id")) 
        return CONTINUE_E;
      const string id = signal->get_vars()->get_id();
      mutex_.lock();
      
      if (visible_menulists_.find(id) != visible_menulists_.end()) {
        visible_menulists_.erase(id);
      }
      mutex_.unlock();

      EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
      return CONTINUE_E;
    }
  }
}
