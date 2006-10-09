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
//    Date   : Fri Aug 11 20:34:34 2006

#ifndef SKINNER_MENUMANAGER_H
#define SKINNER_MENUMANAGER_H

#include <Core/Skinner/Parent.h>
#include <Core/Thread/Mutex.h>
#include <map>
#include <set>
#include <vector>

namespace SCIRun {
  class MenuManager;
  class MenuList;
  class MenuButton;

  namespace Skinner {
    class MenuManager : public Parent {
    public:
      MenuManager (Variables *variables);
      virtual ~MenuManager();
      virtual propagation_state_e       process_event(event_handle_t);
    protected:
      Mutex                             mutex_;
      CatcherFunction_t                 MenuList_Maker;
      CatcherFunction_t                 MenuButton_Maker;
      CatcherFunction_t                 show_MenuList;
      CatcherFunction_t                 hide_MenuList;
      typedef vector<MenuList *>        MenuLists_t;
      MenuLists_t                       menu_lists_;

      typedef vector<MenuButton *>        MenuButtons_t;
      MenuButtons_t                       menu_buttons_;


      typedef map<string, MenuList *>   VisibleMenuLists_t;
      VisibleMenuLists_t                visible_menulists_;

    };
  }
}

#endif
