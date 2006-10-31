/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
  University of Utah.

  
  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation
  the rights to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  DEALINGS IN THE SOFTWARE.
*/


/*
 *  TxtMenu.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   September 2005
 *
 */

#ifndef CCA_Components_TxtBuilder_TxtMenu_h
#define CCA_Components_TxtBuilder_TxtMenu_h

#include <curses.h>
#include <menu.h>
#include <string>
#include <vector>

namespace SCIRun {

  class DropMenu;

  class TxtMenu{

  public:
    TxtMenu(const sci::cca::Services::pointer& svc);
    void setup(WINDOW* win_menu);
    int enter(WINDOW* win_menu);
    void drawInactiveMenu(WINDOW* win_menu);
    void exec(int main_menu_index, int sub_menu_index);
  private:
    void buildPackageMenus();
    MENU *main_menu;
    vector<DropMenu*>drop_menu;
    vector<string> main_menu_choices;
    map<std::string, DropMenu *> menus;
    sci::cca::Services::pointer svc;
  };


} //namespace SCIRun

#endif
