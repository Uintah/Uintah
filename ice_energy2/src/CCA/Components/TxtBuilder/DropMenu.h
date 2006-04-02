/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
  University of Utah.

  License for the specific language governing rights and limitations under
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
 *  DropMenu.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   September 2005
 *
 */

#ifndef CCA_Components_TxtBuilder_DropMenu_h
#define CCA_Components_TxtBuilder_DropMenu_h

#include <menu.h>
#include <panel.h>
#include <map>
#include <Core/CCA/spec/cca_sidl.h>

using namespace std;
namespace SCIRun {

  class DropMenu{

  public:
    DropMenu();
    DropMenu(char* choices[], int cnt);
    ~DropMenu();
    void show(int lines, int cols);
    void hide();
    void mv_up();
    void mv_down();
    void addItem(const sci::cca::ComponentClassDescription::pointer &cd, string name);
    //build the menu from item_map;
    void build();
    //execute the menu command
    int exec();
    int item_index();
  private:
    map<string, sci::cca::ComponentClassDescription::pointer> item_map;
    ITEM **items;
    MENU *menu;
    WINDOW *win;
    PANEL *panel;
  };

} //namespace SCIRun

#endif
