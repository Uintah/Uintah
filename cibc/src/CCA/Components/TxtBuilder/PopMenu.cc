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
 *  PopMenu.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   September 2005
 *
 */

#include <CCA/Components/TxtBuilder/PopMenu.h>
#include <string.h>

using namespace SCIRun;

PopMenu::PopMenu(char** choices, int cnt){
  items = (ITEM **)new (ITEM*)[cnt + 1];
  int w=0;
  for(int i = 0; i < cnt; ++i){
    items[i] = new_item(choices[i], NULL);
    int len=strlen(choices[i]);
    if(w<len)w=len;
  }
  items[cnt] = (ITEM *)NULL;
  menu = new_menu(items);

  w++; //leave space for menu item indicator
  win = newwin(cnt+2,w+2, 0, 0); 
  WINDOW *dwin = derwin(win,cnt,w, 1, 1);
  panel= new_panel(win); 	
  set_menu_win(menu,win);
  set_menu_sub(menu,dwin);
  box(win,'|','-');
  //initially hide the drop menu panel
  //do not call hide() here, it does not work
  hide_panel(panel);
  update_panels();
  doupdate();
}

PopMenu::PopMenu(std::vector<std::string> choices){
  int cnt=choices.size();
  items = (ITEM **)new (ITEM*)[cnt + 1];
  int w=0;
  for(int i = 0; i < cnt; ++i){
    items[i] = new_item(choices[i].c_str(), NULL);
    int len=choices[i].size();
    if(w<len)w=len;
  }
  items[cnt] = (ITEM *)NULL;
  menu = new_menu(items);

  w++; //leave space for menu item indicator
  win = newwin(cnt+2,w+2, 0, 0); 
  WINDOW *dwin = derwin(win,cnt,w, 1, 1);
  panel= new_panel(win); 	
  set_menu_win(menu,win);
  set_menu_sub(menu,dwin);
  box(win,'|','-');
  //initially hide the drop menu panel
  //do not call hide() here, it does not work
  hide_panel(panel);
  update_panels();
  doupdate();
}

PopMenu::~PopMenu(){
  del_panel(panel);
  delwin(win);
  int cnt=item_count(menu);
  for(int i = 0; i < cnt; ++i) free_item(items[i]);
  delete []items;
}

int
PopMenu::enter(int lines, int cols, int &flag){
  move_panel(panel, lines, cols);
  post_menu(menu);
  wrefresh(win);
  show_panel(panel);
  update_panels();
  doupdate();
  int selection=-1;
  
  bool quit=false;
  while(!quit){
    switch(wgetch(stdscr)){	
    case KEY_DOWN:
      mv_down();
      break;
    case KEY_UP:
      mv_up();
      break;
    case ' ':  //SPACE
      selection=item_index(current_item(menu));
      flag=1; //disconnect
      quit=true;
      break;
    case 0x0A:  //ENTER:
      flag=0;
      selection=item_index(current_item(menu));
      quit=true;
      break;
    case 'q': //Q
    case 0x1b: //ESC
      flag=-1; //show compatible ports
      quit=true;
      break;
    }
  }
  hide();
  return selection;
}

int
PopMenu::select(int lines, int cols){
  move_panel(panel, lines, cols);
  post_menu(menu);
  wrefresh(win);
  show_panel(panel);
  update_panels();
  doupdate();
  int selection=-1;
  bool quit=false;
  while(!quit){
    switch(wgetch(stdscr)){	
    case KEY_DOWN:
      mv_down();
      break;
    case KEY_UP:
      mv_up();
      break;
    case 'q': //Q
    case 0x1b: //ESC
      quit=true;
      break;
    case 0x0A:  //ENTER:
      selection=item_index(current_item(menu));
      quit=true;
      break;
    }
  }
  hide();
  return selection;
}


void
PopMenu::hide(){
  unpost_menu(menu);
  hide_panel(panel);
  update_panels();
  doupdate();
}

void
PopMenu::mv_up(){
  menu_driver(menu, REQ_UP_ITEM);
  wrefresh(win);
}

void
PopMenu::mv_down(){
  menu_driver(menu, REQ_DOWN_ITEM);
  wrefresh(win);
}
