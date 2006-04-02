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
 *  TxtMenu.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   September 2005
 *
 */

#include <SCIRun/CCA/CCAException.h>
#include <curses.h>
#include <menu.h>
#include <CCA/Components/TxtBuilder/DropMenu.h>
#include <CCA/Components/TxtBuilder/TxtMenu.h>
#include <CCA/Components/TxtBuilder/TxtBuilder.h>
#include <CCA/Components/TxtBuilder/TxtNetwork.h>
#include <CCA/Components/TxtBuilder/TxtModule.h>
#include <Core/Thread/Thread.h>
#include <Core/CCA/spec/cca_sidl.h>

#define ARRAY_SIZE(a) (sizeof(a) / sizeof(a[0]))

using namespace sci::cca;
using namespace SCIRun;

TxtMenu::TxtMenu(const sci::cca::Services::pointer& svc){
  this->svc=svc;
}

void
TxtMenu::setup(WINDOW* win_menu){
  //setup main menu

  main_menu_choices.push_back("File");
  main_menu_choices.push_back("Proxy Framework");

  static char *choices0[] = {
    "Load",
    "Insert",
    "Save",
    "Save As",
    "Clear Network",
    "Add Info",
    "Add Sidl XML Path",
    "Quit GUI",
    "Exit"
  };
  
  
  static char *choices1[] = {
    "Add Loader",
    "Remove Loader",
    "Refresh Menu"
  };

  static char *choices2[] = {
    "About"
  };

  //setup drop drop menus
  drop_menu.push_back(new DropMenu(choices0, ARRAY_SIZE(choices0)));
  drop_menu.push_back(new DropMenu(choices1, ARRAY_SIZE(choices1)));
  buildPackageMenus();
  drop_menu.push_back(new DropMenu(choices2, ARRAY_SIZE(choices2)));
  
  //  mvwprintw(win_main,5, 10, "Click menu to see what happened");
  //  if(! ( mousemask(REPORT_MOUSE_POSITION, &oldmask) & REPORT_MOUSE_POSITION) )return 0;


  // build main menu
  int n_main_menu_choices = main_menu_choices.size();
  ITEM** main_menu_items = new (ITEM *)[n_main_menu_choices + 1];
  for(int i = 0; i < n_main_menu_choices; ++i)
    main_menu_items[i] = new_item(main_menu_choices[i].c_str(), NULL);

  main_menu_items[n_main_menu_choices] = NULL;
  main_menu = new_menu(main_menu_items);

  set_menu_format(main_menu, 1, n_main_menu_choices);

  ///////////////////////////
  set_menu_win(main_menu,win_menu);
  set_menu_sub(main_menu,win_menu);
  
  drawInactiveMenu(win_menu);

}

int
TxtMenu::enter(WINDOW* win_menu){
  werase(win_menu);
  post_menu(main_menu);
  wrefresh(win_menu);
  update_panels();
  doupdate();

  //assuming main menu is activiated here
  bool isMainMenu=true;
  bool quitMenu=false;
  int menu_x=0;
  int menu_y=1;
  int main_menu_index=item_index(current_item(main_menu));
  while(!quitMenu){
    /* refresh, accept single keystroke of input */
    int c;
    //    MEVENT mevent;
    c=wgetch(stdscr);//win_menu);
    //    if(c==KEY_MOUSE){
    //      getmouse(&mevent);
    //      if(mevent.bstate&BUTTON1_CLICKED
    //	 && mevent.x>=1 && mevent.x<=7 && mevent.y==0){

    if(isMainMenu){
      switch(c){	
      case KEY_DOWN:
      case 0xa:  //ENTER:
	isMainMenu=false;
	//popup sub menu here
	menu_x=main_menu_index*16;
	drop_menu[main_menu_index]->show(menu_y, menu_x);
	break;
      case KEY_LEFT:
	menu_driver(main_menu, REQ_LEFT_ITEM);
	main_menu_index=item_index(current_item(main_menu));
	wrefresh(win_menu);
	break;
      case KEY_RIGHT:
	menu_driver(main_menu, REQ_RIGHT_ITEM);
	main_menu_index=item_index(current_item(main_menu));
	wrefresh(win_menu);
	break;
      case 'q':
      case 0x1b: //ESC
	//TODO: deactivate the main menu
	quitMenu=true;
	//	menu_driver(main_menu, REQ_CLEAR_PATTERN);
	//	refresh();
	break;
      }
    }else{
      switch(c){	
      case KEY_DOWN:
	drop_menu[main_menu_index]->mv_down();
	break;
      case KEY_UP:
	drop_menu[main_menu_index]->mv_up();
	break;
      case KEY_LEFT:
	drop_menu[main_menu_index]->hide();
	menu_driver(main_menu, REQ_LEFT_ITEM);
	main_menu_index=item_index(current_item(main_menu));
	menu_x=main_menu_index*16;
	drop_menu[main_menu_index]->show(menu_y, menu_x);
	break;
      case KEY_RIGHT:
	drop_menu[main_menu_index]->hide();
	menu_driver(main_menu, REQ_RIGHT_ITEM);
	main_menu_index=item_index(current_item(main_menu));
	menu_x=main_menu_index*16;
	drop_menu[main_menu_index]->show(menu_y, menu_x);
	break;
      case 'q':
      case 0x1b: //ESC
	//TODO: deactivate the main menu
	quitMenu=true;
	drop_menu[main_menu_index]->hide();
	break;
      case 0xa:  //ENTER
	//execute the command
	drop_menu[main_menu_index]->hide();
	//components might be instantiated here
	drop_menu[main_menu_index]->exec();
	//other commands might be executed here
	exec(main_menu_index, drop_menu[main_menu_index]->item_index());
	quitMenu=true;
	break;
      }	
    }
  }
  unpost_menu(main_menu);
  drawInactiveMenu(win_menu);
  return 0; //TODO: return the command
}


void
TxtMenu::drawInactiveMenu(WINDOW* win_menu){
  int n_main_menu_choices = main_menu_choices.size();
  werase(win_menu);
  unsigned max_item_size=0;
  for(int i = 0; i < n_main_menu_choices; ++i){
    if(max_item_size< main_menu_choices[i].size())max_item_size=main_menu_choices[i].size();
  }

  for(int i = 0; i < n_main_menu_choices; ++i){
    mvwprintw(win_menu,0, 1+i*(max_item_size+2), main_menu_choices[i].c_str()); 
  }
  wrefresh(win_menu);
  // Update the stacking order
  update_panels();

  // Show it on the screen 
  doupdate();
}



void 
TxtMenu::buildPackageMenus()
{
    sci::cca::ports::ComponentRepository::pointer reg =
        pidl_cast<sci::cca::ports::ComponentRepository::pointer>(
            svc->getPort("cca.ComponentRepository")
        );
    if (reg.isNull()) {
      //        displayMsg("Error: cannot find component registry, not building component menus.");
        return;
    }
    std::vector<sci::cca::ComponentClassDescription::pointer> list =
        reg->getAvailableComponentClasses();

    for (std::vector<sci::cca::ComponentClassDescription::pointer>::iterator iter =
            list.begin();
            iter != list.end(); iter++) {
        // model name could be obtained somehow locally.
        // and we can assume that the remote component model is always "CCA"
        std::string model = (*iter)->getComponentModelName();
        std::string loaderName = (*iter)->getLoaderName();

        std::string name = (*iter)->getComponentClassName();

        // component class has a loader that is not in this address space?
        if (loaderName != "") {
            std::string::size_type i = name.find_first_of(".");
            name.insert(i, "@" + loaderName);
        }

	//insert main menu item (component models) here
        if (menus.find(model) == menus.end()) {
	  main_menu_choices.push_back(model);
	  menus[model] = new DropMenu;
	  drop_menu.push_back(menus[model]);
        }
	//add the component to the drop menu
	menus[model]->addItem(*iter, name); //TODO: need implement this
    }

    for(map<std::string, DropMenu *>::iterator iter= menus.begin(); iter!=menus.end(); iter++){
      iter->second->build();
    }
    svc->releasePort("cca.ComponentRepository");
}

void 
TxtMenu::exec(int main_menu_index, int sub_menu_index){
  if(main_menu_index==0){
    switch(sub_menu_index){
    case 0: //LOAD
      TxtNetwork::setFilename(SCIRun::TxtBuilder::getString("Load:"));
      TxtNetwork::loadFile();
      break;
    case 1: //Insert
      SCIRun::TxtBuilder::displayMessage("Insert is not implemented!");
      break;
    case 2: //Save
      if (TxtNetwork::getFilename().size()==0) {
	TxtNetwork::setFilename(SCIRun::TxtBuilder::getString("Save As:"));
      }
      if (TxtNetwork::getFilename().size()>0) {
        TxtNetwork::writeFile();
      }      
      break;
    case 3: //SaveAs
      TxtNetwork::setFilename(SCIRun::TxtBuilder::getString("Save As:"));
      if (TxtNetwork::getFilename().size()>0) {
        TxtNetwork::writeFile();
      }      
      break;
    case 4: //Clear Network
      TxtNetwork::clear();
      break;
    case 5: //Add Info
      SCIRun::TxtBuilder::displayMessage("Add Info is not implemented!");
      break;
    case 6: //Add SIDL XML Path
      SCIRun::TxtBuilder::displayMessage("Add SIDL XML Path is not implemented!");
      break;    
    case 7: //Quit GUI
      SCIRun::TxtBuilder::displayMessage("Quit GUI is not implemented!");
      break;    
    case 8: //Quit SCIRun2
      refresh();
      endwin();
      SCIRun::Thread::exitAll(0);
      break;
    }
  }else if(main_menu_index==1){
    
    
  }
}

