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
 *  DropMenu.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   September 2005
 *
 */

#include <CCA/Components/TxtBuilder/DropMenu.h>
#include <CCA/Components/TxtBuilder/TxtNetwork.h>
#include <CCA/Components/TxtBuilder/TxtBuilder.h>
#include <CCA/Components/TxtBuilder/TxtModule.h>
#include <SCIRun/TypeMap.h>
#include <SCIRun/CCA/CCAException.h>

using namespace SCIRun;

DropMenu::DropMenu()
{
}

DropMenu::DropMenu(char** choices, int cnt)
{
  //items = (ITEM **) new (ITEM*)[cnt + 1];
  items = (ITEM **) new ITEM* [cnt + 1];
  for (int i = 0; i < cnt; ++i) {
    items[i] = new_item(choices[i], NULL);
  }
  items[cnt] = (ITEM *)NULL;
  menu = new_menu(items);


  int rows, cols;
  //calculate the size of the menu window
  scale_menu(menu, &rows, &cols);
  win = newwin(rows+2, cols+2, 0, 0);
  WINDOW *dwin = derwin(win,rows, cols, 1, 1);
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

DropMenu::~DropMenu()
{
  del_panel(panel);
  delwin(win);
  const int cnt = item_count(menu);
  for (int i = 0; i < cnt; ++i) {
    free_item(items[i]);
  }
  delete [] items;
}

void
DropMenu::show(int lines, int cols)
{
  move_panel(panel, lines, cols);
  post_menu(menu);
  wrefresh(win);
  show_panel(panel);
  update_panels();
  doupdate();
}

void
DropMenu::hide()
{
  unpost_menu(menu);
  hide_panel(panel);
  update_panels();
  doupdate();
}

void
DropMenu::mv_up()
{
  menu_driver(menu, REQ_UP_ITEM);
  wrefresh(win);
}

void
DropMenu::mv_down()
{
  menu_driver(menu, REQ_DOWN_ITEM);
  wrefresh(win);
}

void
DropMenu::addItem(const sci::cca::ComponentClassDescription::pointer &cd, string name)
{
  item_map[name]=cd;
}

void
DropMenu::build()
{
  int cnt = item_map.size();
  items = (ITEM **) new ITEM* [cnt + 1];
  int i = 0;
  for(map<string, sci::cca::ComponentClassDescription::pointer>::iterator iter=item_map.begin();
      iter!=item_map.end(); iter++, i++){
    items[i] = new_item(iter->first.c_str(), NULL);
  }
  items[cnt] = (ITEM *)NULL;
  menu = new_menu(items);

  int rows, cols;
  //calculate the size of the menu window
  scale_menu(menu, &rows, &cols);
  win = newwin(rows+2, cols+2, 0, 0);
  WINDOW *dwin = derwin(win,rows, cols, 1, 1);
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

int
DropMenu::exec()
{
  if (item_map.size() > 0) {
    //instantiate a component
    sci::cca::ComponentClassDescription::pointer cd=item_map[item_name(current_item(menu))];

    sci::cca::TypeMap *tm = new SCIRun::TypeMap;
    tm->putString("LOADER NAME", cd->getLoaderName());
    tm->putString("cca.className", cd->getComponentModelName()); // component type

    sci::cca::ComponentID::pointer cid;
    try {
      sci::cca::ports::BuilderService::pointer builder =
        pidl_cast<sci::cca::ports::BuilderService::pointer>(
                                                            TxtBuilder::getService()->getPort("cca.BuilderService")
                                                            );
      if (builder.isNull()) {
        std::cerr << "Fatal Error: Cannot find builder service" << std::endl;
        //unsetCursor();
        return 1;
      }
      cid = builder->createInstance(cd->getComponentClassName(),
                                    cd->getComponentClassName(), sci::cca::TypeMap::pointer(tm));

      TxtNetwork::addModule(new TxtModule(cid,cd->getComponentClassName()));

      if (cid.isNull()) {
        //displayMsg("Error: could not instantiate component of type " +
        //   cd->getComponentClassName());
        //statusBar()->message("Instantiate failed.", 2000);
        return 1;
      } else {
        //statusBar()->clear();
      }
      TxtBuilder::getService()->releasePort("cca.BuilderService");
    }
    catch(...) {
      //  displayMsg("Caught unexpected exception while instantiating " +
      //   cd->getComponentClassName());
    }

  }

  return 0;
}

int
DropMenu::item_index()
{
  return ::item_index(current_item(menu));
}
