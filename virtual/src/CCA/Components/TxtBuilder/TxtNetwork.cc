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
 *  TxtNetwork.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   September 2005
 *
 */

#include <curses.h>
#include <menu.h>
#include <Core/CCA/spec/cca_sidl.h>
#include "TxtModule.h"
#include "TxtNetwork.h"
#include "TxtBuilder.h"

using namespace sci::cca;
using namespace SCIRun;

std::string 
TxtNetwork:: filename;

int
TxtNetwork:: location=1;

TxtModule*
TxtNetwork:: active_module=NULL;

std::vector<TxtModule*>
TxtNetwork:: module_list;

WINDOW*
TxtNetwork::win;

TxtNetwork::TxtNetwork(){
  
}

void
TxtNetwork::setup(WINDOW* win_main){
  active_module=NULL;
  win=win_main;
}

int
TxtNetwork::enter(){
  if(module_list.size()==0)return 0;
  if(active_module)  active_module->reverse();
  bool quit=false;
  unsigned i;
  while(!quit){
    switch(wgetch(stdscr)){	
    case ' ':  //SPACE:
      active_module->pop_menu();
      drawConnections();
      if(active_module==NULL) quit=true;
      break;
    case 0x0A:  //ENTER:
      active_module->show_ports();
      break;
    case 0x09: //KEY_TAB:
      for(i=0; active_module!=module_list[i];i++);
      if(++i==module_list.size())i=0;
      active_module->reverse();
      active_module=module_list[i];
      active_module->reverse();
      break;
    case KEY_DOWN:
      active_module->mv_down();
      drawConnections();
      break;
    case KEY_UP:
      active_module->mv_up();
      drawConnections();
      break;
    case KEY_LEFT:
      active_module->mv_left();
      drawConnections();
      break;
    case KEY_RIGHT:
      active_module->mv_right();
      drawConnections();
      break;
    case 'q':
    case 0x1b: //ESC
      //TODO: deactivate the main menu
      quit=true;
      break;
    }
  }
  if(active_module)  active_module->reverse();
  return 0; //TODO: return 
}

void
TxtNetwork::addModule(TxtModule *com){
  //TODO: need adjust these values accroding to the screen size
  int w=100; 
  int h=50; 
  int nrows=h/4;
  int ncols=w/15;
  
  module_list.push_back(com);
  location=location%(nrows*ncols);
  int col=location/nrows;
  int row=location%nrows;
  com->show(row*4, col*15);
  location++;
  active_module=com;
}

void
TxtNetwork::addModule(TxtModule *com, int row, int col){
  module_list.push_back(com);
  com->show(row, col);
  active_module=com;
}


void
TxtNetwork::delModule(TxtModule *com){
  for(unsigned i=0; i<module_list.size();i++){
    if(module_list[i]==com) module_list.erase(module_list.begin()+i);
  }
  if(module_list.size()>0){
    active_module=module_list[0];
    active_module->reverse();
  }
  else active_module=NULL;
}

void
TxtNetwork::delModule(ComponentID::pointer cid){
  for(unsigned i=0; i<module_list.size();i++){
    if(module_list[i]->getCID()==cid){
      delete module_list[i];
      module_list.erase(module_list.begin()+i);
      break;
    }
  }
  if(module_list.size()>0){
    active_module=module_list[0];
  }
  else active_module=NULL;
}

void 
TxtNetwork::clear(){
  unsigned int num=module_list.size();
  for(unsigned i=0; i<num;i++){
    delete module_list[i];
  }
  active_module=NULL;
  module_list.erase(module_list.begin(),module_list.end());
  drawConnections();
  update_panels();
  doupdate();
}


void
TxtNetwork::drawConnections(){
  wclear(win);
  sci::cca::ports::BuilderService::pointer bs=SCIRun::TxtBuilder::getBuilderService();
  std::vector<ComponentID::pointer> cids=bs->getComponentIDs();
  SSIDL::array1<ConnectionID::pointer> connIDs=bs->getConnectionIDs(cids);
  // for all pairs of modules
  for(unsigned int i=0; i<module_list.size(); i++){
    for(unsigned int j=i+1; j<module_list.size(); j++){
      // check if there is a connection
      for(unsigned int c=0; c<connIDs.size();c++){
	if(connIDs[c]->getUser()==module_list[i]->getCID() &&
	   connIDs[c]->getProvider()==module_list[j]->getCID()){
	  TxtConnection conn(win);
	  conn.resetPoints(module_list[i]->rect(), module_list[j]->rect()); 
	  conn.draw();
	  break;
	}else if(connIDs[c]->getUser()==module_list[j]->getCID() &&
		 connIDs[c]->getProvider()==module_list[i]->getCID()){
	  TxtConnection conn(win);
	  conn.resetPoints(module_list[j]->rect(), module_list[i]->rect()); 
	  conn.draw();
	  break;
	}
      }
    }
  }
  update_panels();
  doupdate();
}

std::vector<TxtModule*>
TxtNetwork::getModuleList(){
  return module_list;
}

void
TxtNetwork::setFilename(const std::string& file){
  filename=file;
}


std::string
TxtNetwork::getFilename(){
  return filename;
}




void TxtNetwork::writeFile()
{
    std::ofstream saveOutputFile(getFilename().c_str());
    std::vector<TxtModule*> module_list=getModuleList();

    sci::cca::ports::BuilderService::pointer bs=SCIRun::TxtBuilder::getBuilderService();
    std::vector<ComponentID::pointer> cids=bs->getComponentIDs();
    SSIDL::array1<ConnectionID::pointer> connIDs=bs->getConnectionIDs(cids);

    saveOutputFile << module_list.size() << std::endl;
    saveOutputFile << connIDs.size() << std::endl;

    if (saveOutputFile.is_open()) {
      //save module names and positions
      for (unsigned int i=0; i<module_list.size(); i++){
	saveOutputFile << module_list[i]->getType() << std::endl;
	//todo: convert cols, lines to pixels (x, y)
	saveOutputFile << module_list[i]->getCols() << std::endl;
	saveOutputFile << module_list[i]->getLines() << std::endl;
      }

      //save connections
      for(unsigned int i=0; i<module_list.size(); i++){
	for(unsigned int j=i+1; j<module_list.size(); j++){
	  // check if there is a connection
	  for(unsigned int c=0; c<connIDs.size();c++){
	    if(connIDs[c]->getUser()==module_list[i]->getCID() &&
	       connIDs[c]->getProvider()==module_list[j]->getCID()){
	      saveOutputFile << i << " " <<
		connIDs[c]->getUserPortName() << " " << j << " " <<
		connIDs[c]->getProviderPortName() << std::endl;
	    }else if(connIDs[c]->getUser()==module_list[j]->getCID() &&
		     connIDs[c]->getProvider()==module_list[i]->getCID()){
	      saveOutputFile << j << " " <<
		connIDs[c]->getUserPortName() << " " << i << " " <<
		connIDs[c]->getProviderPortName() << std::endl;
	    }
	  }
	}
      }
    }
    saveOutputFile.close();
}


void
TxtNetwork::loadFile()
{
    if (getFilename().empty()) return;

    std::ifstream is( getFilename().c_str() ); 

    int numMod = 0;
    int numConn = 0;
    std::string type;
    int x;
    int y;
    std::vector<sci::cca::ComponentID::pointer> cidTable;

    is >> numMod >> numConn;

    sci::cca::ports::BuilderService::pointer bs=SCIRun::TxtBuilder::getBuilderService();

    // If there's an error creating a component, stop trying to load
    // the network file until there are improvements to the
    // network file format.
    try {
        for (int i = 0; i < numMod; i++) {
            is >> type >> x >> y;

            sci::cca::ComponentID::pointer cid;
	    //            TypeMap *tm = new TypeMap;
	    //            tm->putInt("x", modName_x);
	    //            tm->putInt("y", modName_y);

            cid = bs->createInstance(type, type, sci::cca::TypeMap::pointer(0));

            if (! cid.isNull()) {
                if (type != "SCIRun.GUIBuilder") {
                    cidTable.push_back(cid);
		    addModule(new TxtModule(cid, type),y,x);
                }
            }
        }

        for (int i = 0; i < numConn; i++) {
            int iu, ip;
            std::string up, pp;
            is >> iu >> up >> ip >> pp;
            //networkCanvasView->addPendingConnection(cidTable[iu], up, cidTable[ip], pp);
            sci::cca::ConnectionID::pointer connID =
                bs->connect(cidTable[iu], up, cidTable[ip], pp);
        }
	drawConnections();
    }

    //    catch(const CCAException &e) {
      //displayMsg(e.message());
    //    }
    //    catch(const Exception &e) {
      // displayMsg(e.message());
    //    }
    catch(...) {
      //  displayMsg("Caught unexpected exception while loading network.");
    }

    is.close();
}


