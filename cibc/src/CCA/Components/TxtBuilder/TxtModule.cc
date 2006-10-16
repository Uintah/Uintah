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
 *  TxtModule.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   September 2005
 *
 */

#include <CCA/Components/TxtBuilder/TxtModule.h>
#include <CCA/Components/TxtBuilder/TxtBuilder.h>
#include <CCA/Components/TxtBuilder/TxtNetwork.h>
#include <CCA/Components/TxtBuilder/PopMenu.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <string.h>
#include <sstream>

using namespace std;
using namespace sci::cca;
using namespace SCIRun;

TxtModule::TxtModule(const sci::cca::ComponentID::pointer &cid, const std::string& type){
  this->cid=cid;
  height=3;
  this->name=cid->getInstanceName();
  this->type=type;
  width=2+this->name.length();
  lines=cols=0;
  is_highlight=false;
  win = newwin(height, width, lines, cols);
  draw();
  panel= new_panel(win); 	

  //initialize popmenu here

  static char *choices[]={
    "destroy",
    "go",
    "ui",
  };
  popmenu=new PopMenu(choices, 3);


  static char *port_list[]={
    "u_port1  ->  component A",
    "p_port2  ->  component B",
    "         ->  component C",
    "p_port3  ->             "
  };
  portmenu=new PopMenu(port_list, 4);





  //initially hide the drop menu panel
  //do not call hide() here, it does not work
  hide_panel(panel);
  update_panels();
  doupdate();
}

TxtModule::~TxtModule(){
  delete portmenu;
  del_panel(panel);
  delwin(win);
}

void
TxtModule::draw(){
  mvwprintw(win,1,1,name.c_str());  
  box(win,0,0);
  wrefresh(win);
}

void
TxtModule::show(int lines, int cols){
  this->lines=lines;
  this->cols=cols;
  move_panel(panel, lines, cols);
  show_panel(panel);
  update_panels();
  doupdate();
}

void
TxtModule::pop_menu(){
  int sel=popmenu->select(lines+2, cols+2);
  update_panels();
  doupdate();
  sci::cca::ports::BuilderService::pointer bs=SCIRun::TxtBuilder::getBuilderService();
  bool hasGoPort=false;
  bool isSciPort=false;
  SSIDL::array1<string> ports = bs->getProvidedPortNames(cid);
  for(unsigned int i=0; i < ports.size(); i++){
    if(ports[i]=="go") hasGoPort=true;
    if(ports[i]=="sci.go"){
      hasGoPort=true;
      isSciPort=true;
    }
  }
  if(sel==0){ //destroy
    vector<ComponentID::pointer> cids;
    cids.push_back(cid);
    SSIDL::array1<ConnectionID::pointer> connIDs=bs->getConnectionIDs(cids);
    for(unsigned i=0; i<connIDs.size();i++){
      bs->disconnect(connIDs[i],0);
    }
    TxtNetwork::delModule(this);
    bs->destroyInstance(cid,0);
    delete this;
  }else if(sel==1){ //go

    if(hasGoPort)
      {
	sci::cca::Services::pointer svc=TxtBuilder::getService();
	string instanceName = cid->getInstanceName();
	string goPortName = instanceName+" goPort";
	svc->registerUsesPort(goPortName, "sci.cca.ports.GoPort",
			      sci::cca::TypeMap::pointer(0));
	ConnectionID::pointer connID=bs->connect(svc->getComponentID(), goPortName, cid, isSciPort?"sci.go":"go");
	Port::pointer p = svc->getPort(goPortName);
	ports::GoPort::pointer goPort = pidl_cast<ports::GoPort::pointer>(p);
	if(goPort.isNull()){
	  cerr << "goPort is not connected, cannot bring up Go!\n";
	} 
	else{
	  int status=goPort->go();
	  cout<<"Done! return value="<<status<<endl;
	}
	svc->releasePort(goPortName);
	bs->disconnect(connID,0);
	svc->unregisterUsesPort(goPortName);
      }
    
  }else if(sel==2){ //ui
    TxtBuilder::displayMessage("UI is not yet supported in TxtBuilder!");
  }
}

void
TxtModule::show_ports(){
  //create portmenu here
  sci::cca::ports::BuilderService::pointer bs=SCIRun::TxtBuilder::getBuilderService();
  SSIDL::array1<std::string> pports=bs->getProvidedPortNames(cid); 
  SSIDL::array1<std::string> uports=bs->getUsedPortNames(cid); 
  std::vector<std::string> item_list;
  std::vector<std::string> port_list;

  std::vector<ConnectionID::pointer> conn_list;

  std::vector<ComponentID::pointer> cids;
  cids.push_back(cid);
  SSIDL::array1<ConnectionID::pointer> connIDs=bs->getConnectionIDs(cids);
  string item;

  for(unsigned int i=0; i<uports.size(); i++){
    bool idle=true;
    for(unsigned j=0; j<connIDs.size();j++){

      if(connIDs[j]->getUserPortName()==uports[i]){
	item= uports[i]+"-->"+connIDs[j]->getProviderPortName()+" ("+connIDs[j]->getProvider()->getInstanceName()+")";
	item_list.push_back(item);
	port_list.push_back(uports[i]);
	conn_list.push_back(connIDs[j]);
	idle=false;
	break;
      }
    }
    if(idle){
      item= uports[i]+"-->";
      item_list.push_back(item);
      port_list.push_back(uports[i]);
      conn_list.push_back(ConnectionID::pointer(0));
    }
  }
    int n_uport=item_list.size();
  for(unsigned int i=0; i<pports.size(); i++){
    bool idle=true;
    for(unsigned j=0; j<connIDs.size();j++){
      if(connIDs[j]->getProviderPortName()==pports[i]){
	item= pports[i]+"<--"+connIDs[j]->getUserPortName()+" ("+connIDs[j]->getUser()->getInstanceName()+")";
	item_list.push_back(item);
	port_list.push_back(pports[i]);
	conn_list.push_back(connIDs[j]);
	idle=false;
      }
    }
    if(idle){
      item= pports[i]+"<--";
      item_list.push_back(item);
      port_list.push_back(pports[i]);
      conn_list.push_back(ConnectionID::pointer(0));
    }
  }

  portmenu=new PopMenu(item_list);
  //start the portmenu
  int flag;
  int sel=portmenu->enter(lines+2, cols+2, flag);
  //refresh windows
  update_panels();
  doupdate();
  delete portmenu;
  //portmenu quits and returns the selected item
  if(flag==0){ //show compatible ports
    std::string selport=port_list[sel];
    //item_list, port_list and cids are reused for the compatible ports
    item_list.erase(item_list.begin(), item_list.end());
    port_list.erase(port_list.begin(), port_list.end());
    cids=bs->getComponentIDs();

    if(sel<n_uport){
      //uses port selected
      item=selport+" --> ";
      item_list.push_back(item);
    }else{
      //provides port selected
      item=selport+" <-- ";
      item_list.push_back(item);
    }
    
    std::vector<sci::cca::ComponentID::pointer> cid_list;

    for(unsigned j=0; j<cids.size(); j++){
      if(cids[j]==cid)continue;
      SSIDL::array1<string> cps=bs->getCompatiblePortList(cid,selport,cids[j]);
      for(unsigned k=0; k<cps.size(); k++){
	port_list.push_back(cps[k]);
	cid_list.push_back(cids[j]);
	item="  "+cps[k]+" ("+cids[j]->getInstanceName()+")";
	item_list.push_back(item);
      }
    }
    
    PopMenu compatiblePortMenu(item_list);
    int sel_compitable=compatiblePortMenu.select(lines+2, cols+2);
    if(sel_compitable>0){
      int i=sel_compitable-1;
      ostringstream os;
      //      os<<"i="<<i<<" "<<selport<<" "<<port_list[i];
      //      SCIRun::TxtBuilder::displayMessage(os.str());
      if(sel<n_uport){
	bs->connect(cid,selport,cid_list[i], port_list[i]);
      }else{
	bs->connect(cid_list[i],port_list[i],cid,selport);
      }
    }
  }else if(flag==1){ //disconnect
    if(!conn_list[sel].isNull()){
      bs->disconnect(conn_list[sel],0);
    }
  }

  TxtNetwork::drawConnections();
  //refresh windows
  update_panels();
  doupdate();
}

void
TxtModule::mv_to(int lines, int cols){
  this->lines=lines;
  this->cols=cols;
  move_panel(panel, lines, cols);
  update_panels();
  doupdate();
}

void
TxtModule::hide(){
  hide_panel(panel);
  update_panels();
  doupdate();
}

void 
TxtModule::reverse(){
  is_highlight=!is_highlight;
  if(is_highlight) 
    wattrset(win,A_REVERSE);
  else  wattrset(win,A_NORMAL);
  draw();
}

void
TxtModule::mv_up(){
  mv_to(lines-1, cols);
}

void
TxtModule::mv_down(){
  show(lines+1, cols);
}

void
TxtModule::mv_left(){
  show(lines, cols-1);
}

void
TxtModule::mv_right(){
  show(lines, cols+1);
}

Rect
TxtModule::rect(){
  //-1 here is to change the coords to win_main.
  //need do it nicely later
  return Rect(cols, lines-1, width, height); 
}

sci::cca::ComponentID::pointer 
TxtModule::getCID(){
  return cid;
}


int
TxtModule::getCols(){
  return cols;
}


int
TxtModule::getLines(){
  return lines;
}

string 
TxtModule::getName(){
  return name;
}

string 
TxtModule::getType(){
  return type;
}

