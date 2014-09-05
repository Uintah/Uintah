/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is Vtk, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  Component.cc: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */


#include <string>
#include <iostream>
#include <vector>
#include <SCIRun/Vtk/InPort.h>
#include <SCIRun/Vtk/OutPort.h>
#include <SCIRun/Vtk/Component.h>

using namespace SCIRun;
using namespace vtk;

Component::Component(){
  have_ui=false;
}

Component::~Component(){
  //TODO
  std::cerr<<" *** ~Component *** called\n";
}

void
Component::enableUI(bool new_thread){
  have_ui=true;
  new_ui_thread=new_thread;
}

void 
Component::disableUI(){
  have_ui=false;
}

bool 
Component::haveUI(){
  return have_ui;
}

Port* 
Component::getPort(const std::string &name){
  for(unsigned int i=0; i<iports.size(); i++){
    if(name==iports[i]->getName()) return iports[i];
  }
  for(unsigned int i=0; i<oports.size(); i++){
    if(name==oports[i]->getName()) return oports[i];
  }
  return 0;
}

void
Component::addPort(Port *port){
  if(port->isInput())
    iports.push_back((InPort*)port);
  else 
    oports.push_back((OutPort*)port);
}

void
Component::removePort(Port *port){
  if(port->isInput()){
    for(unsigned int i=0; i<iports.size(); i++){
      if(iports[i]==port){
	iports.erase(iports.begin()+i);
	return;
      }
    }
  }
  else{ 
    for(unsigned int i=0; i<oports.size(); i++){
      if(oports[i]==port){
	oports.erase(oports.begin()+i);
	return;
      }
    }
  }
}

void
Component::removePort(const std::string &name){
  removePort(getPort(name));
}

int 
Component::numIPorts(){
  return iports.size();
}

int 
Component::numOPorts(){
  return oports.size();
}

bool
Component::isThreadedUI(){
  return new_ui_thread;
}



InPort* 
Component::getIPort(unsigned int index){
  if(index>iports.size()) return 0;
  return iports[index];
}

OutPort* 
Component::getOPort(unsigned int index){
  if(index>oports.size()) return 0;
  return oports[index];
}

int 
Component::popupUI(){
  return 0;
}
