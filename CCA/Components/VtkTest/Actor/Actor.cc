/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  Actor.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#include <iostream>
#include <SCIRun/Vtk/Port.h>
#include <CCA/Components/VtkTest/Actor/Actor.h>

#include "vtkActor.h"
#include "vtkMapper.h"

using namespace std;
using namespace SCIRun;
using namespace vtk;

extern "C" vtk::Component* make_Vtk_Actor()
{
  return new Actor;
}

//Input Port
IPort::IPort(vtkActor *actor){
  this->actor=actor;  
}

IPort::~IPort(){

}

bool
IPort::isInput(){
  return true;
}

std::string
IPort::getName(){
  return "Actor::input";
}

bool 
IPort::accept(Port* port){
  return dynamic_cast<vtkMapper*>(port->getObj())!=0;
}

void
IPort::connect(Port* port){
  actor->SetMapper(dynamic_cast<vtkMapper*>(port->getObj()));
}

//Output Port

OPort::OPort(vtkActor *actor){
  this->actor=actor;
}

OPort::~OPort(){

}

bool
OPort::isInput(){
  return false;
}

std::string
OPort::getName(){
  return "Actor::output";
}

vtkObject *
OPort::getObj(){
  return actor;
}


Actor::Actor(){

  actor=vtkActor::New();
  iports.push_back(new IPort(actor));
  oports.push_back(new OPort(actor));
}

Actor::~Actor(){
  for(unsigned int i=0; i<iports.size(); i++){
    delete iports[i];
  }
  for(unsigned int i=0; i<oports.size(); i++){
    delete oports[i];
  }
  actor->Delete();
}
