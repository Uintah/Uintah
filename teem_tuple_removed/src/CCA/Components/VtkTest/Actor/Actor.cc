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
#include <vtkActor.h>
#include <vtkMapper.h>

#include <CCA/Components/VtkTest/Actor/Actor.h>

using namespace std;
using namespace SCIRun;
using namespace vtk;

extern "C" vtk::Component* make_Vtk_Actor()
{
  return new Actor;
}

bool 
Actor::accept(OutPort* port){
  return dynamic_cast<vtkMapper*>(port->getOutput())!=0;
}

void
Actor::connect(OutPort* port){
  actor->SetMapper(dynamic_cast<vtkMapper*>(port->getOutput()));
}

Actor::Actor(){
  //set input port name
  InPort::setName("Actor::input");

  //set output port name
  OutPort::setName("Actor::output");

  actor=vtkActor::New();

  setOutput(actor);

  addPort( dynamic_cast<InPort*>(this));
  addPort( dynamic_cast<OutPort*>(this));
}

Actor::~Actor(){
  actor->Delete();
}
