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
 *  PolyDataMapper.cc:
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
#include <CCA/Components/VtkTest/PolyDataMapper/PolyDataMapper.h>

#include "vtkPolyDataMapper.h"

using namespace std;
using namespace SCIRun;
using namespace vtk;

extern "C" vtk::Component* make_Vtk_PolyDataMapper()
{
  return new PolyDataMapper;
}

//Input Port
IPort::IPort(){
  
}

IPort::~IPort(){

}

bool
IPort::isInput(){
  return true;
}

std::string
IPort::getName(){
  return "PolyDataMapper::input";
}

//Output Port

OPort::OPort(){
  
}

OPort::~OPort(){

}

bool
OPort::isInput(){
  return false;
}

std::string
OPort::getName(){
  return "PolyDataMapper::output";
}


PolyDataMapper::PolyDataMapper(){
  
  iports.push_back(new IPort);
  oports.push_back(new OPort);
  mapper=vtkPolyDataMapper::New();
}

PolyDataMapper::~PolyDataMapper(){
  for(unsigned int i=0; i<iports.size(); i++){
    delete iports[i];
  }
  for(unsigned int i=0; i<oports.size(); i++){
    delete oports[i];
  }
  mapper->Delete();
}
