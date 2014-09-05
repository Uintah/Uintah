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
#include <vtkPolyDataMapper.h>
#include <vtkPolyData.h>
#include <vtkContourFilter.h>
#include <SCIRun/Vtk/Port.h>
#include <CCA/Components/VtkTest/PolyDataMapper/PolyDataMapper.h>

using namespace std;
using namespace SCIRun;
using namespace vtk;

extern "C" vtk::Component* make_Vtk_PolyDataMapper()
{
  return new PolyDataMapper;
}

bool 
PolyDataMapper::accept(OutPort* port){
  return dynamic_cast<vtkPolyData*>(port->getOutput())!=0;
}

void
PolyDataMapper::connect(OutPort* port){
  mapper->SetInput(dynamic_cast<vtkPolyData*>(port->getOutput()));
  mapper->ScalarVisibilityOff();
}

PolyDataMapper::PolyDataMapper(){
  //set input port name
  InPort::setName("PolyDataMapper::input");

  //set output port name
  OutPort::setName("PolyDataMapper::output");

  mapper=vtkPolyDataMapper::New();

  setOutput(mapper);

  addPort(dynamic_cast<InPort*>(this));
  addPort(dynamic_cast<OutPort*>(this));
}

PolyDataMapper::~PolyDataMapper(){
  mapper->Delete();
}
