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
 *  ContourFilter.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#include <iostream>
#include <vtkContourFilter.h>
#include <vtkDataSet.h>
#include <vtkStructuredPoints.h>
#include <vtkPolyData.h>

#include <SCIRun/Vtk/Port.h>
#include <CCA/Components/VtkTest/ContourFilter/ContourFilter.h>

using namespace std;
using namespace SCIRun;
using namespace vtk;

extern "C" vtk::Component* make_Vtk_ContourFilter()
{
  return new ContourFilter;
}

bool 
ContourFilter::accept(OutPort* port){
  return dynamic_cast<vtkDataSet*>(port->getOutput())!=0;
}

void
ContourFilter::connect(OutPort* port){
  filter->SetInput(dynamic_cast<vtkDataSet*>(port->getOutput()));
  //TODO: use GUI
  filter->SetValue(0,70);
}

ContourFilter::ContourFilter(){
  //set input port name
  InPort::setName("ContourFilter::input");

  //set output port name
  OutPort::setName("ContourFilter::output");

  filter=vtkContourFilter::New();

  setOutput(filter->GetOutput());

  addPort(dynamic_cast<InPort*>(this));
  addPort(dynamic_cast<OutPort*>(this));
}

ContourFilter::~ContourFilter(){
  filter->Delete();
}

