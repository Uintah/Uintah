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
#include <SCIRun/Vtk/Port.h>
#include <CCA/Components/VtkTest/ContourFilter/ContourFilter.h>

#include "vtkContourFilter.h"
#include "vtkDataSet.h"
#include "vtkStructuredPoints.h"
#include "vtkPolyData.h"
using namespace std;
using namespace SCIRun;
using namespace vtk;


extern "C" vtk::Component* make_Vtk_ContourFilter()
{
  return new ContourFilter;
}

//Input Port
IPort::IPort(vtkContourFilter *filter){
  this->filter=filter;
}
  

IPort::~IPort(){

}

bool
IPort::isInput(){
  return true;
}

std::string
IPort::getName(){
  return "ContourFilter::input";
}

bool 
IPort::accept(Port* port){
  return dynamic_cast<vtkDataSet*>(port->getObj())!=0;
}

void
IPort::connect(Port* port){
  filter->SetInput(dynamic_cast<vtkDataSet*>(port->getObj()));
  //TODO: use GUI
  filter->SetValue(0,70);
}

//Output Port

OPort::OPort(vtkContourFilter *filter){
  this->filter=filter;
}

OPort::~OPort(){

}

bool
OPort::isInput(){
  return false;
}

std::string
OPort::getName(){
  return "ContourFilter::output";
}

vtkObject *
OPort::getObj(){
  return filter->GetOutput();
}

ContourFilter::ContourFilter(){

  filter=vtkContourFilter::New();
  iports.push_back(new IPort(filter));
  oports.push_back(new OPort(filter));
}

ContourFilter::~ContourFilter(){
  for(unsigned int i=0; i<iports.size(); i++){
    delete iports[i];
  }
  for(unsigned int i=0; i<oports.size(); i++){
    delete oports[i];
  }
  filter->Delete();
}

