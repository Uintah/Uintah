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



#include "vtkActor.h"
#include "vtkAxes.h"
#include "vtkCamera.h"
#include "vtkContourFilter.h"
#include "vtkDataSet.h"
#include "vtkFloatArray.h"
#include "vtkGaussianSplatter.h"
#include "vtkImageData.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkPolyDataMapper.h"
#include "vtkProperty.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkStructuredPointsReader.h"
#include "vtkStructuredPoints.h"
#include "vtkContourFilter.h"
#include "vtkRenderer.h"
#include "vtkTubeFilter.h"
#include "vtkUnstructuredGrid.h"



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
  filter->SetValue(0,7);

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


bool
ContourFilter::haveUI(){
  return true;
}

int
ContourFilter::popupUI(){

  std::cerr<<"Executing popupUI in ContourFilter...\n";

  vtkActor* actor=vtkActor::New();

  vtkPolyDataMapper* mapper =vtkPolyDataMapper::New();

  vtkRenderer *ren1=vtkRenderer::New();
  
  vtkRenderWindow *renWin=vtkRenderWindow::New();

  vtkRenderWindowInteractor *iren=vtkRenderWindowInteractor::New();

  mapper->SetInput(filter->GetOutput());

  mapper->ScalarVisibilityOff();

  actor->SetMapper(mapper);

  ren1->AddActor(actor);

  ren1->GetActiveCamera()->Azimuth(20);
  
  ren1->GetActiveCamera()->Elevation(30);

  ren1->SetBackground(0.1,0.2,0.4);
  
  ren1->GetActiveCamera()->Zoom(1.4);

  ren1->ResetCameraClippingRange();

  renWin->AddRenderer(ren1);
    
  iren->SetRenderWindow(renWin);

  renWin->SetSize( 500, 500);

  //Now start working...
  renWin->Render();

  iren->Initialize();

  iren->Start();

  return 0;
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

