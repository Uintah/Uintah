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
#include <CCA/Components/VtkTest/Renderer/Renderer.h>

#include "vtkRenderer.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderWindow.h"
#include "vtkCamera.h"
#include "vtkMapper.h"
#include "vtkProp.h"
#include "vtkActor.h"

using namespace std;
using namespace SCIRun;
using namespace vtk;

extern "C" vtk::Component* make_Vtk_Renderer()
{
  return new Renderer;
}

//Input Port
IPort::IPort(vtkRenderer *ren){
  this->ren=ren;
}

IPort::~IPort(){

}

bool
IPort::isInput(){
  return true;
}

std::string
IPort::getName(){
  return "Renderer::input";
}


bool 
IPort::accept(Port* port){
  return dynamic_cast<vtkProp*>(port->getObj())!=0;
}

void
IPort::connect(Port* port){
  ren->AddActor(dynamic_cast<vtkProp*>(port->getObj()));
}

Renderer::Renderer(){
  ren1=vtkRenderer::New();
  ren1->SetBackground(0.1,0.2,0.4);
  ren1->ResetCameraClippingRange();
  iports.push_back(new IPort(ren1));
  


  //  ren1->GetActiveCamera()->Azimuth(20);
  
  //ren1->GetActiveCamera()->Elevation(30);


  
  //ren1->GetActiveCamera()->Zoom(1.4);

}

Renderer::~Renderer(){
  ren1->Delete();

}

bool
Renderer::haveUI(){
  return true;
}

int
Renderer::popupUI(){
  renWin=vtkRenderWindow::New();
  renWin->SetSize( 500, 500);
  renWin->AddRenderer(ren1);

  iren=vtkRenderWindowInteractor::New();
  iren->SetRenderWindow(renWin);

  renWin->Render();

  iren->Initialize();

  iren->Start();  

  renWin->Delete();
  iren->Delete();
  return 0;
}
