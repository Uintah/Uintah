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

using namespace std;
using namespace SCIRun;
using namespace vtk;

extern "C" vtk::Component* make_Vtk_Renderer()
{
  return new Renderer;
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
  return "Renderer::input";
}


Renderer::Renderer(){
  iPort=new IPort;

  iports.push_back(iPort);
  
  ren1=vtkRenderer::New();
  renWin=vtkRenderWindow::New();
  iren=vtkRenderWindowInteractor::New();
  ren1->GetActiveCamera()->Azimuth(20);
  
  ren1->GetActiveCamera()->Elevation(30);

  ren1->SetBackground(0.1,0.2,0.4);
  
  renWin->SetSize( 500, 500);

  ren1->GetActiveCamera()->Zoom(1.4);

  ren1->ResetCameraClippingRange();
}

Renderer::~Renderer(){
  delete iPort;
  ren1->Delete();
  renWin->Delete();
  iren->Delete();

}

bool
Renderer::haveUI(){
  return true;
}

int
Renderer::popupUI(){
  renWin->Render();

  iren->Initialize();

  iren->Start();  
  return 0;
}
