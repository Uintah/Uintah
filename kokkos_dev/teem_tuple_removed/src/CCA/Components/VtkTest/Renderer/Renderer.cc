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


#include <CCA/Components/VtkTest/Renderer/Renderer.h>

#include <iostream>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderWindow.h>
#include <vtkCamera.h>
#include <vtkMapper.h>
#include <vtkProp.h>
#include <vtkActor.h>

using namespace std;
using namespace SCIRun;
using namespace vtk;

extern "C" vtk::Component* make_Vtk_Renderer()
{
  return new Renderer;
}

bool 
Renderer::accept(OutPort* port){
  return dynamic_cast<vtkProp*>(port->getOutput())!=0;
}

void
Renderer::connect(OutPort* port){
  ren1->AddActor(dynamic_cast<vtkProp*>(port->getOutput()));
}

Renderer::Renderer(){
  //Set Input port name
  setName("Renderer::input");

  ren1=vtkRenderer::New();
  ren1->SetBackground(0.1,0.2,0.4);
  ren1->ResetCameraClippingRange();

  //register the inport
  addPort(this);

  //enable UI (using new thread)
  enableUI(true);
}

Renderer::~Renderer(){
  ren1->Delete();
}

int
Renderer::popupUI(){
  vtkRenderWindow *renWin;
  vtkRenderWindowInteractor *iren;

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
