/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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


#include "Renderer.h"

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
  renWindowActive=false;
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
  //  renWin->Delete();
}

void 
Renderer::refresh(int flag){
  if(!renWindowActive) return;
  /*  if(flag==Port::RESETCAMERA){
    ren1->ResetCamera();
    renWin->Render();
    iren->Render(); 
  }
  if(flag==Port::REFRESH){
    ren1->ResetCamera();
    renWin->Render();
    iren->Render(); 
  }
  */
}

int
Renderer::popupUI(){

  renWin=vtkRenderWindow::New();
  renWin->SetSize( 500, 500);
  renWin->AddRenderer(ren1);
  ren1->ResetCamera();
  // set the user-defined C-style ExitMethod as the new exit method
  iren=vtkRenderWindowInteractor::New();
  iren->SetRenderWindow(renWin);
  renWindowActive=true;
  iren->Initialize();
  iren->Start();
  /*  while(1){
    iren->Render(); 
    sleep(1);
    }*/
  //  iren->Delete();
  //  renWin->Delete();
  //  renWindowActive=false;
  return 0;
}

