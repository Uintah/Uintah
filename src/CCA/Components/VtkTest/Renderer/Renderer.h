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
 *  Renderer.h
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#ifndef SCIRun_VTK_Components_Renderer_h
#define SCIRun_VTK_Components_Renderer_h

#include <SCIRun/Vtk/Port.h>
#include <SCIRun/Vtk/Component.h>
#include <vector>

class vtkRenderer;
class vtkRenderWindow;
class vtkRenderWindowInteractor;

namespace SCIRun {
  namespace vtk{
  class IPort : public vtk::Port {
  public:
    IPort();
    virtual ~IPort();
    bool isInput();
    std::string getName();
  };
  
  class Renderer: public vtk::Component{
    
  public:
    Renderer();
    virtual ~Renderer();
    bool haveUI();
    int popupUI();
  private:
    IPort *iPort;
    vtkRenderer *ren1; 
    vtkRenderWindow *renWin;
    vtkRenderWindowInteractor *iren;

    Renderer(const Renderer&);
    Renderer& operator=(const Renderer&);
  };
  
  
} //namespace vtk
} //namespace SCIRun


#endif
