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

#include <SCIRun/Vtk/InPort.h>
#include <SCIRun/Vtk/OutPort.h>
#include <SCIRun/Vtk/Component.h>
#include <vector>

class vtkRenderer;
class vtkRenderWindow;
class vtkRenderWindowInteractor;

namespace SCIRun {
  namespace vtk{
    class Renderer: public Component, public InPort{
      
    public:
      //constructor
      Renderer();

      //destructor
      ~Renderer();

      //Component interface
      int popupUI();

      void refresh(int flag); //virtual
      
      //InPort interfaces
      bool accept(OutPort* port);

      //InPort interfaces
      void connect(OutPort* port);
    private:
      bool renWindowActive;
      vtkRenderer *ren1; 
      vtkRenderWindowInteractor *iren;
  
      vtkRenderWindow *renWin;
      Renderer(const Renderer&);
      Renderer& operator=(const Renderer&);
    };
  } //namespace vtk
} //namespace SCIRun


#endif
