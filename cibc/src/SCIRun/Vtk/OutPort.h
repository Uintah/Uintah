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
 *  OutPort.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#ifndef SCIRun_Vtk_OutPort_h
#define SCIRun_Vtk_OutPort_h

#include <SCIRun/Vtk/Port.h>

class vtkObject;
namespace SCIRun {
  namespace vtk {
    /**
     * \class OutPort
     *
     * A virtual class that defines the interface for a SCIRun::vtk output port.
     * An output port is a component port that provides data. Connects between
     * input and output ports are made in the InPort object.
     */
    class OutPort : public SCIRun::vtk::Port{
    public:
      friend class VtkPortInstance;
      OutPort();
      virtual ~OutPort();
      
      /** Returns the vtkObject that will receive the output data. */
      virtual vtkObject* getOutput()=0;
      
      void update(int flag); //virtual 
      
    private:
      /** Sets the vtkObject that will receive the output data. */
      void setOutput(vtkObject* object);
      
      //the output vtk object
      vtkObject* vtkobj;
    };
    
  }
}

#endif
