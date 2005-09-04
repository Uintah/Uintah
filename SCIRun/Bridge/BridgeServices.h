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
 *  BridgeServices.h:
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   September 2003
 *
 */


#ifndef SCIRUN_BRIDGE_BRIDGESERVICES_H
#define SCIRUN_BRIDGE_BRIDGESERVICES_H
                                                                                                            
#include <sci_defs/vtk_defs.h>

//CCA:
#include <Core/CCA/spec/sci_sidl.h>
//Babel:
#include <SCIRun/Babel/framework.hh>
#include <SCIRun/Babel/gov_cca.hh>
//Dataflow:
#include <Dataflow/Network/Port.h>

//Vtk:
#if HAVE_VTK
 #include <SCIRun/Vtk/InPort.h>
 #include <SCIRun/Vtk/OutPort.h>
 #include <SCIRun/Vtk/VtkPortInstance.h>
#endif


namespace SCIRun {

  typedef enum {
    CCA = 1,
    Babel,
    Dataflow,
    Vtk,
    Tao
  } modelT;
  
  class BridgeServices {
  public:
    BridgeServices::BridgeServices() { }
    virtual BridgeServices::~BridgeServices() { }

    virtual Port* getDataflowIPort(const std::string& name) = 0;    
    virtual Port* getDataflowOPort(const std::string& name) = 0;
    virtual sci::cca::Port::pointer getCCAPort(const std::string& name) = 0;
    virtual gov::cca::Port getBabelPort(const std::string& name) = 0;

#if HAVE_VTK
    virtual vtk::Port* getVtkPort(const std::string& name) = 0; 
    virtual void addVtkPort(vtk::Port* vtkport, VtkPortInstance::PortType portT) = 0;
#endif
 
    virtual void releasePort(const std::string& name, const modelT model) = 0;
    virtual void registerUsesPort(const std::string& name, const std::string& type,
			  const modelT model) = 0;
    virtual void unregisterUsesPort(const std::string& name, const modelT model) = 0;
    virtual void addProvidesPort(void* port,
			 const std::string& name,
			 const std::string& type,
			 const modelT model) = 0;
    virtual void removeProvidesPort(const std::string& name, const modelT model) = 0;
    virtual sci::cca::ComponentID::pointer getComponentID() = 0;
  };    
}

#endif
