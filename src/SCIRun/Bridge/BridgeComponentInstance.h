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
 *  BridgeComponentInstance.h: 
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   September 2003
 *
 */

#ifndef SCIRun_Framework_BridgeComponentInstance_h
#define SCIRun_Framework_BridgeComponentInstance_h

#include <sci_defs/vtk_defs.h>

#include <SCIRun/ComponentInstance.h>
#include <SCIRun/PortInstance.h>
#include <SCIRun/PortInstanceIterator.h>

#include <SCIRun/CCA/CCAPortInstance.h>
#include <SCIRun/Babel/BabelPortInstance.h>
#include <SCIRun/Dataflow/SCIRunPortInstance.h>

#if HAVE_VTK
 #include <SCIRun/Vtk/VtkPortInstance.h>
 #include <SCIRun/Vtk/Component.h>
#endif

#include <SCIRun/Bridge/BridgeModule.h>

#include <SCIRun/Bridge/BridgeServices.h>
#include <SCIRun/Bridge/BridgeComponent.h>

#include <Core/CCA/PIDL/Object.h>
#include <map>
#include <string>

namespace SCIRun {
  class Services;
  class Mutex;
  class BridgeComponentInstance : public ComponentInstance, public BridgeServices {
  public:
    BridgeComponentInstance(SCIRunFramework* framework,
			 const std::string& instanceName,
			 const std::string& className,
			 BridgeComponent* component);
    virtual ~BridgeComponentInstance();

    // Methods from BridgeServices
    Port* getDataflowIPort(const std::string& name);
    Port* getDataflowOPort(const std::string& name);
    sci::cca::Port::pointer getCCAPort(const std::string& name);
    gov::cca::Port getBabelPort(const std::string& name);

#if HAVE_VTK
    vtk::Port* getVtkPort(const std::string& name);
    void addVtkPort(vtk::Port* vtkport, VtkPortInstance::PortType portT);
#endif

    void releasePort(const std::string& name,const modelT model);
    void registerUsesPort(const std::string& name, const std::string& type,
			  const modelT model);
    void unregisterUsesPort(const std::string& name, const modelT model);
    void addProvidesPort(void* port,
			 const std::string& name,
			 const std::string& type,
			 const modelT model);
    void removeProvidesPort(const std::string& name, const modelT model);
    sci::cca::ComponentID::pointer getComponentID();

    // Methods from ComponentInstance
    virtual PortInstance* getPortInstance(const std::string& name);
    virtual PortInstanceIterator* getPorts();

  private:
    //ITERATOR CLASS
    class Iterator : public PortInstanceIterator {
      std::map<std::string, PortInstance*>::iterator iter;
      BridgeComponentInstance* comp;
    public:
      Iterator(BridgeComponentInstance*);
      virtual ~Iterator();
      virtual PortInstance* get();
      virtual bool done();
      virtual void next();
    private:
      Iterator(const Iterator&);
      Iterator& operator=(const Iterator&);
    };
    //EOF ITERATOR CLASS

    std::map<std::string, PortInstance*> ports;
 
    BridgeComponent* component;
    Mutex *mutex;

    //Dataflow:
    BridgeModule* bmdl; 
   
    BridgeComponentInstance(const BridgeComponentInstance&);
    BridgeComponentInstance& operator=(const BridgeComponentInstance&);
  };
}

#endif
