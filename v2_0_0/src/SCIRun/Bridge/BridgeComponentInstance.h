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

#include <SCIRun/ComponentInstance.h>
#include <SCIRun/PortInstance.h>
#include <SCIRun/PortInstanceIterator.h>

#include <SCIRun/CCA/CCAPortInstance.h>

#include <SCIRun/Babel/BabelPortInstance.h>

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
    sci::cca::Port::pointer getCCAPort(const std::string& name);
    gov::cca::Port getBabelPort(const std::string& name);
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

    BridgeComponentInstance(const BridgeComponentInstance&);
    BridgeComponentInstance& operator=(const BridgeComponentInstance&);
  };
}

#endif
