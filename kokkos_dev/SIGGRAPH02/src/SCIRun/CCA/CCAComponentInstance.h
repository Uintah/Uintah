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
 *  CCAComponentInstance.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_CCAComponentInstance_h
#define SCIRun_Framework_CCAComponentInstance_h

#include <SCIRun/ComponentInstance.h>
#include <SCIRun/PortInstanceIterator.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <map>
#include <string>

namespace SCIRun {
  class CCAPortInstance;
  class Services;

  class CCAComponentInstance : public ComponentInstance, public gov::cca::Services {
  public:
    CCAComponentInstance(SCIRunFramework* framework,
			 const std::string& instanceName,
			 const std::string& className,
			 const gov::cca::TypeMap::pointer& typemap,
			 const gov::cca::Component::pointer& component);
    virtual ~CCAComponentInstance();

    // Methods from gov::cca::Services
    gov::cca::Port::pointer getPort(const std::string& name);
    gov::cca::Port::pointer getPortNonblocking(const std::string& name);
    void releasePort(const std::string& name);
    gov::cca::TypeMap::pointer createTypeMap();
    void registerUsesPort(const std::string& name, const std::string& type,
			  const gov::cca::TypeMap::pointer& properties);
    void unregisterUsesPort(const std::string& name);
    void addProvidesPort(const gov::cca::Port::pointer& port,
			 const std::string& name,
			 const std::string& type,
			 const gov::cca::TypeMap::pointer& properties);
    void removeProvidesPort(const std::string& name);
    gov::cca::TypeMap::pointer getPortProperties(const std::string& portName);
    gov::cca::ComponentID::pointer getComponentID();

    // Methods from ComponentInstance
    virtual PortInstance* getPortInstance(const std::string& name);
    virtual PortInstanceIterator* getPorts();
 private:
    class Iterator : public PortInstanceIterator {
      std::map<std::string, CCAPortInstance*>::iterator iter;
      CCAComponentInstance* comp;
    public:
      Iterator(CCAComponentInstance*);
      virtual ~Iterator();
      virtual PortInstance* get();
      virtual bool done();
      virtual void next();
    private:
      Iterator(const Iterator&);
      Iterator& operator=(const Iterator&);
      //gov::cca::ComponentID::pointer cid;
    };
    std::map<std::string, CCAPortInstance*> ports;
    gov::cca::Component::pointer component;

    CCAComponentInstance(const CCAComponentInstance&);
    CCAComponentInstance& operator=(const CCAComponentInstance&);
  };
}

#endif
