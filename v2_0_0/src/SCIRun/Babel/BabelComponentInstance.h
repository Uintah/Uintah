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
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   July 2002
 *
 */
#ifndef SCIRun_Framework_BabelComponentInstance_h
#define SCIRun_Framework_BabelComponentInstance_h

#include <SCIRun/Babel/BabelPortInstance.h>
#include <SCIRun/ComponentInstance.h>
#include <SCIRun/PortInstanceIterator.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/Babel/framework.hh>
#include <SCIRun/Babel/gov_cca.hh>
#include <map>
#include <string>

namespace SCIRun {

  class BabelCCAGoPort : public virtual sci::cca::ports::GoPort{
  public:
    BabelCCAGoPort(const gov::cca::ports::GoPort &port);
    virtual int go();
  private:
    gov::cca::ports::GoPort port;
  };

  class BabelCCAUIPort : public virtual sci::cca::ports::UIPort{
  public:
    BabelCCAUIPort(const gov::cca::ports::UIPort &port);
    virtual int ui();
  private:
    gov::cca::ports::UIPort port;
  };


  class BabelPortInstance;
  class Services;

  class BabelComponentInstance : public ComponentInstance{
  public:
    BabelComponentInstance(SCIRunFramework* framework,
			 const std::string& instanceName,
			 const std::string& className,
			 const gov::cca::TypeMap& typemap,
			 const gov::cca::Component& component,
			 const framework::Services& svc);
    virtual ~BabelComponentInstance();

    // Methods from gov::cca::Services
    gov::cca::Port getPort(const std::string& name);
    gov::cca::Port getPortNonblocking(const std::string& name);
    void releasePort(const std::string& name);
    gov::cca::TypeMap createTypeMap();
    void registerUsesPort(const std::string& name, const std::string& type,
			  const gov::cca::TypeMap& properties);
    void unregisterUsesPort(const std::string& name);
    void addProvidesPort(const gov::cca::Port& port,
			 const std::string& name,
			 const std::string& type,
			 const gov::cca::TypeMap& properties);
    void removeProvidesPort(const std::string& name);
    gov::cca::TypeMap getPortProperties(const std::string& portName);
    gov::cca::ComponentID getComponentID();

    // Methods from ComponentInstance
    virtual PortInstance* getPortInstance(const std::string& name);
    virtual PortInstanceIterator* getPorts();
 private:
    framework::Services svc;
    class Iterator : public PortInstanceIterator {
      std::map<std::string, PortInstance*> *ports;
      std::map<std::string, PortInstance*>::iterator iter;
    public:
      Iterator(BabelComponentInstance*);
      virtual ~Iterator();
      virtual PortInstance* get();
      virtual bool done();
      virtual void next();
    private:
      Iterator(const Iterator&);
      Iterator& operator=(const Iterator&);
      //sci::cca::ComponentID::pointer cid;
    };

    gov::cca::Component component;
    BabelComponentInstance(const BabelComponentInstance&);
    BabelComponentInstance& operator=(const BabelComponentInstance&);
  };
}

#endif




















