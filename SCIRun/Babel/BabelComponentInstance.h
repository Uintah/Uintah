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
#include <SCIRun/Babel/govcca.hh>
#include <map>
#include <string>

namespace SCIRun {

  class BabelCCAGoPort : public virtual gov::cca::ports::GoPort{
  public:
    BabelCCAGoPort(const govcca::GoPort &port);
    virtual int go();
  private:
    govcca::GoPort port;
  };

  class BabelCCAUIPort : public virtual gov::cca::ports::UIPort{
  public:
    BabelCCAUIPort(const govcca::UIPort &port);
    virtual int ui();
  private:
    govcca::UIPort port;
  };


  class BabelPortInstance;
  class Services;

  class BabelComponentInstance : public ComponentInstance{
  public:
    BabelComponentInstance(SCIRunFramework* framework,
			 const std::string& instanceName,
			 const std::string& className,
			 const govcca::TypeMap& typemap,
			 const govcca::Component& component,
			 const framework::Services& svc);
    virtual ~BabelComponentInstance();

    // Methods from govcca::Services
    govcca::Port getPort(const std::string& name);
    govcca::Port getPortNonblocking(const std::string& name);
    void releasePort(const std::string& name);
    govcca::TypeMap createTypeMap();
    void registerUsesPort(const std::string& name, const std::string& type,
			  const govcca::TypeMap& properties);
    void unregisterUsesPort(const std::string& name);
    void addProvidesPort(const govcca::Port& port,
			 const std::string& name,
			 const std::string& type,
			 const govcca::TypeMap& properties);
    void removeProvidesPort(const std::string& name);
    govcca::TypeMap getPortProperties(const std::string& portName);
    govcca::ComponentID getComponentID();

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
      //gov::cca::ComponentID::pointer cid;
    };

    govcca::Component component;
    BabelComponentInstance(const BabelComponentInstance&);
    BabelComponentInstance& operator=(const BabelComponentInstance&);
  };
}

#endif




















