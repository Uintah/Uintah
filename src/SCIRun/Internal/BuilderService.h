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
 *  BuilderService.h: Implementation of the CCA BuilderService interface for SCIRun
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_BuilderService_h
#define SCIRun_BuilderService_h

#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/Internal/InternalComponentModel.h>
#include <SCIRun/Internal/InternalComponentInstance.h>

namespace SCIRun {
  class SCIRunFramework;
  class BuilderService : public gov::cca::ports::BuilderService, public InternalComponentInstance {
  public:
    virtual ~BuilderService();

    static InternalComponentInstance* create(SCIRunFramework* framework,
					     const std::string& name);


    virtual gov::cca::ComponentID::pointer createInstance(const std::string& instanceName,
						 const std::string& className,
						 const gov::cca::TypeMap::pointer& properties);
    virtual SIDL::array1<gov::cca::ComponentID::pointer> getComponentIDs();
    virtual gov::cca::TypeMap::pointer getComponentProperties(const gov::cca::ComponentID::pointer& cid);
    virtual void setComponentProperties(const gov::cca::ComponentID::pointer& cid,
					const gov::cca::TypeMap::pointer& map);
    virtual gov::cca::ComponentID::pointer getDeserialization(const std::string& s);
    virtual gov::cca::ComponentID::pointer getComponentID(const std::string& componentInstanceName);
    virtual void destroyInstance(const gov::cca::ComponentID::pointer& toDie,
				 float timeout);
    virtual SIDL::array1<std::string> getProvidedPortNames(const gov::cca::ComponentID::pointer& cid);
    virtual SIDL::array1<std::string> getUsedPortNames(const gov::cca::ComponentID::pointer& cid);
    virtual gov::cca::TypeMap::pointer getPortProperties(const gov::cca::ComponentID::pointer& cid,
						const std::string& portname);
    virtual void setPortProperties(const gov::cca::ComponentID::pointer& cid,
				   const std::string& portname,
				   const gov::cca::TypeMap::pointer& map);
    virtual gov::cca::ConnectionID::pointer connect(const gov::cca::ComponentID::pointer& user,
					   const std::string& usingPortName,
					   const gov::cca::ComponentID::pointer& provider,
					   const ::std::string& providingPortName);
    virtual SIDL::array1<gov::cca::ConnectionID::pointer> getConnectionIDs(const SIDL::array1<gov::cca::ComponentID::pointer>& componentList);
    virtual gov::cca::TypeMap::pointer getConnectionProperties(const gov::cca::ConnectionID::pointer& connID);
    virtual void setConnectionProperties(const gov::cca::ConnectionID::pointer& connID,
					 const gov::cca::TypeMap::pointer& map);
    virtual void disconnect(const gov::cca::ConnectionID::pointer& connID,
			    float timeout);
    virtual void disconnectAll(const gov::cca::ComponentID::pointer& id1,
			       const gov::cca::ComponentID::pointer& id2,
			       float timeout);

    virtual SIDL::array1<std::string> getCompatiblePortList(
		const gov::cca::ComponentID::pointer& c1,
		const std::string& port1,
		const gov::cca::ComponentID::pointer& c2);

    gov::cca::Port::pointer getService(const std::string&);
    //virtual void registerFramework(const std::string &frameworkURL); 
    //virtual void registerServices(const gov::cca::Services::pointer &svc);
  private:
    BuilderService(SCIRunFramework* fwk, const std::string& name);
    std::vector<gov::cca::Services::pointer> servicesList;	
  };
}

#endif
