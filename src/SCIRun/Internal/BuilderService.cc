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
 *  BuilderService.cc: Implementation of CCA BuilderService for SCIRun
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/Internal/BuilderService.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <SCIRun/PortInstanceIterator.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/CCA/CCAException.h>
#include <SCIRun/PortInstance.h>
#include <SCIRun/CCA/ComponentID.h>
#include <SCIRun/CCA/CCAComponentInstance.h>
#include <SCIRun/ComponentInstance.h>
#include <SCIRun/CCA/ConnectionID.h>
#include <SCIRun/Internal/ConnectionEvent.h>
#include <SCIRun/Internal/ConnectionEventService.h>
#include <iostream>
#include <string>

using namespace std;

namespace SCIRun {

BuilderService::BuilderService(SCIRunFramework* framework,
                   const std::string& name)
  : InternalComponentInstance(framework, name, "internal:BuilderService")
{
    this->framework=framework;
}

BuilderService::~BuilderService()
{
}

sci::cca::ComponentID::pointer
BuilderService::createInstance(const std::string& instanceName,
                   const std::string& className,
                   const sci::cca::TypeMap::pointer& properties)
{
    return framework->createComponentInstance(instanceName, className, properties);
}

sci::cca::ConnectionID::pointer
BuilderService::connect(const sci::cca::ComponentID::pointer &user,
                        const std::string &usesPortName,
                        const sci::cca::ComponentID::pointer &provider,
                        const ::std::string &providesPortName)
{
    ComponentID* uCID = dynamic_cast<ComponentID*>(user.getPointer());
    ComponentID* pCID = dynamic_cast<ComponentID*>(provider.getPointer());
    if (! uCID) {
        throw sci::cca::CCAException::pointer(new CCAException("Cannot connect: invalid user componentID"));
    }
    if (! pCID) {
        throw sci::cca::CCAException::pointer(new CCAException("Cannot connect: invalid provider componentID"));
    }
    if (uCID->framework != framework || pCID->framework != framework) {
        throw sci::cca::CCAException::pointer(new CCAException("Cannot connect components from different frameworks"));
    }
    ComponentInstance* uCI =
        framework->lookupComponent(user->getInstanceName());
    if (! uCI) {
        throw sci::cca::CCAException::pointer(new CCAException("Unknown ComponentInstance " + user->getInstanceName()));
    }
    sci::cca::TypeMap::pointer uProps = uCI->getComponentProperties();

    PortInstance* usesPort = uCI->getPortInstance(usesPortName);
    if (! usesPort) {
      throw sci::cca::CCAException::pointer(new CCAException("Unknown port " + usesPortName, sci::cca::BadPortName));
    }

    ComponentInstance* pCI =
        framework->lookupComponent(provider->getInstanceName());
    if (! pCI) {
        throw sci::cca::CCAException::pointer(new CCAException("Unknown ComponentInstance " + provider->getInstanceName()));
    }
    sci::cca::TypeMap::pointer pProps = pCI->getComponentProperties();
    PortInstance* providesPort = pCI->getPortInstance(providesPortName);
    if (! providesPort) {
        throw sci::cca::CCAException::pointer(new CCAException("Unknown port " + providesPortName));
    }

    if (! usesPort->connect(providesPort)) {
std::cerr << "BuilderService::connect: attempt to connect " << usesPortName << " with " << providesPortName << " failed." << std::endl;
        throw sci::cca::CCAException::pointer(new CCAException("Cannot connect " + usesPortName + " with " + providesPortName));
    }

    bool isBridge = uProps->getBool("bridge", false);
    if (! isBridge) {
        isBridge = pProps->getBool("bridge", false);
    }

    sci::cca::TypeMap::pointer properties = framework->createTypeMap();
    properties->putString("user", uCID->getInstanceName());
    properties->putString("provider", pCID->getInstanceName());
    properties->putString("uses port", usesPortName);
    properties->putString("provides port", providesPortName);
    properties->putBool("bridge", isBridge);

    sci::cca::ConnectionID::pointer conn(new ConnectionID(user, usesPortName, provider, providesPortName));
    framework->connIDs.push_back(conn);
    emitConnectionEvent(
        new ConnectionEvent(sci::cca::ports::Connected, properties));
    return conn;
}

InternalComponentInstance*
BuilderService::create(SCIRunFramework* framework, const std::string& name)
{
    BuilderService* n = new BuilderService(framework, name);
    n->addReference();
    return n;
}

sci::cca::Port::pointer
BuilderService::getService(const std::string&)
{
    return sci::cca::Port::pointer(this);
}

SSIDL::array1<sci::cca::ComponentID::pointer>
BuilderService::getComponentIDs()
{
    return framework->compIDs;
}

sci::cca::TypeMap::pointer
BuilderService::getComponentProperties(const sci::cca::ComponentID::pointer &cid)
{
    if (cid.isNull()) {
        throw sci::cca::CCAException::pointer(new CCAException("Invalid ComponentID"));
    }

    ComponentInstance *ci = framework->lookupComponent(cid->getInstanceName());
    if (! ci) {
        throw sci::cca::CCAException::pointer(new CCAException("Framework could not locate component " + cid->getInstanceName()));
    }
    return ci->getComponentProperties();
}

void
BuilderService::setComponentProperties(const sci::cca::ComponentID::pointer &cid,
                                       const sci::cca::TypeMap::pointer &map)
{
    if (cid.isNull()) {
        throw sci::cca::CCAException::pointer(new CCAException("Invalid ComponentID"));
    }
    if (map.isNull()) {
        throw sci::cca::CCAException::pointer(new CCAException("Invalid TypeMap"));
    }

    ComponentInstance *ci = framework->lookupComponent(cid->getInstanceName());
    if (! ci) {
        throw sci::cca::CCAException::pointer(new CCAException("Framework could not locate component " + cid->getInstanceName()));
    }
    ci->setComponentProperties(map);
}

sci::cca::ComponentID::pointer
BuilderService::getDeserialization(const std::string& /*s*/)
{
    // TODO: finish this!
  std::cerr << "BuilderService::getDeserialization not finished\n";
  return sci::cca::ComponentID::pointer(0);
}

sci::cca::ComponentID::pointer
BuilderService::getComponentID(const std::string &componentInstanceName)
{
    sci::cca::ComponentID::pointer cid =
        framework->lookupComponentID(componentInstanceName);
    if (cid.isNull()) {
        throw sci::cca::CCAException::pointer(new CCAException("ComponentID not found"));
    }
    return cid;
}

void
BuilderService::destroyInstance(const sci::cca::ComponentID::pointer &toDie, float timeout)
{
    framework->destroyComponentInstance(toDie, timeout);
    return;
}

SSIDL::array1<std::string>
BuilderService::getProvidedPortNames(const sci::cca::ComponentID::pointer &cid)
{
    SSIDL::array1<std::string> result;
    ComponentInstance *ci =
        framework->lookupComponent(cid->getInstanceName());
    if (! ci) {
        throw sci::cca::CCAException::pointer(new CCAException("Invalid component " + cid->getInstanceName()));
    }

    for (PortInstanceIterator* iter = ci->getPorts();
            !iter->done(); iter->next()) {
        PortInstance* port = iter->get();
        if (port->portType() == PortInstance::To) {
            result.push_back(port->getUniqueName());
        }
    }
    return result;
}

SSIDL::array1<std::string>
BuilderService::getUsedPortNames(const sci::cca::ComponentID::pointer &cid)
{
    SSIDL::array1<std::string> result;
    ComponentInstance *ci =
        framework->lookupComponent(cid->getInstanceName());
    if (! ci) {
        throw sci::cca::CCAException::pointer(new CCAException("Invalid component " + cid->getInstanceName()));
    }

    for (PortInstanceIterator* iter = ci->getPorts();
            !iter->done(); iter->next()) {
        PortInstance* port = iter->get();
        if (port->portType() == PortInstance::From) {
            result.push_back(port->getUniqueName());
        }
    }
    return result;
}

// TODO: make sure that port properties are actually created
// TODO: extend to other component models
sci::cca::TypeMap::pointer
BuilderService::getPortProperties(const sci::cca::ComponentID::pointer &cid, const std::string &portname)
{
    ComponentInstance* comp = framework->lookupComponent(cid->getInstanceName());
    if (! comp) {
        return framework->createTypeMap();
    }
    CCAComponentInstance* ccaComp = dynamic_cast<CCAComponentInstance*>(comp);
    if (! ccaComp) {
        return framework->createTypeMap();
    }
    return ccaComp->getPortProperties(portname);
}

void BuilderService::setPortProperties(const sci::cca::ComponentID::pointer& /*cid*/,
                       const std::string& /*portname*/,
                       const sci::cca::TypeMap::pointer& /*map*/)
{
    // TODO: finish this!!!
    std::cerr << "BuilderService::setPortProperties not finished\n";
}

SSIDL::array1<sci::cca::ConnectionID::pointer>
BuilderService::getConnectionIDs(const SSIDL::array1<sci::cca::ComponentID::pointer> &componentList)
{
    SSIDL::array1<sci::cca::ConnectionID::pointer> conns;
    for (unsigned i = 0; i < framework->connIDs.size(); i++) {
        for (unsigned j = 0; j < componentList.size(); j++) {
            sci::cca::ComponentID::pointer userCID =
                framework->connIDs[i]->getUser();
            sci::cca::ComponentID::pointer provCID =
                framework->connIDs[i]->getProvider();
            if (userCID == componentList[j] || provCID == componentList[j]) {
                conns.push_back(framework->connIDs[i]);
                break;
            }
        }
    }
    return conns;
}

sci::cca::TypeMap::pointer
BuilderService::getConnectionProperties(const sci::cca::ConnectionID::pointer& connID)
{
    for (unsigned i = 0; i < framework->connIDs.size(); i++) {
        if (connID == framework->connIDs[i]) {
            ConnectionID *connIDPtr = dynamic_cast<ConnectionID*>(connID.getPointer());
            if (connIDPtr) {
                return connIDPtr->getProperties();
            }
        }
    }
    return framework->createTypeMap();
}

void
BuilderService::setConnectionProperties(const sci::cca::ConnectionID::pointer &connID,
                                        const sci::cca::TypeMap::pointer &map)
{
    for (unsigned i = 0; i < framework->connIDs.size(); i++) {
        if (connID == framework->connIDs[i]) {
            ConnectionID *connIDPtr = dynamic_cast<ConnectionID*>(connID.getPointer());
            if (connIDPtr) {
                return connIDPtr->setProperties(map);
            }
        }
    }
}

// TODO: timeout never used
// TODO: disconnect event
void
BuilderService::disconnect(const sci::cca::ConnectionID::pointer& connID,
                           float /*timeout*/)
{
  ComponentID* userID=dynamic_cast<ComponentID*>(connID->getUser().getPointer());
  ComponentID* providerID=dynamic_cast<ComponentID*>(connID->getProvider().getPointer());

  ComponentInstance* user=framework->lookupComponent(userID->name);
  ComponentInstance* provider=framework->lookupComponent(providerID->name);

  PortInstance* userPort=user->getPortInstance(connID->getUserPortName());
  PortInstance* providerPort=provider->getPortInstance(connID->getProviderPortName());
  userPort->disconnect(providerPort);
  for(unsigned i=0; i<framework->connIDs.size();i++) {
    if (framework->connIDs[i]==connID) {
      framework->connIDs.erase(framework->connIDs.begin()+i);
      break;
    }
  }
  //std::cerr << "BuilderService::disconnect: timeout or safty check needed "<<std::endl;
}

void
BuilderService::disconnectAll(const sci::cca::ComponentID::pointer& /*id1*/,
                              const sci::cca::ComponentID::pointer& /*id2*/,
                              float /*timeout*/)
{
  std::cerr << "BuilderService::disconnectAll not finished\n";
}


// port properties useful here?
SSIDL::array1<std::string>
BuilderService::getCompatiblePortList(
    const sci::cca::ComponentID::pointer &user,
    const std::string &usesPortName,
    const sci::cca::ComponentID::pointer &provider)
{
    ComponentID* uCID = dynamic_cast<ComponentID*>(user.getPointer());
    ComponentID* pCID = dynamic_cast<ComponentID*>(provider.getPointer());
    if (! uCID) {
        throw sci::cca::CCAException::pointer(new CCAException("Cannot connect: invalid user componentID"));
    }
    if (! pCID) {
        throw sci::cca::CCAException::pointer(new CCAException("Cannot connect: invalid provider componentID"));
    }

    if (uCID->framework != framework || pCID->framework != framework) {
        throw sci::cca::CCAException::pointer(new CCAException("Cannot connect components from different frameworks"));
    }
    ComponentInstance* uCI = framework->lookupComponent(uCID->name);
    ComponentInstance* pCI = framework->lookupComponent(pCID->name);

    PortInstance* usesPort = uCI->getPortInstance(usesPortName);
    if (! usesPort) {
        throw sci::cca::CCAException::pointer(new CCAException("Unknown uses port"));
    }

    SSIDL::array1<std::string> availablePorts;
    if (uCID == pCID) { // same component
        return availablePorts;
    }
    for (PortInstanceIterator* iter = pCI->getPorts();
            !iter->done(); iter->next()) {
        PortInstance* providesPort = iter->get();
        if (usesPort->canConnectTo(providesPort)) {
            availablePorts.push_back(providesPort->getUniqueName());
        }
    }  

    return availablePorts;
}

SSIDL::array1<std::string>
BuilderService::getBridgablePortList(
     const sci::cca::ComponentID::pointer& c1,
     const std::string& port1,
     const sci::cca::ComponentID::pointer& c2)
{
  SSIDL::array1<std::string> availablePorts;

#ifdef HAVE_RUBY
  ComponentID* cid1 = dynamic_cast<ComponentID*>(c1.getPointer());
  ComponentID* cid2 = dynamic_cast<ComponentID*>(c2.getPointer());
  if (!cid1 || !cid2)
    throw sci::cca::CCAException::pointer(new CCAException("Cannot understand this ComponentID"));
  if (cid1->framework != framework || cid2->framework != framework) {
    throw sci::cca::CCAException::pointer(new CCAException("Cannot connect components from different frameworks"));
  }
  ComponentInstance* comp1=framework->lookupComponent(cid1->name);
  ComponentInstance* comp2=framework->lookupComponent(cid2->name);

  std::cerr<<"Component: "<<cid2->getInstanceName()<<std::endl;
  PortInstance* pr1=comp1->getPortInstance(port1);
  if (!pr1)
    throw sci::cca::CCAException::pointer(new CCAException("Unknown port"));

  if (cid1 == cid2) { // same component
    return availablePorts;
  }
  for(PortInstanceIterator* iter = comp2->getPorts(); !iter->done();
      iter->next()) {
    PortInstance* pr2 = iter->get();
    if ((pr1->getModel() != pr2->getModel())&&(autobr.canBridge(pr1,pr2)))
      availablePorts.push_back(pr2->getUniqueName());
  }
#endif

  return availablePorts;
}

std::string 
BuilderService::generateBridge(const sci::cca::ComponentID::pointer& c1,
                               const std::string& port1,
                               const sci::cca::ComponentID::pointer& c2,
                               const std::string& port2)
{
#ifdef HAVE_RUBY
  ComponentID* cid1 = dynamic_cast<ComponentID*>(c1.getPointer());
  ComponentID* cid2 = dynamic_cast<ComponentID*>(c2.getPointer());
  if (!cid1 || !cid2) {
    throw sci::cca::CCAException::pointer(new CCAException("Cannot understand this ComponentID"));
  }
  if (cid1->framework != framework || cid2->framework != framework) {
    throw sci::cca::CCAException::pointer(new CCAException("Cannot connect components from different frameworks"));
  }
  ComponentInstance* comp1=framework->lookupComponent(cid1->name);
  ComponentInstance* comp2=framework->lookupComponent(cid2->name);
  PortInstance* pr1=comp1->getPortInstance(port1);
  if (!pr1) {
    throw sci::cca::CCAException::pointer(new CCAException("Unknown uses port"));
  }
  PortInstance* pr2=comp2->getPortInstance(port2);
  if (!pr2) {
    throw sci::cca::CCAException::pointer(new CCAException("Unknown provides port"));
  }
  return (autobr.genBridge(pr1->getModel(),cid1->name,pr2->getModel(),cid2->name));
#else
  return std::string();
#endif
}

std::string
BuilderService::getFrameworkURL() {
  return framework->getURL().getString();
}


int BuilderService::addComponentClasses(const std::string &loaderName)
{
    std::cerr<<"BuiderService::addComponentClasses not implemented" << std::endl;
    return 0;
}

int BuilderService::removeComponentClasses(const std::string &loaderName)
{
    std::cerr<<"BuiderService::removeComponentClasses not implemented" << std::endl;
    return 0;
}

void BuilderService::emitConnectionEvent(ConnectionEvent* event)
{
    sci::cca::ports::ConnectionEventService::pointer service =
    pidl_cast<sci::cca::ports::ConnectionEventService::pointer>(
        framework->getFrameworkService("cca.ConnectionEventService", "")
    );
    if (service.isNull()) {
        std::cerr << "Error: could not find ConnectionEventService" << std::endl;
    } else {
        ConnectionEventService* ces =
            dynamic_cast<ConnectionEventService*>(service.getPointer());
        sci::cca::ports::ConnectionEvent::pointer ce =
            ConnectionEvent::pointer(event);
        ces->emitConnectionEvent(ce);
        framework->releaseFrameworkService("cca.ConnectionEventService", "");
    }

}

} // end namespace SCIRun

