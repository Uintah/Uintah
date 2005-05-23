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
BuilderService::connect(const sci::cca::ComponentID::pointer& c1,
                        const string& port1,
                        const sci::cca::ComponentID::pointer& c2,
                        const string& port2)
{
    ComponentID* cid1 = dynamic_cast<ComponentID*>(c1.getPointer());
    ComponentID* cid2 = dynamic_cast<ComponentID*>(c2.getPointer());
    if (!cid1 || !cid2) {
        throw CCAException("Cannot understand this ComponentID");
    }
    if (cid1->framework != framework || cid2->framework != framework) {
        throw CCAException("Cannot connect components from different frameworks");
    }
    ComponentInstance* comp1 = framework->lookupComponent(cid1->name);
    if (!comp1) {
        throw CCAException("Unknown ComponentInstance " + cid1->name);
    }
    ComponentInstance* comp2 = framework->lookupComponent(cid2->name);
    if (!comp2) {
        throw CCAException("Unknown ComponentInstance " + cid2->name);
    }
    PortInstance* pr1 = comp1->getPortInstance(port1);
    if (!pr1) {
        throw CCAException("Unknown port " + port1);
    }
    PortInstance* pr2 = comp2->getPortInstance(port2);
    if (!pr2) {
        throw CCAException("Unknown port " + port2);
    }
    if (!pr1->connect(pr2)) {
        throw CCAException("Cannot connect " + port1 + " with " + port2);
    }
    sci::cca::ConnectionID::pointer conn(new ConnectionID(c1, port1, c2, port2));
    framework->connIDs.push_back(conn);
    emitConnectionEvent(
        new ConnectionEvent(sci::cca::ports::Connected, getConnectionProperties(conn))
    );
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
BuilderService::getComponentProperties(const sci::cca::ComponentID::pointer& /*cid*/)
{
    std::cerr << "BuilderService::getComponentProperties not finished" << std::endl;
    return sci::cca::TypeMap::pointer(0);
}

void
BuilderService::setComponentProperties(const sci::cca::ComponentID::pointer& /*cid*/, const sci::cca::TypeMap::pointer& /*map*/)
{
  std::cerr << "BuilderService::setComponentProperties not finished\n";
}

sci::cca::ComponentID::pointer
BuilderService::getDeserialization(const std::string& /*s*/)
{
  std::cerr << "BuilderService::getDeserialization not finished\n";
  return sci::cca::ComponentID::pointer(0);
}

sci::cca::ComponentID::pointer
BuilderService::getComponentID(const std::string &componentInstanceName)
{
  sci::cca::ComponentID::pointer cid=framework->lookupComponentID(componentInstanceName);
  if (cid.isNull()) throw CCAException("ComponentID not found");
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
  ComponentInstance *ci=framework->lookupComponent(cid->getInstanceName());
  //  std::cerr<<"Component: "<<cid->getInstanceName()<<std::endl;
  for(PortInstanceIterator* iter = ci->getPorts(); !iter->done(); iter->next()) {
    PortInstance* port = iter->get();
    if (port->portType() == PortInstance::To)
      result.push_back(port->getUniqueName());
  }
  return result;
}

SSIDL::array1<std::string>
BuilderService::getUsedPortNames(const sci::cca::ComponentID::pointer &cid)
{
  SSIDL::array1<std::string> result;
  ComponentInstance *ci=framework->lookupComponent(cid->getInstanceName());
  for(PortInstanceIterator* iter = ci->getPorts(); !iter->done(); iter->next()) {
    PortInstance* port = iter->get();
    if (port->portType() == PortInstance::From)
      result.push_back(port->getUniqueName());
  }
  return result;
}

sci::cca::TypeMap::pointer
BuilderService::getPortProperties(const sci::cca::ComponentID::pointer &cid, const std::string &portname)
{
  ComponentInstance* comp = framework->lookupComponent(cid->getInstanceName());
  if (comp == NULL) {
    return sci::cca::TypeMap::pointer(0);
  }
  CCAComponentInstance* ccacomp = dynamic_cast<CCAComponentInstance*>(comp);
  if (ccacomp == NULL) {
    return sci::cca::TypeMap::pointer(0);
  }
  return ccacomp->getPortProperties(portname);
}

void BuilderService::setPortProperties(const sci::cca::ComponentID::pointer& /*cid*/,
                       const std::string& /*portname*/,
                       const sci::cca::TypeMap::pointer& /*map*/)
{
  std::cerr << "BuilderService::setPortProperties not finished\n";
}

SSIDL::array1<sci::cca::ConnectionID::pointer>
BuilderService::getConnectionIDs(const SSIDL::array1<sci::cca::ComponentID::pointer> &componentList)
{
  SSIDL::array1<sci::cca::ConnectionID::pointer> conns;
  for(unsigned i=0; i<framework->connIDs.size(); i++) {
    for(unsigned j=0; j<componentList.size(); j++) {
      sci::cca::ComponentID::pointer cid1=framework->connIDs[i]->getUser();
      sci::cca::ComponentID::pointer cid2=framework->connIDs[i]->getProvider();
      if (cid1==componentList[j]||cid2==componentList[j]) {
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
  std::cerr << "BuilderService::getConnectionProperties not finished\n";
  return sci::cca::TypeMap::pointer(0);
}

void BuilderService::setConnectionProperties(const sci::cca::ConnectionID::pointer& /*connID*/, const sci::cca::TypeMap::pointer& /*map*/)
{
  std::cerr << "BuilderService::setConnectionProperties not finished\n";
}

void BuilderService::disconnect(const sci::cca::ConnectionID::pointer& connID,
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

void BuilderService::disconnectAll(const sci::cca::ComponentID::pointer& /*id1*/,
                   const sci::cca::ComponentID::pointer& /*id2*/,
                   float /*timeout*/)
{
  std::cerr << "BuilderService::disconnectAll not finished\n";
}


SSIDL::array1<std::string>  BuilderService::getCompatiblePortList(
    const sci::cca::ComponentID::pointer& c1,
    const std::string& port1,
    const sci::cca::ComponentID::pointer& c2)
{
  ComponentID* cid1 = dynamic_cast<ComponentID*>(c1.getPointer());
  ComponentID* cid2 = dynamic_cast<ComponentID*>(c2.getPointer());
  if (!cid1 || !cid2) {
    throw CCAException("Cannot understand this ComponentID");
  }
  if (cid1->framework != framework || cid2->framework != framework) {
    throw CCAException("Cannot connect components from different frameworks");
  }
  ComponentInstance* comp1=framework->lookupComponent(cid1->name);
  ComponentInstance* comp2=framework->lookupComponent(cid2->name);

  //  std::cerr << "Component: "<<cid2->getInstanceName() << std::endl;
  PortInstance* pr1=comp1->getPortInstance(port1);
  if (!pr1) {
    throw CCAException("Unknown port");
  }

  SSIDL::array1<std::string> availablePorts;
  if (cid1 == cid2) { // same component
    return availablePorts;
  }
  for(PortInstanceIterator* iter = comp2->getPorts(); !iter->done();
      iter->next()) {
    PortInstance* pr2 = iter->get();
    if (pr1->canConnectTo(pr2))
      availablePorts.push_back(pr2->getUniqueName());
  }  

  return availablePorts;
}

SSIDL::array1<std::string> BuilderService::getBridgablePortList(
     const sci::cca::ComponentID::pointer& c1,
     const std::string& port1,
     const sci::cca::ComponentID::pointer& c2)
{
  SSIDL::array1<std::string> availablePorts;

#if HAVE_RUBY
  ComponentID* cid1 = dynamic_cast<ComponentID*>(c1.getPointer());
  ComponentID* cid2 = dynamic_cast<ComponentID*>(c2.getPointer());
  if (!cid1 || !cid2)
    throw CCAException("Cannot understand this ComponentID");
  if (cid1->framework != framework || cid2->framework != framework) {
    throw CCAException("Cannot connect components from different frameworks");
  }
  ComponentInstance* comp1=framework->lookupComponent(cid1->name);
  ComponentInstance* comp2=framework->lookupComponent(cid2->name);

  //  std::cerr<<"Component: "<<cid2->getInstanceName()<<std::endl;
  PortInstance* pr1=comp1->getPortInstance(port1);
  if (!pr1)
    throw CCAException("Unknown port");

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
#if HAVE_RUBY
  ComponentID* cid1 = dynamic_cast<ComponentID*>(c1.getPointer());
  ComponentID* cid2 = dynamic_cast<ComponentID*>(c2.getPointer());
  if (!cid1 || !cid2) {
    throw CCAException("Cannot understand this ComponentID");
  }
  if (cid1->framework != framework || cid2->framework != framework) {
    throw CCAException("Cannot connect components from different frameworks");
  }
  ComponentInstance* comp1=framework->lookupComponent(cid1->name);
  ComponentInstance* comp2=framework->lookupComponent(cid2->name);
  PortInstance* pr1=comp1->getPortInstance(port1);
  if (!pr1) {
    throw CCAException("Unknown port");
  }
  PortInstance* pr2=comp2->getPortInstance(port2);
  if (!pr2) {
    throw CCAException("Unknown port");
  }
  return (autobr.genBridge(pr1->getModel(),cid1->name,pr2->getModel(),cid2->name));
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

