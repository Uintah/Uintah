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
 *  BridgeComponentInstance.cc:
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   September, 2003
 *
 */

#include <SCIRun/Bridge/BridgeComponentInstance.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/TypeMap.h>
#include <SCIRun/CCA/CCAException.h>
#include <SCIRun/Dataflow/SCIRunComponentModel.h>
#include <SCIRun/Tao/TaoPortInstance.h>
#include <iostream>
#include <Core/Thread/Mutex.h>
#include <Core/Exceptions/InternalError.h>

using namespace std;
using namespace SCIRun;

BridgeComponentInstance::BridgeComponentInstance(SCIRunFramework* framework,
					   const std::string& instanceName,
					   const std::string& typeName,
					   BridgeComponent* component
)
  : ComponentInstance(framework, instanceName, typeName), component(component)
{
  mutex=new Mutex("getPort mutex");
  if(!SCIRunComponentModel::gui) {
    SCIRunComponentModel::initGuiInterface();
  }
  bmdl = new BridgeModule(component);
}

BridgeComponentInstance::~BridgeComponentInstance()
{
  delete mutex;
  delete bmdl;
}

PortInstance*
BridgeComponentInstance::getPortInstance(const std::string& portname)
{
  map<string, PortInstance*>::iterator iter = ports.find(portname);
  if(iter == ports.end()) {
    //!!!!! Check if it is a dataflow port:
    // SCIRun ports can potentially have the same name for both, so
    // SCIRunPortInstance tags them with a prefix of "Input: " or
    // "Output: ", so we need to check that first.
    if(portname.substr(0, 7) == "Input: "){
      map<string, PortInstance*>::iterator iter = ports.find(portname.substr(7));
      return iter->second;
    } else if(portname.substr(0,8) == "Output: "){
      map<string, PortInstance*>::iterator iter = ports.find(portname.substr(8));
      return iter->second;
    } else {
      return 0;
    }    
  } else {
    return iter->second;
  }
}

sci::cca::Port::pointer BridgeComponentInstance::getCCAPort(const std::string& name)
{
  mutex->lock();
  sci::cca::Port::pointer svc = framework->getFrameworkService(name, instanceName);
  if(!svc.isNull()){
    return svc;
  }
  map<string, PortInstance*>::iterator iter = ports.find(name);
  if(iter == ports.end())
    return sci::cca::Port::pointer(0);
  CCAPortInstance* pr = dynamic_cast<CCAPortInstance*>(iter->second);
  if(pr == NULL)
    return sci::cca::Port::pointer(0);
  if(pr->porttype == CCAPortInstance::Provides)
    throw CCAException("Cannot call getPort on a Provides port");
  pr->incrementUseCount();
  if(pr->connections.size() != 1)
    return sci::cca::Port::pointer(0); 
  CCAPortInstance *pi=dynamic_cast<CCAPortInstance*> (pr->getPeer());
  mutex->unlock();
  return pi->port;
}

gov::cca::Port BridgeComponentInstance::getBabelPort(const std::string& name)
{
  mutex->lock();
  map<string, PortInstance*>::iterator iter = ports.find(name);
  if(iter == ports.end())
    return 0;
  BabelPortInstance* pr = dynamic_cast<BabelPortInstance*>(iter->second);
  if(pr == NULL)
    return 0;
  if(pr->porttype != BabelPortInstance::Uses) {
    throw InternalError("Cannot call getPort on a Provides port");
  }
  pr->incrementUseCount();
  if(pr->connections.size() != 1)
    return 0; 
  BabelPortInstance *pi=dynamic_cast<BabelPortInstance*> (pr->getPeer());
  mutex->unlock();
  return pi->port;
}

Port* BridgeComponentInstance::getDataflowIPort(const std::string& name)
{
  IPort *ip = bmdl->get_iport(name);
  if(!ip) {
    cerr << "Unable to initialize iport: "<< name << "\n";
    return 0;
  }
  return static_cast<Port*>(ip);
}

Port* BridgeComponentInstance::getDataflowOPort(const std::string& name)
{
  OPort *op = bmdl->get_oport(name);
  if(!op) {
    cerr << "Unable to initialize iport: "<< name << "\n";
    return 0;
   }
   return static_cast<Port*>(op);
}

#if HAVE_VTK
vtk::Port* BridgeComponentInstance::getVtkPort(const std::string& name)
{

  mutex->lock();
  map<string, PortInstance*>::iterator iter = ports.find(name);
  if(iter == ports.end())
    return 0;
  VtkPortInstance* pr = dynamic_cast<VtkPortInstance*>(iter->second);
  if(pr == NULL)
    return 0;
  mutex->unlock();
  return pr->port;
}

void BridgeComponentInstance::addVtkPort(vtk::Port* vtkport, VtkPortInstance::PortType portT) 
{
  map<string, PortInstance*>::iterator iter;
  std::string portName = vtkport->getName();

  iter = ports.find(portName);
  if(iter != ports.end()){
    throw InternalError("port name conflicts with another one");
  }
                                                                                                      
  ports.insert(make_pair(portName, new VtkPortInstance(NULL, vtkport, portT)));
}
#endif

void BridgeComponentInstance::releasePort(const std::string& name, const modelT model)
{
  CCAPortInstance* cpr;
  BabelPortInstance* bpr;
  map<string, PortInstance*>::iterator iter;

  switch (model) {
  case CCA:
    if(framework->releaseFrameworkService(name, instanceName))
      return;
    
    iter = ports.find(name);
    if(iter == ports.end()){
      throw CCAException("Released an unknown port: "+name);
    }
    cpr = dynamic_cast<CCAPortInstance*>(iter->second);
    if(cpr == NULL)
      throw CCAException("Trying to release a port of the wrong type");
    
    if(cpr->porttype != CCAPortInstance::Uses)
      throw CCAException("Cannot call releasePort on a Provides port");
    
    if(!cpr->decrementUseCount())
      throw CCAException("Port released without correspond get");
    break;

  case Babel:
    iter = ports.find(name);
    if(iter == ports.end()){
      cerr << "Released an unknown port: " << name << '\n';
      throw InternalError("Released an unknown port: "+name);
    }
    bpr = dynamic_cast<BabelPortInstance*>(iter->second);
    if(bpr == NULL)
      throw InternalError("Trying to release a port of the wrong type");
    
    if(bpr->porttype != BabelPortInstance::Uses)
      throw InternalError("Cannot call releasePort on a Provides port");
    
    if(!bpr->decrementUseCount())
      throw InternalError("Port released without correspond get");
    break;

  case Dataflow:
    ::std::cerr << "Don't know how to release a dataflow port\n";
    break;

  case Vtk:
  case Tao:
    //We aren't releasing Vtk ports at this point of time
    break;

  }
  return;
}

void BridgeComponentInstance::registerUsesPort(const std::string& portName,
					    const std::string& portType,
					    const modelT model)
{
  CCAPortInstance* cpr;
  BabelPortInstance* bpr;
  TaoPortInstance* tpr;
  map<string, PortInstance*>::iterator iter;

  Port* dflowport;
  SCIRunPortInstance::PortType portT;

  switch (model) {
  case CCA:	
    iter = ports.find(portName);
    if(iter != ports.end()){
      cpr = dynamic_cast<CCAPortInstance*>(iter->second);
      if(cpr == NULL)
	throw CCAException("Trying to register a port of the wrong type");
      if(cpr->porttype == CCAPortInstance::Provides)
	throw CCAException("name conflict between uses and provides ports");
      else {
        throw CCAException("registerUsesPort called twice for " + portName + " " + portType + " " + instanceName);
      } 
    }
    ports.insert(make_pair(portName, new CCAPortInstance(portName, portType, sci::cca::TypeMap::pointer(0), CCAPortInstance::Uses)));
    break;

  case Babel:
    iter = ports.find(portName);
    if(iter != ports.end()){
      bpr = dynamic_cast<BabelPortInstance*>(iter->second);
      if(bpr == NULL)
	throw InternalError("Trying to register a port of the wrong type");
      if(bpr->porttype == BabelPortInstance::Provides)
	throw InternalError("name conflict between uses and provides ports");
      else {
	cerr << "registerUsesPort called twice, instance=" << instanceName << ", portName = " << portName << ", portType = " << portType << '\n';
	throw InternalError("registerUsesPort called twice for " + portName);
      }     
    }
    ports.insert(make_pair(portName, new BabelPortInstance(portName, portType, 0, BabelPortInstance::Uses)));
    break;

  case Dataflow:
    iter = ports.find(portName);
    if(iter != ports.end()){
      throw InternalError("port name conflicts with another one");
    }

    portT = SCIRunPortInstance::Output;
    bmdl->addOPortByName(portName, portType);
    dflowport = bmdl->getOPort(portName);
    
    if(!dflowport)
      throw InternalError("Wrong port model for addProvidesPort");

    //NO SCIRunComponentInstance to pass into SCIRunPortInstance, hopefully NULL is okay
    ports.insert(make_pair(portName, new SCIRunPortInstance(NULL, dflowport, portT)));
    break;

  case Tao:
    iter = ports.find(portName);
    if(iter != ports.end()){
      tpr = dynamic_cast<TaoPortInstance*>(iter->second);
      if(tpr == NULL)
        throw InternalError("Trying to register a port of the wrong type");
      if(tpr->porttype == TaoPortInstance::Provides)
        throw InternalError("name conflict between uses and provides ports");
      else {
        cerr << "registerUsesPort called twice, instance=" << instanceName 
             << ", portName = " << portName << ", portType = " << portType << '\n';
        throw InternalError("registerUsesPort called twice for " + portName);
      }
    }
    ports.insert(make_pair(portName, new TaoPortInstance(portName, portType, TaoPortInstance::Uses)));
    break;

  case Vtk:
    throw InternalError("Use addVtkPort for Vtk component model");
    break;

  }
  return;
}

void BridgeComponentInstance::unregisterUsesPort(const std::string& portName, const modelT model)
{
  CCAPortInstance* cpr;
  BabelPortInstance* bpr;
  map<string, PortInstance*>::iterator iter;

  switch (model) {
  case CCA:  
    iter = ports.find(portName);
    if(iter != ports.end()){
      cpr = dynamic_cast<CCAPortInstance*>(iter->second);
      if(cpr == NULL)
	throw CCAException("Trying to unregister a port of the wrong type");
      if(cpr->porttype == CCAPortInstance::Provides)
	throw CCAException("name conflict between uses and provides ports");
      else {
	ports.erase(portName);
      }
    }
    else{
      throw CCAException("port name " + portName +" not found, cannot unregister uses port");
    }
    break;

  case Babel:
    iter = ports.find(portName);
    if(iter != ports.end()){
      bpr = dynamic_cast<BabelPortInstance*>(iter->second);
      if(bpr == NULL)
	throw InternalError("Trying to unregister a port of the wrong type");
      if(bpr->porttype == BabelPortInstance::Provides)
	throw InternalError("name conflict between uses and provides ports");
      else {
	ports.erase(portName);
      }
    }
    else{
      throw CCAException("port name " + portName +" not found, cannot unregister uses port");
    }
    break;

  case Dataflow:
    cerr<<"Don't know how to unregisterUsesPort for Dataflow ports\n";
    break;

  case Vtk:
  case Tao:
    //Not implemented for now
    break;
  }
  return;
}

void BridgeComponentInstance::addProvidesPort(void* port,
					   const std::string& portName,
					   const std::string& portType,
					   const modelT model)
{
  CCAPortInstance* cpr;
  BabelPortInstance* bpr;
  TaoPortInstance* tpr;
  map<string, PortInstance*>::iterator iter;

  sci::cca::Port::pointer* ccaport;
  gov::cca::Port* babelport;
  Port* dflowport;
  
  SCIRunPortInstance::PortType portT;

  switch (model) {
  case CCA:
    iter = ports.find(portName);
    if(iter != ports.end()){
      cpr = dynamic_cast<CCAPortInstance*>(iter->second);
      if(cpr == NULL)
	throw CCAException("port name conflicts with another one of a different type");
      if(cpr->porttype == CCAPortInstance::Uses)
	throw CCAException("name conflict between uses and provides ports");
      else
	throw CCAException("addProvidesPort called twice for " + portName);
    }

    ccaport = reinterpret_cast<sci::cca::Port::pointer*>(port);
    if(!ccaport)
      throw CCAException("Wrong port model for addProvidesPort");
    ports.insert(make_pair(portName, new CCAPortInstance(portName, portType, sci::cca::TypeMap::pointer(0), *ccaport, CCAPortInstance::Provides)));
    break;

  case Babel:
    iter = ports.find(portName);
    if(iter != ports.end()){
      bpr = dynamic_cast<BabelPortInstance*>(iter->second);
      if(bpr == NULL)
	throw InternalError("port name conflicts with another one of a different type");
      if(bpr->porttype == BabelPortInstance::Uses)
	throw InternalError("name conflict between uses and provides ports");
      else
	throw InternalError("addProvidesPort called twice for " + portName);
    }
    babelport = reinterpret_cast<gov::cca::Port*>(port);
    if(!babelport)
      throw InternalError("Wrong port model for addProvidesPort");
    ports.insert(make_pair(portName, new BabelPortInstance(portName, portType, 0, *babelport, BabelPortInstance::Provides)));
    break;

  case Dataflow:
    
    iter = ports.find(portName);
    if(iter != ports.end()){
      throw InternalError("port name conflicts with another one");
    }

    portT = SCIRunPortInstance::Input;
    bmdl->addIPortByName(portName, portType);
    dflowport = bmdl->getIPort(portName);
    
    if(!dflowport)
      throw InternalError("Wrong port model for addProvidesPort");

    //NO SCIRunComponentInstance to pass into SCIRunPortInstance, hopefully NULL is okay
    ports.insert(make_pair(portName, new SCIRunPortInstance(NULL, dflowport, portT)));
    
    break;

  case Vtk:
    throw InternalError("Use addVtkPort for Vtk component model");
    break;

  case Tao:
    iter = ports.find(portName);
    if(iter != ports.end()){
      tpr = dynamic_cast<TaoPortInstance*>(iter->second);
      if(tpr == NULL)
        throw InternalError("Trying to register a port of the wrong type");
      if(tpr->porttype == TaoPortInstance::Provides)
        throw InternalError("name conflict between uses and provides ports");
      else {
        cerr << "registerUsesPort called twice, instance=" << instanceName
             << ", portName = " << portName << ", portType = " << portType << '\n';
        throw InternalError("registerUsesPort called twice for " + portName);
      }
    }
    ports.insert(make_pair(portName, new TaoPortInstance(portName, portType, TaoPortInstance::Uses)));
    break;

  }
  return;
}

void BridgeComponentInstance::removeProvidesPort(const std::string& name, const modelT model)
{
  switch (model) {
  case CCA:
    cerr << "removeProvidesPort not done, name=" << name << '\n';
    break;
  case Babel:
    cerr << "removeProvidesPort not done, name=" << name << '\n';
    break;
  case Dataflow:
    cerr << "removeProvidesPort not done, name=" << name << '\n';
    break;
  case Vtk:
  case Tao:
    cerr << "removeProvidesPort not done, name=" << name << '\n';
    break;

  }
  return;
}


sci::cca::ComponentID::pointer BridgeComponentInstance::getComponentID()
{
  return sci::cca::ComponentID::pointer(new ComponentID(framework, instanceName));
}

PortInstanceIterator* BridgeComponentInstance::getPorts()
{
  return new Iterator(this);
}

BridgeComponentInstance::Iterator::Iterator(BridgeComponentInstance* comp)
  :iter(comp->ports.begin()), comp(comp)
{
}

BridgeComponentInstance::Iterator::~Iterator()
{
}

PortInstance* BridgeComponentInstance::Iterator::get()
{
  return iter->second;
}

bool BridgeComponentInstance::Iterator::done()
{
  return iter == comp->ports.end();
}

void BridgeComponentInstance::Iterator::next()
{
  ++iter;
}
