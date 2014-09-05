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
 *  TaoComponentInstance.cc:
 *
 *  Written by:
 *   Kosta Damevski
 *   Department of Computer Science
 *   University of Utah
 *   May 2005
 *
 */

#include <Framework/Tao/TaoComponentInstance.h>
#include <Framework/Tao/TaoPortInstance.h>
#include <Framework/CCA/CCAPortInstance.h>
#include <Framework/Tao/TaoException.h>
#include <Framework/Tao/TaoGoPort.h>
#include <Framework/SCIRunFramework.h>
#include <Framework/TypeMap.h>
#include <Core/Thread/Mutex.h>

#include <iostream>

namespace SCIRun {

TaoComponentInstance::TaoComponentInstance(
    SCIRunFramework *framework,
    const std::string &instanceName,
    const std::string &typeName,
    const sci::cca::TypeMap::pointer &tm,
    tao::Component* component)
  : ComponentInstance(framework, instanceName, typeName, tm),
    lock_ports("TaoComponentInstance::ports lock"),
    component(component)
{
    mutex = new Mutex("getPort mutex");
}

TaoComponentInstance::~TaoComponentInstance()
{
    delete mutex;
}

PortInstance*
TaoComponentInstance::getPortInstance(const std::string& portname)
{
  lock_ports.lock();
  std::map<std::string, TaoPortInstance*>::iterator iter = ports.find(portname);
  lock_ports.unlock();
  if (iter == ports.end()) {
    return 0;
  } else {
    if(portname == "go") {
      return new CCAPortInstance("go", "sci.cca.ports.GoPort",
				 sci::cca::TypeMap::pointer(0),
				 sci::cca::Port::pointer(new TaoGoPort(this)),
				 CCAPortInstance::Provides);

    }
    return iter->second;
  }
}

void TaoComponentInstance::registerUsesPort(const std::string& portName,
					    const std::string& portType,
					    const sci::cca::TypeMap::pointer& properties)
{
    SCIRun::Guard g1(&lock_ports);
    std::map<std::string, TaoPortInstance*>::iterator iter = ports.find(portName);
    if (iter != ports.end()) {
	if (iter->second->porttype == TaoPortInstance::Provides) {
	    throw TaoException("name conflict between uses and provides ports for " + portName + " " + portType + " " + instanceName);
	} else {
	    throw TaoException("registerUsesPort called twice for " + portName + " " + portType + " " + instanceName);
	}
    }
    ports.insert(make_pair(portName,
			   new TaoPortInstance(portName, portType, properties, TaoPortInstance::Uses)));
}

void TaoComponentInstance::unregisterUsesPort(const std::string& portName)
{
std::cerr << "TaoComponentInstance::unregisterUsesPort: " << portName << std::endl;
    SCIRun::Guard g1(&lock_ports);
    std::map<std::string, TaoPortInstance*>::iterator iter = ports.find(portName);
    if (iter != ports.end()) {
	if (iter->second->porttype == TaoPortInstance::Provides) {
	    throw TaoException("name conflict between uses and provides ports for " + portName);
	} else {
	    ports.erase(portName);
	}
    } else {
	throw TaoException("port name not found for " + portName);
    }
}

void TaoComponentInstance::addProvidesPort(const std::string& portName,
					   const std::string& portType,
					   const sci::cca::TypeMap::pointer& properties)
{
  SCIRun::Guard g1(&lock_ports);
  std::map<std::string, TaoPortInstance*>::iterator iter = ports.find(portName);
  if (iter != ports.end()) {
    if (iter->second->porttype == TaoPortInstance::Uses) {
      throw TaoException("name conflict between uses and provides ports for " + portName);
    } else {
      throw TaoException("addProvidesPort called twice for " + portName);
    }
  }

  ports.insert(make_pair(portName,
			 new TaoPortInstance(portName, portType, properties, TaoPortInstance::Provides)));
}

void TaoComponentInstance::removeProvidesPort(const std::string& name)
{
  std::cerr << "removeProvidesPort not done, name=" << name << std::endl;
}

sci::cca::ComponentID::pointer TaoComponentInstance::getComponentID()
{
  return sci::cca::ComponentID::pointer(new ComponentID(framework, instanceName));
}

PortInstanceIterator* TaoComponentInstance::getPorts()
{
  return new Iterator(this);
}

sci::cca::TypeMap::pointer
TaoComponentInstance::getPortProperties(const std::string& portName)
{
  lock_ports.lock();
  std::map<std::string, TaoPortInstance*>::iterator iter = ports.find(portName);
  lock_ports.unlock();
  if (iter == ports.end()) {
    return sci::cca::TypeMap::pointer(new TypeMap);
  }

  return iter->second->getProperties();
}

void
TaoComponentInstance::setPortProperties(const std::string& portName, const sci::cca::TypeMap::pointer& tm)
{
  lock_ports.lock();
  std::map<std::string, TaoPortInstance*>::iterator iter = ports.find(portName);
  lock_ports.unlock();
  if (iter == ports.end()) {
    // with warning?
    return;
  }

  iter->second->setProperties(tm);
}

TaoComponentInstance::Iterator::Iterator(TaoComponentInstance* comp)
  :iter(comp->ports.begin()), comp(comp)
{
}

TaoComponentInstance::Iterator::~Iterator()
{
}

PortInstance* TaoComponentInstance::Iterator::get()
{
  return iter->second;
}

bool TaoComponentInstance::Iterator::done()
{
  return iter == comp->ports.end();
}

void TaoComponentInstance::Iterator::next()
{
  ++iter;
}

} // end namespace SCIRun
