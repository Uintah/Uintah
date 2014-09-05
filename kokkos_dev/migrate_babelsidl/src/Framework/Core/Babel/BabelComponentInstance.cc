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
 *  BabelComponentInstance.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   July 2002
 *
 */

#include <Framework/sidl/Impl/glue/scijump.hxx>

#include <Framework/Core/Babel/BabelPortInstance.h>
#include <Framework/Core/Babel/BabelComponentInstance.h>
#include <iostream>

#ifndef DEBUG
#  define DEBUG 0
#endif

namespace SCIRun {

BabelComponentInstance::BabelComponentInstance(SCIJumpFramework* framework,
                                               const std::string& instanceName,
                                               const std::string& typeName,
                                               const ::gov::cca::TypeMap& properties,
                                               const ::gov::cca::Component& component,
                                               const ::scijump::Services& svc)
  : ComponentInstance(framework, instanceName, typeName, NULL)
{
  // Babel component properties are ignored for now.
  this->component = component;
  this->svc = svc;

  /*
  BabelPortInstance *go = dynamic_cast<BabelPortInstance*>(getPortInstance("go"));
  if (go != 0) {
    // locate a GoPort provided by the Babel component instance
    PortInstanceMap *pports = (PortInstanceMap*) (this->svc.getData());
    UCXX ::gov::cca::ports::GoPort g = UCXX ::sidl::babel_cast<UCXX ::gov::cca::ports::GoPort>(go->getPort());
    sci::cca::ports::GoPort::pointer goPort(new BabelCCAGoPort(g));
    CCAPortInstance *piGo = new CCAPortInstance("go",
						"sci.cca.ports.GoPort",
						sci::cca::TypeMap::pointer(0),
						goPort,
						CCAPortInstance::Provides);

    // Override the original gov.cca.GoPort in the pports map
    // (the original port is stored in a BabelCCAGoPort) instantiated above.
    // This is needed to connect GoPorts with CCA UI components such as the GUIBuilder.
    (*pports)["go"] = (PortInstance*) piGo;
  }

  // locate a UIPort provided by the Babel component instance
  BabelPortInstance *ui = dynamic_cast<BabelPortInstance*>(getPortInstance("ui"));
  if (ui != 0) {
    PortInstanceMap *pports= (PortInstanceMap*) (this->svc.getData());

    // If there is a UIPort, create a corresponding CCAPortInstance
    UCXX ::gov::cca::ports::UIPort u = UCXX ::sidl::babel_cast<UCXX ::gov::cca::ports::UIPort>(ui->getPort());
    sci::cca::ports::UIPort::pointer uiPort(new BabelCCAUIPort(u));
    CCAPortInstance *piUI = new CCAPortInstance("ui",
						"sci.cca.ports.UIPort",
						sci::cca::TypeMap::pointer(0),
						uiPort,
						CCAPortInstance::Provides);

    // Override the original gov.cca.UIPort in the pports map
    // (the original port is stored in a BabelCCAUIPort) instantiated above.
    // This is needed to connect UIPorts with CCA UI components such as the GUIBuilder.
    (*pports)["ui"] = (PortInstance*) piUI;
  }
  */
}

BabelComponentInstance::~BabelComponentInstance()
{
#if DEBUG
  std::cerr << "BabelComponentInstance destroyed..." << std::endl;
#endif
}

PortInstance*
BabelComponentInstance::getPortInstance(const std::string& portname)
{
  /*
  PortInstanceMap *pports = (PortInstanceMap*) svc.getData();

  if (pports != 0) {
    PortInstanceMap::iterator iter = pports->find(portname);
    if (iter == pports->end()) {
      return 0;
    } else {
      return iter->second;
    }
  } else {
    std::cerr << "Warning: NULL pports!" << std::endl;
    return 0;
  }
  */
}

::gov::cca::Port
BabelComponentInstance::getPort(const std::string& name)
{
  return svc.getPort(name);
}

::gov::cca::Port
BabelComponentInstance::getPortNonblocking(const std::string& name)
{
  return svc.getPortNonblocking(name);
}

void
BabelComponentInstance::releasePort(const std::string& name)
{
  return svc.releasePort(name);
}

::gov::cca::TypeMap
BabelComponentInstance::createTypeMap()
{
  return svc.createTypeMap();
}

void
BabelComponentInstance::registerUsesPort(const std::string& portName,
                                         const std::string& portType,
                                         const ::gov::cca::TypeMap& properties)
{
  return svc.registerUsesPort(portName, portType, properties);
}

void BabelComponentInstance::unregisterUsesPort(const std::string& name)
{
  return svc.unregisterUsesPort(name);
}

void
BabelComponentInstance::addProvidesPort(const ::gov::cca::Port& port,
                                        const std::string& portName,
                                        const std::string& portType,
                                        const ::gov::cca::TypeMap& properties)
{
  return svc.addProvidesPort(port, portName, portType, properties);
}

void BabelComponentInstance::removeProvidesPort(const std::string& name)
{
  svc.removeProvidesPort(name);
  return;
}

// gov::cca::TypeMap BabelComponentInstance::getPortProperties(const std::string& portName)
// {
//   return svc.getPortProperties(portName);
// }
// these do nothing at the moment - implement after compiler changeover
gov::cca::TypeMap
BabelComponentInstance::getPortProperties(const std::string& portName)
{
  //return sci::cca::TypeMap::pointer(new TypeMap);
}

::gov::cca::ComponentID
BabelComponentInstance::getComponentID()
{
  return svc.getComponentID();
}

PortInstanceIterator*
BabelComponentInstance::getPorts()
{
  return new Iterator(this);
}

BabelComponentInstance::Iterator::Iterator(BabelComponentInstance* comp)
{  
  /*
  ports = (PortInstanceMap*) (comp->svc.getData());
  iter = ports->begin();
  */
}

BabelComponentInstance::Iterator::~Iterator()
{
}

PortInstance* BabelComponentInstance::Iterator::get()
{
  return iter->second;
}

bool BabelComponentInstance::Iterator::done()
{
  return iter == ports->end();
}

void BabelComponentInstance::Iterator::next()
{
  ++iter;
}

} // end namespace SCIRun
