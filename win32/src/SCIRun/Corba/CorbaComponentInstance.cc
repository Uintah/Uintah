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
 *  CorbaComponentInstance.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#include <SCIRun/Corba/CorbaComponentInstance.h>
#include <SCIRun/Corba/CorbaPortInstance.h>
#include <SCIRun/Corba/CorbaUIPort.h>
#include <SCIRun/Corba/Component.h>
#include <SCIRun/CCA/CCAPortInstance.h>

namespace SCIRun {

CorbaComponentInstance::CorbaComponentInstance(
    SCIRunFramework* framework,
    const std::string &instanceName,
    const std::string &className,
    const sci::cca::TypeMap::pointer &tm,
    corba::Component* component)
  : ComponentInstance(framework, instanceName, className, tm), component(component)

{
    // See if we have a user-interface...
    if (component->haveUI()) {
        specialPorts.push_back(new CCAPortInstance("ui", "sci.cca.ports.UIPort",
                               sci::cca::TypeMap::pointer(0),
                               sci::cca::Port::pointer(new CorbaUIPort(this)),
                               CCAPortInstance::Provides));
    }
}

CorbaComponentInstance::~CorbaComponentInstance()
{
}

PortInstance* CorbaComponentInstance::getPortInstance(const std::string& name)
{
    //if the port is CCA port, find it from the specialPorts
    if (name=="ui" || name=="go") {
        for(unsigned int i=0; i < specialPorts.size(); i++) {
            if (specialPorts[i]->getName() == name) {
                return specialPorts[i];
            }
            return 0;
        }
    }

    //otherwise it is corba port
    corba::Port* port = component->getPort(name);
    if (!port) {
        return 0;
    }
    //TODO: check memory leak
    return new CorbaPortInstance(this, port, port->isUses() ?
        CorbaPortInstance::Uses : CorbaPortInstance::Provides);
}

PortInstanceIterator* CorbaComponentInstance::getPorts()
{
    return new Iterator(this);
}

CorbaComponentInstance::Iterator::Iterator(CorbaComponentInstance* ci)
  : ci(ci), index(0)
{
}

CorbaComponentInstance::Iterator::~Iterator()
{
}

void CorbaComponentInstance::Iterator::next()
{
    index++;
}

bool CorbaComponentInstance::Iterator::done()
{
  return index >= (int)ci->specialPorts.size()
    +ci->component->numProvidesPorts()
    +ci->component->numUsesPorts();
}

PortInstance* CorbaComponentInstance::Iterator::get()
{
  
  corba::Component* component = ci->component;
  int spsize = static_cast<int>(ci->specialPorts.size());
  if (index < spsize) {
    return ci->specialPorts[index];
  } else if (index < spsize+component->numProvidesPorts()) {
    //TODO: check memory leak
    return new CorbaPortInstance(ci,
                               component->getProvidesPort(index-spsize),
                               CorbaPortInstance::Provides);
  } else if (index < spsize+component->numProvidesPorts() +component->numUsesPorts()) {
    return new CorbaPortInstance(ci,
                               component->getUsesPort(index-spsize-component->numProvidesPorts()),
                               CorbaPortInstance::Uses);
  } else {
    return 0; // Illegal
  }
}

} // end namespace SCIRun
