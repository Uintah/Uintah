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
 *  VtkComponentInstance.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#include <SCIRun/Vtk/VtkComponentInstance.h>
#include <SCIRun/Vtk/VtkPortInstance.h>
#include <SCIRun/Vtk/VtkUIPort.h>
#include <SCIRun/Vtk/Component.h>
#include <SCIRun/CCA/CCAPortInstance.h>

namespace SCIRun {

VtkComponentInstance::VtkComponentInstance(SCIRunFramework* framework,
                                           const std::string& instanceName,
                                           const std::string& className,
                                           vtk::Component* component)
  : ComponentInstance(framework, instanceName, className), component(component)
{
  // See if we have a user-interface...
  if(component->haveUI()){
  specialPorts.push_back(new CCAPortInstance("ui", "sci.cca.ports.UIPort",
                                             sci::cca::TypeMap::pointer(0),
                                             sci::cca::Port::pointer(new VtkUIPort(this)),
                                             CCAPortInstance::Provides));
  }
}

VtkComponentInstance::~VtkComponentInstance()
{
}

PortInstance* VtkComponentInstance::getPortInstance(const std::string& name)
{
  //if the port is CCA port, find it from the specialPorts
  if(name=="ui" || name=="go"){
  for(unsigned int i=0;i<specialPorts.size();i++)
    if(specialPorts[i]->getName() == name)
      return specialPorts[i];
  return 0;
  }

  //otherwise it is vtk port
  vtk::Port* port = component->getPort(name);
  if(!port){
  return 0;
  }
  //TODO: check memory leak
  return new VtkPortInstance(this, port, port->isInput()?VtkPortInstance::Input:VtkPortInstance::Output);
}

PortInstanceIterator* VtkComponentInstance::getPorts()
{
  return new Iterator(this);
}

VtkComponentInstance::Iterator::Iterator(VtkComponentInstance* ci)
  : ci(ci), index(0)
{
}

VtkComponentInstance::Iterator::~Iterator()
{
}

void VtkComponentInstance::Iterator::next()
{
  index++;
}

bool VtkComponentInstance::Iterator::done()
{
  return index >= (int)ci->specialPorts.size()
    +ci->component->numOPorts()
    +ci->component->numIPorts();
}

PortInstance* VtkComponentInstance::Iterator::get()
{
  
  vtk::Component* component = ci->component;
  int spsize = static_cast<int>(ci->specialPorts.size());
  if(index < spsize)
    return ci->specialPorts[index];
  else if(index < spsize+component->numOPorts())
    //TODO: check memory leak
    return new VtkPortInstance(ci,
                               component->getOPort(index-spsize),
                               VtkPortInstance::Output);
  else if(index < spsize+component->numOPorts()
          +component->numIPorts())
    return new VtkPortInstance(ci,
                               component->getIPort(index-spsize-component->numOPorts()),
                               VtkPortInstance::Input);
  else
    return 0; // Illegal
}

} // end namespace SCIRun
