/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is Vtk, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
using namespace std;
using namespace SCIRun;
using namespace vtk;

VtkComponentInstance::VtkComponentInstance(SCIRunFramework* framework,
						 const string& instanceName,
						 const string& className,
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

PortInstance* VtkComponentInstance::getPortInstance(const string& name)
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

  Component* component = ci->component;
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
