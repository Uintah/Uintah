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
 *  SCIRunComponentInstance.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/Dataflow/SCIRunComponentInstance.h>
#include <SCIRun/Dataflow/SCIRunPortInstance.h>
#include <SCIRun/Dataflow/SCIRunUIPort.h>
#include <SCIRun/CCA/CCAPortInstance.h>
#include <Dataflow/Network/Module.h>
using namespace std;
using namespace SCIRun;

SCIRunComponentInstance::SCIRunComponentInstance(SCIRunFramework* framework,
						 const string& instanceName,
						 const string& className,
						 Module* module)
  : ComponentInstance(framework, instanceName, className), module(module)
{
  // See if we have a user-interface...
  if(module->haveUI()){
    specialPorts.push_back(new CCAPortInstance("ui", "sci.cca.UIPort",
					       sci::cca::TypeMap::pointer(0),
					       sci::cca::Port::pointer(new SCIRunUIPort(this)),
					       CCAPortInstance::Provides));
  }
}

SCIRunComponentInstance::~SCIRunComponentInstance()
{
}

PortInstance* SCIRunComponentInstance::getPortInstance(const string& name)
{
  // SCIRun ports can potentially have the same name for both, so
  // SCIRunPortInstance tags them with a prefix of "Input: " or
  // "Output: ", so we need to check that first.
  if(name.substr(0, 7) == "Input: "){
    IPort* port = module->getIPort(name.substr(7));
    if(!port)
      return 0;
    return new SCIRunPortInstance(this, port, SCIRunPortInstance::Input);
  } else if(name.substr(0,8) == "Output: "){
    OPort* port = module->getOPort(name.substr(8));
    if(!port)
      return 0;
    return new SCIRunPortInstance(this, port, SCIRunPortInstance::Output);
  } else {
    for(unsigned int i=0;i<specialPorts.size();i++)
      if(specialPorts[i]->getName() == name)
	return specialPorts[i];
    return 0;
  }
}

PortInstanceIterator* SCIRunComponentInstance::getPorts()
{
  return new Iterator(this);
}

SCIRunComponentInstance::Iterator::Iterator(SCIRunComponentInstance* component)
  : component(component), idx(0)
{
}

SCIRunComponentInstance::Iterator::~Iterator()
{
}

void SCIRunComponentInstance::Iterator::next()
{
  idx++;
}

bool SCIRunComponentInstance::Iterator::done()
{
  return idx >= (int)component->specialPorts.size()
    +component->module->numOPorts()
    +component->module->numIPorts();
}

PortInstance* SCIRunComponentInstance::Iterator::get()
{
  Module* module = component->module;
  int spsize = static_cast<int>(component->specialPorts.size());
  if(idx < spsize)
    return component->specialPorts[idx];
  else if(idx < spsize+module->numOPorts())
    return new SCIRunPortInstance(component,
				  module->getOPort(idx-spsize),
				  SCIRunPortInstance::Output);
  else if(idx < spsize+module->numOPorts()
	  +module->numIPorts())
    return new SCIRunPortInstance(component,
				  module->getIPort(idx-spsize-module->numOPorts()),
				  SCIRunPortInstance::Input);
  else
    return 0; // Illegal
}
