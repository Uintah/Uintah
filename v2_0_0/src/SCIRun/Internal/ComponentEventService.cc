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
 *  ComponentEventService.cc: Implementation of CCA ComponentEventService for SCIRun
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/Internal/ComponentEventService.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/CCA/CCAException.h>
#include <iostream>
using namespace std;
using namespace SCIRun;

ComponentEventService::ComponentEventService(SCIRunFramework* framework,
			       const std::string& name)
  : InternalComponentInstance(framework, name, "internal:ComponentEventService")
{
}

ComponentEventService::~ComponentEventService()
{
  cerr << "EventService destroyed...\n";
}

InternalComponentInstance* ComponentEventService::create(SCIRunFramework* framework,
						  const std::string& name)
{
  ComponentEventService* n = new ComponentEventService(framework, name);
  n->addReference();
  return n;
}

sci::cca::Port::pointer ComponentEventService::getService(const std::string&)
{
  return sci::cca::Port::pointer(this);
}

void
ComponentEventService::addComponentEventListener(sci::cca::ports::ComponentEventType type,
						 const sci::cca::ports::ComponentEventListener::pointer& l,
						 bool playInitialEvents)
{
  listeners.push_back(new Listener(type, l));
  if(playInitialEvents){
    cerr << "addComponentEventListener not done!\n";
  }
}

void
ComponentEventService::removeComponentEventListener(sci::cca::ports::ComponentEventType /*type*/,
						    const sci::cca::ports::ComponentEventListener::pointer& /*l*/)
{
  cerr << "removeComponentEventListener not done!\n";
}

void ComponentEventService::moveComponent(const sci::cca::ComponentID::pointer& /*id*/,
					  int /*x*/, int /*y*/)
{
  cerr << "moveComponent not done!\n";
}

