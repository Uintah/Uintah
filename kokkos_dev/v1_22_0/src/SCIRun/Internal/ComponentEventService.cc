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

