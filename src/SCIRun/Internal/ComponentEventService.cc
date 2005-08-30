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
#include <SCIRun/Internal/ComponentEvent.h>
#include <SCIRun/CCA/CCAComponentInstance.h>
#include <SCIRun/CCA/CCAException.h>
#include <SCIRun/TypeMap.h>
#include <iostream>

namespace SCIRun {

  ComponentEventService::ComponentEventService(SCIRunFramework* framework)
  : InternalFrameworkServiceInstance(framework, "internal:ComponentEventService"),
    lock_listeners("ComponentEventService::listeners lock"),
    lock_events("ComponentEventService::events lock")
{
}

ComponentEventService::~ComponentEventService()
{
  std::cerr << "EventService destroyed..." << std::endl;
}

InternalFrameworkServiceInstance* ComponentEventService::create(SCIRunFramework* framework)
{
    ComponentEventService* n = new ComponentEventService(framework);
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
    Listener *listener = new Listener(type, l);
    lock_listeners.lock();
    listeners.push_back(listener);
    lock_listeners.unlock();
    if (playInitialEvents) {
        // send listener all events stored in vector
        lock_events.lock();
        for (std::vector<sci::cca::ports::ComponentEvent::pointer>::iterator iter =
                events.begin();
                iter != events.end(); iter++) {
            listener->l->componentActivity((*iter));
        }
        lock_events.unlock();
    }
}

void
ComponentEventService::removeComponentEventListener(
                                  sci::cca::ports::ComponentEventType type,
                const sci::cca::ports::ComponentEventListener::pointer& l)
{
    SCIRun::Guard g1(&lock_listeners);
    for (std::vector<Listener*>::iterator iter = listeners.begin();
            iter != listeners.end(); iter++) {
        if ((*iter)->type == type && (*iter)->l == l) {
            delete *iter;
        }
    }
}

void
ComponentEventService::moveComponent(const sci::cca::ComponentID::pointer& id, int x, int y)
{
    ComponentInstance* ci = framework->lookupComponent(id->getInstanceName());
    if (ci) {
        std::string cn = ci->getClassName();
        unsigned int firstColon = cn.find(':');
        std::string modelName;
        if (firstColon != std::string::npos) {
            modelName = cn.substr(0, firstColon);
        } else {
            modelName = cn;
        }
        sci::cca::TypeMap::pointer properties;
        if (modelName == "CCA") {
            // empty string argument gets component properties;
            properties =
                ((CCAComponentInstance*)ci)->getPortProperties("");
            if (properties.isNull()) {
                properties = TypeMap::pointer(new TypeMap);
            }
        } else {
            properties = TypeMap::pointer(new TypeMap);
        }
        properties->putInt("x", x);
        properties->putInt("y", y);

        sci::cca::ports::ComponentEvent::pointer ce =
            ComponentEvent::pointer(
                new ComponentEvent(sci::cca::ports::ComponentMoved, id, properties)
            );
        emitComponentEvent(ce);
    } else {
      // throw exception?
      std::cerr << "Error: could not locate component instance for "
                << id->getInstanceName() << " in ComponentEventService::moveComponent."
                << std::endl;
    }
}

void
ComponentEventService::emitComponentEvent(const sci::cca::ports::ComponentEvent::pointer& event)
{
    // iterate through listeners and call connectionActivity
    // should the event type to be emitted be ALL?
    if (event->getEventType() == sci::cca::ports::AllComponentEvents) {
        return;
    }

    lock_listeners.lock();
    for (std::vector<Listener*>::iterator iter=listeners.begin();
            iter != listeners.end(); iter++) {
        if ((*iter)->type == sci::cca::ports::AllComponentEvents ||
                (*iter)->type == event->getEventType()) {
            (*iter)->l->componentActivity(event);
        }
    }
    lock_listeners.unlock();

    // how to keep track of events?
    lock_events.lock();
    events.push_back(event);
    lock_events.unlock();
}

} // end namespace SCIRun
