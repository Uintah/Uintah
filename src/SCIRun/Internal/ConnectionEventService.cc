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

#include <SCIRun/Internal/ConnectionEventService.h>
#include <SCIRun/SCIRunFramework.h>


namespace SCIRun {

  ConnectionEventService::ConnectionEventService(SCIRunFramework* framework)
    : InternalFrameworkServiceInstance(framework, "internal:ConnectionEventService")
  {
  }
  
  ConnectionEventService::~ConnectionEventService()
  {
    std::cerr << "ConnectionEventService destroyed..." << std::endl;
  }
  
  InternalFrameworkServiceInstance* ConnectionEventService::create(SCIRunFramework* framework)
  {
    ConnectionEventService* n = new ConnectionEventService(framework);
    n->addReference();
    return n;
  }

sci::cca::Port::pointer ConnectionEventService::getService(const std::string&)
{
    return sci::cca::Port::pointer(this);
}

void ConnectionEventService::addConnectionEventListener(
        sci::cca::ports::EventType et,
        const sci::cca::ports::ConnectionEventListener::pointer& cel)
{
    std::cerr << "ConnectionEventService::addConnectionEventListener" << std::endl;
    listeners.push_back(new Listener(et, cel));
}

void ConnectionEventService::removeConnectionEventListener(
        sci::cca::ports::EventType et,
        const sci::cca::ports::ConnectionEventListener::pointer& cel)
{
    for (std::vector<Listener*>::iterator iter = listeners.begin();
            iter != listeners.end(); iter++) {
        if ((*iter)->type == et && (*iter)->l == cel) {
            delete *iter;
        }
    }
}

void
ConnectionEventService::emitConnectionEvent(const sci::cca::ports::ConnectionEvent::pointer& event)
{
    // iterate through listeners and call connectionActivity
    // should the event type to be emitted be ALL?
    if (event->getEventType() == sci::cca::ports::ALL) {
        return;
    }

    for (std::vector<Listener*>::iterator iter=listeners.begin();
            iter != listeners.end(); iter++) {
        if ((*iter)->type == sci::cca::ports::ALL ||
                (*iter)->type == event->getEventType()) {
            (*iter)->l->connectionActivity(event);
        }
    }
}


}
