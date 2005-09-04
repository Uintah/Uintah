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
 *  ConnectionEventService.h: Implementation of the CCA
 *                            ConnectionEventService interface for SCIRun
 *
 *  Written by:
 *   Ayla Khan
 *   Scientific Computing and Imaging Institute
 *   University of Utah
 *   October 2004
 *
 *  Copyright (C) 2004 SCI Institute
 *
 */

#ifndef SCIRun_ConnectionEventService_h
#define SCIRun_ConnectionEventService_h

#include <Core/Thread/Mutex.h>
#include <Core/Thread/Guard.h>
#include <Core/CCA/spec/sci_sidl.h>
#include <SCIRun/Internal/BuilderService.h>
#include <SCIRun/Internal/InternalComponentModel.h>
#include <SCIRun/Internal/InternalFrameworkServiceInstance.h>
#include <vector>

namespace SCIRun {

class BuilderService;
class ConnectionEvent;

/**
 * \class ConnectionEventService
 *
 * The connection event service is a CCA port that is used to register
 * connection activity between components (analogous to callback methods)
 * with events dispatched to components that implement the
 * ConnectionEventListener interface.
 */

class ConnectionEventService
    : public sci::cca::ports::ConnectionEventService,
      public InternalFrameworkServiceInstance
{
public:
    virtual ~ConnectionEventService();

    /** Factory method for allocating new ConnectionEventService objects.
    Returns a smart pointer to the newly allocated object registered in
    the framework \em fwk with the instance name \em name. */
  static InternalFrameworkServiceInstance* create(SCIRunFramework* fwk);

    /** Returns this service. */
    virtual sci::cca::Port::pointer getService(const std::string& name);

    /** Sign up to be told about connection activity of a given \em EventType. */
    virtual void addConnectionEventListener(sci::cca::ports::EventType et,
        const sci::cca::ports::ConnectionEventListener::pointer& cel);

    /** Ignore future ConnectionEvents of the given \em EventType. */
    virtual void removeConnectionEventListener(sci::cca::ports::EventType et,
        const sci::cca::ports::ConnectionEventListener::pointer& cel);

private:
    friend void BuilderService::emitConnectionEvent(ConnectionEvent* event);
    struct Listener
    {
        sci::cca::ports::EventType type;
        sci::cca::ports::ConnectionEventListener::pointer l;
        Listener(sci::cca::ports::EventType type,
                const sci::cca::ports::ConnectionEventListener::pointer& l) :
                type(type), l(l) {}
    };
    std::vector<Listener*> listeners;
    SCIRun::Mutex lock_listeners;

    ConnectionEventService(SCIRunFramework* framework);
    void emitConnectionEvent(const sci::cca::ports::ConnectionEvent::pointer &event);
};

}

#endif
