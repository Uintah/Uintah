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
 *  ConnectionEventService.h: Baseementation of the CCA ConnectionEventService interface for SCIRun
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_ConnectionEventService_h
#define SCIRun_ConnectionEventService_h

#include <Core/Thread/Mutex.h>
#include <Core/Thread/Guard.h>
#include <SCIRun/Core/ConnectionEventServiceBase.h>
#include <vector>

namespace SCIRun {

  namespace CorePorts = core::ports;

  class CoreFramework;
  class ConnectionEvent;
  
  /**
   * \class ConnectionEventService
   *
   * The component event service is a CCA port that is used to register
   * command objects (analogous to callback methods) with events broadcast
   * from the framework (?)
   *
   */
  class ConnectionEventService : public ConnectionEventServiceBase<CorePorts::ConnectionEventService>
  {
  public:
    typedef internal::Service::pointer pointer;
    
    ConnectionEventService(const CoreFramework::internalPointer &framework);

    virtual ~ConnectionEventService();
    
    /** Factory method for allocating new ConnectionEventService objects.  Returns
	a smart pointer to the newly allocated object registered in the framework
	\em fwk with the instance name \em name. */

    static pointer create(const CoreFramework::internalPointer &framework);
    
    /** ? */
    void emitConnectionEvent(const Ports::ConnectionEvent::pointer& event);

  private:
    CoreFramework *framework;

  };


} // end namespace SCIRun

#endif
