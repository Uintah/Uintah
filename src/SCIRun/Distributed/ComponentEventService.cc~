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

#include <SCIRun/Distributed/ComponentEventService.h>
#include <iostream>

namespace SCIRun {

  ComponentEventService::~ComponentEventService()
  {
    std::cerr << "EventService destroyed..." << std::endl;
  }

  ComponentEventService::pointer ComponentEventService::create(DistributedFramework *framework)
  {
    return pointer(new ComponentEventService(framework));
  }

  void ComponentEventService::emitComponentEvent(const Distributed::ComponentEvent::pointer& event)
  {
    // iterate through listeners and call connectionActivity
    // should the event type to be emitted be ALL?
    if (event->getType() == Distributed::AllComponentEvents) {
      return;
    }

    {
      SCIRun::Guard lock(&listeners_lock);

      for (std::vector<Listener*>::iterator iter=listeners.begin();
	   iter != listeners.end(); iter++) {
        if ((*iter)->type == Distributed::AllComponentEvents ||
	    (*iter)->type == event->getType()) {
	  (*iter)->listener->componentActivity(event);
        }
      }
    }
  }
  
} // end namespace SCIRun
