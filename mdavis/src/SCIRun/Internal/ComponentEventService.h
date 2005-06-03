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
 *  ComponentEventService.h: Implementation of the CCA ComponentEventService interface for SCIRun
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_ComponentEventService_h
#define SCIRun_ComponentEventService_h

#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/Internal/InternalComponentModel.h>
#include <SCIRun/Internal/InternalComponentInstance.h>
#include <vector>

namespace SCIRun {

class SCIRunFramework;
class ComponentEvent;

/**
 * \class ComponentEventService
 *
 * The component event service is a CCA port that is used to register
 * command objects (analogous to callback methods) with events broadcast
 * from the framework (?)
 *
 */
class ComponentEventService : public sci::cca::ports::ComponentEventService,
                              public InternalComponentInstance
{
public:
  virtual ~ComponentEventService();

  /** Factory method for allocating new ComponentEventService objects.  Returns
      a smart pointer to the newly allocated object registered in the framework
      \em fwk with the instance name \em name. */
  static InternalComponentInstance* create(SCIRunFramework* fwk,
                                           const std::string& name);
  
  /** ? */
  sci::cca::Port::pointer getService(const std::string&);
  
  /** ? */
  virtual void
  addComponentEventListener(sci::cca::ports::ComponentEventType type,
                            const sci::cca::ports::ComponentEventListener::pointer& l,
                            bool playInitialEvents);
  
  /** ? */
  virtual void
  removeComponentEventListener(sci::cca::ports::ComponentEventType type,
                               const sci::cca::ports::ComponentEventListener::pointer& l);
  
  virtual void moveComponent(const sci::cca::ComponentID::pointer& id, int x, int y);

private:
  friend void SCIRunFramework::emitComponentEvent(ComponentEvent* event);
  struct Listener
  {
    sci::cca::ports::ComponentEventType type;
    sci::cca::ports::ComponentEventListener::pointer l;
    Listener(sci::cca::ports::ComponentEventType type,
             const sci::cca::ports::ComponentEventListener::pointer& l)
      : type(type), l(l)
    {  }
  };

  std::vector<Listener*> listeners;
  std::vector<sci::cca::ports::ComponentEvent::pointer> events;

  ComponentEventService(SCIRunFramework* fwk, const std::string& name);
  void emitComponentEvent(const sci::cca::ports::ComponentEvent::pointer& event);
};


} // end namespace SCIRun

#endif
