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
#include <SCIRun/Internal/InternalComponentModel.h>
#include <SCIRun/Internal/InternalComponentInstance.h>
#include <vector>

namespace SCIRun {
  class SCIRunFramework;
  class ComponentEventService : public gov::cca::ports::ComponentEventService, public InternalComponentInstance {
  public:
    virtual ~ComponentEventService();
    static InternalComponentInstance* create(SCIRunFramework* fwk,
					     const std::string& name);
    gov::cca::Port::pointer getService(const std::string&);

    virtual void addComponentEventListener(gov::cca::ports::ComponentEventType type,
					   const gov::cca::ports::ComponentEventListener::pointer& l,
					   bool playInitialEvents);
    virtual void removeComponentEventListener(gov::cca::ports::ComponentEventType type,
					      const gov::cca::ports::ComponentEventListener::pointer& l);
    virtual void moveComponent(const gov::cca::ComponentID::pointer& id, int x, int y);
  private:

    struct Listener {
      gov::cca::ports::ComponentEventType type;
      gov::cca::ports::ComponentEventListener::pointer l;
      Listener(gov::cca::ports::ComponentEventType type, const gov::cca::ports::ComponentEventListener::pointer& l)
	: type(type), l(l)
      {
      }
    };
    std::vector<Listener*> listeners;
    ComponentEventService(SCIRunFramework* fwk, const std::string& name);
  };
}

#endif
