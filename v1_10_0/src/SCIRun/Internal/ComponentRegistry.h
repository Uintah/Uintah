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
 *  ComponentRegistry.h: Implementation of the CCA ComponentRegistry interface for SCIRun
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_ComponentRegistry_h
#define SCIRun_ComponentRegistry_h

#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/Internal/InternalComponentModel.h>
#include <SCIRun/Internal/InternalComponentInstance.h>
#include <vector>

namespace SCIRun {
  class SCIRunFramework;
  class ComponentRegistry : public sci::cca::ports::ComponentRepository, public InternalComponentInstance {
  public:
    virtual ~ComponentRegistry();
    static InternalComponentInstance* create(SCIRunFramework* fwk,
					     const std::string& name);
    sci::cca::Port::pointer getService(const std::string&);

    virtual SSIDL::array1<sci::cca::ComponentClassDescription::pointer> getAvailableComponentClasses();
  private:
    ComponentRegistry(SCIRunFramework* fwk, const std::string& name);
  };
}

#endif
