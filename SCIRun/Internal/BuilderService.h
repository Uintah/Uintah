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
 *  BuilderService.h: Implementation of the CCA BuilderService interface for SCIRun
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_BuilderService_h
#define SCIRun_BuilderService_h

#include <Core/CCA/ccaspec/cca_sidl.h>
#include <SCIRun/Internal/InternalComponentModel.h>
#include <SCIRun/Internal/InternalComponentInstance.h>

namespace SCIRun {
  class SCIRunFramework;
  class BuilderService : public gov::cca::BuilderService_interface, public InternalComponentInstance {
  public:
    virtual ~BuilderService();
    gov::cca::ComponentID createComponentInstance(const std::string& name,
						  const std::string& type);
    void connect(const gov::cca::ComponentID& c1, const std::string& port1,
		 const gov::cca::ComponentID& c2, const std::string& port2);
    static InternalComponentInstance* create(SCIRunFramework* fwk,
					     const std::string& name);
    gov::cca::Port getService(const std::string&);
  private:
    BuilderService(SCIRunFramework* fwk, const std::string& name);
  };
}

#endif
