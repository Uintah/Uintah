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
 *  SCIRunFramework.h: An instance of the SCIRun framework
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_SCIRunFramework_h
#define SCIRun_Framework_SCIRunFramework_h

#include <Core/CCA/spec/cca_sidl.h>
#include <vector>
#include <map>
#include <string>

namespace SCIRun {
  class ComponentDescription;
  class ComponentModel;
  class ComponentInstance;
  class InternalComponentModel;
  class CCAComponentModel;
  class SCIRunFramework : public gov::cca::AbstractFramework {
  public:
    SCIRunFramework();
    virtual ~SCIRunFramework();

    virtual gov::cca::TypeMap::pointer createTypeMap();
    virtual gov::cca::Services::pointer getServices(const ::std::string& selfInstanceName,
						    const ::std::string& selfClassName,
						    const gov::cca::TypeMap::pointer& selfProperties);
    virtual void releaseServices(const gov::cca::Services::pointer& svc);
    virtual void shutdownFramework();
    virtual gov::cca::AbstractFramework::pointer createEmptyFramework();


    // Semi-private:
    // Used by builderService
    gov::cca::ComponentID::pointer createComponentInstance(const std::string& name, const std::string& type);
    gov::cca::Port::pointer getFrameworkService(const std::string& type);
    void registerComponent(ComponentInstance* ci, const std::string& name);
    void listAllComponentTypes(std::vector<ComponentDescription*>&,
			       bool);
  protected:
    friend class Services;
    // Put these in a private structure to avoid #include bloat?
    std::vector<ComponentModel*> models;
    std::map<std::string, ComponentInstance*> activeInstances;
    InternalComponentModel* internalServices;
    CCAComponentModel* cca;

    friend class BuilderService;
    ComponentInstance* getComponent(const std::string& name);
  };
}

#endif
