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
 *  SCIRunComponentModel.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_InternalComponentModel_h
#define SCIRun_Framework_InternalComponentModel_h

#include <SCIRun/ComponentModel.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <map>
#include <string>

namespace SCIRun {
  class ComponentDescription;
  class InternalComponentDescription;
  class SCIRunFramework;
  class InternalComponentModel : public ComponentModel {
  public:
      
    InternalComponentModel(SCIRunFramework* framework);
    virtual ~InternalComponentModel();

    virtual bool haveComponent(const std::string& type);
    virtual ComponentInstance* createInstance(const std::string& name,
					      const std::string& type);
    virtual bool destroyInstance(ComponentInstance *ci);
    sci::cca::Port::pointer getFrameworkService(const std::string& type,
						const std::string& componentName);
    bool releaseFrameworkService(const std::string& type,
				 const std::string& componentName);
    virtual std::string getName() const;
    virtual void listAllComponentTypes(std::vector<ComponentDescription*>&,
				       bool);
  private:
    std::map<std::string, InternalComponentDescription*> services;
    SCIRunFramework* framework;

    void addService(InternalComponentDescription* cd);

    InternalComponentModel(const InternalComponentModel&);
    InternalComponentModel& operator=(const InternalComponentModel&);
  };
}

#endif
