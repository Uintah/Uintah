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
 *  ComponentModel.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_ComponentModel_h
#define SCIRun_Framework_ComponentModel_h

#include <string>
#include <vector>
#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/resourceReference.h>
namespace SCIRun {
  class ComponentDescription;
  class ComponentInstance;
  class ComponentModel {
  public:
    ComponentModel(const std::string& prefixName);
    virtual ~ComponentModel();

    virtual bool haveComponent(const std::string& type) = 0;
    virtual ComponentInstance* createInstance(const std::string& name,
					      const std::string& type);
    virtual bool destroyInstance(ComponentInstance* ci)= 0;
    virtual std::string getName() const = 0;
    virtual void listAllComponentTypes(std::vector<ComponentDescription*>&,
				       bool) = 0;
    std::string prefixName;
  private:
    ComponentModel(const ComponentModel&);
    ComponentModel& operator=(const ComponentModel&);
  };
}

#endif
