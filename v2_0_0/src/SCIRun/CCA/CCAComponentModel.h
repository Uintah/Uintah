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
 *  CCAComponentModel.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_CCA_CCAComponentModel_h
#define SCIRun_CCA_CCAComponentModel_h

#include <SCIRun/ComponentModel.h>
#include <SCIRun/ComponentInstance.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <string>
#include <map>

namespace SCIRun {
  class SCIRunFramework;
  class CCAComponentDescription;
  class CCAComponentInstance;

  class CCAComponentModel : public ComponentModel {
  public:
    CCAComponentModel(SCIRunFramework* framework);
    virtual ~CCAComponentModel();

    sci::cca::Services::pointer createServices(const std::string& instanceName,
					       const std::string& className,
					       const sci::cca::TypeMap::pointer& properties);
    virtual bool haveComponent(const std::string& type);
    virtual ComponentInstance* createInstance(const std::string& name,
					      const std::string& type,
					      const sci::cca::TypeMap::pointer& properties);
    
    virtual bool destroyInstance(ComponentInstance *ci);
    virtual std::string getName() const;
    virtual void listAllComponentTypes(std::vector<ComponentDescription*>&,
				       bool);
    int addLoader(resourceReference *rr);
    int removeLoader(const std::string &loaderName);

  private:
    SCIRunFramework* framework;
    typedef std::map<std::string, CCAComponentDescription*> componentDB_type;
    componentDB_type components;

    void destroyComponentList();
    void buildComponentList();
    void readComponentDescription(const std::string& file);
    resourceReference *getLoader(std::string loaderName);
    CCAComponentModel(const CCAComponentModel&);
    CCAComponentModel& operator=(const CCAComponentModel&);

    std::vector<resourceReference* > loaderList;

  };
}

#endif
