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
 *  BridgeComponentModel.h: 
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   September, 2003
 *
 */

#ifndef SCIRun_CCA_BridgeComponentModel_h
#define SCIRun_CCA_BridgeComponentModel_h

#include <SCIRun/ComponentModel.h>
#include <SCIRun/ComponentInstance.h>
#include <SCIRun/Bridge/BridgeServices.h>
#include <SCIRun/Bridge/BridgeComponent.h>
#include <string>
#include <map>

namespace SCIRun {
  class SCIRunFramework;
  class BridgeComponentDescription;
  class BridgeComponentInstance;

  class BridgeComponentModel : public ComponentModel {
  public:
    BridgeComponentModel(SCIRunFramework* framework);
    virtual ~BridgeComponentModel();

    BridgeServices* createServices(const std::string& instanceName,
					       const std::string& className);
    virtual bool haveComponent(const std::string& type);
    virtual ComponentInstance* createInstance(const std::string& name,
					      const std::string& type);
    
    virtual bool destroyInstance(ComponentInstance *ci);
    virtual std::string getName() const;
    virtual void listAllComponentTypes(std::vector<ComponentDescription*>&,
				       bool);
    int addLoader(resourceReference *rr);
    int removeLoader(const std::string &loaderName);

  private:
    SCIRunFramework* framework;
    typedef std::map<std::string, BridgeComponentDescription*> componentDB_type;
    componentDB_type components;

    void destroyComponentList();
    void buildComponentList();
    void readComponentDescription(const std::string& file);
    resourceReference *getLoader(std::string loaderName);
    BridgeComponentModel(const BridgeComponentModel&);
    BridgeComponentModel& operator=(const BridgeComponentModel&);

    std::vector<resourceReference* > loaderList;

  };
}

#endif
