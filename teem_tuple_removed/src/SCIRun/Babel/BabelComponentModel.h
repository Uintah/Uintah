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
 *  BabelComponentModel.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   July 2002
 *
 */

#ifndef SCIRun_Babel_BabelComponentModel_h
#define SCIRun_Babel_BabelComponentModel_h

#include <SCIRun/ComponentModel.h>
#include <SCIRun/ComponentInstance.h>
#include <SCIRun/Babel/gov_cca.hh>
#include <string>
#include <map>

namespace SCIRun {
  class SCIRunFramework;
  class BabelComponentDescription;
  class BabelComponentInstance;

  class BabelComponentModel : public ComponentModel {
  public:
    BabelComponentModel(SCIRunFramework* framework);
    virtual ~BabelComponentModel();

    gov::cca::Services createServices(const std::string& instanceName,
					       const std::string& className,
					       const gov::cca::TypeMap& properties);
    virtual bool haveComponent(const std::string& type);
    virtual ComponentInstance* createInstance(const std::string& name,
					      const std::string& type);

    virtual std::string createComponent(const std::string& name,
					 const std::string& type);
						     
    virtual bool destroyInstance(ComponentInstance *ci);
    virtual std::string getName() const;
    virtual void listAllComponentTypes(std::vector<ComponentDescription*>&,
				       bool);

  private:
    SCIRunFramework* framework;
    typedef std::map<std::string, BabelComponentDescription*> componentDB_type;
    componentDB_type components;

    void destroyComponentList();
    void buildComponentList();
    void readComponentDescription(const std::string& file);

    BabelComponentModel(const BabelComponentModel&);
    BabelComponentModel& operator=(const BabelComponentModel&);
  };
} //namespace SCIRun

#endif
