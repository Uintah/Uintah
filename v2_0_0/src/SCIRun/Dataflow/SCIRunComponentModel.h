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

#ifndef SCIRun_Framework_SCIRunComponentModel_h
#define SCIRun_Framework_SCIRunComponentModel_h

#include <SCIRun/ComponentModel.h>

namespace SCIRun {
  class Network;
  class SCIRunComponentDescription;
  class SCIRunFramework;
  class GuiInterface;
  class SCIRunComponentModel : public ComponentModel {
  public:
    SCIRunComponentModel(SCIRunFramework* framework);
    virtual ~SCIRunComponentModel();

    virtual bool haveComponent(const std::string& type);
    virtual ComponentInstance* createInstance(const std::string& name,
					      const std::string& type);
    virtual bool destroyInstance(ComponentInstance * ic);
    virtual std::string getName() const;
    virtual void listAllComponentTypes(std::vector<ComponentDescription*>&,
				       bool);
  private:
    SCIRunFramework* framework;
    GuiInterface* gui;
    Network* net;

    SCIRunComponentModel(const SCIRunComponentModel&);
    SCIRunComponentModel& operator=(const SCIRunComponentModel&);
  };
}

#endif
