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
 *  SCIRunComponentDescription.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_SCIRunComponentDescription_h
#define SCIRun_Framework_SCIRunComponentDescription_h

#include <SCIRun/ComponentDescription.h>
#include <string>

namespace SCIRun {
  class ComponentModel;
  class SCIRunComponentModel;
  class SCIRunComponentDescription : public ComponentDescription {
  public:
    SCIRunComponentDescription(SCIRunComponentModel* model,
			       const std::string& package,
			       const std::string& category,
			       const std::string& module);
    virtual ~SCIRunComponentDescription();

    virtual std::string getType() const;
    virtual const ComponentModel* getModel() const;
  private:
    SCIRunComponentModel* model;
    std::string package;
    std::string category;
    std::string module;

    SCIRunComponentDescription(const SCIRunComponentDescription&);
    SCIRunComponentDescription& operator=(const SCIRunComponentDescription&);
  };
}

#endif
