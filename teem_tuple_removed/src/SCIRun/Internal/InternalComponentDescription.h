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
 *  InternalComponentDescription.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Internal_InternalComponentDescription_h
#define SCIRun_Internal_InternalComponentDescription_h

#include <SCIRun/ComponentDescription.h>
#include <string>

namespace SCIRun {
  class InternalComponentInstance;
  class InternalComponentModel;
  class InternalComponentDescription : public ComponentDescription {
  public:
    InternalComponentDescription(InternalComponentModel* model,
				 const std::string& serviceType,
				 InternalComponentInstance* (*create)(SCIRunFramework*, const std::string&),
				 bool isSingleton);
    virtual ~InternalComponentDescription();
    virtual std::string getType() const;
    virtual const ComponentModel* getModel() const;

  private:
    friend class InternalComponentModel;
    InternalComponentModel* model;
    std::string serviceType;
    InternalComponentInstance* (*create)(SCIRunFramework*, const std::string&);
    InternalComponentInstance* singleton_instance;
    bool isSingleton;
    InternalComponentDescription(const InternalComponentDescription&);
    InternalComponentDescription& operator=(const InternalComponentDescription&);
  };
}

#endif
