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
 *  ComponentID.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_ComponentID_h
#define SCIRun_Framework_ComponentID_h

#include <Core/CCA/spec/cca_sidl.h>

namespace SCIRun {
  class SCIRunFramework;
  class ComponentID : public sci::cca::ComponentID {
  public:
    ComponentID(SCIRunFramework* framework, const std::string& name);
    virtual ~ComponentID();

    virtual std::string getInstanceName();
    virtual std::string getSerialization();

    SCIRunFramework* framework;
    const std::string name;
  private:
    ComponentID(const ComponentID&);
    ComponentID& operator=(const ComponentID&);
  };
}

#endif
