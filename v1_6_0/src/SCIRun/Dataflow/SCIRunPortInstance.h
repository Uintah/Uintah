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
 *  SCIRunPortInstance.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_CCA_SCIRunPortInstance_h
#define SCIRun_CCA_SCIRunPortInstance_h

#include <SCIRun/PortInstance.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <map>
#include <string>
#include <vector>

namespace SCIRun {
  class Port;
  class SCIRunComponentInstance;
  class SCIRunPortInstance : public PortInstance {
  public:
    enum PortType {
      Output, Input
    };
    SCIRunPortInstance(SCIRunComponentInstance*, Port* port, PortType type);
    ~SCIRunPortInstance();

    virtual bool connect(PortInstance*);
    virtual PortInstance::PortType portType();
    virtual std::string getUniqueName();
    virtual bool disconnect(PortInstance*);
    virtual bool canConnectTo(PortInstance *);

  private:
    SCIRunPortInstance(const SCIRunPortInstance&);
    SCIRunPortInstance& operator=(const SCIRunPortInstance&);

    SCIRunComponentInstance* component;
    Port* port;
    PortType porttype;
  };
}

#endif
