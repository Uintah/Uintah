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
 *  SCIRunComponentInstance.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_SCIRunComponentInstance_h
#define SCIRun_Framework_SCIRunComponentInstance_h

#include <SCIRun/ComponentInstance.h>
#include <SCIRun/PortInstanceIterator.h>
#include <string>
#include <vector>

namespace SCIRun {
  class CCAPortInstance;
  class Module;
  class SCIRunComponentInstance : public ComponentInstance {
  public:
    SCIRunComponentInstance(SCIRunFramework* fwk,
			    const std::string& instanceName,
			    const std::string& className,
			    Module* module);
    virtual ~SCIRunComponentInstance();

    // Methods from ComponentInstance
    virtual PortInstance* getPortInstance(const std::string& name);
    virtual PortInstanceIterator* getPorts();
    Module* getModule() {
      return module;
    }
  private:
    class Iterator : public PortInstanceIterator {
    public:
      Iterator(SCIRunComponentInstance*);
      virtual ~Iterator();
      virtual PortInstance* get();
      virtual bool done();
      virtual void next();
    private:
      Iterator(const Iterator&);
      Iterator& operator=(const Iterator&);

      SCIRunComponentInstance* component;
      int idx;
    };
    Module* module;
    std::vector<CCAPortInstance*> specialPorts;
    SCIRunComponentInstance(const SCIRunComponentInstance&);
    SCIRunComponentInstance& operator=(const SCIRunComponentInstance);
  };
}

#endif
