/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is Vtk, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  VtkComponentInstance.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#ifndef SCIRun_Vtk_VtkComponentInstance_h
#define SCIRun_Vtk_VtkComponentInstance_h

#include <SCIRun/ComponentInstance.h>
#include <SCIRun/PortInstanceIterator.h>
#include <SCIRun/Vtk/Port.h>
#include <SCIRun/Vtk/Component.h>
#include <string>
#include <vector>

namespace SCIRun {
  class CCAPortInstance;
  class Module;
  class VtkComponentInstance : public ComponentInstance {
  public:
    VtkComponentInstance(SCIRunFramework* fwk,
			    const std::string& instanceName,
			    const std::string& className,
			    vtk::Component * component);
    virtual ~VtkComponentInstance();

    // Methods from ComponentInstance
    virtual PortInstance* getPortInstance(const std::string& name);
    virtual PortInstanceIterator* getPorts();
    vtk::Component* getComponent() {
      return component;
    }
  private:
    class Iterator : public PortInstanceIterator {
    public:
      Iterator(VtkComponentInstance*);
      virtual ~Iterator();
      virtual PortInstance* get();
      virtual bool done();
      virtual void next();
    private:
      Iterator(const Iterator&);
      Iterator& operator=(const Iterator&);

      VtkComponentInstance* ci;
      int index;
    };
    vtk::Component* component;
    std::vector<CCAPortInstance*> specialPorts;
    VtkComponentInstance(const VtkComponentInstance&);
    VtkComponentInstance& operator=(const VtkComponentInstance);
  };
}

#endif
