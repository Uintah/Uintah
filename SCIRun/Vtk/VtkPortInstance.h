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
 *  VtkPortInstance.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#ifndef SCIRun_Vtk_VtkPortInstance_h
#define SCIRun_Vtk_VtkPortInstance_h

#include <SCIRun/PortInstance.h>
#include <map>
#include <string>
#include <vector>
#include <SCIRun/Vtk/Port.h>

namespace SCIRun {
  class vtk::Port;
  class VtkComponentInstance;
  class VtkPortInstance : public PortInstance {
  public:
    enum PortType {
      Output, Input
    };
    VtkPortInstance(VtkComponentInstance* ci, vtk::Port* port, PortType type);
    ~VtkPortInstance();

    virtual bool connect(PortInstance*);
    virtual PortInstance::PortType portType();
    virtual std::string getUniqueName();
    virtual bool disconnect(PortInstance*);
    virtual bool canConnectTo(PortInstance *);

  private:
    friend class BridgeComponentInstance;
    
    VtkPortInstance(const VtkPortInstance&);
    VtkPortInstance& operator=(const VtkPortInstance&);

    VtkComponentInstance* ci;
    vtk::Port* port;
    PortType porttype;
  };
}

#endif
