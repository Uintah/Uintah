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
 *  PortInstance.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_PortInstance_h
#define SCIRun_PortInstance_h

#include <string>

namespace SCIRun {
  class PortInstance {
  public:
    PortInstance();
    virtual ~PortInstance();

    enum PortType {
      From=0,
      To=1
    };

    virtual bool connect(PortInstance*) = 0;
    virtual PortType portType() = 0;
    virtual std::string getType();
    virtual std::string getModel();
    virtual std::string getUniqueName() = 0;
    virtual bool disconnect(PortInstance*) =0;
    virtual bool canConnectTo(PortInstance *)=0;
    virtual bool available();
    virtual PortInstance* getPeer();
  private:
    PortInstance(const PortInstance&);
    PortInstance& operator=(const PortInstance&);
  };
}

#endif
