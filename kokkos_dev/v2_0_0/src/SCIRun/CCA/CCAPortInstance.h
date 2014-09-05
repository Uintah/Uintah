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
 *  CCAPortInstance.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_CCA_CCAPortInstance_h
#define SCIRun_CCA_CCAPortInstance_h

#include <SCIRun/PortInstance.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <map>
#include <string>
#include <vector>

namespace SCIRun {
  class CCAPortInstance : public PortInstance {
  public:
    enum PortType {
      Uses=0, Provides=1
    };
    CCAPortInstance(const std::string& portname, const std::string& classname,
		    const sci::cca::TypeMap::pointer& properties,
		    PortType porttype);
    CCAPortInstance(const std::string& portname, const std::string& classname,
		    const sci::cca::TypeMap::pointer& properties,
		    const sci::cca::Port::pointer& port,
		    PortType porttype);
    ~CCAPortInstance();
    virtual bool connect(PortInstance*);
    virtual PortInstance::PortType portType();
    virtual std::string getType();
    virtual std::string getModel();
    virtual std::string getUniqueName();
    virtual bool disconnect(PortInstance*);
    virtual bool canConnectTo(PortInstance*);
    virtual bool available();
    virtual PortInstance *getPeer();
    std::string getName();
    void incrementUseCount();
    bool decrementUseCount();
  private:
    friend class CCAComponentInstance;
    friend class BridgeComponentInstance;
    std::string name;
    std::string type;
    sci::cca::TypeMap::pointer properties;
    std::vector<PortInstance*> connections;
    sci::cca::Port::pointer port;
    PortType porttype;
    int useCount;

    CCAPortInstance(const CCAPortInstance&);
    CCAPortInstance& operator=(const CCAPortInstance&);
  };
}

#endif
