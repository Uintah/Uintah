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
      Uses, Provides
    };
    CCAPortInstance(const std::string& portname, const std::string& classname,
		    const gov::cca::TypeMap::pointer& properties,
		    PortType porttype);
    CCAPortInstance(const std::string& portname, const std::string& classname,
		    const gov::cca::TypeMap::pointer& properties,
		    const gov::cca::Port::pointer& port,
		    PortType porttype);
    ~CCAPortInstance();
    std::string name;
    std::string type;
    gov::cca::TypeMap::pointer properties;
    std::vector<CCAPortInstance*> connections;
    gov::cca::Port::pointer port;
    PortType porttype;
    virtual bool connect(PortInstance*);

  private:
    CCAPortInstance(const CCAPortInstance&);
    CCAPortInstance& operator=(const CCAPortInstance&);
  };
}

#endif
