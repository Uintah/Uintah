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
 *  BabelPortInstance.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   July 2002
 *
 */

#ifndef SCIRun_Babel_BabelPortInstance_h
#define SCIRun_Babel_BabelPortInstance_h

#include <SCIRun/PortInstance.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/Babel/gov_cca.hh>
#include <map>
#include <string>
#include <vector>

namespace SCIRun{
  class BabelPortInstance : public PortInstance {
  public:
    enum PortType {
      Uses, Provides
    };
    BabelPortInstance(const std::string& portname, const std::string& classname,
		    const gov::cca::TypeMap& properties,
		    PortType porttype);
    BabelPortInstance(const std::string& portname, const std::string& classname,
		    const gov::cca::TypeMap& properties,
		    const gov::cca::Port& port,
		    PortType porttype);
    ~BabelPortInstance();
    virtual bool connect(PortInstance*);
    virtual PortInstance::PortType portType();
    virtual std::string getType();
    virtual std::string getModel();
    virtual std::string getUniqueName();
    virtual bool disconnect(PortInstance*);
    virtual bool canConnectTo(PortInstance*);
    virtual bool available();
    virtual PortInstance* getPeer();
    std::string getName();
    void incrementUseCount();
    bool decrementUseCount();
  public:
    PortType porttype;
    std::vector<PortInstance*> connections;
    friend class BabelComponentInstance;
    std::string name;
    std::string type;
    gov::cca::TypeMap properties;
    gov::cca::Port port;

    int useCount;

    BabelPortInstance(const BabelPortInstance&);
    BabelPortInstance& operator=(const BabelPortInstance&);
  };
} //namespace SCIRun

#endif

