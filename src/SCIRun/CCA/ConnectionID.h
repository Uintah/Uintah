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
 *  ConnectionID.h: 
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 */

#ifndef SCIRun_Framework_ConnectionID_h
#define SCIRun_Framework_ConnectionID_h

#include <Core/CCA/spec/cca_sidl.h>

namespace SCIRun {
  class SCIRunFramework;
  class ComponentID;
  class ConnectionID : public gov::cca::ConnectionID {
  public:
    ConnectionID(const gov::cca::ComponentID::pointer& user,
		 const std::string& userPortName,
		 const gov::cca::ComponentID::pointer& provider,
		 const std::string& providerPortName);
    virtual ~ConnectionID();
    gov::cca::ComponentID::pointer getProvider();
    gov::cca::ComponentID::pointer getUser();
    std::string getProviderPortName();
    std::string getUserPortName();
  private:
    ConnectionID(const ConnectionID&);
    ConnectionID& operator=(const ConnectionID&);
    std::string userPortName;
    std::string providerPortName;
    gov::cca::ComponentID::pointer user;
    gov::cca::ComponentID::pointer provider;
  };
}

#endif

