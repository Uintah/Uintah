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
 *  BridgeServices.h:
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   September 2003
 *
 */


#ifndef SCIRUN_BRIDGE_BRIDGESERVICES_H
#define SCIRUN_BRIDGE_BRIDGESERVICES_H

//CCA:
#include <Core/CCA/spec/cca_sidl.h>
//Babel:
#include <SCIRun/Babel/framework.hh>
#include <SCIRun/Babel/gov_cca.hh>

namespace SCIRun {

  typedef enum {
    CCA = 1,
    Babel
  } modelT;
  
  class BridgeServices {
  public:
    BridgeServices::BridgeServices() { }
    virtual BridgeServices::~BridgeServices() { }
    
    virtual sci::cca::Port::pointer getCCAPort(const std::string& name) = 0;
    virtual gov::cca::Port getBabelPort(const std::string& name) = 0;
    
    virtual void releasePort(const std::string& name, const modelT model) = 0;
    virtual void registerUsesPort(const std::string& name, const std::string& type,
			  const modelT model) = 0;
    virtual void unregisterUsesPort(const std::string& name, const modelT model) = 0;
    virtual void addProvidesPort(void* port,
			 const std::string& name,
			 const std::string& type,
			 const modelT model) = 0;
    virtual void removeProvidesPort(const std::string& name, const modelT model) = 0;
    virtual sci::cca::ComponentID::pointer getComponentID() = 0;
  };    
}

#endif
