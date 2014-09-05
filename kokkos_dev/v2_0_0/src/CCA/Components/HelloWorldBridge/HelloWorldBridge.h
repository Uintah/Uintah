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
 *  HelloWorldBridge.h
 *
 *  Written by:
 *   Kostadin Damevski 
 *   Department of Computer Science
 *   University of Utah
 *   September 2003
 *
 */

#ifndef SCIRun_CCA_Components_World_h
#define SCIRun_CCA_Components_World_h

#include <Core/CCA/spec/cca_sidl.h>
//#include <CCA/Components/Hello/Hello_sidl.h>

#include <CCA/Components/BabelTest/who/gov_cca_ports_IDPort.hh>

#include <SCIRun/Bridge/BridgeComponent.h>
#include <SCIRun/Bridge/BridgeServices.h>


namespace SCIRun {

  class HelloWorldBridge : public BridgeComponent{
  public:
    HelloWorldBridge();
    virtual ~HelloWorldBridge();
    virtual void setServices(const BridgeServices* svc);
  private:
    HelloWorldBridge(const HelloWorldBridge&);
    HelloWorldBridge& operator=(const HelloWorldBridge&);
    BridgeServices* services;
  };

  class StringPort: public sci::cca::ports::StringPort{
  public:
    StringPort(BridgeServices* svc) : mysvcs(svc) { }
    std::string getString(); 
  private:
    BridgeServices* mysvcs;
  };

} //namepace SCIRun


#endif
