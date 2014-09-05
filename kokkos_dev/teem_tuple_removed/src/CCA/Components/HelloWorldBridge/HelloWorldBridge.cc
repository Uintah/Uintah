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
 *  HelloWorldBridge.cc:
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   September 2003
 *
 */

#include <CCA/Components/HelloWorldBridge/HelloWorldBridge.h>
#include <iostream>

using namespace std;
using namespace SCIRun;

extern "C" BridgeComponent* make_Bridge_HelloWorldBridge()
{
  return static_cast<BridgeComponent*>(new HelloWorldBridge());
}


HelloWorldBridge::HelloWorldBridge(){
}

HelloWorldBridge::~HelloWorldBridge(){
}

void HelloWorldBridge::setServices(const BridgeServices* svc){
  ::std::cerr << "Begin HWB::setSVcs\n";
  services=const_cast<BridgeServices*>(svc);
  StringPort::pointer* strport = new StringPort::pointer;
  (*strport) = StringPort::pointer(new StringPort(services));
  ::std::cerr << "MIdway HWB::setSVcs\n";
  services->addProvidesPort((void*)strport,"stringport","sci.cca.ports.StringPort",CCA);
  services->registerUsesPort("idport","gov.cca.ports.IDPort",Babel);
  ::std::cerr << "End HWB::setSVcs\n";
}

std::string StringPort::getString() {
  gov::cca::ports::IDPort s = mysvcs->getBabelPort("idport");
  if(s._is_nil()) return "NIL\n"; 
  return s.getID();
}

