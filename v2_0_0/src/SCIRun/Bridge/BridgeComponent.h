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
 *  BridgeComponent.h:
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   September 2003
 *
 */

#ifndef SCIRUN_BRIDGE_BRIDGECOMPONENT
#define SCIRUN_BRIDGE_BRIDGECOMPONENT

#include <SCIRun/Bridge/BridgeServices.h>

namespace SCIRun {
  
  class BridgeComponent {
  public:
    BridgeComponent::BridgeComponent() { }
    virtual BridgeComponent::~BridgeComponent() { }
    virtual void setServices(const BridgeServices* svc) { }
  };    

}

#endif
