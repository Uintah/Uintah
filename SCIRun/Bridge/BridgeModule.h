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
 *  BridgeModule.h: 
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   September 2003
 *
 */

#ifndef SCIRun_Framework_BridgeModule_h
#define SCIRun_Framework_BridgeModule_h

#include <Dataflow/Network/Module.h>
#include <SCIRun/Dataflow/SCIRunComponentModel.h>

namespace SCIRun {
  class BridgeModule : public Module {
  public:
    BridgeModule() : Module("BridgeModule", SCIRunComponentModel::gui->createContext("BridgeModule"), Filter) {}
    virtual ~BridgeModule() {}
    virtual void execute() {}
  private:
  };
}

#endif
