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
 *  AutoBridge.h: Interface from framework to automatic bridge gen tools 
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   February 2004
 *
 */

#ifndef SCIRun_Bridge_AutoBridge_h
#define SCIRun_Bridge_AutoBridge_h

#include <SCIRun/PortInstance.h>

namespace SCIRun {
  class AutoBridge {
  public:
    AutoBridge(/*BuilderService* bsvc*/) {} 
    virtual ~AutoBridge() {}
    std::string execute(std::string cFrom, std::string cTo);
    bool canBridge(PortInstance* pr1, PortInstance* pr2);
  private:
  };
}

#endif
