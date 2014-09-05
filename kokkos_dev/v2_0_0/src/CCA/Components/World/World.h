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
 *  World.h
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   May 2003
 *
 */

#ifndef SCIRun_CCA_Components_World_h
#define SCIRun_CCA_Components_World_h

#include <Core/CCA/spec/cca_sidl.h>

namespace SCIRun {

  class World : public sci::cca::Component{
    
  public:
    World();
    ~World();
    void setServices(const sci::cca::Services::pointer& svc);
  private:
    World(const World&);
    World& operator=(const World&);
    sci::cca::Services::pointer services;
  };
  
  
  class StringPort: public sci::cca::ports::StringPort{
  public:
    std::string getString(){
      return "World";
    }
  };
  
} //namepace SCIRun


#endif
