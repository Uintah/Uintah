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
 *  Scheduler Service for Dataflow components
 *
 *  Written by:
 *   Kostadin Damevski
 *   School of Computing
 *   University of Utah
 *   November, 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#ifndef SCIRun_DataflowScheduler_h
#define SCIRun_DataflowScheduler_h

#include <Core/CCA/spec/cca_sidl.h>
#include <Dataflow/Network/Scheduler.h>
#include <SCIRun/Internal/InternalComponentModel.h>
#include <SCIRun/Internal/InternalComponentInstance.h>

namespace SCIRun {
  class SCIRunFramework;
  class DataflowScheduler : public ::sci::cca::ports::DataflowScheduler, public InternalComponentInstance {
  public:
    virtual ~DataflowScheduler();
    static InternalComponentInstance* create(SCIRunFramework* fwk,
					     const std::string& name);
    sci::cca::Port::pointer getService(const std::string&);
    bool toggleOnOffScheduling();
    void do_scheduling();
  private:
    DataflowScheduler(SCIRunFramework* fwk, const std::string& name, Scheduler* _sched);
    Scheduler* sched;
  };
}

#endif
