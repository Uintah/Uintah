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

#include <SCIRun/Dataflow/DataflowScheduler.h>
#include <SCIRun/SCIRunFramework.h>
#include <Dataflow/Network/Network.h>

using namespace SCIRun;

DataflowScheduler::DataflowScheduler(SCIRunFramework* framework,
                                     const std::string& name,
				     Scheduler* _sched)
  : InternalComponentInstance(framework, name, "internal:DataflowScheduler"), 
    sched(_sched)
{
}

DataflowScheduler::~DataflowScheduler()
{
}

InternalComponentInstance* DataflowScheduler::create(SCIRunFramework* framework,
						     const std::string& name)
{
  DataflowScheduler* n = new DataflowScheduler(framework, name, framework->dflow->net->get_scheduler());
  n->addReference();
  return n;
}

sci::cca::Port::pointer DataflowScheduler::getService(const std::string&)
{
  return sci::cca::Port::pointer(this);
}

bool DataflowScheduler::toggleOnOffScheduling() 
{ 
  return sched->toggleOnOffScheduling();
} 

void DataflowScheduler::do_scheduling() 
{ 
  sched->do_scheduling();
}
