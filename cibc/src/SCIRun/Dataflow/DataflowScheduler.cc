/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

namespace SCIRun {

DataflowScheduler::DataflowScheduler(SCIRunFramework* framework,
                                     Scheduler* _sched)
  : InternalFrameworkServiceInstance(framework, "internal:DataflowScheduler"), 
    sched(_sched)
{
}

DataflowScheduler::~DataflowScheduler()
{
}

InternalFrameworkServiceInstance* DataflowScheduler::create(SCIRunFramework* framework)
{
  DataflowScheduler* n = new DataflowScheduler(framework, 
                                               framework->dflow->net->get_scheduler());
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

} // end namespace SCIRun
