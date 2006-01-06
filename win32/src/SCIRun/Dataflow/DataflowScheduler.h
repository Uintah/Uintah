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

#ifndef SCIRun_DataflowScheduler_h
#define SCIRun_DataflowScheduler_h

#include <Core/CCA/spec/cca_sidl.h>
#include <Dataflow/Network/Scheduler.h>
#include <SCIRun/Internal/InternalComponentModel.h>
#include <SCIRun/Internal/InternalFrameworkServiceInstance.h>

namespace SCIRun {

class SCIRunFramework;
/**
 * \class DataflowScheduler
 *
 * A scheduler service for the SCIRun Dataflow components.  An implementation
 * of the sci::cca::ports::DataflowScheduler interface.  This class is a
 * wrapper around the SCIRun(1) Scheduler object.
 */
class DataflowScheduler : public ::sci::cca::ports::DataflowScheduler,
                          public InternalFrameworkServiceInstance
{
public:
  virtual ~DataflowScheduler();

  /** Object factory method for allocating DataflowScheduler instances. Returns
      a smart pointer to the newly allocated object. */
  static InternalFrameworkServiceInstance* create(SCIRunFramework* fwk);

  /** Returns a pointer to this DataflowSchedular port. */
  sci::cca::Port::pointer getService(const std::string&);

  /** Turns the scheduler on and off.  Returns \code true is scheduler is on
      and \code false if scheduler is now off. */
  bool toggleOnOffScheduling();

  /** Trigger analysis of a network to build the cue for module execution.*/
  void do_scheduling();
  
private:
  DataflowScheduler(SCIRunFramework* fwk, Scheduler* _sched);
  Scheduler* sched;
};

} // end namespace SCIRun

#endif
