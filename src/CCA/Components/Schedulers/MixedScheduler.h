/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef UINTAH_HOMEBREW_MIXEDSCHEDULER_H
#define UINTAH_HOMEBREW_MIXEDSCHEDULER_H

#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <CCA/Components/Schedulers/Relocate.h>
#include <CCA/Components/Schedulers/MessageLog.h>
#include <CCA/Components/Schedulers/ThreadPool.h>
#include <CCA/Ports/DataWarehouseP.h>
 
#include <Core/Grid/Task.h>

#include <vector>
#include <sci_defs/mpi_defs.h> // For MPIPP_H on SGI

namespace Uintah {
using std::vector;

  class OnDemandDataWarehouse;
  class Task;

/**************************************

CLASS
   MixedScheduler
   
   Implements a mixed MPI/Threads version of the scheduler.

GENERAL INFORMATION

   MixedScheduler.h

   J. Davison de St. Germain
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   Scheduler MPI Thread

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class MixedScheduler : public MPIScheduler {

public:

  MixedScheduler(const ProcessorGroup* myworld, Output* oport);
  virtual ~MixedScheduler();
      
  virtual void problemSetup( const ProblemSpecP& prob_spec,
                             SimulationStateP& /* state */ );
      
  virtual SchedulerP createSubScheduler();
private:

  MessageLog log;

  virtual void initiateTask( DetailedTask* task,
			     bool only_old_recvs, int abort_point, int /* iteration */ );

  virtual void initiateReduction( DetailedTask* task );  

  // Waits until all tasks have finished. (Ie: talks to the ThreadPool
  // and waits until the threadpool in empty (ie: all tasks done.))
  virtual void wait_till_all_done();

  virtual bool useInternalDeps();

  MixedScheduler(const MixedScheduler&);
  MixedScheduler& operator=(const MixedScheduler&);

  ThreadPool * d_threadPool;
};

} // End namespace Uintah
   
#endif
