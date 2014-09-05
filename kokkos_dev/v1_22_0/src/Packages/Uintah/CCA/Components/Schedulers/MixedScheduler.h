
#ifndef UINTAH_HOMEBREW_MIXEDSCHEDULER_H
#define UINTAH_HOMEBREW_MIXEDSCHEDULER_H

#include <Packages/Uintah/CCA/Components/Schedulers/MPIScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/Relocate.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MessageLog.h>
#include <Packages/Uintah/CCA/Components/Schedulers/ThreadPool.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/TaskProduct.h>
#include <Packages/Uintah/Core/Grid/Task.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <sci_config.h> // For MPIPP_H on SGI
#include <mpi.h>

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
  
   Copyright (C) 2000 SCI Group

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
      
  virtual void problemSetup(const ProblemSpecP& prob_spec);
      
  virtual SchedulerP createSubScheduler();
private:

  MessageLog log;

  virtual void initiateTask( DetailedTask* task,
			     bool only_old_recvs, int abort_point );
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
