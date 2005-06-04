#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerFactory.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SimpleScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SingleProcessorScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MPIScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/NullScheduler.h>

#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <iostream>

using std::cerr;
using std::endl;

using namespace Uintah;

SchedulerCommon* SchedulerFactory::create(ProblemSpecP& ps, 
                                          const ProcessorGroup* world,
                                          Output* output)
{
  SchedulerCommon* sch = 0;
  string scheduler = "";
  
  ps->get("Scheduler",scheduler);

  // Default settings
  if (Uintah::Parallel::usingMPI()) {
    if (scheduler == "")
      scheduler = "MPIScheduler";
    Uintah::Parallel::noThreading();
  }
  else // No MPI
    if (scheduler == "")
      scheduler = "SingleProcessorScheduler";


  if(scheduler == "SingleProcessorScheduler"){
    SingleProcessorScheduler* sched = 
      scinew SingleProcessorScheduler(world, output);
    sch=sched;
  } else if(scheduler == "SimpleScheduler"){
    SimpleScheduler* sched = 
      scinew SimpleScheduler(world, output);
    sch=sched;
  } else if(scheduler == "MPIScheduler"){
    MPIScheduler* sched = 
      scinew MPIScheduler(world, output);
    sch=sched;
  }  else if(scheduler == "NullScheduler"){
    NullScheduler* sched =
      scinew NullScheduler(world, output);
    sch=sched;
  } else {
    sch = 0;   
    cerr << "Unknown scheduler: " + scheduler << endl;
  }
  
  return sch;

}
