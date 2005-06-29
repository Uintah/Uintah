#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerFactory.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SimpleScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SingleProcessorScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MPIScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/NullScheduler.h>

#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
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
  
  ProblemSpecP sc_ps = ps->findBlock("Scheduler");
  if (sc_ps)
    sc_ps->get("type",scheduler);

  // Default settings
  if (Uintah::Parallel::usingMPI()) {
    if (scheduler == "")
      scheduler = "MPIScheduler";
    Uintah::Parallel::noThreading();

  }
  else {// No MPI
    if (scheduler == "")
      scheduler = "SingleProcessorScheduler";
  }

  if (world->myrank() == 0)
    cout << "Using scheduler " << scheduler << endl;

  if(scheduler == "SingleProcessorScheduler"){
    sch = scinew SingleProcessorScheduler(world, output);
  } else if(scheduler == "SimpleScheduler"){
    sch = scinew SimpleScheduler(world, output);
  } else if(scheduler == "MPIScheduler"){
    sch = scinew MPIScheduler(world, output);
  } else if(scheduler == "NullScheduler"){
    sch = scinew NullScheduler(world, output);
  } else {
    sch = 0;   
    throw ProblemSetupException("Unknown scheduler");
  }
  
  return sch;

}
