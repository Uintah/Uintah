/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

#include <CCA/Components/Schedulers/SchedulerFactory.h>
#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <CCA/Components/Schedulers/DynamicMPIScheduler.h>
#include <CCA/Components/Schedulers/KokkosScheduler.h>
#include <CCA/Components/Schedulers/KokkosOpenMPScheduler.h>
#include <CCA/Components/Schedulers/UnifiedScheduler.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <sci_defs/cuda_defs.h>
#include <sci_defs/kokkos_defs.h>

#include <iostream>
#include <string>

using namespace Uintah;


SchedulerCommon*
SchedulerFactory::create( const ProblemSpecP   & ps
                        , const ProcessorGroup * world
                        )
{
  SchedulerCommon* sch  = nullptr;
  std::string scheduler = "";

  ProblemSpecP sc_ps = ps->findBlock("Scheduler");
  if (sc_ps) {
    sc_ps->getAttribute("type", scheduler);
  }

  /////////////////////////////////////////////////////////////////////
  // Default settings - nothing specified in the input file
  if (scheduler == "") {
    if ((Uintah::Parallel::getNumPartitions() > 0) && (Uintah::Parallel::getThreadsPerPartition() > 0)) {
      if (Uintah::Parallel::usingDevice()) {
        scheduler = "Kokkos";       // User passed '-npartitions <#> -nthreadsperpartition <#> -gpu'
      }
      else {
        scheduler = "KokkosOpenMP"; // User passed '-npartitions <#> -nthreadsperpartition <#>'
      }
    }
    else if (Uintah::Parallel::getNumThreads() > 0) {
      scheduler = "Unified";        // User passed '-nthreads <#>'
    }
    else {
      scheduler = "MPI";            // User passed no scheduler-specific run-time parameters
    }
  }

  /////////////////////////////////////////////////////////////////////
  // Check for specific scheduler request from the input file

  if (scheduler == "MPI") {
    sch = scinew MPIScheduler(world, nullptr);
    Parallel::setCpuThreadEnvironment(Parallel::CpuThreadEnvironment::PTHREADS);
  }

  else if (scheduler == "DynamicMPI") {
    sch = scinew DynamicMPIScheduler(world, nullptr);
    Parallel::setCpuThreadEnvironment(Parallel::CpuThreadEnvironment::PTHREADS);
  }

  else if (scheduler == "Unified") {
    sch = scinew UnifiedScheduler(world, nullptr);
    Parallel::setCpuThreadEnvironment(Parallel::CpuThreadEnvironment::PTHREADS);
  }

  else if (scheduler == "Kokkos") {
    sch = scinew KokkosScheduler(world, nullptr);
    Parallel::setCpuThreadEnvironment(Parallel::CpuThreadEnvironment::OPEN_MP_THREADS);
  }

  else if (scheduler == "KokkosOpenMP") {
    sch = scinew KokkosOpenMPScheduler(world, nullptr);
    Parallel::setCpuThreadEnvironment(Parallel::CpuThreadEnvironment::OPEN_MP_THREADS);
  }

  else {
    sch = nullptr;
    std::string error = "\nERROR<Scheduler>: Unknown scheduler: '" + scheduler
                        + "' Please check UPS Spec for valid scheduler options (.../src/StandAlone/inputs/UPS_SPEC/ups_spec.xml)'.\n";
    throw ProblemSetupException(error, __FILE__, __LINE__);
  }

  //__________________________________
  //  bulletproofing

  // "-nthreads" at command line, something other than "Unified" specified in UPS file (w/ -do_not_validate)
  if ((Uintah::Parallel::getNumThreads() > 0) && (scheduler != "Unified")) {
    throw ProblemSetupException("\nERROR<Scheduler>: Unified Scheduler needed for '-nthreads <n>' option.\n", __FILE__, __LINE__);
  }

  // "-gpu" provided at command line, but not using "Unified"
  if ((scheduler != "Unified") && (scheduler != "Kokkos") && Uintah::Parallel::usingDevice()) {
    std::string error = "\nERROR<Scheduler>: To use '-gpu' option you must invoke the Kokkos Scheduler or Unified Scheduler.  Add  '-npartitions <n> -nthreadsperpartition <n>' or '-nthreads <n>' to the sus command line.\n";
    throw ProblemSetupException(error, __FILE__, __LINE__);
  }

  // "Unified" specified in UPS file, but "-nthreads" not given at command line
  if ((scheduler == "Unified") && !(Uintah::Parallel::getNumThreads() > 0)) {
    std::string error = "\nERROR<Scheduler>: Add '-nthreads <n>' to the sus command line if you are specifying Unified in your input file.\n";
    throw ProblemSetupException(error, __FILE__, __LINE__);
  }

  // Output which scheduler will be used
  proc0cout << "Scheduler: \t\t" << scheduler << std::endl;

  return sch;

}
