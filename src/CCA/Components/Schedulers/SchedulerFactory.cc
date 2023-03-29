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

#include <sci_defs/gpu_defs.h>

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
    if ((Uintah::Parallel::getNumPartitions() > 0) &&
        (Uintah::Parallel::getThreadsPerPartition() > 0)) {
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
    std::string error = "\nERROR<Scheduler>: "
      "Unified Scheduler needed for '-nthreads <n>' option.\n";
    throw ProblemSetupException(error, __FILE__, __LINE__);
  }

  // "-gpu" provided at command line, but not using "Unified"
  if ((scheduler != "Unified") &&
      (scheduler != "Kokkos" ) && Uintah::Parallel::usingDevice()) {
    std::string error = "\nERROR<Scheduler>: "
      "To use '-gpu' option you must invoke the Kokkos Scheduler or "
      "Unified Scheduler.  Add '-npartitions <n> -nthreadsperpartition <n>' "
      "or '-nthreads <n>' to the sus command line.\n";
    throw ProblemSetupException(error, __FILE__, __LINE__);
  }

  // "Unified" specified in UPS file, but "-nthreads" not given at command line
  if ((scheduler == "Unified") && !(Uintah::Parallel::getNumThreads() > 0)) {
    std::string error = "\nERROR<Scheduler>: "
      "Add '-nthreads <n>' to the sus command line if 'Unified' is specified "
      "in the input file.\n";
    throw ProblemSetupException(error, __FILE__, __LINE__);
  }

  // Output which scheduler will be used
  proc0cout << "Scheduler: \t\t" << scheduler << std::endl;

  // The parallel class is tied closely to the scheduler so print it here.
  Parallel::printManager();

#if defined(HAVE_KOKKOS)
  // Output Execution Space and Memory Space that will be used with Kokkos builds
  proc0cout << "Kokkos HOST Execution Space: \t"
            << Kokkos::DefaultHostExecutionSpace::name() << std::endl;
  proc0cout << "Kokkos HOST Memory Space: \t"
            << Kokkos::DefaultHostExecutionSpace::memory_space::name() << std::endl;
  proc0cout << "Kokkos DEVICE Execution Space: \t"
            << Kokkos::DefaultExecutionSpace::name() << std::endl;
  proc0cout << "Kokkos DEVICE Memory Space: \t"
            << Kokkos::DefaultExecutionSpace::memory_space::name() << std::endl;

  if ( Parallel::usingDevice() ) {
    proc0cout << "Kokkos DEVICE Execution Policy: ";
    switch ( Parallel::getKokkosPolicy() )
    {
    case Parallel::Kokkos_Team_Policy:
      proc0cout << "Team" << std::endl;
      break;
    case Parallel::Kokkos_Range_Policy:
      proc0cout << "Range" << std::endl;
      break;
    case Parallel::Kokkos_MDRange_Policy:
      proc0cout << "MDRange" << std::endl;
      break;
    case Parallel::Kokkos_MDRange_Reverse_Policy:
      proc0cout << "MDRange reversed" << std::endl;
      break;
    default:
      proc0cout << "Unknown" << std::endl;
      break;
    }

    switch ( Parallel::getKokkosPolicy() )
    {
    case Parallel::Kokkos_Team_Policy:
    case Parallel::Kokkos_Range_Policy:
      proc0cout << "Kokkos DEVICE Chunk Size: \t";
      if ( Parallel::getKokkosChunkSize() > 0 )
        proc0cout << Parallel::getKokkosChunkSize() << std::endl;
      else
        proc0cout << "Default" << std::endl;
      break;
    case Parallel::Kokkos_MDRange_Policy:
    case Parallel::Kokkos_MDRange_Reverse_Policy:
      {
        int i, j, k;

        proc0cout << "Kokkos DEVICE tile Size: \t";
        Parallel::getKokkosTileSize(i,j,k);
        if ( i > 0 || j > 0 || k > 0 )
          proc0cout << i << " " << j << " " << k << std::endl;
        else
          proc0cout << "Default" << std::endl;
      }
      break;
    }

    if( Parallel::getKokkosPolicy() == Parallel::Kokkos_Team_Policy)
    {
      int blocksPerLoop = Parallel::getCudaBlocksPerLoop();
      if( blocksPerLoop > 0) {
        std::string plural = blocksPerLoop > 1 ? "s" : "";
        proc0cout << "Kokkos Team Block" << plural
                  << " per loop: \t" << blocksPerLoop <<std::endl;
      }

      int threadsPerBlock = Parallel::getCudaThreadsPerBlock();
      if(threadsPerBlock > 0) {
        std::string plural = threadsPerBlock > 1 ? "s" : "";
        proc0cout << "Kokkos Team Thread" << plural
                  << " per block: \t" << threadsPerBlock << std::endl;
      }
    }
  }

#endif  // #if defined(HAVE_KOKKOS)

  return sch;
}
