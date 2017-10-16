/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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
#include <CCA/Components/Schedulers/KokkosOpenMPScheduler.h>
#include <CCA/Components/Schedulers/UnifiedScheduler.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <sci_defs/cuda_defs.h>

#include <iostream>
#include <string>

using namespace Uintah;


SchedulerCommon*
SchedulerFactory::create( const ProblemSpecP   & ps
                        , const ProcessorGroup * world
                        , const Output         * output
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

#ifdef UINTAH_ENABLE_KOKKOS

    scheduler = "KokkosOpenMP";

#else

    if (Uintah::Parallel::getNumThreads() > 0) {
      scheduler = "Unified";
    }

    else {
      scheduler = "MPI";
    }

#endif

  }

  /////////////////////////////////////////////////////////////////////
  // Check for specific scheduler request from the input file

  if (scheduler == "MPI") {
    sch = scinew MPIScheduler(world, output, nullptr);
  }

  else if (scheduler == "DynamicMPI") {
    sch = scinew DynamicMPIScheduler(world, output, nullptr);
  }

#ifdef UINTAH_ENABLE_KOKKOS

  else if (scheduler == "KokkosOpenMP") {
    sch = scinew KokkosOpenMPScheduler(world, output, nullptr);
  }

  else if (scheduler == "Unified") {
    std::string error = "\n \tTo use the Unified Scheduler, you must disable Kokkos. Build Uintah without Kokkos.";
    throw ProblemSetupException(error, __FILE__, __LINE__);
  }

#else

  else if (scheduler == "Unified") {
    sch = scinew UnifiedScheduler(world, output, nullptr);
  }

#endif

  else {
    sch = nullptr;
    std::string error = "Unknown scheduler: '" + scheduler
                        + "' Please check UPS Spec for valid scheduler options (.../src/StandAlone/inputs/UPS_SPEC/ups_spec.xml)'";
    throw ProblemSetupException(error, __FILE__, __LINE__);
  }

  //__________________________________
  //  bulletproofing

#ifdef UINTAH_ENABLE_KOKKOS

  // Kokkos parallel patterns are enabled, but trying to use Unified Scheduler from command line with "-nthreads"
  if ( (scheduler == "Unified") || (Uintah::Parallel::getNumThreads() > 0) || (Uintah::Parallel::usingDevice()) ) {
    std::string error = "\n \tTo use the Unified Scheduler, '-nthreads <n>' OR '-gpu', you must disable Kokkos. Build Uintah without Kokkos.";
    throw ProblemSetupException(error, __FILE__, __LINE__);
  }

#else

  // "-nthreads" at command line, something other than "Unified" specified in UPS file (w/ -do_not_validate)
  if ((Uintah::Parallel::getNumThreads() > 0) && (scheduler != "Unified")) {
    throw ProblemSetupException("Unified Scheduler needed for '-nthreads <n>' option", __FILE__, __LINE__);
  }


  // "-gpu" provided at command line, but not using "Unified"
  if ((scheduler != "Unified") && Uintah::Parallel::usingDevice()) {
    std::string error = "\n \tTo use '-gpu' option you must invoke the Unified Scheduler.  Add '-nthreads <n>' to the sus command line.";
    throw ProblemSetupException(error, __FILE__, __LINE__);
  }

  // "Unified" specified in UPS file, but "-nthreads" not given at command line
  if ((scheduler == "Unified") && !(Uintah::Parallel::getNumThreads() > 0)) {
    std::string error = "\n \tAdd '-nthreads <n>' to the sus command line if you are specifying Unified in your input file.";
    throw ProblemSetupException(error, __FILE__, __LINE__);
  }

#endif

  // Output which scheduler will be used
  proc0cout << "Scheduler: \t\t" << scheduler << std::endl;

  return sch;

}
