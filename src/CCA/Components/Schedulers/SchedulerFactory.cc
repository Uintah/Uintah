/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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
#include <CCA/Components/Schedulers/SingleProcessorScheduler.h>
#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <CCA/Components/Schedulers/DynamicMPIScheduler.h>
#include <CCA/Components/Schedulers/ThreadedMPIScheduler.h>
#include <CCA/Components/Schedulers/ThreadedTaskScheduler.h>
#include <CCA/Components/Schedulers/UnifiedScheduler.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DebugStream.h>

#include <sci_defs/cuda_defs.h>

#include <iostream>
#include <string>

using namespace Uintah;

// Enable specific schedulers via environment variable
namespace {

DebugStream singleProcessor("SingleProcessorScheduler", false);
DebugStream dynamicMPI(     "DynamicMPIScheduler"     , false);
DebugStream thread_funneled("ThreadedTaskScheduler"   , false);
DebugStream unified(        "UnifiedScheduler"        , false);

}

SchedulerCommon*
SchedulerFactory::create( const ProblemSpecP   & ps,
                          const ProcessorGroup * world,
                          const Output         * output )
{
  SchedulerCommon* sch = 0;
  std::string scheduler = "";

  ProblemSpecP sc_ps = ps->findBlock("Scheduler");
  if (sc_ps) {
    sc_ps->getAttribute("type", scheduler);
  }

  // Default settings - nothing specified in the input file
  if (scheduler == "") {

    // Using MPI
    if (Uintah::Parallel::usingMPI()) {

      // Using MPI without threads
      if (!(Uintah::Parallel::getNumThreads() > 0)) {
        if (singleProcessor.active()) {
          throw ProblemSetupException("Cannot use Single Processor Scheduler with MPI.", __FILE__, __LINE__);
        }
        else if (dynamicMPI.active()) {
          scheduler = "DynamicMPI";
        }
        else {
          scheduler = "MPI";
        }
      }

      // Using MPI and threads (Unified and ThreadedMPI schedulers)
      else if (Uintah::Parallel::usingDevice() || unified.active()) {
        scheduler = "Unified";
      }
      else if (thread_funneled.active()) {
        scheduler = "ThreadedTask";
      }
      else {
        scheduler = "ThreadedMPI";
      }
    }

    // No MPI
    else if (Uintah::Parallel::getNumThreads() > 0) {

      // Threads + GPU
      if (Uintah::Parallel::usingDevice()) {
        scheduler = "Unified";
      }

      // Threads
      else {
        scheduler = "ThreadedMPI";
      }
    }

    // Throw an exception of all the above fails
    else if (dynamicMPI.active() || unified.active()) {
      std::string message =
          "Cannot use MPI schedulers without -mpi option. SCI_DEBUG flags: DynamicMPI or Unified may also be active.";
      throw ProblemSetupException(message, __FILE__, __LINE__);
    }

    // Otherwise use the SingleProcessorScheduler
    else {
      scheduler = "SingleProcessor";
    }
  }

  // Check for specific scheduler request from the input file
  if (scheduler == "SingleProcessor") {
    sch = scinew SingleProcessorScheduler(world, output, nullptr);
  }
  else if (scheduler == "MPI") {
    sch = scinew MPIScheduler(world, output, nullptr);
  }
  else if (scheduler == "DynamicMPI") {
    sch = scinew DynamicMPIScheduler(world, output, nullptr);
  }
  else if (scheduler == "ThreadedMPI") {
    sch = scinew ThreadedMPIScheduler(world, output, nullptr);
  }
  else if (scheduler == "ThreadedTask") {
    sch = scinew ThreadedTaskScheduler(world, output);
  }
  else if (scheduler == "Unified") {
    sch = scinew UnifiedScheduler(world, output, nullptr);
  }
  else {
    sch = 0;
    std::string error = "Unknown scheduler: '" + scheduler
                        + "' Please check UPS Spec for valid scheduler options (.../src/StandAlone/inputs/UPS_SPEC/ups_spec.xml)'";
    throw ProblemSetupException(error, __FILE__, __LINE__);
  }

  //__________________________________
  //  bulletproofing
  // "-nthreads" at command line, something other than "ThreadedMPI" specified in UPS file (w/ -do_not_validate)
  if ((Uintah::Parallel::getNumThreads() > 0) && ((scheduler != "Unified") && (scheduler != "ThreadedMPI") && (scheduler != "ThreadedTask"))) {
    throw ProblemSetupException("ThreadedMPI or Unified Scheduler needed for '-nthreads <n>' option", __FILE__, __LINE__);
  }

  if ( (scheduler != "Unified") && Uintah::Parallel::usingDevice() ){
    std::string error = "\n \tTo use '-gpu' option you must invoke the Unified Scheduler.  Add '-nthreads <n>' to the sus command line.";
    throw ProblemSetupException( error , __FILE__, __LINE__);
  }

  // Output which scheduler will be used
  proc0cout << "Scheduler: \t\t" << scheduler << std::endl;

  return sch;

}
