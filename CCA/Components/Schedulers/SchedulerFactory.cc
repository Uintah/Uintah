/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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
#include <CCA/Components/Schedulers/UnifiedScheduler.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DebugStream.h>

#include <sci_defs/cuda_defs.h>

#include <iostream>

using namespace Uintah;

// Enable specific schedulers via environment variable
static DebugStream SingleProcessor("SingleProcessor", false);
static DebugStream DynamicMPI(     "DynamicMPI",      false);
static DebugStream ThreadedMPI(    "ThreadedMPI",     false);

SchedulerCommon*
SchedulerFactory::create(const ProblemSpecP&   ps,
                         const ProcessorGroup* world,
                         const Output*         output)
{
  SchedulerCommon* sch = 0;
  std::string scheduler = "";

  ProblemSpecP sc_ps = ps->findBlock("Scheduler");
  if (sc_ps) {
    sc_ps->getAttribute("type", scheduler);
  }

  // Default settings
  if (scheduler == "") {
    if (Uintah::Parallel::usingMPI()) {
      if (!(Uintah::Parallel::getNumThreads() > 0)) {
        if (SingleProcessor.active()) {
          throw ProblemSetupException("Cannot use Single Processor Scheduler with MPI.", __FILE__, __LINE__);
        }
        else if (DynamicMPI.active()) {
          scheduler = "DynamicMPI";
        }
        else {
          scheduler = "MPI";
        }
      }
      else if (ThreadedMPI.active()) {
        scheduler = "ThreadedMPI";
      }
      else {
        scheduler = "Unified";
      }
    }
    else {
      if (SingleProcessor.active()) {
        scheduler = "SingleProcessor";
      }
      else {
        // Defaults are: MPI (w/o threads) and Unified (with threads)
        scheduler = "Unified";
      }
    }
  }

  // Check for specific scheduler request from the input file
  if (scheduler == "SingleProcessor") {
    sch = scinew SingleProcessorScheduler(world, output, NULL);
  }
  else if (scheduler == "MPI") {
    sch = scinew MPIScheduler(world, output, NULL);
  }
  else if (scheduler == "DynamicMPI") {
    sch = scinew DynamicMPIScheduler(world, output, NULL);
  }
  else if (scheduler == "ThreadedMPI") {
    sch = scinew ThreadedMPIScheduler(world, output, NULL);
  }
  else if (scheduler == "Unified") {
    sch = scinew UnifiedScheduler(world, output, NULL);
  }
  else {
    sch = 0;
    std::string error = "Unknown scheduler: '" + scheduler + "' Please check UPS Spec for valid scheduler options";
    throw ProblemSetupException(error, __FILE__, __LINE__);
  }

  // "-nthreads" at command line, something other than "Unified" specified in UPS file
  if ((Uintah::Parallel::getNumThreads() > 0) && ((scheduler != "Unified") && (scheduler != "ThreadedMPI"))) {
    throw ProblemSetupException("ThreadedMPI or Unified Scheduler needed for '-nthreads <n>' option", __FILE__, __LINE__);
  }

  // Output which scheduler will be used
  if (world->myrank() == 0) {
    std::cout << "Scheduler: \t\t" << scheduler << std::endl;
  }

  return sch;

}
