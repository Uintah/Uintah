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
#include <CCA/Components/Schedulers/UnifiedScheduler.h>

#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DebugStream.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <sci_defs/cuda_defs.h>

#include <iostream>

using namespace std;
using namespace Uintah;

static DebugStream SingleProcessor("SingleProcessor", false);
static DebugStream MPI("MPI", false);
static DebugStream DynamicMPI("DynamicMPI", false);

SchedulerCommon*
SchedulerFactory::create( const ProblemSpecP   & ps,
                          const ProcessorGroup * world,
                          const Output         * output )
{
  SchedulerCommon* sch = 0;
  string scheduler = "";

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
          scheduler = "DynamicMPIScheduler";
        }
        else {
          scheduler = "MPIScheduler";
        }
      }
      else {
        scheduler = "UnifiedScheduler";
      }
    }
    else {
      if (SingleProcessor.active()) {
        scheduler = "SingleProcessorScheduler";
      }
      else {
        // Unified Scheduler without threads or MPI (Single-Processor mode)
        scheduler = "UnifiedScheduler";
      }
    }
  }

  // Output which scheduler will be used
  if (world->myrank() == 0) {
    cout << "Scheduler: \t\t" << scheduler << endl;
  }

  // Check for specific scheduler request
  if (scheduler == "SingleProcessorScheduler" || scheduler == "SingleProcessor") {
    sch = scinew SingleProcessorScheduler(world, output, NULL);
  }
  else if (scheduler == "MPIScheduler" || scheduler == "MPI") {
    sch = scinew MPIScheduler(world, output, NULL);
  }
  else if (scheduler == "DynamicMPIScheduler" || scheduler == "DynamicMPI") {
    sch = scinew DynamicMPIScheduler(world, output, NULL);
  }
  else if (scheduler == "UnifiedScheduler" || scheduler == "Unified") {
    sch = scinew UnifiedScheduler(world, output, NULL);
  }
  else {
    sch = 0;
    string error = "Unknown scheduler: '" + scheduler + "'";
    throw ProblemSetupException("Unknown scheduler", __FILE__, __LINE__);
  }

  if ((Uintah::Parallel::getNumThreads() > 0) && (scheduler != "UnifiedScheduler")) {
    throw ProblemSetupException("Unified Scheduler needed for -nthreads", __FILE__, __LINE__);
  }

  return sch;

}
