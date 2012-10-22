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

#include <CCA/Components/Schedulers/SchedulerFactory.h>
#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Components/Schedulers/SingleProcessorScheduler.h>
#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <CCA/Components/Schedulers/DynamicMPIScheduler.h>
#include <CCA/Components/Schedulers/ThreadedMPIScheduler.h>
#include <CCA/Components/Schedulers/ThreadedMPIScheduler2.h>
#include <CCA/Components/Schedulers/UnifiedScheduler.h>

#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DebugStream.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <sci_defs/cuda_defs.h>

#ifdef HAVE_CUDA
#include <CCA/Components/Schedulers/GPUThreadedMPIScheduler.h>
#endif

#include <iostream>

using namespace std;
using namespace Uintah;

static DebugStream SingleProcessor("SingleProcessor", false);
static DebugStream MPI("MPI", false);
static DebugStream DynamicMPI("DynamicMPI", false);
static DebugStream Threaded("Threaded", false);
static DebugStream Threaded2("Threaded2", false);
static DebugStream GPU("GPU", false);

SchedulerCommon* SchedulerFactory::create(ProblemSpecP& ps,
                                          const ProcessorGroup* world,
                                          Output* output)
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
        } else if (Threaded.active() || Threaded2.active() || GPU.active()) {
          throw ProblemSetupException("Cannot run Threaded Schedulers without -nthreads <num>.", __FILE__, __LINE__);
        } else if (GPU.active() && !(Uintah::Parallel::usingGPU())) {
          throw ProblemSetupException("Cannot use GPU Scheduler without configuring with --enable-cuda.", __FILE__, __LINE__);
        } else if (DynamicMPI.active()) {
          scheduler = "DynamicMPIScheduler";
        } else {
          scheduler = "MPIScheduler";
        }
      } else if (Threaded.active()) {
        scheduler = "ThreadedMPIScheduler";
      } else if (Threaded2.active()) {
        scheduler = "ThreadedMPI2Scheduler";
      }
#ifdef HAVE_CUDA
      else if (GPU.active()) {
        scheduler = "GPUThreadedMPIScheduler";
      }
#endif
      else {
        scheduler = "UnifiedScheduler";
      }
    } else {
      if (SingleProcessor.active()) {
        scheduler = "SingleProcessorScheduler";
      } else {
        scheduler = "UnifiedScheduler";
      }
    }
    if ((Uintah::Parallel::getNumThreads() > 0) && (scheduler != "ThreadedMPIScheduler")
                                                && (scheduler != "ThreadedMPI2Scheduler")
                                                && (scheduler != "GPUThreadedMPIScheduler")
                                                && (scheduler != "UnifiedScheduler")) {
      throw ProblemSetupException("Threaded, GPU or Unified Scheduler needed for -nthreads", __FILE__, __LINE__);
    }
  }

  // Output which scheduler will be used
  if (world->myrank() == 0) {
    cout << "Scheduler: \t\t" << scheduler << endl;
  }

  // Check for specific scheduler request
  if (scheduler == "SingleProcessorScheduler" || scheduler == "SingleProcessor") {
    sch = scinew SingleProcessorScheduler(world, output, NULL);
  } else if (scheduler == "MPIScheduler" || scheduler == "MPI") {
    sch = scinew MPIScheduler(world, output, NULL);
  } else if (scheduler == "DynamicMPIScheduler" || scheduler == "DynamicMPI") {
    sch = scinew DynamicMPIScheduler(world, output, NULL);
  } else if (scheduler == "ThreadedMPIScheduler" || scheduler == "ThreadedMPI") {
    sch = scinew ThreadedMPIScheduler(world, output, NULL);
  } else if (scheduler == "ThreadedMPI2Scheduler" || scheduler == "ThreadedMPI2") {
    sch = scinew ThreadedMPIScheduler2(world, output);
  } else if (scheduler == "UnifiedScheduler" || scheduler == "Unified") {
    sch = scinew UnifiedScheduler(world, output, NULL);
  }
#ifdef HAVE_CUDA
  else if (scheduler == "GPUThreadedMPIScheduler" || scheduler == "GPUThreadedMPI") {
    sch = scinew GPUThreadedMPIScheduler(world, output, NULL);
  }
#endif
  else {
    sch = 0;
    throw ProblemSetupException("Unknown scheduler", __FILE__, __LINE__);
  }

  return sch;

}
