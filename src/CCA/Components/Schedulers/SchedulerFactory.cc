/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <CCA/Components/Schedulers/SchedulerFactory.h>
#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Components/Schedulers/SingleProcessorScheduler.h>
#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <CCA/Components/Schedulers/DynamicMPIScheduler.h>
#include <CCA/Components/Schedulers/ThreadedMPIScheduler.h>
#include <CCA/Components/Schedulers/GPUThreadedMPIScheduler.h>

#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <iostream>

using namespace std;
using namespace Uintah;

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
  if (Uintah::Parallel::usingMPI()) {
    if (scheduler == "") {
      if (Uintah::Parallel::getMaxThreads() > 0) {
        scheduler = (Uintah::Parallel::usingGPU() ? "GPUThreadedMPI" : "ThreadedMPI");
      } else {
        scheduler = "MPIScheduler";
      }
    }
  } else {  // No MPI
    if (scheduler == "") {
      scheduler = "SingleProcessorScheduler";
    }
  }

  if ((Uintah::Parallel::getMaxThreads() > 0) && ((scheduler != "ThreadedMPI") && (scheduler != "GPUThreadedMPI"))) {
    throw ProblemSetupException("Threaded or GPU Threaded Scheduler needed for -nthreads", __FILE__, __LINE__);
  }

  if (world->myrank() == 0) {
    cout << "Scheduler: \t\t" << scheduler << endl;
  }

  if (scheduler == "SingleProcessorScheduler") {
    sch = scinew SingleProcessorScheduler(world, output, NULL);
  } else if (scheduler == "MPIScheduler" || scheduler == "MPI") {
    sch = scinew MPIScheduler(world, output, NULL);
  } else if (scheduler == "DynamicMPI") {
    sch = scinew DynamicMPIScheduler(world, output, NULL);
  } else if (scheduler == "ThreadedMPI") {
    sch = scinew ThreadedMPIScheduler(world, output, NULL);
  } else if (scheduler == "GPUThreadedMPI") {
    sch = scinew GPUThreadedMPIScheduler(world, output, NULL);
  } else {
    sch = 0;
    throw ProblemSetupException("Unknown scheduler", __FILE__, __LINE__);
  }

  return sch;

}
