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


#include <CCA/Components/Schedulers/TaskWorker.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/TaskGraph.h>

#include <Core/Exceptions/ProblemSetupException.h>

#include <CCA/Ports/Output.h>

#include <Core/Thread/Time.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>

#include <cstring>

#define USE_PACKING

using namespace std;
using namespace Uintah;
using namespace SCIRun;

#undef UINTAHSHARE
#if defined(_WIN32) && !defined(BUILD_UINTAH_STATIC)
#define UINTAHSHARE __declspec(dllimport)
#else
#define UINTAHSHARE
#endif

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern UINTAHSHARE SCIRun::Mutex cerrLock;
extern DebugStream taskdbg;

static DebugStream threaddbg("ThreadDBG",false);


TaskWorker::TaskWorker(ThreadedMPIScheduler* scheduler, int id) :
   d_id(id), d_scheduler(scheduler), d_schedulergpu(NULL), d_task(NULL), d_iteration(0),
   d_runmutex("run mutex"),  d_runsignal("run condition"), d_quit(false),
   d_waittime(0.0), d_waitstart(0.0), d_rank(scheduler->getProcessorGroup()->myrank())
{
  d_runmutex.lock();
}

TaskWorker::TaskWorker(GPUThreadedMPIScheduler* scheduler, int id) :
   d_id(id), d_scheduler(NULL), d_schedulergpu(scheduler), d_task(NULL), d_iteration(0),
   d_runmutex("run mutex"),  d_runsignal("run condition"), d_quit(false),
   d_waittime(0.0), d_waitstart(0.0), d_rank(scheduler->getProcessorGroup()->myrank())
{
  d_runmutex.lock();
}

TaskWorker::~TaskWorker()
{
}

void TaskWorker::run()
{
//  WAIT_FOR_DEBUGGER();
  threaddbg << "Binding thread id " << d_id+1 << " to cpu " << d_id+1 << endl;
  bool useGPU = Uintah::Parallel::usingGPU() && d_schedulergpu;

  Thread::self()->set_myid(d_id+1);
  Thread::self()->set_affinity(d_id+1);

  while(true) {
    //wait for main thread signal
    d_runsignal.wait(d_runmutex);
    d_runmutex.unlock();
    d_waittime += Time::currentSeconds()-d_waitstart;

    if (d_quit) {
      if(taskdbg.active()) {
        cerrLock.lock();
        taskdbg << "Worker " << d_rank  << "-" << d_id << " quitting   " << "\n";
        cerrLock.unlock();
      }
      return;
    }

    if(taskdbg.active()) {
      cerrLock.lock();
      taskdbg << "Worker " << d_rank  << "-" << d_id << ": executeTask:   " << *d_task << "\n";
      cerrLock.unlock();
    }
    ASSERT(d_task!=NULL);

    try {
      if (d_task->getTask()->getType() == Task::Reduction) {
        if (useGPU) {
          d_schedulergpu->initiateReduction(d_task);
        } else {
          d_scheduler->initiateReduction(d_task);
        }
      } else {
        if (useGPU) {
          d_schedulergpu->runTask(d_task, d_iteration, d_id);
        } else {
            d_scheduler->runTask(d_task, d_iteration, d_id);
        }
      }
    } catch (Exception& e) {
      cerrLock.lock();
      cerr << "Worker " << d_rank << "-" << d_id << ": Caught exception: " << e.message() << "\n";
      if(e.stackTrace()) {
        cerr << "Stack trace: " << e.stackTrace() << '\n';
      }
      cerrLock.unlock();
    }

    if(taskdbg.active()) {
      cerrLock.lock();
      taskdbg << "Worker " << d_rank << "-" << d_id << ": finishTask:   " << *d_task << "\n";
      cerrLock.unlock();
    }

    //signal main thread for next task
    if (useGPU) {
      d_schedulergpu->d_nextmutex.lock();
    } else {
      d_scheduler->d_nextmutex.lock();
    }

    d_runmutex.lock();
    d_task = NULL;
    d_iteration = 0;
    d_waitstart = Time::currentSeconds();

    if (useGPU) {
      d_schedulergpu->d_nextsignal.conditionSignal();
      d_schedulergpu->d_nextmutex.unlock();
    } else {
        d_scheduler->d_nextsignal.conditionSignal();
        d_scheduler->d_nextmutex.unlock();
    }
  }
}

double TaskWorker::getWaittime()
{
    return  d_waittime;
}

void TaskWorker::resetWaittime(double start)
{
    d_waitstart  = start;
    d_waittime = 0.0;
}

