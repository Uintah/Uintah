/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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


#include <CCA/Components/Schedulers/SingleProcessorScheduler.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/TaskGraph.h>
#include <CCA/Ports/LoadBalancer.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>

using namespace Uintah;
using namespace SCIRun;

extern DebugStream taskdbg;
extern DebugStream taskLevel_dbg;

static DebugStream dbg("SingleProcessorScheduler", false);

SingleProcessorScheduler::SingleProcessorScheduler( const ProcessorGroup           * myworld,
                                                    const Output                   * oport,
                                                          SingleProcessorScheduler * parent )
  : SchedulerCommon(myworld, oport)
{
  d_generation = 0;
  m_parent     = parent;
}

SingleProcessorScheduler::~SingleProcessorScheduler()
{
}

SchedulerP
SingleProcessorScheduler::createSubScheduler()
{
  SingleProcessorScheduler* subsched = scinew SingleProcessorScheduler(d_myworld, m_outPort, this);
  UintahParallelPort* lbp = getPort("load balancer");
  subsched->attachPort("load balancer", lbp);
  subsched->d_sharedState = d_sharedState;
  return subsched;
}

void
SingleProcessorScheduler::verifyChecksum()
{
  // Not used in SingleProcessorScheduler
}


void
SingleProcessorScheduler::execute( int tgnum     /*=0*/,
                                   int iteration /*=0*/ )
{
  ASSERTRANGE(tgnum, 0, (int)graphs.size());
  TaskGraph* tg = graphs[tgnum];
  tg->setIteration(iteration);
  currentTG_ = tgnum;
  DetailedTasks* dts = tg->getDetailedTasks();

  if (graphs.size() > 1) {
    // tg model is the multi TG model, where each graph is going to need to
    // have its dwmap reset here (even with the same tgnum)
    tg->remapTaskDWs(dwmap);
  }

  if (dts == 0) {
    std::cerr << "SingleProcessorScheduler skipping execute, no tasks\n";
    return;
  }

  int ntasks = dts->numTasks();
  if (ntasks == 0) {
    std::cerr << "WARNING: Scheduler executed, but no tasks\n";
  }

  ASSERT(dws.size()>=2);
  std::vector<DataWarehouseP> plain_old_dws(dws.size());
  for (unsigned int i = 0; i < dws.size(); i++) {
    plain_old_dws[i] = dws[i].get_rep();
  }

  if (dbg.active()) {
    dbg << "Executing " << ntasks << " tasks, ";
    for (int i = 0; i < numOldDWs; i++) {
      dbg << "from DWs: ";
      if (dws[i]) {
        dbg << dws[i]->getID() << ", ";
      }
      else {
        dbg << "Null, ";
      }
    }
    if (dws.size() - numOldDWs > 1) {
      dbg << "intermediate DWs: ";
      for (unsigned int i = numOldDWs; i < dws.size() - 1; i++) {
        dbg << dws[i]->getID() << ", ";
      }
    }
    if (dws[dws.size() - 1]) {
      dbg << " to DW: " << dws[dws.size() - 1]->getID();
    }
    else {
      dbg << " to DW: Null";
    }
    dbg << "\n";
  }

  makeTaskGraphDoc(dts);

  dts->initializeScrubs(dws, dwmap);

  for (int i = 0; i < ntasks; i++) {
    double start = Time::currentSeconds();
    DetailedTask* task = dts->getTask(i);

    taskdbg << d_myworld->myrank() << " SPS: Initiating: ";
    printTask(taskdbg, task);
    taskdbg << '\n';

    if (trackingVarsPrintLocation_ & SchedulerCommon::PRINT_BEFORE_EXEC) {
      printTrackedVars(task, SchedulerCommon::PRINT_BEFORE_EXEC);
    }

    task->doit(d_myworld, dws, plain_old_dws);

    if (trackingVarsPrintLocation_ & SchedulerCommon::PRINT_AFTER_EXEC) {
      printTrackedVars(task, SchedulerCommon::PRINT_AFTER_EXEC);
    }

    task->done(dws);

    if (taskdbg.active()) {
      taskdbg << d_myworld->myrank() << " SPS: Completed:  ";
      printTask(taskdbg, task);
      taskdbg << '\n';
      printTaskLevels(d_myworld, taskLevel_dbg, task);
    }

    double delT = Time::currentSeconds() - start;
    if (dws[dws.size() - 1] && dws[dws.size() - 1]->timestepAborted()) {
      dbg << "Aborting timestep after task: " << *task->getTask() << '\n';
      break;
    }

    if (dbg.active()) {
      dbg << "Completed task: " << *task->getTask() << " (" << delT << " seconds)\n";
    }

    emitNode(task, start, delT, delT);
  }
  finalizeTimestep();
}
