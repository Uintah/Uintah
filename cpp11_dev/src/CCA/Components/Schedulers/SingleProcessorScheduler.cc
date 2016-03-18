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

extern DebugStream taskdbg;
extern DebugStream taskLevel_dbg;

static DebugStream dbg("SingleProcessorScheduler", false);

SingleProcessorScheduler::SingleProcessorScheduler( const ProcessorGroup           * myworld,
                                                    const Output                   * oport,
                                                          SingleProcessorScheduler * parent )
  : SchedulerCommon(myworld, oport)
{
  m_generation = 0;
  m_parent     = parent;
}

SingleProcessorScheduler::~SingleProcessorScheduler()
{
}

SchedulerP
SingleProcessorScheduler::createSubScheduler()
{
  SingleProcessorScheduler * subsched = scinew SingleProcessorScheduler( d_myworld, m_output_port, this );
  UintahParallelPort       * lbp      = getPort("load balancer");
  subsched->attachPort( "load balancer", lbp );
  subsched->m_shared_state = m_shared_state;
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
  ASSERTRANGE(tgnum, 0, (int)m_graphs.size());
  TaskGraph* tg = m_graphs[tgnum];
  tg->setIteration(iteration);
  m_currentTG = tgnum;
  DetailedTasks* dts = tg->getDetailedTasks();

  if (m_graphs.size() > 1) {
    // tg model is the multi TG model, where each graph is going to need to
    // have its dwmap reset here (even with the same tgnum)
    tg->remapTaskDWs(m_dw_map);
  }

  if (dts == 0) {
    std::cerr << "SingleProcessorScheduler skipping execute, no tasks\n";
    return;
  }

  int ntasks = dts->numTasks();
  if (ntasks == 0) {
    std::cerr << "WARNING: Scheduler executed, but no tasks\n";
  }

  ASSERT(m_dws.size()>=2);
  std::vector<DataWarehouseP> plain_old_dws(m_dws.size());
  for (unsigned int i = 0; i < m_dws.size(); i++) {
    plain_old_dws[i] = m_dws[i].get_rep();
  }

  if (dbg.active()) {
    dbg << "Executing " << ntasks << " tasks, ";
    for (int i = 0; i < m_num_old_dws; i++) {
      dbg << "from DWs: ";
      if (m_dws[i]) {
        dbg << m_dws[i]->getID() << ", ";
      }
      else {
        dbg << "Null, ";
      }
    }
    if (m_dws.size() - m_num_old_dws > 1) {
      dbg << "intermediate DWs: ";
      for (unsigned int i = m_num_old_dws; i < m_dws.size() - 1; i++) {
        dbg << m_dws[i]->getID() << ", ";
      }
    }
    if (m_dws[m_dws.size() - 1]) {
      dbg << " to DW: " << m_dws[m_dws.size() - 1]->getID();
    }
    else {
      dbg << " to DW: Null";
    }
    dbg << "\n";
  }

  makeTaskGraphDoc(dts);

  dts->initializeScrubs(m_dws, m_dw_map);

  for (int i = 0; i < ntasks; i++) {
    double start = Time::currentSeconds();
    DetailedTask* task = dts->getTask(i);

    taskdbg << d_myworld->myrank() << " SPS: Initiating: ";
    printTask(taskdbg, task);
    taskdbg << '\n';

    if (m_tracking_vars_print_location & SchedulerCommon::PRINT_BEFORE_EXEC) {
      printTrackedVars(task, SchedulerCommon::PRINT_BEFORE_EXEC);
    }

    task->doit(d_myworld, m_dws, plain_old_dws);

    if (m_tracking_vars_print_location & SchedulerCommon::PRINT_AFTER_EXEC) {
      printTrackedVars(task, SchedulerCommon::PRINT_AFTER_EXEC);
    }

    task->done(m_dws);

    if (taskdbg.active()) {
      taskdbg << d_myworld->myrank() << " SPS: Completed:  ";
      printTask(taskdbg, task);
      taskdbg << '\n';
      printTaskLevels(d_myworld, taskLevel_dbg, task);
    }

    double delT = Time::currentSeconds() - start;
    if (m_dws[m_dws.size() - 1] && m_dws[m_dws.size() - 1]->timestepAborted()) {
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
