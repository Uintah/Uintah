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
#include <Core/Util/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>

#include <sci_defs/visit_defs.h>

using namespace Uintah;

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
  SingleProcessorScheduler * subsched = scinew SingleProcessorScheduler( d_myworld, m_out_port, this );
  UintahParallelPort       * lbp      = getPort("load balancer");
  subsched->attachPort( "load balancer", lbp );
  subsched->m_shared_state = m_shared_state;

#ifdef HAVE_VISIT
  if( m_shared_state->getVisIt() )
  {
    m_shared_state->d_debugStreams.push_back( &dbg );
  }
#endif

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
  ASSERTRANGE(tgnum, 0, (int)m_task_graphs.size());
  TaskGraph* tg = m_task_graphs[tgnum];
  tg->setIteration(iteration);
  m_current_task_graph = tgnum;
  DetailedTasks* dts = tg->getDetailedTasks();

  if (m_task_graphs.size() > 1) {
    // tg model is the multi TG model, where each graph is going to need to
    // have its dwmap reset here (even with the same tgnum)
    tg->remapTaskDWs(m_dwmap);
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

  makeTaskGraphDoc(dts);

  dts->initializeScrubs(m_dws, m_dwmap);

  for (int i = 0; i < ntasks; i++) {
    double start = Time::currentSeconds();
    DetailedTask* task = dts->getTask(i);


    if (m_tracking_vars_print_location & SchedulerCommon::PRINT_BEFORE_EXEC) {
      printTrackedVars(task, SchedulerCommon::PRINT_BEFORE_EXEC);
    }

    task->doit(d_myworld, m_dws, plain_old_dws);

    if (m_tracking_vars_print_location & SchedulerCommon::PRINT_AFTER_EXEC) {
      printTrackedVars(task, SchedulerCommon::PRINT_AFTER_EXEC);
    }

    task->done(m_dws);


    double delT = Time::currentSeconds() - start;
    if (m_dws[m_dws.size() - 1] && m_dws[m_dws.size() - 1]->timestepAborted()) {
      dbg << "Aborting timestep after task: " << *task->getTask() << '\n';
      break;
    }

    emitNode(task, start, delT, delT);
  }
  finalizeTimestep();
}
