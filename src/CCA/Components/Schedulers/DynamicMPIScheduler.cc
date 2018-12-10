/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <CCA/Components/Schedulers/DynamicMPIScheduler.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/TaskGraph.h>
#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Output.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DOUT.hpp>

#include <iomanip>
#include <sstream>

using namespace Uintah;


namespace Uintah {
  extern Dout g_task_dbg;
  extern Dout g_task_order;
}

namespace {
  Dout g_dbg(          "DynamicMPI_DBG",         "DynamicMPIScheduler", "general debugging info for DynamicMPIScheduler"  , false );
  Dout g_queue_length( "DynamicMPI_QueueLength", "DynamicMPIScheduler", "report task queue length for DynamicMPIScheduler", false );
}


//______________________________________________________________________
//
DynamicMPIScheduler::DynamicMPIScheduler( const ProcessorGroup*      myworld
                                        ,       DynamicMPIScheduler* parentScheduler
                                        )
  : MPIScheduler( myworld, parentScheduler )
{
  m_task_queue_alg =  MostMessages;
}

//______________________________________________________________________
//
DynamicMPIScheduler::~DynamicMPIScheduler()
{

}

//______________________________________________________________________
//
void
DynamicMPIScheduler::problemSetup( const ProblemSpecP&     prob_spec
                                 , const MaterialManagerP& materialManager
                                 )
{
  std::string taskQueueAlg = "";

  ProblemSpecP params = prob_spec->findBlock("Scheduler");
  if (params) {
    params->get("taskReadyQueueAlg", taskQueueAlg);
  }
  if (taskQueueAlg == "") {
    taskQueueAlg = "MostMessages";  //default taskReadyQueueAlg
  }

  if (taskQueueAlg == "FCFS") {
    m_task_queue_alg = FCFS;
  }
  else if (taskQueueAlg == "Random") {
    m_task_queue_alg = Random;
  }
  else if (taskQueueAlg == "Stack") {
    m_task_queue_alg = Stack;
  }
  else if (taskQueueAlg == "MostMessages") {
    m_task_queue_alg = MostMessages;
  }
  else if (taskQueueAlg == "LeastMessages") {
    m_task_queue_alg = LeastMessages;
  }
  else if (taskQueueAlg == "PatchOrder") {
    m_task_queue_alg = PatchOrder;
  }
  else if (taskQueueAlg == "PatchOrderRandom") {
    m_task_queue_alg = PatchOrderRandom;
  }
  else {
    throw ProblemSetupException("Unknown task ready queue algorithm", __FILE__, __LINE__);
  }

  SchedulerCommon::problemSetup(prob_spec, materialManager);
}

//______________________________________________________________________
//
SchedulerP
DynamicMPIScheduler::createSubScheduler()
{
  DynamicMPIScheduler * newsched = scinew DynamicMPIScheduler( d_myworld, this );
  newsched->setComponents( this );
  newsched->m_materialManager = m_materialManager;
  return newsched;
}

//______________________________________________________________________
//
void
DynamicMPIScheduler::execute( int tgnum     /*=0*/
                            , int iteration /*=0*/
                            )
{
  if (m_is_copy_data_timestep) {
    MPIScheduler::execute(tgnum, iteration);
    return;
  }

  // track total scheduler execution time across timesteps
  m_exec_timer.reset(true);

  RuntimeStats::initialize_timestep(m_task_graphs);

  ASSERTRANGE(tgnum, 0, static_cast<int>(m_task_graphs.size()));
  TaskGraph* tg = m_task_graphs[tgnum];
  tg->setIteration(iteration);
  m_current_task_graph = tgnum;

  // multi TG model - each graph needs have its dwmap reset here (even with the same tgnum)
  if (static_cast<int>(m_task_graphs.size()) > 1) {
    tg->remapTaskDWs(m_dwmap);
  }

  DetailedTasks* dts = tg->getDetailedTasks();

  if(!dts) {
    if (d_myworld->myRank() == 0) {
      DOUT(true, "DynamicMPIScheduler skipping execute, no tasks");
    }
    return;
  }
  
  int ntasks = dts->numLocalTasks();
  dts->initializeScrubs(m_dws, m_dwmap);
  dts->initTimestep();

  for (int i = 0; i < ntasks; i++) {
    dts->localTask(i)->resetDependencyCounts();
  }

  int me = d_myworld->myRank();

  // This only happens if "-emit_taskgraphs" is passed to sus
  makeTaskGraphDoc(dts, me);

  mpi_info_.reset( 0 );

  if( m_reloc_new_pos_label && m_dws[m_dwmap[Task::OldDW]] != nullptr ) {
    m_dws[m_dwmap[Task::OldDW]]->exchangeParticleQuantities(dts, m_loadBalancer, m_reloc_new_pos_label, iteration);
  }

#if 0
  // hook to post all the messages up front
  if (!m_is_copy_data_timestep) {
    // post the receives in advance
    for (int i = 0; i < ntasks; i++) {
      initiateTask( dts->localTask(i), abort, abort_point, iteration );
    }
  }
#endif

  int currphase = 0;
  std::map<int, int> phaseTasks;
  std::map<int, int> phaseTasksDone;
  std::map<int,  DetailedTask *> phaseSyncTask;
  dts->setTaskPriorityAlg(m_task_queue_alg);

  for (int i = 0; i < ntasks; i++) {
    phaseTasks[dts->localTask(i)->getTask()->m_phase]++;
  }
  
  if (g_dbg) {
    std::ostringstream message;
    message << "Rank-" << me << " Executing " << dts->numTasks() << " tasks (" << ntasks << " local)";
    for (std::map<int, int>::iterator it = phaseTasks.begin(); it != phaseTasks.end(); it++) {
      message << ", phase[" << (*it).first << "] = " << (*it).second;
    }
    DOUT(true, message.str());
  }

  static std::vector<int> histogram;
  static int totaltasks;
  std::set<DetailedTask*> pending_tasks;

  int numTasksDone = 0;
  bool abort       = false;
  int  abort_point = 987654;
  int i            = 0;

  while( numTasksDone < ntasks ) {

    i++;

    DetailedTask * task = nullptr;

    // if we have an internally-ready task, initiate its recvs
    while(dts->numInternalReadyTasks() > 0) { 
      DetailedTask * task = dts->getNextInternalReadyTask();

      if ((task->getTask()->getType() == Task::Reduction) || (task->getTask()->usesMPI())) {  //save the reduction task for later
        phaseSyncTask[task->getTask()->m_phase] = task;
        DOUT(g_task_dbg, "Rank-" << d_myworld->myRank() << " Task Reduction ready " << *task << " deps needed: " << task->getExternalDepCount());
      } else {
        initiateTask(task, abort, abort_point, iteration);
        task->markInitiated();
        task->checkExternalDepCount();
        DOUT(g_task_dbg, "Rank-" << d_myworld->myRank() << " Task internal ready " << *task << " deps needed: " << task->getExternalDepCount());

        // if MPI has completed, it will run on the next iteration
        pending_tasks.insert(task);
      }
    }

    if (dts->numExternalReadyTasks() > 0) {
      // run a task that has its communication complete
      // tasks get in this queue automatically when their receive count hits 0
      //   in DependencyBatch::received, which is called when a message is delivered.
      if (g_queue_length) {
        if ((int)histogram.size() < dts->numExternalReadyTasks() + 1) {
          histogram.resize(dts->numExternalReadyTasks() + 1);
        }
        histogram[dts->numExternalReadyTasks()]++;
      }
     
      DetailedTask * task = dts->getNextExternalReadyTask();

      DOUT(g_task_dbg,
           "Rank-" << d_myworld->myRank() << " Running task " << *task << "(" << dts->numExternalReadyTasks() << "/" << pending_tasks.size() << " tasks in queue)");;

      pending_tasks.erase(pending_tasks.find(task));
      ASSERTEQ(task->getExternalDepCount(), 0);
      runTask(task, iteration);
      numTasksDone++;

      if (g_task_order && d_myworld->myRank() == d_myworld->nRanks() / 2) {
        std::ostringstream task_name;
        task_name << "  Running task: \"" << task->getTask()->getName() << "\" ";

        std::ostringstream task_type;
        task_type << "(" << task->getTask()->getType() << ") ";

        // task ordering debug info - please keep this here, APH 05/30/18
        DOUT(true, "Rank-" << d_myworld->myRank()
                           << std::setw(60) << std::left << task_name.str()
                           << std::setw(14) << std::left << task_type.str()
                           << std::setw(15) << " static order: "    << std::setw(3) << std::left << task->getStaticOrder()
                           << std::setw(18) << " scheduled order: " << std::setw(3) << std::left << numTasksDone);
      }

      phaseTasksDone[task->getTask()->m_phase]++;
    } 

    if ((phaseSyncTask.find(currphase) != phaseSyncTask.end()) && (phaseTasksDone[currphase] == phaseTasks[currphase] - 1)) {  //if it is time to run the reduction task
      if (g_queue_length) {
        if ((int)histogram.size() < dts->numExternalReadyTasks() + 1) {
          histogram.resize(dts->numExternalReadyTasks() + 1);
        }
        histogram[dts->numExternalReadyTasks()]++;
      }
      DetailedTask *reducetask = phaseSyncTask[currphase];
      if (reducetask->getTask()->getType() == Task::Reduction) {
        if (!abort) {
          DOUT(g_task_dbg, "Rank-" << d_myworld->myRank() << " Running Reduce task " << reducetask->getTask()->getName());
        }
        initiateReduction(reducetask);
      }
      else {  // Task::OncePerProc task
        ASSERT(reducetask->getTask()->usesMPI());
        initiateTask(reducetask, abort, abort_point, iteration);
        reducetask->markInitiated();
        ASSERT(reducetask->getExternalDepCount() == 0);
        runTask(reducetask, iteration);

        DOUT(g_task_dbg, "Rank-" << d_myworld->myRank() << " Running OPP task:");;

      }
      ASSERT(reducetask->getTask()->m_phase == currphase);

      numTasksDone++;

      // task ordering debug info - please keep this here, APH 05/30/18
      if (g_task_order && d_myworld->myRank() == d_myworld->nRanks() / 2) {
        std::ostringstream task_name;
        task_name << "  Running task: \"" << reducetask->getTask()->getName() << "\" ";

        std::ostringstream task_type;
        task_type << "(" << reducetask->getTask()->getType() << ") ";

        DOUT(true, "Rank-" << d_myworld->myRank()
                           << std::setw(60) << std::left << task_name.str()
                           << std::setw(14) << std::left << task_type.str()
                           << std::setw(15) << " static order: "    << std::setw(3) << std::left << reducetask->getStaticOrder()
                           << std::setw(18) << " scheduled order: " << std::setw(3) << std::left << numTasksDone);
      }

      phaseTasksDone[reducetask->getTask()->m_phase]++;
    }

    if (numTasksDone < ntasks) {
      if (phaseTasks[currphase] == phaseTasksDone[currphase]) {
        currphase++;
      }
      else if (dts->numExternalReadyTasks() > 0 || dts->numInternalReadyTasks() > 0
               || (phaseSyncTask.find(currphase) != phaseSyncTask.end() && phaseTasksDone[currphase] == phaseTasks[currphase] - 1))  // if there is work to do
          {
        processMPIRecvs(TEST);  // receive what is ready and do not block
      }
      else {
        // we have nothing to do, so wait until we get something
        processMPIRecvs(WAIT_ONCE);  // there is no other work to do so block until some receives are completed
      }
    }

    // ARS - FIXME CHECK THE WAREHOUSE
    OnDemandDataWarehouseP dw = m_dws[m_dws.size() - 1];
    if (!abort && dw && dw->abortTimeStep()) {
      // TODO - abort might not work with external queue...
      abort = true;
      abort_point = task->getTask()->getSortedOrder();

      DOUT(g_dbg,  "Rank-" << d_myworld->myRank()
	                   << "  WARNING: Aborting time step after task: "
                           << task->getTask()->getName());
    }
  } // end while( numTasksDone < ntasks )


  if (g_queue_length) {
    float lengthsum = 0;
    totaltasks += ntasks;
    for (unsigned int i = 1; i < histogram.size(); i++) {
      lengthsum = lengthsum + i * histogram[i];
    }
    float queuelength = lengthsum / totaltasks;
    float allqueuelength = 0;
    Uintah::MPI::Reduce(&queuelength, &allqueuelength, 1, MPI_FLOAT, MPI_SUM, 0, d_myworld->getComm());
    proc0cout << "average queue length:" << allqueuelength / d_myworld->nRanks() << std::endl;
  }

  //---------------------------------------------------------------------------
  // New way of managing single MPI requests - avoids MPI_Waitsome & MPI_Donesome - APH 07/20/16
  // ---------------------------------------------------------------------------
  // wait on all pending requests
  auto ready_request = [](CommRequest const& r)->bool { return r.wait(); };
  while ( m_sends.size() != 0u ) {
    CommRequestPool::iterator comm_sends_iter;
    if ( (comm_sends_iter = m_sends.find_any(ready_request)) ) {
      m_sends.erase(comm_sends_iter);
    } else {
      // TODO - make this a sleep? APH 07/20/16
    }
  }
  //---------------------------------------------------------------------------

  ASSERT(m_sends.size() == 0u);
  ASSERT(m_recvs.size() == 0u);

  finalizeTimestep();
  
  m_exec_timer.stop();

  // compute the net timings
  MPIScheduler::computeNetRuntimeStats();

  // only do on top-level scheduler
  if ( m_parent_scheduler == nullptr ) {
    MPIScheduler::outputTimingStats( "DynamicMPIScheduler" );
  }

  RuntimeStats::report(d_myworld->getComm());

} // end execute()

