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

#include <TauProfilerForSCIRun.h>

#include <CCA/Components/Schedulers/DynamicMPIScheduler.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/TaskGraph.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Time.h>

#include <cstring>

using namespace Uintah;
using namespace SCIRun;

// Used to sync cout/cerr so it is readable when output by multiple threads
extern SCIRun::Mutex      coutLock;
extern SCIRun::Mutex      cerrLock;

extern DebugStream        taskdbg;
extern std::map<std::string,double> waittimes;
extern std::map<std::string,double> exectimes;
extern DebugStream        waitout;
extern DebugStream        execout;
extern DebugStream        taskorder;

static DebugStream dynamicmpi_dbg(        "DynamicMPI_DBG",         false);
static DebugStream dynamicmpi_timeout(    "DynamicMPI_TimingsOut",  false);
static DebugStream dynamicmpi_queuelength("DynamicMPI_QueueLength", false);

#ifdef USE_TAU_PROFILING
extern int create_tau_mapping( const string&      taskname,
                               const PatchSubset* patches );  // ThreadPool.cc
#endif

//______________________________________________________________________
//
DynamicMPIScheduler::DynamicMPIScheduler( const ProcessorGroup*      myworld,
                                          const Output*              oport,
                                                DynamicMPIScheduler* parentScheduler )
  : MPIScheduler( myworld, oport, parentScheduler )
{
  taskQueueAlg_ =  MostMessages;
}

//______________________________________________________________________
//
void
DynamicMPIScheduler::problemSetup( const ProblemSpecP&     prob_spec,
                                         SimulationStateP& state )
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
    taskQueueAlg_ = FCFS;
  }
  else if (taskQueueAlg == "Random") {
    taskQueueAlg_ = Random;
  }
  else if (taskQueueAlg == "Stack") {
    taskQueueAlg_ = Stack;
  }
  else if (taskQueueAlg == "MostChildren") {
    taskQueueAlg_ = MostChildren;
  }
  else if (taskQueueAlg == "LeastChildren") {
    taskQueueAlg_ = LeastChildren;
  }
  else if (taskQueueAlg == "MostAllChildren") {
    taskQueueAlg_ = MostChildren;
  }
  else if (taskQueueAlg == "LeastAllChildren") {
    taskQueueAlg_ = LeastChildren;
  }
  else if (taskQueueAlg == "MostL2Children") {
    taskQueueAlg_ = MostL2Children;
  }
  else if (taskQueueAlg == "LeastL2Children") {
    taskQueueAlg_ = LeastL2Children;
  }
  else if (taskQueueAlg == "MostMessages") {
    taskQueueAlg_ = MostMessages;
  }
  else if (taskQueueAlg == "LeastMessages") {
    taskQueueAlg_ = LeastMessages;
  }
  else if (taskQueueAlg == "PatchOrder") {
    taskQueueAlg_ = PatchOrder;
  }
  else if (taskQueueAlg == "PatchOrderRandom") {
    taskQueueAlg_ = PatchOrderRandom;
  }
  else {
    throw ProblemSetupException("Unknown task ready queue algorithm", __FILE__, __LINE__);
  }
  log.problemSetup(prob_spec);
  SchedulerCommon::problemSetup(prob_spec, state);
}

//______________________________________________________________________
//
DynamicMPIScheduler::~DynamicMPIScheduler()
{
  if (dynamicmpi_timeout.active()) {
    timingStats.close();
    if (d_myworld->myrank() == 0) {
      avgStats.close();
      maxStats.close();
    }
  }
}

//______________________________________________________________________
//
SchedulerP
DynamicMPIScheduler::createSubScheduler()
{
  DynamicMPIScheduler* newsched = scinew DynamicMPIScheduler(d_myworld, m_outPort, this);
  newsched->d_sharedState = d_sharedState;
  UintahParallelPort* lbp = getPort("load balancer");
  newsched->attachPort("load balancer", lbp);
  newsched->d_sharedState=d_sharedState;
  return newsched;
}

//______________________________________________________________________
//
void
DynamicMPIScheduler::execute( int tgnum     /*=0*/,
                              int iteration /*=0*/ )
{
  if (d_sharedState->isCopyDataTimestep()) {
    MPIScheduler::execute(tgnum, iteration);
    return;
  }

  MALLOC_TRACE_TAG_SCOPE("DynamicMPIScheduler::execute");

  TAU_PROFILE("DynamicMPIScheduler::execute()", " ", TAU_USER); 
  TAU_PROFILE_TIMER(reducetimer, "Reductions", "[DynamicMPIScheduler::execute()] " , TAU_USER); 
  TAU_PROFILE_TIMER(sendtimer, "Send Dependency", "[DynamicMPIScheduler::execute()] " , TAU_USER); 
  TAU_PROFILE_TIMER(recvtimer, "Recv Dependency", "[DynamicMPIScheduler::execute()] " , TAU_USER); 
  TAU_PROFILE_TIMER(outputtimer, "Task Graph Output", "[DynamicMPIScheduler::execute()] ", TAU_USER);
  TAU_PROFILE_TIMER(testsometimer, "Test Some", "[DynamicMPIScheduler::execute()] ", TAU_USER);
  TAU_PROFILE_TIMER(finalwaittimer, "Final Wait", "[DynamicMPIScheduler::execute()] ", TAU_USER);
  TAU_PROFILE_TIMER(sorttimer, "Topological Sort", "[DynamicMPIScheduler::execute()] ",TAU_USER);
  TAU_PROFILE_TIMER(sendrecvtimer, "Initial Send Recv", "[DynamicMPIScheduler::execute()] ", TAU_USER);

  ASSERTRANGE(tgnum, 0, (int)graphs.size());
  TaskGraph* tg = graphs[tgnum];
  tg->setIteration(iteration);
  currentTG_ = tgnum;

  if (graphs.size() > 1) {
    // tg model is the multi TG model, where each graph is going to need to
    // have its dwmap reset here (even with the same tgnum)
    tg->remapTaskDWs(dwmap);
  }

  DetailedTasks* dts = tg->getDetailedTasks();

  if(dts == 0) {
    if (d_myworld->myrank() == 0) {
      std::cerr << "DynamicMPIScheduler skipping execute, no tasks\n";
    }
    return;
  }
  
  int ntasks = dts->numLocalTasks();
  dts->initializeScrubs(dws, dwmap);
  dts->initTimestep();

  for (int i = 0; i < ntasks; i++) {
    dts->localTask(i)->resetDependencyCounts();
  }

  if(dynamicmpi_timeout.active()) {
    d_labels.clear();
    d_times.clear();
    //emitTime("time since last execute");
  }

  int me = d_myworld->myrank();
  makeTaskGraphDoc(dts, me);

  //if(timeout.active())
    //emitTime("taskGraph output");

  mpi_info_.totalreduce = 0;
  mpi_info_.totalsend = 0;
  mpi_info_.totalrecv = 0;
  mpi_info_.totaltask = 0;
  mpi_info_.totalreducempi = 0;
  mpi_info_.totalsendmpi = 0;
  mpi_info_.totalrecvmpi = 0;
  mpi_info_.totaltestmpi = 0;
  mpi_info_.totalwaitmpi = 0;

  int numTasksDone = 0;

  bool abort=false;
  int abort_point = 987654;

  int i = 0;

  if (reloc_new_posLabel_ && dws[dwmap[Task::OldDW]] != 0) {
    dws[dwmap[Task::OldDW]]->exchangeParticleQuantities(dts, getLoadBalancer(), reloc_new_posLabel_, iteration);
  }

  TAU_PROFILE_TIMER(doittimer, "Task execution", "[DynamicMPIScheduler::execute() loop] ", TAU_USER); 
  TAU_PROFILE_START(doittimer);

#if 0
  // hook to post all the messages up front
  if (useDynamicScheduling_ && !d_sharedState->isCopyDataTimestep()) {
    // post the receives in advance
    for (int i = 0; i < ntasks; i++) {
      initiateTask( dts->localTask(i), abort, abort_point, iteration );
    }
  }
#endif
  int currphase=0;
  std::map<int, int> phaseTasks;
  std::map<int, int> phaseTasksDone;
  std::map<int,  DetailedTask *> phaseSyncTask;
  dts->setTaskPriorityAlg(taskQueueAlg_ );

  for (int i = 0; i < ntasks; i++) {
    phaseTasks[dts->localTask(i)->getTask()->d_phase]++;
  }
  
  if (dynamicmpi_dbg.active()) {
    cerrLock.lock();
    dynamicmpi_dbg << me << " Executing " << dts->numTasks() << " tasks (" << ntasks << " local)";
    for (std::map<int, int>::iterator it = phaseTasks.begin(); it != phaseTasks.end(); it++) {
      dynamicmpi_dbg << ", phase[" << (*it).first << "] = " << (*it).second;
    }
    dynamicmpi_dbg << std::endl;
    cerrLock.unlock();
  }

  static std::vector<int> histogram;
  static int totaltasks;
  std::set<DetailedTask*> pending_tasks;

  while( numTasksDone < ntasks ) {
    i++;

    // 
    // The following checkMemoryUse() is commented out to allow for
    // maintaining the same functionality as before this commit...
    // In other words, so that memory highwater checking is only done
    // at the end of a timestep, and not between tasks... Once the
    // RT settles down we will uncomment this section and then
    // memory use checks will occur before every task.
    //
    // Note, the results (memuse, highwater, maxMemUse) from the following
    // checkMemoryUse call are not used... the call, however, records
    // the maxMemUse for future reference, and that is why we are calling
    // it.
    //
    //unsigned long memuse, highwater, maxMemUse;
    //checkMemoryUse( memuse, highwater, maxMemUse );

    DetailedTask * task = 0;

    // if we have an internally-ready task, initiate its recvs
    while(dts->numInternalReadyTasks() > 0) { 
      DetailedTask * task = dts->getNextInternalReadyTask();

      if ((task->getTask()->getType() == Task::Reduction) || (task->getTask()->usesMPI())) {  //save the reduction task for later
        phaseSyncTask[task->getTask()->d_phase] = task;
        taskdbg << d_myworld->myrank() << " Task Reduction ready " << *task << " deps needed: " << task->getExternalDepCount() << std::endl;
      } else {
        initiateTask(task, abort, abort_point, iteration);
        task->markInitiated();
        task->checkExternalDepCount();
        taskdbg << d_myworld->myrank() << " Task internal ready " << *task << " deps needed: " << task->getExternalDepCount() << std::endl;

        // if MPI has completed, it will run on the next iteration
        pending_tasks.insert(task);
      }
    }

    if (dts->numExternalReadyTasks() > 0) {
      // run a task that has its communication complete
      // tasks get in this queue automatically when their receive count hits 0
      //   in DependencyBatch::received, which is called when a message is delivered.
      if (dynamicmpi_queuelength.active()) {
        if ((int)histogram.size() < dts->numExternalReadyTasks() + 1) {
          histogram.resize(dts->numExternalReadyTasks() + 1);
        }
        histogram[dts->numExternalReadyTasks()]++;
      }
     
      DetailedTask * task = dts->getNextExternalReadyTask();
#ifdef USE_TAU_PROFILING
      int id;
      const PatchSubset* patches = task->getPatches();
      id = create_tau_mapping( task->getTask()->getName(), patches );

      string phase_name = "no patches";
      if (patches && patches->size() > 0) {
        phase_name = "level";
        for(int i=0;i<patches->size();i++) {

          ostringstream patch_num;
          patch_num << patches->get(i)->getLevel()->getIndex();

          if (i == 0) {
            phase_name = phase_name + " " + patch_num.str();
          } else {
            phase_name = phase_name + ", " + patch_num.str();
          }
        }
      }

      static map<string,int> phase_map;
      static int unique_id = 99999;
      int phase_id;
      map<string,int>::iterator iter = phase_map.find( phase_name );
      if( iter != phase_map.end() ) {
        phase_id = (*iter).second;
      } else {
        TAU_MAPPING_CREATE( phase_name, "",
            (TauGroup_t) unique_id, "TAU_USER", 0 );
        phase_map[ phase_name ] = unique_id;
        phase_id = unique_id++;
      }
      // Task name
      TAU_MAPPING_OBJECT(tautimer)
      TAU_MAPPING_LINK(tautimer, (TauGroup_t)id);  // EXTERNAL ASSOCIATION
      TAU_MAPPING_PROFILE_TIMER(doitprofiler, tautimer, 0)
      TAU_MAPPING_PROFILE_START(doitprofiler,0);
#endif
      if (taskdbg.active()) {
        cerrLock.lock();
        taskdbg << d_myworld->myrank() << " Running task " << *task << "(" << dts->numExternalReadyTasks() << "/"
                << pending_tasks.size() << " tasks in queue)" << std::endl;
        cerrLock.unlock();
      }

      pending_tasks.erase(pending_tasks.find(task));
      ASSERTEQ(task->getExternalDepCount(), 0);
      runTask(task, iteration);
      numTasksDone++;
      if (taskorder.active()) {
        if (d_myworld->myrank() == d_myworld->size() / 2) {
          cerrLock.lock();
          taskorder << d_myworld->myrank() << " Running task static order: " << task->getSaticOrder() << " , scheduled order: "
                    << numTasksDone << std::endl;
          cerrLock.unlock();
        }
      }
      phaseTasksDone[task->getTask()->d_phase]++;

#ifdef USE_TAU_PROFILING
      TAU_MAPPING_PROFILE_STOP(doitprofiler);
#endif
    } 

    if ((phaseSyncTask.find(currphase) != phaseSyncTask.end()) && (phaseTasksDone[currphase] == phaseTasks[currphase] - 1)) {  //if it is time to run the reduction task
      if (dynamicmpi_queuelength.active()) {
        if ((int)histogram.size() < dts->numExternalReadyTasks() + 1)
          histogram.resize(dts->numExternalReadyTasks() + 1);
        histogram[dts->numExternalReadyTasks()]++;
      }
      DetailedTask *reducetask = phaseSyncTask[currphase];
      if (reducetask->getTask()->getType() == Task::Reduction) {
        if (!abort) {
          cerrLock.lock();
          taskdbg << d_myworld->myrank() << " Running Reduce task " << reducetask->getTask()->getName() << std::endl;
          cerrLock.unlock();
        }
        initiateReduction(reducetask);
      }
      else {  // Task::OncePerProc task
        ASSERT(reducetask->getTask()->usesMPI());
        initiateTask(reducetask, abort, abort_point, iteration);
        reducetask->markInitiated();
        ASSERT(reducetask->getExternalDepCount() == 0);
        runTask(reducetask, iteration);

        if (taskdbg.active()) {
          cerrLock.lock();
          taskdbg << d_myworld->myrank() << " Runnding OPP task:  \t";
          printTask(taskdbg, reducetask);
          taskdbg << '\n';
          cerrLock.unlock();
        }

      }
      ASSERT(reducetask->getTask()->d_phase == currphase);

      numTasksDone++;
      if (taskorder.active()) {
        if (d_myworld->myrank() == d_myworld->size() / 2) {
          taskorder << d_myworld->myrank() << " Running task static order: " << reducetask->getSaticOrder()
                    << " , scheduled order: " << numTasksDone << std::endl;
        }
      }
      phaseTasksDone[reducetask->getTask()->d_phase]++;
    }

    if (numTasksDone < ntasks) {
      if (phaseTasks[currphase] == phaseTasksDone[currphase]) {
        currphase++;
      }
      else if (dts->numExternalReadyTasks() > 0 || dts->numInternalReadyTasks() > 0
               || (phaseSyncTask.find(currphase) != phaseSyncTask.end() && phaseTasksDone[currphase] == phaseTasks[currphase] - 1))  //if there is work to do
          {
        processMPIRecvs(TEST);  // receive what is ready and do not block
      }
      else {
        // we have nothing to do, so wait until we get something
        processMPIRecvs(WAIT_ONCE);  //There is no other work to do so block until some receives are completed
      }
    }

    if (!abort && dws[dws.size() - 1] && dws[dws.size() - 1]->timestepAborted()) {
      // TODO - abort might not work with external queue...
      abort = true;
      abort_point = task->getTask()->getSortedOrder();
      dynamicmpi_dbg << "Aborting timestep after task: " << *task->getTask() << '\n';
    }
  } // end while( numTasksDone < ntasks )

  TAU_PROFILE_STOP(doittimer);
  
  if (dynamicmpi_queuelength.active()) {
    float lengthsum = 0;
    totaltasks += ntasks;
    for (unsigned int i = 1; i < histogram.size(); i++) {
      lengthsum = lengthsum + i * histogram[i];
    }
    // if (me==0) cout << endl;
    float queuelength = lengthsum / totaltasks;
    float allqueuelength = 0;

    MPI_Reduce(&queuelength, &allqueuelength, 1, MPI_FLOAT, MPI_SUM, 0, d_myworld->getComm());

    proc0cout << "average queue length:" << allqueuelength / d_myworld->size() << std::endl;
  }
  
  if (dynamicmpi_timeout.active()) {
    emitTime("MPI send time", mpi_info_.totalsendmpi);
    emitTime("MPI Testsome time", mpi_info_.totaltestmpi);
    emitTime("Total send time", mpi_info_.totalsend - mpi_info_.totalsendmpi - mpi_info_.totaltestmpi);
    emitTime("MPI recv time", mpi_info_.totalrecvmpi);
    emitTime("MPI wait time", mpi_info_.totalwaitmpi);
    emitTime("Total recv time", mpi_info_.totalrecv - mpi_info_.totalrecvmpi - mpi_info_.totalwaitmpi);
    emitTime("Total task time", mpi_info_.totaltask);
    emitTime("Total MPI reduce time", mpi_info_.totalreducempi);
    emitTime("Total reduction time", mpi_info_.totalreduce - mpi_info_.totalreducempi);
    emitTime("Total comm time", mpi_info_.totalrecv + mpi_info_.totalsend + mpi_info_.totalreduce);

    double time = Time::currentSeconds();
    double totalexec = time - d_lasttime;

    d_lasttime = time;

    emitTime("Other excution time",
             totalexec - mpi_info_.totalsend - mpi_info_.totalrecv - mpi_info_.totaltask - mpi_info_.totalreduce);
  }

  if (d_sharedState != 0) {  // subschedulers don't have a sharedState
    d_sharedState->taskExecTime += mpi_info_.totaltask - d_sharedState->outputTime;  // don't count output time...
    d_sharedState->taskLocalCommTime += mpi_info_.totalrecv + mpi_info_.totalsend;
    d_sharedState->taskWaitCommTime += mpi_info_.totalwaitmpi;
    d_sharedState->taskGlobalCommTime += mpi_info_.totalreduce;
  }

  // Don't need to lock sends 'cause all threads are done at this point.
  sends_[0].waitall(d_myworld);
  ASSERT(sends_[0].numRequests() == 0);
  //if(timeout.active())
    //emitTime("final wait");
  if (restartable && tgnum == (int)graphs.size() - 1) {
    // Copy the restart flag to all processors
    int myrestart = dws[dws.size() - 1]->timestepRestarted();
    int netrestart;
    MPI_Allreduce(&myrestart, &netrestart, 1, MPI_INT, MPI_LOR, d_myworld->getComm());
    if (netrestart) {
      dws[dws.size() - 1]->restartTimestep();
      if (dws[0]) {
        dws[0]->setRestarted();
      }
    }
  }

  finalizeTimestep();
  

  log.finishTimestep();
  if( dynamicmpi_timeout.active() && !parentScheduler_ ){ // only do on toplevel scheduler
    //emitTime("finalize");

    // add number of cells, patches, and particles
    int numCells = 0, numParticles = 0;
    OnDemandDataWarehouseP dw = dws[dws.size()-1];
    const GridP grid(const_cast<Grid*>(dw->getGrid()));
    const PatchSubset* myPatches = getLoadBalancer()->getPerProcessorPatchSet(grid)->getSubset(d_myworld->myrank());
    for (int p = 0; p < myPatches->size(); p++) {
      const Patch* patch = myPatches->get(p);
      IntVector range = patch->getExtraCellHighIndex() - patch->getExtraCellLowIndex();
      numCells += range.x()*range.y()*range.z();

      // go through all materials since getting an MPMMaterial correctly would depend on MPM
      for (int m = 0; m < d_sharedState->getNumMatls(); m++) {
        if (dw->haveParticleSubset(m, patch))
          numParticles += dw->getParticleSubset(m, patch)->numParticles();
      }
    }

    emitTime("NumPatches", myPatches->size());
    emitTime("NumCells", numCells);
    emitTime("NumParticles", numParticles);
    std::vector<double> d_totaltimes(d_times.size());
    std::vector<double> d_maxtimes(d_times.size());
    std::vector<double> d_avgtimes(d_times.size());
    double avgTask = -1, maxTask = -1;
    double avgComm = -1, maxComm = -1;
    double avgCell = -1, maxCell = -1;

    MPI_Reduce(&d_times[0], &d_totaltimes[0], static_cast<int>(d_times.size()), MPI_DOUBLE, MPI_SUM, 0, d_myworld->getComm());
    MPI_Reduce(&d_times[0], &d_maxtimes[0],   static_cast<int>(d_times.size()), MPI_DOUBLE, MPI_MAX, 0, d_myworld->getComm());

    double total = 0, avgTotal = 0, maxTotal = 0;
    for (int i = 0; i < static_cast<int>(d_totaltimes.size()); i++) {
      d_avgtimes[i] = d_totaltimes[i] / d_myworld->size();
      if (strcmp(d_labels[i], "Total task time") == 0) {
        avgTask = d_avgtimes[i];
        maxTask = d_maxtimes[i];
      }
      else if (strcmp(d_labels[i], "Total comm time") == 0) {
        avgComm = d_avgtimes[i];
        maxComm = d_maxtimes[i];
      }
      else if (strncmp(d_labels[i], "Num", 3) == 0) {
        if (strcmp(d_labels[i], "NumCells") == 0) {
          avgCell = d_avgtimes[i];
          maxCell = d_maxtimes[i];
        }
        // these are independent stats - not to be summed
        continue;
      }

      total += d_times[i];
      avgTotal += d_avgtimes[i];
      maxTotal += d_maxtimes[i];
    }

    // to not duplicate the code
    std::vector <std::ofstream*> files;
    std::vector <std::vector<double>* > data;
    files.push_back(&timingStats);
    data.push_back(&d_times);

    if (me == 0) {
      files.push_back(&avgStats);
      files.push_back(&maxStats);
      data.push_back(&d_avgtimes);
      data.push_back(&d_maxtimes);
    }

    for (unsigned file = 0; file < files.size(); file++) {
      std::ofstream& out = *files[file];
      out << "Timestep " << d_sharedState->getCurrentTopLevelTimeStep() << std::endl;
      for (int i = 0; i < static_cast<int>((*data[file]).size()); i++) {
        out << "DynamicMPIScheduler: " << d_labels[i] << ": ";
        int len = (int)(strlen(d_labels[i]) + strlen("DynamicMPIScheduler: ") + strlen(": "));
        for (int j = len; j < 55; j++) {
          out << ' ';
        }
        double percent;
        if (strncmp(d_labels[i], "Num", 3) == 0) {
          percent = d_totaltimes[i] == 0 ? 100 : (*data[file])[i] / d_totaltimes[i] * 100;
        }
        else {
          percent = (*data[file])[i] / total * 100;
        }
        out << (*data[file])[i] << " (" << percent << "%)\n";
      }
      out << std::endl << std::endl;
    }

    if (me == 0) {
      dynamicmpi_timeout << "  Avg. exec: " << avgTask << ", max exec: " << maxTask << " = " << (1-avgTask/maxTask)*100 << " load imbalance (exec)%\n";
      dynamicmpi_timeout << "  Avg. comm: " << avgComm << ", max comm: " << maxComm << " = " << (1-avgComm/maxComm)*100 << " load imbalance (comm)%\n";
      dynamicmpi_timeout << "  Avg.  vol: " << avgCell << ", max  vol: " << maxCell << " = " << (1-avgCell/maxCell)*100 << " load imbalance (theoretical)%\n";
    }

    double time = Time::currentSeconds();
    d_lasttime = time;
  }

  if (execout.active()) {
    static int count = 0;

    // only output the exec times every 10 timesteps
    if (++count % 10 == 0) {
      std::ofstream fout;
      char filename[100];
      sprintf(filename, "exectimes.%d.%d", d_myworld->size(), d_myworld->myrank());
      fout.open(filename);
      for (std::map<std::string, double>::iterator iter = exectimes.begin(); iter != exectimes.end(); iter++) {
        fout << std::fixed << d_myworld->myrank() << ": TaskExecTime(s): " << iter->second << " Task:" << iter->first << std::endl;
      }
      fout.close();
      exectimes.clear();
    }
  }

  if (waitout.active()) {
    static int count = 0;

    // only output the wait times every 10 timesteps
    if (++count % 10 == 0) {

      if (d_myworld->myrank() == 0 || d_myworld->myrank() == d_myworld->size() / 2
          || d_myworld->myrank() == d_myworld->size() - 1) {

        std::ofstream wout;
        char fname[100];
        sprintf(fname, "WaitTimes.%d.%d", d_myworld->size(), d_myworld->myrank());
        wout.open(fname);

        for (std::map<std::string, double>::iterator iter = waittimes.begin(); iter != waittimes.end(); iter++) {
          wout << std::fixed << d_myworld->myrank() << ": TaskWaitTime(TO): " << iter->second << " Task:" << iter->first << std::endl;
        }

        for (std::map<std::string, double>::iterator iter = DependencyBatch::waittimes.begin(); iter != DependencyBatch::waittimes.end();
            iter++) {
          wout << std::fixed << d_myworld->myrank() << ": TaskWaitTime(FROM): " << iter->second << " Task:" << iter->first << std::endl;
        }

        wout.close();
      }

      waittimes.clear();
      DependencyBatch::waittimes.clear();
    }
  }

  if( dynamicmpi_dbg.active()) {
    coutLock.lock();
    dynamicmpi_dbg << me << " DynamicMPIScheduler finished\n";
    coutLock.unlock();
  }
}

