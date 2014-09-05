/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <CCA/Components/Schedulers/ThreadedMPIScheduler2.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/TaskGraph.h>

#include <Core/Exceptions/ProblemSetupException.h>

#include <CCA/Ports/Output.h>

#include <Core/Thread/Time.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>

#include   <cstring>

#define USE_PACKING

using namespace std;
using namespace Uintah;
using namespace SCIRun;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern SCIRun::Mutex       cerrLock;
extern DebugStream taskdbg;
extern DebugStream mpidbg;
extern map<string,double> waittimes;
extern map<string,double> exectimes;
extern DebugStream waitout;
extern DebugStream execout;

static double CurrentWaitTime=0;

static DebugStream dbg("ThreadedMPIScheduler2", false);
static DebugStream timeout("ThreadedMPIScheduler2.timings", false);
static DebugStream queuelength("QueueLength",false);
static DebugStream threaddbg("ThreadDBG",false);
static DebugStream affinity("CPUAffinity", true);

ThreadedMPIScheduler2::ThreadedMPIScheduler2( const ProcessorGroup * myworld,
			          Output         * oport,
			          ThreadedMPIScheduler2   * parentScheduler) :
  MPIScheduler( myworld, oport, parentScheduler ), 
  d_nextsignal("next condition"), d_nextmutex("next mutex"), dlbLock( "loadbalancer lock"),
  schedulerLock("scheduler lock"), recvLock("MPI receive Lock")
{

}

void
ThreadedMPIScheduler2::problemSetup(const ProblemSpecP& prob_spec,
                           SimulationStateP& state)
{
  //default taskReadyQueueAlg
  taskQueueAlg_ = MostMessages;
  string taskQueueAlg = "MostMessages";

  ProblemSpecP params = prob_spec->findBlock("Scheduler");
  if(params){
    params->get("taskReadyQueueAlg", taskQueueAlg);
    if (taskQueueAlg == "FCFS") 
      taskQueueAlg_ =  FCFS;
    else if (taskQueueAlg == "Random")
      taskQueueAlg_ =  Random;
    else if (taskQueueAlg == "Stack")
      taskQueueAlg_ =  Stack;
    else if (taskQueueAlg == "MostChildren")
      taskQueueAlg_ =  MostChildren;
    else if (taskQueueAlg == "LeastChildren")
      taskQueueAlg_ =  LeastChildren;
    else if (taskQueueAlg == "MostAllChildren")
      taskQueueAlg_ =  MostChildren;
    else if (taskQueueAlg == "LeastAllChildren")
      taskQueueAlg_ =  LeastChildren;
    else if (taskQueueAlg == "MostL2Children")
      taskQueueAlg_ =  MostL2Children;
    else if (taskQueueAlg == "LeastL2Children")
      taskQueueAlg_ =  LeastL2Children;
    else if (taskQueueAlg == "MostMessages")
      taskQueueAlg_ =  MostMessages;
    else if (taskQueueAlg == "LeastMessages")
      taskQueueAlg_ =  LeastMessages;
    else if (taskQueueAlg == "PatchOrder")
      taskQueueAlg_ =  PatchOrder;
    else if (taskQueueAlg == "PatchOrderRandom")
      taskQueueAlg_ =  PatchOrderRandom;
  }
  if (d_myworld->myrank() == 0)
      cout << "   Using \"" << taskQueueAlg <<  "\" Algorithm" <<endl;

  numThreads_ = Uintah::Parallel::getNumThreads() - 1;
  if (numThreads_ < 1) {
    if (d_myworld->myrank() == 0) {
	    cerr << "Error: no thread number specified" << endl;
	    throw ProblemSetupException("This scheduler requires number of threads > 1, use  -nthreads <num> ", __FILE__, __LINE__);
     }
  } else if (numThreads_ > MAX_THREADS) {
    if (d_myworld->myrank() == 0) {
	    cerr << "Error: thread number too large" << endl;
	    throw ProblemSetupException("Too many number of threads. Try to increase MAX_THREADS and recompile.", __FILE__, __LINE__);
    }
  }
  
  if ( d_myworld->myrank()==0){
    cout <<"   WARNING: Multi-thread/MPI hybrid scheduler is EXPERIMENTAL, not all tasks are thread safe yet." << endl;
    cout <<"   Creating " << numThreads_ << " more threads for scheduling and task execution."<< endl;
  }

 /* d_nextsignal = scinew ConditionVariable("NextCondition");
  d_nextmutex = scinew Mutex("NextMutex");*/
  char name[1024];
  
  for( int i = 0; i < numThreads_; i++ ){
    SchedulerWorker * worker = scinew SchedulerWorker( this, i );
    t_worker[i] = worker;
    sprintf( name, "Computing Worker %d-%d",
	     Parallel::getRootProcessorGroup()->myrank(), i );
    Thread * t = scinew Thread( worker, name );
    t_thread[i] = t;
    //t->detach();
  }
  
  log.problemSetup(prob_spec);
  SchedulerCommon::problemSetup(prob_spec, state);
  if (affinity.active()) Thread::self()->set_affinity(0); //bind main thread to cpu 0

}


ThreadedMPIScheduler2::~ThreadedMPIScheduler2()
{
  for( int i = 0; i < numThreads_; i++ ){
    t_worker[i]->d_runmutex.lock();
    t_worker[i]->quit();
    t_worker[i]->d_runsignal.conditionSignal();
    t_worker[i]->d_runmutex.unlock();
    t_thread[i]->setCleanupFunction( NULL );
    t_thread[i]->join();
  }

  if (timeout.active()) {
    timingStats.close();
    if (d_myworld->myrank() == 0) {
      avgStats.close();
      maxStats.close();
    }
  }
}

SchedulerP
ThreadedMPIScheduler2::createSubScheduler()
{
  ThreadedMPIScheduler2* newsched = scinew ThreadedMPIScheduler2(d_myworld, m_outPort, this);
  newsched->d_sharedState = d_sharedState;
  UintahParallelPort* lbp = getPort("load balancer");
  newsched->attachPort("load balancer", lbp);
  newsched->d_sharedState=d_sharedState;
  return newsched;
}

void
ThreadedMPIScheduler2::initiateTask( DetailedTask          * task,
			    bool only_old_recvs, int abort_point, int iteration)
{
  MALLOC_TRACE_TAG_SCOPE("MPIScheduler::initiateTask");
  TAU_PROFILE("MPIScheduler::initiateTask()", " ", TAU_USER); 

  postMPIRecvs(task, only_old_recvs, abort_point, iteration);
  if(only_old_recvs) {
    return;
  }
} // end initiateTask()


void
ThreadedMPIScheduler2::runTask( DetailedTask         * task, int iteration, int t_id /*=0*/)
{
  TAU_PROFILE("ThreadedMPIScheduler2::runTask()", " ", TAU_USER); 

  if(waitout.active())
  {
    waittimes[task->getTask()->getName()]+=CurrentWaitTime;
    CurrentWaitTime=0;
  }

  double taskstart = Time::currentSeconds();
  
  if (trackingVarsPrintLocation_ & SchedulerCommon::PRINT_BEFORE_EXEC) {
    printTrackedVars(task, SchedulerCommon::PRINT_BEFORE_EXEC);
  }

  vector<DataWarehouseP> plain_old_dws(dws.size());
  for(int i=0;i<(int)dws.size();i++)
    plain_old_dws[i] = dws[i].get_rep();
  //const char* tag = AllocatorSetDefaultTag(task->getTask()->getName());

  task->doit(d_myworld, dws, plain_old_dws);
  //AllocatorSetDefaultTag(tag);

  if (trackingVarsPrintLocation_ & SchedulerCommon::PRINT_AFTER_EXEC) {
    printTrackedVars(task, SchedulerCommon::PRINT_AFTER_EXEC);
  }

  double dtask = Time::currentSeconds()-taskstart;
 
  dlbLock.lock();
  if(execout.active())
  {
    exectimes[task->getTask()->getName()]+=dtask;
  }

  //if i do not have a sub scheduler 
  if(!task->getTask()->getHasSubScheduler())
  {
    //add my task time to the total time
    mpi_info_.totaltask += dtask;  
    //if(d_myworld->myrank()==0)
    //  cout << "adding: " << dtask << " to counters, new total: " << mpi_info_.totaltask << endl;
    if(!d_sharedState->isCopyDataTimestep() && task->getTask()->getType()!=Task::Output)
    {
      //if(d_myworld->myrank()==0 && task->getPatches()!=0)
      //  cout << d_myworld->myrank() << " adding: " << task->getTask()->getName() << " to profile:" << dtask << " on patches:" << *(task->getPatches()) << endl;
      //add contribution for patchlist
      getLoadBalancer()->addContribution(task,dtask);
    }
  }
  dlbLock.unlock();

  postMPISends( task, iteration, t_id );
  task->done(dws); // should this be timed with taskstart? - BJW
  double teststart = Time::currentSeconds();

 // sendsLock.lock(); // Dd... could do better?
  sends_[t_id].testsome( d_myworld );
 // sendsLock.unlock(); // Dd... could do better?


  mpi_info_.totaltestmpi += Time::currentSeconds() - teststart;
  
 
  if(parentScheduler) //add my timings to the parent scheduler
  {
  //  if(d_myworld->myrank()==0)
  //    cout << "adding: " << mpi_info_.totaltask << " to parent counters, new total: " << parentScheduler->mpi_info_.totaltask << endl;
    parentScheduler->mpi_info_.totaltask+=mpi_info_.totaltask;
    parentScheduler->mpi_info_.totaltestmpi+=mpi_info_.totaltestmpi;
    parentScheduler->mpi_info_.totalrecv+=mpi_info_.totalrecv;
    parentScheduler->mpi_info_.totalsend+=mpi_info_.totalsend;
    parentScheduler->mpi_info_.totalwaitmpi+=mpi_info_.totalwaitmpi;
    parentScheduler->mpi_info_.totalreduce+=mpi_info_.totalreduce;
  }


} // end runTask()

void
ThreadedMPIScheduler2::execute(int tgnum /*=0*/, int iteration /*=0*/)
{
  if (d_sharedState->isCopyDataTimestep()) {
    MPIScheduler::execute(tgnum, iteration);
    return;
  }
  MALLOC_TRACE_TAG_SCOPE("ThreadedMPIScheduler2::execute");
  TAU_PROFILE("ThreadedMPIScheduler2::execute()", " ", TAU_USER); 

  TAU_PROFILE_TIMER(reducetimer, "Reductions", "[ThreadedMPIScheduler2::execute()] " , TAU_USER); 
  TAU_PROFILE_TIMER(sendtimer, "Send Dependency", "[ThreadedMPIScheduler2::execute()] " , TAU_USER); 
  TAU_PROFILE_TIMER(recvtimer, "Recv Dependency", "[ThreadedMPIScheduler2::execute()] " , TAU_USER); 
  TAU_PROFILE_TIMER(outputtimer, "Task Graph Output", "[ThreadedMPIScheduler2::execute()] ", 
      TAU_USER); 
  TAU_PROFILE_TIMER(testsometimer, "Test Some", "[ThreadedMPIScheduler2::execute()] ", 
      TAU_USER); 
  TAU_PROFILE_TIMER(finalwaittimer, "Final Wait", "[ThreadedMPIScheduler2::execute()] ", 
      TAU_USER); 
  TAU_PROFILE_TIMER(sorttimer, "Topological Sort", "[ThreadedMPIScheduler2::execute()] ", 
      TAU_USER); 
  TAU_PROFILE_TIMER(sendrecvtimer, "Initial Send Recv", "[ThreadedMPIScheduler2::execute()] ", 
      TAU_USER); 

  ASSERTRANGE(tgnum, 0, (int)graphs.size());
  TaskGraph* tg = graphs[tgnum];
  tg->setIteration(iteration);
  currentTG_ = tgnum;

  if (graphs.size() > 1) {
    // tg model is the multi TG model, where each graph is going to need to
    // have its dwmap reset here (even with the same tgnum)
    tg->remapTaskDWs(dwmap);
  }

  dts = tg->getDetailedTasks();

  if(dts == 0){
    if (d_myworld->myrank() == 0)
      cerr << "ThreadedMPIScheduler2 skipping execute, no tasks\n";
    return;
  }

  //ASSERT(pg_ == 0);
  //pg_ = pg;

  ntasks = dts->numLocalTasks();
  dts->initializeScrubs(dws, dwmap);
  dts->initTimestep();

  for (int i = 0; i < ntasks; i++)
    dts->localTask(i)->resetDependencyCounts();

  if(timeout.active()){
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

  numTasksDone = 0;


  abort=false;
  abort_point = 987654;

  if (reloc_new_posLabel_ && dws[dwmap[Task::OldDW]] != 0)
    dws[dwmap[Task::OldDW]]->exchangeParticleQuantities(dts, getLoadBalancer(), reloc_new_posLabel_, iteration);

  TAU_PROFILE_TIMER(doittimer, "Task execution", 
      "[ThreadedMPIScheduler2::execute() loop] ", TAU_USER); 
  TAU_PROFILE_START(doittimer);

  curriteration=iteration;
  currphase=0;
  numPhase=tg->getNumTaskPhases();
  phaseTasks.clear();
  phaseTasks.resize(numPhase, 0);
  phaseTasksDone.clear();
  phaseTasksDone.resize(numPhase, 0);
  phaseSyncTask.clear();
  phaseSyncTask.resize(numPhase, NULL);
  dts->setTaskPriorityAlg(taskQueueAlg_ );
  for (int i = 0; i < ntasks; i++){
    phaseTasks[dts->localTask(i)->getTask()->d_phase]++;
  }
  
  if( dbg.active()) {
    cerrLock.lock();
    dbg << me << " Executing " << dts->numTasks() << " tasks (" 
      << ntasks << " local)"<< endl;
    cerrLock.unlock();
  }

  static int totaltasks;

  taskdbg << d_myworld->myrank() << " Switched to Task Phase " << currphase  << " , total task  " <<  phaseTasks[currphase] << endl;
  for (int i=0; i < numThreads_; i++){
      t_worker[i]->resetWaittime(Time::currentSeconds()); //reset wait time counter
      /*sending signal to threads to wake them up*/
      t_worker[i]->d_runmutex.lock();  
      t_worker[i]->d_idle=false;
      t_worker[i]->d_runsignal.conditionSignal();
      t_worker[i]->d_runmutex.unlock();
  }

  /*control loop for all tasks of task graph*/
  runTasks(0);
   // end while( numTasksDone < ntasks )
  TAU_PROFILE_STOP(doittimer);
  
// wait for all tasks to finish 
  wait_till_all_done();
  //if any thread is busy, conditional wait here
  d_nextmutex.lock();
  while ( getAviableThreadNum() < numThreads_) {
    d_nextsignal.wait(d_nextmutex);
  }
  d_nextmutex.unlock();

  //if (me==0)
  //  cout <<"AviableThreads : " << getAviableThreadNum()  << ", task worked: " << numTasksDone << endl;

  //if (d_generation > 2)
  //dws[dws.size()-2]->printParticleSubsets();

  if(queuelength.active())
  {
    float lengthsum=0;
    totaltasks += ntasks;
    // if (me == 0) cout << d_myworld->myrank() << " queue length histogram: ";
    for (unsigned int i=1; i<histogram.size(); i++)
    {
      // if (me == 0)cout << histogram[i] << " ";
      //cout << iter->first << ":" << iter->second << " ";
      lengthsum = lengthsum + i*histogram[i];
    }
    // if (me==0) cout << endl;
    float queuelength = lengthsum/totaltasks;
    float allqueuelength = 0;
    MPI_Reduce(&queuelength, &allqueuelength, 1 , MPI_FLOAT, MPI_SUM, 0, d_myworld->getComm());
    if (me == 0) cout  << "average queue length:" << allqueuelength/d_myworld->size() << endl;
  }

  if(timeout.active()){
    emitTime("MPI send time", mpi_info_.totalsendmpi);
    //emitTime("MPI Testsome time", mpi_info_.totaltestmpi);
    emitTime("Total send time", 
        mpi_info_.totalsend - mpi_info_.totalsendmpi - mpi_info_.totaltestmpi);
    emitTime("MPI recv time", mpi_info_.totalrecvmpi);
    emitTime("MPI wait time", mpi_info_.totalwaitmpi);
    emitTime("Total recv time", 
        mpi_info_.totalrecv - mpi_info_.totalrecvmpi - mpi_info_.totalwaitmpi);
    emitTime("Total task time", mpi_info_.totaltask);
    emitTime("Total MPI reduce time", mpi_info_.totalreducempi);
    //emitTime("Total reduction time", 
    //         mpi_info_.totalreduce - mpi_info_.totalreducempi);
    emitTime("Total comm time", 
        mpi_info_.totalrecv + mpi_info_.totalsend + mpi_info_.totalreduce);

    double time      = Time::currentSeconds();
    double totalexec = time - d_lasttime;

    d_lasttime = time;

    emitTime("Other excution time", totalexec - mpi_info_.totalsend -
        mpi_info_.totalrecv - mpi_info_.totaltask - mpi_info_.totalreduce);
  }

  if (d_sharedState != 0) { // subschedulers don't have a sharedState
    d_sharedState->taskExecTime += mpi_info_.totaltask - d_sharedState->outputTime; // don't count output time...
    d_sharedState->taskLocalCommTime += mpi_info_.totalrecv + mpi_info_.totalsend;
    d_sharedState->taskWaitCommTime += mpi_info_.totalwaitmpi;
    d_sharedState->taskGlobalCommTime += mpi_info_.totalreduce;
    for (int i=0; i < numThreads_; i++)
      d_sharedState->taskWaitThreadTime += t_worker[i]->getWaittime();
  }

  //if(timeout.active())
  //emitTime("final wait");
  if(restartable && tgnum == (int) graphs.size() -1) {
    // Copy the restart flag to all processors
    int myrestart = dws[dws.size()-1]->timestepRestarted();
    int netrestart;
    MPI_Allreduce(&myrestart, &netrestart, 1, MPI_INT, MPI_LOR,
        d_myworld->getComm());
    if(netrestart) {
      dws[dws.size()-1]->restartTimestep();
      if (dws[0])
        dws[0]->setRestarted();
    }
  }

  finalizeTimestep();


  log.finishTimestep();
  if(timeout.active() && !parentScheduler){ // only do on toplevel scheduler
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
    d_labels.push_back("NumPatches");
    d_labels.push_back("NumCells");
    d_labels.push_back("NumParticles");
    d_times.push_back(myPatches->size());
    d_times.push_back(numCells);
    d_times.push_back(numParticles);
    vector<double> d_totaltimes(d_times.size());
    vector<double> d_maxtimes(d_times.size());
    vector<double> d_avgtimes(d_times.size());
    double maxTask = -1, avgTask = -1;
    double maxComm = -1, avgComm = -1;
    double maxCell = -1, avgCell = -1;
    MPI_Reduce(&d_times[0], &d_totaltimes[0], (int)d_times.size(), MPI_DOUBLE,
        MPI_SUM, 0, d_myworld->getComm());
    MPI_Reduce(&d_times[0], &d_maxtimes[0], (int)d_times.size(), MPI_DOUBLE,
        MPI_MAX, 0, d_myworld->getComm());

    double total = 0, avgTotal = 0, maxTotal = 0;
    for(int i=0;i<(int)d_totaltimes.size();i++) {
      d_avgtimes[i] = d_totaltimes[i]/d_myworld->size();
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

      total+= d_times[i];
      avgTotal += d_avgtimes[i];
      maxTotal += d_maxtimes[i];
    }

    // to not duplicate the code
    vector <ofstream*> files;
    vector <vector<double>* > data;
    files.push_back(&timingStats);
    data.push_back(&d_times);

    if (me == 0) {
      files.push_back(&avgStats);
      files.push_back(&maxStats);
      data.push_back(&d_avgtimes);
      data.push_back(&d_maxtimes);
    }

    for (unsigned file = 0; file < files.size(); file++) {
      ofstream& out = *files[file];
      out << "Timestep " << d_sharedState->getCurrentTopLevelTimeStep() << endl;
      for(int i=0;i<(int)(*data[file]).size();i++){
        out << "ThreadedMPIScheduler2: " << d_labels[i] << ": ";
        int len = (int)(strlen(d_labels[i])+strlen("ThreadedMPIScheduler2: ")+strlen(": "));
        for(int j=len;j<55;j++)
          out << ' ';
        double percent;
        if (strncmp(d_labels[i], "Num", 3) == 0)
          percent = d_totaltimes[i] == 0 ? 100 : (*data[file])[i]/d_totaltimes[i]*100;
        else
          percent = (*data[file])[i]/total*100;
        out << (*data[file])[i] <<  " (" << percent << "%)\n";
      }
      out << endl << endl;
    }

    if (me == 0) {
      timeout << "  Avg. exec: " << avgTask << ", max exec: " << maxTask << " = " << (1-avgTask/maxTask)*100 << " load imbalance (exec)%\n";
      timeout << "  Avg. comm: " << avgComm << ", max comm: " << maxComm << " = " << (1-avgComm/maxComm)*100 << " load imbalance (comm)%\n";
      timeout << "  Avg.  vol: " << avgCell << ", max  vol: " << maxCell << " = " << (1-avgCell/maxCell)*100 << " load imbalance (theoretical)%\n";
    }
    double time = Time::currentSeconds();
    //double rtime=time-d_lasttime;
    d_lasttime=time;
    //timeout << "ThreadedMPIScheduler2: TOTAL                                    "
    //        << total << '\n';
    //timeout << "ThreadedMPIScheduler2: time sum reduction (one processor only): " 
    //        << rtime << '\n';
  }

  if(execout.active())
  {
    static int count=0;

    if(++count%10==0)
    {
      ofstream fout;
      char filename[100];
      sprintf(filename,"exectimes.%d.%d",d_myworld->size(),d_myworld->myrank());
      fout.open(filename);

      for(map<string,double>::iterator iter=exectimes.begin();iter!=exectimes.end();iter++)
        fout << fixed << d_myworld->myrank() << ": TaskExecTime: " << iter->second << " Task:" << iter->first << endl;
      fout.close();
      //exectimes.clear();
    }
  }
  if(waitout.active())
  {
    static int count=0;

    //only output the wait times every so many timesteps
    if(++count%100==0)
    {
      for(map<string,double>::iterator iter=waittimes.begin();iter!=waittimes.end();iter++)
        waitout << fixed << d_myworld->myrank() << ": TaskWaitTime(TO): " << iter->second << " Task:" << iter->first << endl;

      for(map<string,double>::iterator iter=DependencyBatch::waittimes.begin();iter!=DependencyBatch::waittimes.end();iter++)
        waitout << fixed << d_myworld->myrank() << ": TaskWaitTime(FROM): " << iter->second << " Task:" << iter->first << endl;

      waittimes.clear();
      DependencyBatch::waittimes.clear();
    }
  }

  if( dbg.active()) {
    dbg << me << " ThreadedMPIScheduler2 finished\n";
  }
  //pg_ = 0;
}

void 
ThreadedMPIScheduler2::runTasks(int t_id)
{
  while( numTasksDone < ntasks) {
    DetailedTask* readyTask=NULL;
    DetailedTask* initTask=NULL;
    int processMPIs=0;
    bool havework=false;

   //Part 1. Check if anything this thread can do concurrently
   //        If can update the scheduler counters. 
    schedulerLock.lock();
    while (!havework) {
      if ((phaseSyncTask[currphase]!= NULL) && (phaseTasksDone[currphase] == phaseTasks[currphase]-1)){ 
        readyTask= phaseSyncTask[currphase];
        havework=true;
        numTasksDone++;
        phaseTasksDone[readyTask->getTask()->d_phase]++;
        while (phaseTasks[currphase] == phaseTasksDone[currphase] && currphase+1 < numPhase) currphase++;
        break;
      }
      else if (dts->numExternalReadyTasks() >0){
        readyTask=dts->getNextExternalReadyTask();
        if (readyTask!=NULL) {
          havework=true;
          numTasksDone++;
          phaseTasksDone[readyTask->getTask()->d_phase]++;
          while (phaseTasks[currphase] == phaseTasksDone[currphase] && currphase+1 < numPhase) currphase++;
          break;
        }
      }
      else if (dts->numInternalReadyTasks() >0) {
         initTask=dts->getNextInternalReadyTask();
         if (initTask!=NULL) {
           if (initTask->getTask()->getType() == Task::Reduction || initTask->getTask()->usesMPI() ) {
             if (taskdbg.active()){
               cerrLock.lock();
                taskdbg << d_myworld->myrank() << " Task internal ready 1 " << *initTask << endl;
               cerrLock.unlock();
             }
             phaseSyncTask[initTask->getTask()->d_phase]= initTask;
             ASSERT(initTask->getRequires().size() == 0)
             initTask=NULL;
           } else if (initTask->getRequires().size() == 0 ) {  //if no ext deps,skip MPI sends
             initTask->markInitiated();
             initTask->checkExternalDepCount(); 
             initTask=NULL;
           } else { 
             havework=true;
             break;
           }
         }
      }
      else {
        processMPIs = pendingMPIRecvs();
        if (processMPIs>0){
          havework=true;
          break;
        }
      }
      if ( numTasksDone == ntasks){
        break;
      } 

    }
    schedulerLock.unlock();


   //Part 2. Concurrent Part

    if (initTask!=NULL){
      initiateTask(initTask, abort, abort_point, curriteration);
      if (taskdbg.active()){
        cerrLock.lock();
        taskdbg << d_myworld->myrank() << " Task internal ready 2 " << *initTask << " deps needed: " << initTask->getExternalDepCount() << endl;
        cerrLock.unlock();
     }
      initTask->markInitiated();
      initTask->checkExternalDepCount();
    }
    else if (readyTask!=NULL){
      if (taskdbg.active()){
        cerrLock.lock();
        taskdbg << d_myworld->myrank() << " Task external ready " << *readyTask <<endl;
        cerrLock.unlock();
      }
      if (readyTask->getTask()->getType() == Task::Reduction){
        initiateReduction(readyTask);
      } else {
        runTask(readyTask, curriteration, t_id);
      }
    } else if (processMPIs>0) {
          processMPIRecvs(TEST);
    } else {
      //This could only happen when finished all tasks
      ASSERT( numTasksDone == ntasks);  
    }

 } //end while tasks
}


struct CompareDep {
bool operator()(DependencyBatch* a, DependencyBatch* b)
{
  return a->messageTag < b->messageTag;
}
};
void
ThreadedMPIScheduler2::postMPIRecvs( DetailedTask * task, bool only_old_recvs, int abort_point, int iteration)
{
  MALLOC_TRACE_TAG_SCOPE("MPIScheduler::postMPIRecvs");
  double recvstart = Time::currentSeconds();
  TAU_PROFILE("MPIScheduler::postMPIRecvs()", " ", TAU_USER); 

  // Receive any of the foreign requires

  if( dbg.active()) {
    cerrLock.lock();dbg << d_myworld->myrank() << " postMPIRecvs - task " << *task << '\n';
    cerrLock.unlock();
  }

  if (trackingVarsPrintLocation_ & SchedulerCommon::PRINT_BEFORE_COMM)
    printTrackedVars(task, SchedulerCommon::PRINT_BEFORE_COMM);

  // sort the requires, so in case there is a particle send we receive it with
  // the right message tag

  vector<DependencyBatch*> sorted_reqs;
  map<DependencyBatch*, DependencyBatch*>::const_iterator iter = 
    task->getRequires().begin();
  for( ; iter != task->getRequires().end(); iter++) {
    sorted_reqs.push_back(iter->first);
  }
  CompareDep comparator;;
  sort(sorted_reqs.begin(), sorted_reqs.end(), comparator);
  vector<DependencyBatch*>::iterator sorted_iter = sorted_reqs.begin();
    recvLock.writeLock();
  for( ; sorted_iter != sorted_reqs.end(); sorted_iter++) {
    DependencyBatch* batch = *sorted_iter;

    // The first thread that calls this on the batch will return true
    // while subsequent threads calling this will block and wait for
    // that first thread to receive the data.

    task->incrementExternalDepCount();
    //cout << d_myworld->myrank() << " Add dep count to task " << *task << " for ext: " << *batch->fromTask << ": " << task->getExternalDepCount() << endl;
    if( !batch->makeMPIRequest() ) {
      //externalRecvs.push_back( batch ); // no longer necessary

      if( dbg.active() ) {
        cerrLock.lock();dbg << "Someone else already receiving it\n";
        cerrLock.unlock();
      }

      continue;
    }

    if(only_old_recvs){
      if(dbg.active()){
        dbg << "abort analysis: " << batch->fromTask->getTask()->getName()
          << ", so=" << batch->fromTask->getTask()->getSortedOrder()
          << ", abort_point=" << abort_point << '\n';
        if(batch->fromTask->getTask()->getSortedOrder() <= abort_point)
          dbg << "posting MPI recv for pre-abort message " 
            << batch->messageTag << '\n';
      }
      if(!(batch->fromTask->getTask()->getSortedOrder() <= abort_point)) {
        continue;
      }
    }
    

    // Prepare to receive a message
    BatchReceiveHandler* pBatchRecvHandler = 
      scinew BatchReceiveHandler(batch);
    PackBufferInfo* p_mpibuff = 0;
#ifdef USE_PACKING
    p_mpibuff = scinew PackBufferInfo();
    PackBufferInfo& mpibuff = *p_mpibuff;
#else
    BufferInfo mpibuff;
#endif

    ostringstream ostr;
    ostr.clear();
    // Create the MPI type
    for(DetailedDep* req = batch->head; req != 0; req = req->next){
      OnDemandDataWarehouse* dw = dws[req->req->mapDataWarehouse()].get_rep();
      //dbg.setActive(req->req->lookInOldTG );
      if ((req->condition == DetailedDep::FirstIteration && iteration > 0) || 
          (req->condition == DetailedDep::SubsequentIterations && iteration == 0) ||
          (notCopyDataVars_.count(req->req->var->getName()) > 0  )){
        // See comment in DetailedDep about CommCondition

        dbg << d_myworld->myrank() << "   Ignoring conditional receive for " << *req << endl;
        continue;
      }
      // if we send/recv to an output task, don't send/recv if not an output timestep
      if (req->toTasks.front()->getTask()->getType() == Task::Output && !oport_->isOutputTimestep() && !oport_->isCheckpointTimestep()) {
        dbg << d_myworld->myrank() << "   Ignoring non-output-timestep receive for " << *req << endl;
        continue;
      }
      if (dbg.active()) {
        ostr << *req << ' ';
        dbg << d_myworld->myrank() << " <-- receiving " << *req << ", ghost: " << req->req->gtype << ", " << req->req->numGhostCells << " into dw " << dw->getID() << '\n';
      }

      OnDemandDataWarehouse* posDW;
      const VarLabel* posLabel;

      // the load balancer is used to determine where data was in the old dw on the prev timestep
      // pass it in if the particle data is on the old dw
      LoadBalancer* lb = 0;
      if(!reloc_new_posLabel_ && parentScheduler){
        posDW = dws[req->req->task->mapDataWarehouse(Task::ParentOldDW)].get_rep();
        posLabel = parentScheduler->reloc_new_posLabel_;
      } else {
        // on an output task (and only on one) we require particle variables from the NewDW
        if (req->toTasks.front()->getTask()->getType() == Task::Output)
          posDW = dws[req->req->task->mapDataWarehouse(Task::NewDW)].get_rep();
        else {
          posDW = dws[req->req->task->mapDataWarehouse(Task::OldDW)].get_rep();
          lb = getLoadBalancer();
        }
        posLabel = reloc_new_posLabel_;
      }

      MPIScheduler* top = this;
      while(top->parentScheduler) top = top->parentScheduler;

      dw->recvMPI(batch, mpibuff, posDW, req, lb);

      if (!req->isNonDataDependency()) {
        graphs[currentTG_]->getDetailedTasks()->setScrubCount(req->req, req->matl, req->fromPatch, dws);
      }

    }

    // Post the receive
    if(mpibuff.count()>0){

      ASSERT(batch->messageTag > 0);
      double start = Time::currentSeconds();
      void* buf;
      int count;
      MPI_Datatype datatype;
#ifdef USE_PACKING
      mpibuff.get_type(buf, count, datatype, d_myworld->getComm());
#else
      mpibuff.get_type(buf, count, datatype);
#endif
      //only recieve message if size is greater than zero
      //we need this empty message to enforce modify after read dependencies 
      //if(count>0)
      //{
        int from = batch->fromTask->getAssignedResourceIndex();
        ASSERTRANGE(from, 0, d_myworld->size());
        MPI_Request requestid;

        if( dbg.active()) {
          cerrLock.lock();
          //if (d_myworld->myrank() == 40 && d_sharedState->getCurrentTopLevelTimeStep() == 2 && from == 43)
          dbg << d_myworld->myrank() << " Recving message number " << batch->messageTag 
            << " from " << from << ": " << ostr.str() << "\n"; cerrLock.unlock();
          //dbg.setActive(false);
        }

        //if (d_myworld->myrank() == 40 && d_sharedState->getCurrentTopLevelTimeStep() == 2 && from == 43) 
        mpidbg << d_myworld->myrank() << " Posting receive for message number " << batch->messageTag << " from " << from << ", length=" << count << "\n";
        MPI_Irecv(buf, count, datatype, from, batch->messageTag,
            d_myworld->getComm(), &requestid);
        int bytes = count;
        recvs_.add(requestid, bytes,
            scinew ReceiveHandler(p_mpibuff, pBatchRecvHandler),
            ostr.str(), batch->messageTag);
        mpi_info_.totalrecvmpi += Time::currentSeconds() - start;
      /*}
      else
      {
        //no message was sent so clean up buffer and handler
        delete p_mpibuff;
        delete pBatchRecvHandler;
      }*/
    }
    else {
      // Nothing really need to be received, but let everyone else know
      // that it has what is needed (nothing).
      batch->received(d_myworld);
#ifdef USE_PACKING
      // otherwise, these will be deleted after it receives and unpacks
      // the data.
      delete p_mpibuff;
      delete pBatchRecvHandler;
#endif	        
    }
  } // end for
    recvLock.writeUnlock();

  double drecv = Time::currentSeconds() - recvstart;
  mpi_info_.totalrecv += drecv;


} // end postMPIRecvs()

int 
ThreadedMPIScheduler2::pendingMPIRecvs()
{
  int num=0;
  recvLock.readLock();
  num = recvs_.numRequests();
  recvLock.readUnlock();
  return num;
}

void
ThreadedMPIScheduler2::processMPIRecvs(int how_much)
{
  MALLOC_TRACE_TAG_SCOPE("MPIScheduler::processMPIRecvs");
  TAU_PROFILE("MPIScheduler::processMPIRecvs()", " ", TAU_USER);

  // Should only have external receives in the MixedScheduler version which
  // shouldn't use this function.
  // ASSERT(outstandingExtRecvs.empty());
  //if (recvs_.numRequests() == 0) return;
  double start = Time::currentSeconds();
  recvLock.writeLock();
  switch (how_much) {
  case TEST:
    recvs_.testsome(d_myworld);
    break;
  case WAIT_ONCE:
    mpidbg << d_myworld->myrank() << " Start waiting once...\n";
    recvs_.waitsome(d_myworld);
    mpidbg << d_myworld->myrank() << " Done  waiting once...\n";
    break;
  case WAIT_ALL:
    // This will allow some receives to be "handled" by their
    // AfterCommincationHandler while waiting for others.  
    mpidbg << d_myworld->myrank() << "  Start waiting...\n";
    while( (recvs_.numRequests() > 0)) {
      bool keep_waiting = recvs_.waitsome(d_myworld);
      if (!keep_waiting)
        break;
    }
    mpidbg << d_myworld->myrank() << "  Done  waiting...\n";
  }
  recvLock.writeUnlock();
  mpi_info_.totalwaitmpi+=Time::currentSeconds()-start;
  CurrentWaitTime+=Time::currentSeconds()-start;

} // end processMPIRecvs()

void
ThreadedMPIScheduler2::postMPISends( DetailedTask         * task, int iteration, int t_id )
{
  MALLOC_TRACE_TAG_SCOPE("ThreadedMPIScheduler2::postMPISends");
  double sendstart = Time::currentSeconds();
  if( dbg.active()) {
    cerrLock.lock();dbg << d_myworld->myrank() << " postMPISends - task " << *task << '\n';
    cerrLock.unlock();
  }

  // Send data to dependendents
  for(DependencyBatch* batch = task->getComputes();
      batch != 0; batch = batch->comp_next){

    // Prepare to send a message
#ifdef USE_PACKING
    PackBufferInfo mpibuff;
#else
    BufferInfo mpibuff;
#endif
    // Create the MPI type
    int to = batch->toTasks.front()->getAssignedResourceIndex();
    ASSERTRANGE(to, 0, d_myworld->size());
    ostringstream ostr;
    ostr.clear();
    for(DetailedDep* req = batch->head; req != 0; req = req->next){
      if ((req->condition == DetailedDep::FirstIteration && iteration > 0) || 
          (req->condition == DetailedDep::SubsequentIterations && iteration == 0) ||
          (notCopyDataVars_.count(req->req->var->getName()) > 0  )){
        // See comment in DetailedDep about CommCondition
        if( dbg.active()) 
        dbg << d_myworld->myrank() << "   Ignoring conditional send for " << *req << endl;
        continue;
      }
      // if we send/recv to an output task, don't send/recv if not an output timestep
      if (req->toTasks.front()->getTask()->getType() == Task::Output && 
          !oport_->isOutputTimestep() && !oport_->isCheckpointTimestep()) {
        if( dbg.active())
        dbg << d_myworld->myrank() << "   Ignoring non-output-timestep send for " << *req << endl;
        continue;
      }
      OnDemandDataWarehouse* dw = dws[req->req->mapDataWarehouse()].get_rep();

      //dbg.setActive(req->req->lookInOldTG);
      if (dbg.active()) {
        ostr << *req << ' ';
        //if (to == 40 && d_sharedState->getCurrentTopLevelTimeStep() == 2 && d_myworld->myrank() == 43)
        dbg << d_myworld->myrank() << " --> sending " << *req << ", ghost: " << req->req->gtype << ", " << req->req->numGhostCells << " from dw " << dw->getID() << '\n';
      }
      const VarLabel* posLabel;
      OnDemandDataWarehouse* posDW;

      // the load balancer is used to determine where data was in the old dw on the prev timestep -
      // pass it in if the particle data is on the old dw
      LoadBalancer* lb = 0;

      if(!reloc_new_posLabel_ && parentScheduler){
	posDW = dws[req->req->task->mapDataWarehouse(Task::ParentOldDW)].get_rep();
	posLabel = parentScheduler->reloc_new_posLabel_;
      } else {
        // on an output task (and only on one) we require particle variables from the NewDW
        if (req->toTasks.front()->getTask()->getType() == Task::Output)
          posDW = dws[req->req->task->mapDataWarehouse(Task::NewDW)].get_rep();
        else {
          posDW = dws[req->req->task->mapDataWarehouse(Task::OldDW)].get_rep();
          lb = getLoadBalancer();
        }
	posLabel = reloc_new_posLabel_;
      }
      MPIScheduler* top = this;
      while(top->parentScheduler) top = top->parentScheduler;

      dw->sendMPI(batch, posLabel, mpibuff, posDW, req, lb);
    }
    // Post the send
    if(mpibuff.count()>0){
      ASSERT(batch->messageTag > 0);
      double start = Time::currentSeconds();
      void* buf;
      int count;
      MPI_Datatype datatype;
     
#ifdef USE_PACKING
      mpibuff.get_type(buf, count, datatype, d_myworld->getComm());
      mpibuff.pack(d_myworld->getComm(), count);
#else
      mpibuff.get_type(buf, count, datatype);
#endif

      //only send message if size is greather than zero
      //we need this empty message to enforce modify after read dependencies 
      //if(count>0)
      //{
        if( dbg.active()) {
          cerrLock.lock();
          //if (to == 40 && d_sharedState->getCurrentTopLevelTimeStep() == 2 && d_myworld->myrank() == 43)
          dbg << d_myworld->myrank() << " Sending message number " << batch->messageTag 
            << " to " << to << ": " << ostr.str() << "\n"; 
          cerrLock.unlock();
          //dbg.setActive(false);
        }
        //if (to == 40 && d_sharedState->getCurrentTopLevelTimeStep() == 2 && d_myworld->myrank() == 43)
        if( mpidbg.active()) 
        mpidbg <<d_myworld->myrank() << " Sending message number " << batch->messageTag << ", to " << to << ", length: " << count << "\n"; 

        numMessages_++;
        int typeSize;

        MPI_Type_size(datatype,&typeSize);
        messageVolume_+=count*typeSize;

        MPI_Request requestid;
        MPI_Isend(buf, count, datatype, to, batch->messageTag,
            d_myworld->getComm(), &requestid);
        int bytes = count;

        //sendsLock.lock(); // Dd: ??
        sends_[t_id].add( requestid, bytes, mpibuff.takeSendlist(), ostr.str(), batch->messageTag );
        //sendsLock.unlock(); // Dd: ??
        mpi_info_.totalsendmpi += Time::currentSeconds() - start;
      //}
    }
  } // end for (DependencyBatch * batch = task->getComputes() )
  double dsend = Time::currentSeconds()-sendstart;
  mpi_info_.totalsend += dsend;

} // end postMPISends();


int 
ThreadedMPIScheduler2::getAviableThreadNum()
{
  int num =0;
  for (int i=0; i < numThreads_; i++) if (t_worker[i]->d_idle) num++;
  return num;
}

/** SchedulerWorker Thread Methods***/
SchedulerWorker::SchedulerWorker(ThreadedMPIScheduler2* scheduler, int id ) : 
   d_id( id ), d_scheduler(scheduler),  d_idle(true),
   d_runmutex("run mutex"),  d_runsignal("run condition"), d_quit(false),
   d_waittime(0.0),d_waitstart(0.0),  d_rank(scheduler->getProcessorGroup()->myrank())
{
  d_runmutex.lock();
}

void
SchedulerWorker::run()
{
  threaddbg << "Binding thread id " << d_id+1 << " to cpu " << d_id+1 << endl;
  Thread::self()->set_myid(d_id+1);
  if (affinity.active()) Thread::self()->set_affinity(d_id+1);
  while(true) {
    //wait for main thread signal
    d_runsignal.wait(d_runmutex);
    d_runmutex.unlock();
    d_waittime += Time::currentSeconds()-d_waitstart;
    if (d_quit) {
      if( taskdbg.active() ) {
      cerrLock.lock();
       taskdbg << "Worker " << d_rank  << "-" << d_id 
        << "quiting   " << "\n";
      cerrLock.unlock();
      }
      return;
    }

    if( taskdbg.active() ) {
      cerrLock.lock();
      taskdbg << "Worker " << d_rank  << "-" << d_id 
        << ": executeTasks \n";
      cerrLock.unlock();
    }

    d_scheduler->runTasks(d_id+1);

    if( taskdbg.active() ) {
      cerrLock.lock();
      taskdbg << "Worker " << d_rank << "-" << d_id 
        << ": finishTasks   \n";
      cerrLock.unlock();
    }

    //signal main thread for next group of tasks
    d_scheduler->d_nextmutex.lock();
    d_runmutex.lock();
    d_waitstart = Time::currentSeconds();
    d_idle=true;
    d_scheduler->d_nextsignal.conditionSignal();
    d_scheduler->d_nextmutex.unlock();
  }
}

double SchedulerWorker::getWaittime()
{
    return  d_waittime;
}

void SchedulerWorker::resetWaittime(double start)
{
    d_waitstart  = start;
    d_waittime = 0.0;
}
