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

#ifndef CCA_COMPONENTS_SCHEDULERS_DETAILEDTASKS_EXP_H
#define CCA_COMPONENTS_SCHEDULERS_DETAILEDTASKS_EXP_H

#include <CCA/Components/Schedulers/DetailedTask_Exp.hpp>
#include <CCA/Components/Schedulers/DetailedDep_Exp.hpp>
#include <CCA/Components/Schedulers/DependencyBatch_Exp.hpp>

#include <CCA/Components/Schedulers/DWDatabase.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>

#include <Core/Containers/FastHashTable.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/PSPatchMatlGhostRange.h>
#include <Core/Grid/Variables/ScrubItem.h>

#include <sci_defs/cuda_defs.h>

#ifdef HAVE_CUDA
#include <CCA/Components/Schedulers/GPUGridVariableGhosts.h>
#include <CCA/Components/Schedulers/GPUGridVariableInfo.h>
#endif

#include <list>
#include <queue>
#include <map>
#include <mutex>
#include <set>
#include <vector>


namespace Uintah {

class ProcessorGroup;
class DataWarehouse;
class TaskGraph;
class SchedulerCommon;


using ParticleExchangeVar = std::map<int, std::set<PSPatchMatlGhostRange> >;
using ScrubCountTable     = Uintah::FastHashTable<ScrubItem>;

//_______________________________________________________________________________________________

//_______________________________________________________________________________________________
enum QueueAlg {
   FCFS
 , Stack
 , Random
 , MostChildren
 , LeastChildren
 , MostAllChildren
 , LeastAllChildren
 , MostMessages
 , LeastMessages
 , MostL2Children
 , LeastL2Children
 , CritialPath
 , PatchOrder
 , PatchOrderRandom
};


//_______________________________________________________________________________________________
//
//    DetailedTaskPriorityComparison
//_______________________________________________________________________________________________
//
class DetailedTaskPriorityComparison
{
public:

  bool operator()( DetailedTask*& ltask, DetailedTask*& rtask );
};
//_______________________________________________________________________________________________




//_______________________________________________________________________________________________
//
//    DetailedTasks
//_______________________________________________________________________________________________
//
class DetailedTasks {

public:

  DetailedTasks(        SchedulerCommon * sc
      ,  const ProcessorGroup  * pg
      ,        DetailedTasks   * first
      ,  const TaskGraph       * taskgraph
      ,  const std::set<int>   & neighborhood_processors
      ,        bool              mustConsiderInternalDependencies = false
      );

  ~DetailedTasks();

  void add( DetailedTask* task );

  void makeDWKeyDatabase();

  void copyoutDWKeyDatabase( OnDemandDataWarehouseP dws ) { dws->copyKeyDB(varKeyDB, levelKeyDB); }

  int numTasks() const { return (int)tasks_.size(); }

  DetailedTask* getTask( int i ) { return tasks_[i]; }

  void assignMessageTags( int me );

  void initializeScrubs( std::vector<OnDemandDataWarehouseP>& dws, int dwmap[] );

  void possiblyCreateDependency(        DetailedTask                       * from
      ,        Task::Dependency                   * comp
      ,  const Patch                              * fromPatch
      ,        DetailedTask                       * to
      ,        Task::Dependency                   * req
      ,  const Patch                              * toPatch
      ,        int                                  matl
      ,  const IntVector                          & low
      ,  const IntVector                          & high
      ,        DetailedDep::CommCondition           cond
      );

  DetailedTask* getOldDWSendTask(int proc);

  void logMemoryUse( std::ostream& out, unsigned long& total, const std::string& tag );

  void initTimestep();

  void computeLocalTasks( int me );

  int numLocalTasks() const { return (int)localtasks_.size(); }

  DetailedTask* localTask( int idx ) { return localtasks_[idx]; }

  void emitEdges( ProblemSpecP edgesElement, int rank );

  DetailedTask* getNextInternalReadyTask();

  int numInternalReadyTasks();

  DetailedTask* getNextExternalReadyTask();

  int numExternalReadyTasks();

  void createScrubCounts();

  bool mustConsiderInternalDependencies() { return mustConsiderInternalDependencies_; }

  unsigned long getCurrentDependencyGeneration() { return currentDependencyGeneration_; }

  const TaskGraph* getTaskGraph() const { return taskgraph_; }

  void setScrubCount( const Task::Dependency                     * req
      ,        int                                   matl
      ,  const Patch                               * patch
      ,        std::vector<OnDemandDataWarehouseP> & dws
      );

  int getExtraCommunication() { return extraCommunication_; }

  ParticleExchangeVar& getParticleSends() { return particleSends_; }

  ParticleExchangeVar& getParticleRecvs() { return particleRecvs_; }

  void setTaskPriorityAlg( QueueAlg alg ) { taskPriorityAlg_=alg; }

  QueueAlg getTaskPriorityAlg() { return taskPriorityAlg_; }


#ifdef HAVE_CUDA
  void addVerifyDataTransferCompletion(DetailedTask* dtask);
  void addFinalizeDevicePreparation(DetailedTask* dtask);
  void addInitiallyReadyDeviceTask( DetailedTask* dtask );
  void addCompletionPendingDeviceTask( DetailedTask* dtask );
  void addFinalizeHostPreparation(DetailedTask* dtask);
  void addInitiallyReadyHostTask(DetailedTask* dtask);

  DetailedTask* getNextVerifyDataTransferCompletionTask();
  DetailedTask* getNextFinalizeDevicePreparationTask();
  DetailedTask* getNextInitiallyReadyDeviceTask();
  DetailedTask* getNextCompletionPendingDeviceTask();
  DetailedTask* getNextFinalizeHostPreparationTask();
  DetailedTask* getNextInitiallyReadyHostTask();

  DetailedTask* peekNextVerifyDataTransferCompletionTask();
  DetailedTask* peekNextFinalizeDevicePreparationTask();
  DetailedTask* peekNextInitiallyReadyDeviceTask();
  DetailedTask* peekNextCompletionPendingDeviceTask();
  DetailedTask* peekNextFinalizeHostPreparationTask();
  DetailedTask* peekNextInitiallyReadyHostTask();

  int numVerifyDataTransferCompletion() { return verifyDataTransferCompletionTasks_.size(); }
  int numFinalizeDevicePreparation() { return finalizeDevicePreparationTasks_.size(); }
  int numInitiallyReadyDeviceTasks() { return initiallyReadyDeviceTasks_.size(); }
  int numCompletionPendingDeviceTasks() { return completionPendingDeviceTasks_.size(); }
  int numFinalizeHostPreparation() { return finalizeHostPreparationTasks_.size(); }
  int numInitiallyReadyHostTasks() { return initiallyReadyHostTasks_.size(); }

  void createInternalDependencyBatch(DetailedTask* from,
      Task::Dependency* comp,
      const Patch* fromPatch,
      DetailedTask* to,
      Task::Dependency* req,
      const Patch *toPatch,
      int matl,
      const IntVector& low,
      const IntVector& high,
      DetailedDep::CommCondition cond);
  // helper of possiblyCreateDependency
  DetailedDep* findMatchingInternalDetailedDep(DependencyBatch* batch, DetailedTask* toTask, Task::Dependency* req,
      const Patch* fromPatch, int matl, IntVector low, IntVector high,
      IntVector& totalLow, IntVector& totalHigh, DetailedDep* &parent_dep);
#endif

protected:

  friend class DetailedTask;

  void internalDependenciesSatisfied( DetailedTask* task );

  SchedulerCommon* getSchedulerCommon() { return sc_; }

private:

  // eliminate copy, assignment and move
  DetailedTasks( const DetailedTasks & )            = delete;
  DetailedTasks& operator=( const DetailedTasks & ) = delete;
  DetailedTasks( DetailedTasks && )                 = delete;
  DetailedTasks& operator=( DetailedTasks && )      = delete;

  void initializeBatches();

  void incrementDependencyGeneration();

  // helper of possiblyCreateDependency
  DetailedDep* findMatchingDetailedDep(       DependencyBatch    * batch
      ,       DetailedTask       * toTask
      ,       Task::Dependency   * req
      , const Patch              * fromPatch
      ,       int                   matl
      ,       IntVector             low
      ,       IntVector             high
      ,       IntVector           & totalLow
      ,       IntVector           & totalHigh
      ,       DetailedDep        *& parent_dep
      );


  void addScrubCount( const VarLabel * var
      ,       int        matlindex
      , const Patch    * patch
      ,       int        dw
      );


  bool getScrubCount( const VarLabel * var
      ,       int        matlindex
      , const Patch    * patch
      ,       int        dw
      ,       int      & count
      );

  std::map<int,int>   sendoldmap_;
  ParticleExchangeVar particleSends_;
  ParticleExchangeVar particleRecvs_;

  SchedulerCommon             * sc_;
  const ProcessorGroup        * d_myworld;
  // store the first so we can share the scrubCountTable
  DetailedTasks               * first;
  std::vector<DetailedTask*>    tasks_;
  KeyDatabase<Patch>            varKeyDB;
  KeyDatabase<Level>            levelKeyDB;

  const TaskGraph               * taskgraph_;
  Task                          * stask_;
  std::vector<DetailedTask*>      localtasks_;
  std::vector<DependencyBatch*>   batches_;
  DetailedDep                   * initreq_;

  // True for mixed scheduler which needs to keep track of internal dependencies.
  bool mustConsiderInternalDependencies_;

  // In the future, we may want to prioritize tasks for the MixedScheduler
  // to run.  I implemented this using topological sort order as the priority
  // but that probably isn't a good way to do unless you make it a breadth
  // first topological order.
  QueueAlg taskPriorityAlg_;

  using TaskQueue  = std::queue<DetailedTask*>;
  using TaskPQueue = std::priority_queue<DetailedTask*, std::vector<DetailedTask*>, DetailedTaskPriorityComparison>;

  TaskQueue   readyTasks_;
  TaskQueue   initiallyReadyTasks_;
  TaskPQueue  mpiCompletedTasks_;

  // This "generation" number is to keep track of which InternalDependency
  // links have been satisfied in the current timestep and avoids the
  // need to traverse all InternalDependency links to reset values.
  unsigned long currentDependencyGeneration_;

  // for logging purposes - how much extra comm is going on
  int extraCommunication_;

  std::mutex  readyQueueLock_;
  std::mutex  mpiCompletedQueueLock_;

  ScrubCountTable scrubCountTable_;


#ifdef HAVE_CUDA
  TaskPQueue            verifyDataTransferCompletionTasks_;    // Some or all ghost cells still need to be processed before a task is ready.
  TaskPQueue            initiallyReadyDeviceTasks_;       // initially ready, h2d copies pending
  TaskPQueue            finalizeDevicePreparationTasks_;  // h2d copies completed, need to mark gpu data as valid and copy gpu ghost cell data internally on device
  TaskPQueue            completionPendingDeviceTasks_;    // execution and d2h copies pending
  TaskPQueue            finalizeHostPreparationTasks_;    // d2h copies completed, need to mark cpu data as valid
  TaskPQueue            initiallyReadyHostTasks_;         // initially ready cpu task, d2h copies pending

  mutable CrowdMonitor  deviceVerifyDataTransferCompletionQueueLock_;
  mutable CrowdMonitor  deviceFinalizePreparationQueueLock_;
  mutable CrowdMonitor  deviceReadyQueueLock_;
  mutable CrowdMonitor  deviceCompletedQueueLock_;
  mutable CrowdMonitor  hostFinalizePreparationQueueLock_;
  mutable CrowdMonitor  hostReadyQueueLock_;
#endif


  friend std::ostream& operator<<( std::ostream& out, const Uintah::DetailedTask& task );
  friend std::ostream& operator<<( std::ostream& out, const Uintah::DetailedDep& task );

}; // DetailedTasks
//_______________________________________________________________________________________________


} // namespace Uintah

#endif // CCA_COMPONENTS_SCHEDULERS_DETAILEDTASKS_EXP_H

