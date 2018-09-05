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

#ifndef CCA_COMPONENTS_SCHEDULERS_UNSTRUCTURED_DETAILEDTASKS_H
#define CCA_COMPONENTS_SCHEDULERS_UNSTRUCTURED_DETAILEDTASKS_H

#include <CCA/Components/Schedulers/UnstructuredDetailedDependency.h>
#include <CCA/Components/Schedulers/UnstructuredDetailedTask.h>
#include <CCA/Components/Schedulers/UnstructuredDWDatabase.h>
#include <CCA/Components/Schedulers/UnstructuredOnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/UnstructuredOnDemandDataWarehouseP.h>

#include <Core/Containers/FastHashTable.h>
#include <Core/Grid/UnstructuredPatch.h>
#include <Core/Grid/UnstructuredTask.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/UnstructuredPSPatchMatlGhostRange.h>
#include <Core/Grid/Variables/ScrubItem.h>

#include <Core/Lockfree/Lockfree_Pool.hpp>


#ifdef HAVE_CUDA
  #include <CCA/Components/Schedulers/GPUGridVariableGhosts.h>
  #include <CCA/Components/Schedulers/GPUGridVariableInfo.h>
#endif

#include <sci_defs/cuda_defs.h>

#include <map>
#include <queue>
#include <set>
#include <vector>
#include <atomic>
#include <list>

namespace Uintah {

class ProcessorGroup;
class UnstructuredDependencyBatch;
class UnstructuredDataWarehouse;
class UnstructuredDetailedTask;
class UnstructuredTaskGraph;
class UnstructuredSchedulerCommon;

using ParticleExchangeVar = std::map<int, std::set<UnstructuredPSPatchMatlGhostRange> >;
using ScrubCountTable     = FastHashTable<UnstructuredScrubItem>;
using DepCommCond         = UnstructuredDetailedDep::CommCondition;


//_____________________________________________________________________________
//
enum QueueAlg {
    FCFS
  , Stack
  , Random
  , MostMessages
  , LeastMessages
  , CritialPath
  , PatchOrder
  , PatchOrderRandom
};


//_____________________________________________________________________________
//
class UnstructuredDetailedTaskPriorityComparison {

public:

  bool operator()( UnstructuredDetailedTask *& ltask, UnstructuredDetailedTask *& rtask );
};


//_____________________________________________________________________________
//
class UnstructuredDetailedTasks {

public:

  UnstructuredDetailedTasks(       UnstructuredSchedulerCommon * sc
               , const ProcessorGroup  * pg
               , const UnstructuredTaskGraph       * taskgraph
               , const std::set<int>   & neighborhood_processors
               ,       bool              mustConsiderInternalDependencies = false
               );

  ~UnstructuredDetailedTasks();

  void add( UnstructuredDetailedTask * dtask );

  void makeDWKeyDatabase();

  void copyoutDWKeyDatabase( UnstructuredOnDemandDataWarehouseP dws )
  {
    dws->copyKeyDB(m_var_keyDB, m_level_keyDB);
  }

  int numTasks() const
  {
    return (int)m_tasks.size();
  }

  UnstructuredDetailedTask* getTask( int i )
  {
    return m_tasks[i];
  }

  void assignMessageTags( int me );

  void initializeScrubs( std::vector<UnstructuredOnDemandDataWarehouseP> & dws, int dwmap[] );

  void possiblyCreateDependency(       UnstructuredDetailedTask     * from
                               ,       UnstructuredTask::Dependency * cmp
                               , const UnstructuredPatch            * fromPatch
                               ,       UnstructuredDetailedTask     * to
                               ,       UnstructuredTask::Dependency * req
                               , const UnstructuredPatch            * toPatch
                               ,       int                matl
                               , const IntVector        & low
                               , const IntVector        & high
                               ,       DepCommCond        cond
                               );

  UnstructuredDetailedTask* getOldDWSendTask( int proc );

  void logMemoryUse(       std::ostream  & out
                   ,       unsigned long & total
                   , const std::string   & tag
                   );

  void initTimestep();

  void computeLocalTasks( int me );

  int numLocalTasks() const
  {
    return static_cast<int>(m_local_tasks.size());
  }

  UnstructuredDetailedTask* localTask( int idx )
  {
    return m_local_tasks[idx];
  }

  void emitEdges( ProblemSpecP edgesElement, int rank );

  UnstructuredDetailedTask* getNextInternalReadyTask();

  int numInternalReadyTasks();

  UnstructuredDetailedTask* getNextExternalReadyTask();

  int numExternalReadyTasks();

  void createScrubCounts();

  bool mustConsiderInternalDependencies()
  {
    return m_must_consider_internal_deps;
  }

  unsigned long getCurrentDependencyGeneration()
  {
    return m_current_dependency_generation;
  }

  const UnstructuredTaskGraph* getTaskGraph() const
  {
    return m_task_graph;
  }

  void setScrubCount( const UnstructuredTask::Dependency                    * req
                    ,       int                                   matl
                    , const UnstructuredPatch                               * patch
                    ,       std::vector<UnstructuredOnDemandDataWarehouseP> & dws
                    );

  int getExtraCommunication()
  {
    return m_extra_comm;
  }

  friend std::ostream& operator<<( std::ostream & out, const Uintah::UnstructuredDetailedTask & dtask );

  friend std::ostream& operator<<( std::ostream & out, const Uintah::UnstructuredDetailedDep & dep );

  ParticleExchangeVar& getParticleSends()
  {
    return m_particle_sends;
  }

  ParticleExchangeVar& getParticleRecvs()
  {
    return m_particle_recvs;
  }

  void setTaskPriorityAlg( QueueAlg alg )
  {
    m_task_priority_alg = alg;
  }

  QueueAlg getTaskPriorityAlg()
  {
    return m_task_priority_alg;
  }

#ifdef HAVE_CUDA

  void addDeviceValidateRequiresCopies( UnstructuredDetailedTask * dtask );

  void addDevicePerformGhostCopies( UnstructuredDetailedTask * dtask );

  void addDeviceValidateGhostCopies( UnstructuredDetailedTask * dtask );

  void addDeviceCheckIfExecutable( UnstructuredDetailedTask * dtask );

  void addDeviceReadyToExecute( UnstructuredDetailedTask * dtask );

  void addDeviceExecutionPending( UnstructuredDetailedTask * dtask );

  void addHostValidateRequiresCopies( UnstructuredDetailedTask * dtask );

  void addHostCheckIfExecutable( UnstructuredDetailedTask * dtask );

  void addHostReadyToExecute( UnstructuredDetailedTask * dtask );

  bool getDeviceValidateRequiresCopiesTask( UnstructuredDetailedTask *& dtask );

  bool getDevicePerformGhostCopiesTask( UnstructuredDetailedTask *& dtask );

  bool getDeviceValidateGhostCopiesTask( UnstructuredDetailedTask *& dtask );

  bool getDeviceCheckIfExecutableTask( UnstructuredDetailedTask *& dtask );

  bool getDeviceReadyToExecuteTask( UnstructuredDetailedTask *& dtask );

  bool getDeviceExecutionPendingTask( UnstructuredDetailedTask *& dtask );

  bool getHostValidateRequiresCopiesTask( UnstructuredDetailedTask *& dtask );

  bool getHostCheckIfExecutableTask( UnstructuredDetailedTask *& dtask );

  bool getHostReadyToExecuteTask( UnstructuredDetailedTask *& dtask );

  void createInternalDependencyBatch(       UnstructuredDetailedTask     * from
                                    ,       UnstructuredTask::Dependency * comp
                                    , const UnstructuredPatch            * fromPatch
                                    ,       UnstructuredDetailedTask     * to
                                    ,       UnstructuredTask::Dependency * req
                                    , const UnstructuredPatch            * toPatch
                                    ,       int                matl
                                    , const IntVector        & low
                                    , const IntVector        & high
                                    ,       DepCommCond        cond
                                    );

  // helper of possiblyCreateDependency
  UnstructuredDetailedDep* findMatchingInternalDetailedDep(UnstructuredDependencyBatch  * batch
                                              ,       UnstructuredDetailedTask     * toTask
                                              ,       UnstructuredTask::Dependency * req
                                              , const UnstructuredPatch            * fromPatch
                                              ,       int                matl
                                              ,       IntVector          low
                                              ,       IntVector          high
                                              ,       IntVector        & totalLow
                                              ,       IntVector        & totalHigh
                                              ,       UnstructuredDetailedDep      * &parent_dep
                                              );
#endif

protected:

  friend class UnstructuredDetailedTask;

  void internalDependenciesSatisfied( UnstructuredDetailedTask * dtask );

  UnstructuredSchedulerCommon* getSchedulerCommon()
  {
    return m_sched_common;
  }

private:

  void initializeBatches();

  void incrementDependencyGeneration();

  // helper of possiblyCreateDependency
  UnstructuredDetailedDep* findMatchingDetailedDep(UnstructuredDependencyBatch  * batch
                                      ,       UnstructuredDetailedTask     * toTask
                                      ,       UnstructuredTask::Dependency * req
                                      , const UnstructuredPatch            * fromPatch
                                      ,       int                matl
                                      ,       IntVector          low
                                      ,       IntVector          high
                                      ,       IntVector        & totalLow
                                      ,       IntVector        & totalHigh
                                      ,       UnstructuredDetailedDep     *& parent_dep
                                      );

  void addScrubCount( const UnstructuredVarLabel * var
                    ,       int        matlindex
                    , const UnstructuredPatch    * patch
                    ,       int        dw
                    );

  bool getScrubCount( const UnstructuredVarLabel * var
                    , int              matlindex
                    , const UnstructuredPatch    * patch
                    , int              dw
                    , int            & count
                    );


  UnstructuredSchedulerCommon               * m_sched_common { nullptr };
  const ProcessorGroup          * m_proc_group;

  UnstructuredKeyDatabase<UnstructuredPatch>              m_var_keyDB;
  UnstructuredKeyDatabase<UnstructuredLevel>              m_level_keyDB;

  const UnstructuredTaskGraph               * m_task_graph { nullptr };

  UnstructuredTask                          * m_send_old_data { nullptr };
  std::map<int, int>              m_send_old_map;
  std::vector<UnstructuredDetailedTask*>      m_local_tasks;
  std::vector<UnstructuredDetailedTask*>      m_tasks;

  std::vector<UnstructuredDependencyBatch*>   m_dep_batches;
  UnstructuredDetailedDep                   * m_init_req { nullptr };

  ParticleExchangeVar             m_particle_sends;
  ParticleExchangeVar             m_particle_recvs;

  // true for any threaded scheduler which needs to keep track of internal dependencies.
  bool m_must_consider_internal_deps;

  QueueAlg m_task_priority_alg { QueueAlg::MostMessages };

  using TaskQueue  = std::queue<UnstructuredDetailedTask*>;
  using TaskPQueue = std::priority_queue<UnstructuredDetailedTask*, std::vector<UnstructuredDetailedTask*>, UnstructuredDetailedTaskPriorityComparison>;

  TaskQueue  m_ready_tasks;
  TaskQueue  m_initial_ready_tasks;
  TaskPQueue m_mpi_completed_tasks;
  std::atomic<int> atomic_readyTasks_size { 0 };
  std::atomic<int> m_atomic_mpi_completed_tasks_size { 0 };

  // This "generation" number is to keep track of which InternalDependency
  // links have been satisfied in the current timestep and avoids the
  // need to traverse all InternalDependency links to reset values.
  unsigned long m_current_dependency_generation { 1 };

  // for logging purposes - how much extra communication is going on
  int m_extra_comm { 0 };

  ScrubCountTable m_scrub_count_table;

  // eliminate copy, assignment and move
  UnstructuredDetailedTasks(const UnstructuredDetailedTasks &)            = delete;
  UnstructuredDetailedTasks& operator=(const UnstructuredDetailedTasks &) = delete;
  UnstructuredDetailedTasks(UnstructuredDetailedTasks &&)                 = delete;
  UnstructuredDetailedTasks& operator=(UnstructuredDetailedTasks &&)      = delete;


#ifdef HAVE_CUDA

  using TaskPool = Lockfree::Pool< UnstructuredDetailedTask *
                                 , uint64_t
                                 , 1
                                 , std::allocator
                                 >;

  TaskPool             device_validateRequiresCopies_pool{};
  TaskPool             device_performGhostCopies_pool{};
  TaskPool             device_validateGhostCopies_pool{};
  TaskPool             device_checkIfExecutable_pool{};
  TaskPool             device_readyToExecute_pool{};
  TaskPool             device_executionPending_pool{};
  TaskPool             host_validateRequiresCopies_pool{};
  TaskPool             host_checkIfExecutable_pool{};
  TaskPool             host_readyToExecute_pool{};

#endif

}; // class UnstructuredDetailedTasks

}  // namespace Uintah

#endif // CCA_COMPONENTS_SCHEDULERS_DETAILEDTASKS_H

