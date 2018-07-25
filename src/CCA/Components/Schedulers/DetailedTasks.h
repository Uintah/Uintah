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

#ifndef CCA_COMPONENTS_SCHEDULERS_DETAILEDTASKS_H
#define CCA_COMPONENTS_SCHEDULERS_DETAILEDTASKS_H

#include <CCA/Components/Schedulers/DetailedDependency.h>
#include <CCA/Components/Schedulers/DetailedTask.h>
#include <CCA/Components/Schedulers/DWDatabase.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouseP.h>

#include <Core/Containers/FastHashTable.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/PSPatchMatlGhostRange.h>
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
class DependencyBatch;
class DataWarehouse;
class DetailedTask;
class TaskGraph;
class SchedulerCommon;

using ParticleExchangeVar = std::map<int, std::set<PSPatchMatlGhostRange> >;
using ScrubCountTable     = FastHashTable<ScrubItem>;
using DepCommCond         = DetailedDep::CommCondition;


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
class DetailedTaskPriorityComparison {

public:

  bool operator()( DetailedTask *& ltask, DetailedTask *& rtask );
};


//_____________________________________________________________________________
//
class DetailedTasks {

public:

  DetailedTasks(       SchedulerCommon         * sc
               , const ProcessorGroup          * pg
               , const TaskGraph               * taskgraph
               , const std::unordered_set<int> & neighborhood_processors
               ,       bool                      mustConsiderInternalDependencies = false
               );

  ~DetailedTasks();

  void add( DetailedTask * dtask );

  void makeDWKeyDatabase();

  void copyoutDWKeyDatabase( OnDemandDataWarehouseP dws )
  {
    dws->copyKeyDB(m_var_keyDB, m_level_keyDB);
  }

  int numTasks() const
  {
    return (int)m_tasks.size();
  }

  DetailedTask* getTask( int i )
  {
    return m_tasks[i];
  }

  void assignMessageTags( int me );

  void initializeScrubs( std::vector<OnDemandDataWarehouseP> & dws, int dwmap[] );

  void possiblyCreateDependency(       DetailedTask     * from
                               ,       Task::Dependency * cmp
                               , const Patch            * fromPatch
                               ,       DetailedTask     * to
                               ,       Task::Dependency * req
                               , const Patch            * toPatch
                               ,       int                matl
                               , const IntVector        & low
                               , const IntVector        & high
                               ,       DepCommCond        cond
                               );

  DetailedTask* getOldDWSendTask( int proc );

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

  DetailedTask* localTask( int idx )
  {
    return m_local_tasks[idx];
  }

  void emitEdges( ProblemSpecP edgesElement, int rank );

  DetailedTask* getNextInternalReadyTask();

  int numInternalReadyTasks();

  DetailedTask* getNextExternalReadyTask();

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

  const TaskGraph* getTaskGraph() const
  {
    return m_task_graph;
  }

  void setScrubCount( const Task::Dependency                    * req
                    ,       int                                   matl
                    , const Patch                               * patch
                    ,       std::vector<OnDemandDataWarehouseP> & dws
                    );

  int getExtraCommunication()
  {
    return m_extra_comm;
  }

  friend std::ostream& operator<<( std::ostream & out, const Uintah::DetailedTask & dtask );

  friend std::ostream& operator<<( std::ostream & out, const Uintah::DetailedDep & dep );

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

  void addDeviceValidateRequiresCopies( DetailedTask * dtask );

  void addDevicePerformGhostCopies( DetailedTask * dtask );

  void addDeviceValidateGhostCopies( DetailedTask * dtask );

  void addDeviceCheckIfExecutable( DetailedTask * dtask );

  void addDeviceReadyToExecute( DetailedTask * dtask );

  void addDeviceExecutionPending( DetailedTask * dtask );

  void addHostValidateRequiresCopies( DetailedTask * dtask );

  void addHostCheckIfExecutable( DetailedTask * dtask );

  void addHostReadyToExecute( DetailedTask * dtask );

  bool getDeviceValidateRequiresCopiesTask( DetailedTask *& dtask );

  bool getDevicePerformGhostCopiesTask( DetailedTask *& dtask );

  bool getDeviceValidateGhostCopiesTask( DetailedTask *& dtask );

  bool getDeviceCheckIfExecutableTask( DetailedTask *& dtask );

  bool getDeviceReadyToExecuteTask( DetailedTask *& dtask );

  bool getDeviceExecutionPendingTask( DetailedTask *& dtask );

  bool getHostValidateRequiresCopiesTask( DetailedTask *& dtask );

  bool getHostCheckIfExecutableTask( DetailedTask *& dtask );

  bool getHostReadyToExecuteTask( DetailedTask *& dtask );

  void createInternalDependencyBatch(       DetailedTask     * from
                                    ,       Task::Dependency * comp
                                    , const Patch            * fromPatch
                                    ,       DetailedTask     * to
                                    ,       Task::Dependency * req
                                    , const Patch            * toPatch
                                    ,       int                matl
                                    , const IntVector        & low
                                    , const IntVector        & high
                                    ,       DepCommCond        cond
                                    );

  // helper of possiblyCreateDependency
  DetailedDep* findMatchingInternalDetailedDep(       DependencyBatch  * batch
                                              ,       DetailedTask     * toTask
                                              ,       Task::Dependency * req
                                              , const Patch            * fromPatch
                                              ,       int                matl
                                              ,       IntVector          low
                                              ,       IntVector          high
                                              ,       IntVector        & totalLow
                                              ,       IntVector        & totalHigh
                                              ,       DetailedDep      * &parent_dep
                                              );
#endif

protected:

  friend class DetailedTask;

  void internalDependenciesSatisfied( DetailedTask * dtask );

  SchedulerCommon* getSchedulerCommon()
  {
    return m_sched_common;
  }

private:

  void initializeBatches();

  void incrementDependencyGeneration();

  // helper of possiblyCreateDependency
  DetailedDep* findMatchingDetailedDep(       DependencyBatch  * batch
                                      ,       DetailedTask     * toTask
                                      ,       Task::Dependency * req
                                      , const Patch            * fromPatch
                                      ,       int                matl
                                      ,       IntVector          low
                                      ,       IntVector          high
                                      ,       IntVector        & totalLow
                                      ,       IntVector        & totalHigh
                                      ,       DetailedDep     *& parent_dep
                                      );

  void addScrubCount( const VarLabel * var
                    ,       int        matlindex
                    , const Patch    * patch
                    ,       int        dw
                    );

  bool getScrubCount( const VarLabel * var
                    , int              matlindex
                    , const Patch    * patch
                    , int              dw
                    , int            & count
                    );


  SchedulerCommon               * m_sched_common { nullptr };
  const ProcessorGroup          * m_proc_group;

  KeyDatabase<Patch>              m_var_keyDB;
  KeyDatabase<Level>              m_level_keyDB;

  const TaskGraph               * m_task_graph { nullptr };

  Task                          * m_send_old_data { nullptr };
  std::map<int, int>              m_send_old_map;
  std::vector<DetailedTask*>      m_local_tasks;
  std::vector<DetailedTask*>      m_tasks;

  std::vector<DependencyBatch*>   m_dep_batches;
  DetailedDep                   * m_init_req { nullptr };

  ParticleExchangeVar             m_particle_sends;
  ParticleExchangeVar             m_particle_recvs;

  // true for any threaded scheduler which needs to keep track of internal dependencies.
  bool m_must_consider_internal_deps;

  QueueAlg m_task_priority_alg { QueueAlg::MostMessages };

  using TaskQueue  = std::queue<DetailedTask*>;
  using TaskPQueue = std::priority_queue<DetailedTask*, std::vector<DetailedTask*>, DetailedTaskPriorityComparison>;

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
  DetailedTasks(const DetailedTasks &)            = delete;
  DetailedTasks& operator=(const DetailedTasks &) = delete;
  DetailedTasks(DetailedTasks &&)                 = delete;
  DetailedTasks& operator=(DetailedTasks &&)      = delete;


#ifdef HAVE_CUDA

  using TaskPool = Lockfree::Pool< DetailedTask *
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

}; // class DetailedTasks

}  // namespace Uintah

#endif // CCA_COMPONENTS_SCHEDULERS_DETAILEDTASKS_H

