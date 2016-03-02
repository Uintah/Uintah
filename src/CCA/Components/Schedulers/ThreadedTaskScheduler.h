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

#ifndef CCA_COMPONENTS_SCHEDULERS_THREADEDTASKSCHEDULER_H
#define CCA_COMPONENTS_SCHEDULERS_THREADEDTASKSCHEDULER_H

#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Components/Schedulers/MessageLog.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouseP.h>
#include <CCA/Ports/DataWarehouseP.h>

#include <Core/Grid/Task.h>
#include <Core/Lockfree/Lockfree_Pool.hpp>
#include <Core/Malloc/Allocators/AllocatorTags.hpp>
#include <Core/Parallel/BufferInfo.h>
#include <Core/Parallel/CommunicationList.h>
#include <Core/Parallel/PackBufferInfo.h>
#include <Core/Util/InfoMapper.h>
#include <Core/Util/Timers/Timers.hpp>

#include <atomic>
#include <fstream>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace Uintah {

class Task;
class DetailedTask;
class TaskRunner;

using clock_type = std::chrono::high_resolution_clock;
using nanoseconds = std::chrono::nanoseconds;

using TaskPool = Lockfree::Pool< DetailedTask*
                               , uint64_t
                               , 1u
                               , Uintah::MallocAllocator      // allocator
                               , Uintah::MallocAllocator      // size_type allocator
                               >;


/**************************************

 CLASS
 ThreadedTaskScheduler


 GENERAL INFORMATION
 ThreadedTaskScheduler.h


 Alan Humphrey & Dan Sunderland - February/March, 2016
 Scientific Computing and Imaging Institute
 University of Utah


 KEYWORDS
   Task Scheduler, Multi-threaded MPI, CPU


 DESCRIPTION
   A multi-threaded scheduler that uses a combination of MPI + Pthreads

   Dynamic task scheduling with non-deterministic execution of tasks at runtime.

   Pthreads are pinned to individual CPU cores where tasks are then executed.

   Uses a decentralized model wherein all threads can access task queues,
   processes there own MPI send and recvs, and have shared access to the DataWarehouse.
   Collective operations are restricted to the main thread.


 WARNING
   This scheduler is still EXPERIMENTAL and undergoing extensive
   development, not all tasks/components are thread-safe yet.

 ****************************************/


class ThreadedTaskScheduler : public SchedulerCommon {

public:

  ThreadedTaskScheduler( const ProcessorGroup * myworld, const Output * oport, ThreadedTaskScheduler * parentScheduler = nullptr);

  virtual ~ThreadedTaskScheduler();

  virtual void problemSetup( const ProblemSpecP & prob_spec, SimulationStateP & state );

  virtual SchedulerP createSubScheduler();

  virtual void execute( int tgnum = 0, int iteration = 0);

  virtual bool useInternalDeps() { return !d_sharedState->isCopyDataTimestep(); }

  const ProcessorGroup* getProcessorGroup() { return d_myworld; }

  void compile() {
    m_num_messages   = 0;
    m_message_volume = 0;
    SchedulerCommon::compile();
  }

  virtual void printMPIStats();

  // timing statistics to test the MPI functionality
  enum TimingStat
  {
    TotalReduce,
    TotalSend,
    TotalRecv,
    TotalTask,
    TotalReduceMPI,
    TotalSendMPI,
    TotalRecvMPI,
    TotalTestMPI,
    TotalWaitMPI,
    MAX_TIMING_STATS
  };

  struct TotalReduceTag{};
  struct TotalReduceMPITag{};

  struct TotalSendTag{};
  struct TotalSendMPITag{};

  struct TotalRecvTag{};
  struct TotalRecvMPITag{};

  struct TotalTestMPITag{};
  struct TotalTaskTag{};

  struct TotalWaitTag{};
  struct TotalWaitMPITag{};

  friend class TaskRunner;



protected:

  virtual void verifyChecksum();



private:

  // eliminate copy, assignment and move
  ThreadedTaskScheduler( const ThreadedTaskScheduler & )            = delete;
  ThreadedTaskScheduler& operator=( const ThreadedTaskScheduler & ) = delete;
  ThreadedTaskScheduler( ThreadedTaskScheduler && )                 = delete;
  ThreadedTaskScheduler& operator=( ThreadedTaskScheduler && )      = delete;

  enum : size_t {
      REQUEST_COLLECTIVE
    , REQUEST_RECV
    , REQUEST_SEND
    , REQUEST_SIZE
  };

  void post_MPI_recvs( DetailedTask * task, bool only_old_recvs, int abort_point, int iteration );

  void post_MPI_sends( DetailedTask * task, int iteration );

  bool process_MPI_requests();

  void run_task( DetailedTask * task, int iteration );

  void run_reduction_task( DetailedTask* task );

  void emit_net_MPI_stats();

  void emit_time( const char * label, double dt );

  void compute_net_runtime_stats( InfoMapper< SimulationState::RunTimeStat, double > & runTimeStats );

  void copy_restart_flag( int task_graph_num );

  void output_timing_stats( const char* label );



  // Methods for TaskRunner management
  void select_tasks();

  void thread_fence();

  static void init_threads( ThreadedTaskScheduler * scheduler, int num_threads );

  static void set_runner( TaskRunner * runner, int tid );

  std::vector<int>             m_phase_tasks{};
  std::vector<DetailedTask*>   m_phase_sync_tasks{};
  DetailedTasks              * m_detailed_tasks{};

  TaskPool   m_task_pool{};

  // Timers for MPI stats
  Timers::Simple  m_mpi_test_time{};
  Timers::Simple  m_last_exec_timer{};
  Timers::Simple  m_task_exec_timer{};

  bool     m_abort{ false };
  int      m_current_iteration{ 0 };
  int      m_num_tasks{ 0 };
  int      m_current_phase{ 0 };
  int      m_num_phases{ 0 };
  int      m_abort_point{ 0 };
  int      m_num_threads{ 0 };

  using atomic_int_array = std::unique_ptr<std::atomic<int>[]>;
  std::atomic<int>            m_num_tasks_done;
  atomic_int_array            m_phase_tasks_done;

  MessageLog                  m_message_log;
  const Output              * m_output_port;

  CommPool                    m_comm_requests{REQUEST_SIZE};

  std::vector<const char*>    m_labels{};
  std::vector<double>         m_times{};

  std::ofstream               m_timings_stats{};
  std::ofstream               m_max_stats{};
  std::ofstream               m_avg_stats{};

  unsigned int                m_num_messages{};
  double                      m_message_volume{};

  ThreadedTaskScheduler     * m_parent_scheduler;

  ReductionInfoMapper< TimingStat, double > m_mpi_info;

  std::map<std::string, std::atomic<uint64_t> > waittimes{};
  std::map<std::string, std::atomic<uint64_t> > exectimes{};

};


class TaskRunner {

  public:

    TaskRunner() = default;

    TaskRunner( ThreadedTaskScheduler* scheduler )
      : m_scheduler{ scheduler }
      , m_task_wait_time{}
      , m_task_exec_time{}
    {}

    ~TaskRunner() {};

    void run() const;


  private:

    ThreadedTaskScheduler*  m_scheduler{ nullptr };

    Timers::Simple            m_task_wait_time{};
    Timers::Simple            m_task_exec_time{};


    TaskRunner( const TaskRunner & )            = delete;
    TaskRunner& operator=( const TaskRunner & ) = delete;
    TaskRunner( TaskRunner &&)                  = delete;
    TaskRunner& operator=( TaskRunner && )      = delete;

    friend class ThreadedTaskScheduler;

}; // class ThreadedTaskScheduler

} // namespace Uintah

#endif  // CCA_COMPONENTS_SCHEDULERS_THREADEDTASKSCHEDULER_H
