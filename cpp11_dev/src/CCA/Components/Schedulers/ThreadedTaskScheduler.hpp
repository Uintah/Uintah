/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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
#include <CCA/Components/Schedulers/CommunicationList.hpp>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouseP.h>
#include <CCA/Components/Schedulers/RuntimeStats.hpp>
#include <CCA/Ports/DataWarehouseP.h>

#include <Core/Grid/Task.h>
#include <Core/Lockfree/Lockfree_Pool.hpp>
#include <Core/Parallel/BufferInfo.h>
#include <Core/Parallel/PackBufferInfo.h>
#include <Core/Util/InfoMapper.h>
#include <Core/Util/Timers/Timers.hpp>

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

namespace Uintah {

class  Task;
class  DetailedTask;
class  TaskRunner;

using TaskPool = Lockfree::Pool< DetailedTask *
                               , uint64_t
                               , 1u
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

  ThreadedTaskScheduler( const ProcessorGroup * myworld, const Output * oport);

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

  void run_task( DetailedTask * dtask, int iteration );

  void run_reduction_task( DetailedTask* task );

  void emit_time( const char * label, double dt );

  void copy_restart_flag( int task_graph_num );


  // Methods for TaskRunner management
  void process_tasks( int iteration );

  void thread_fence();

  static void init_threads( ThreadedTaskScheduler * scheduler, int num_threads );

  static void set_runner( TaskRunner * runner, int tid );

  std::vector<int>             m_phase_tasks{};
  std::vector<DetailedTask*>   m_phase_sync_tasks{};
  DetailedTasks              * m_detailed_tasks{};

  CommPool   m_comm_requests{REQUEST_SIZE};
  TaskPool   m_init_tasks{};
  TaskPool   m_ready_tasks{};

  bool       m_abort{ false };
  int        m_current_iteration{ 0 };
  int        m_num_tasks{ 0 };
  int        m_num_phases{ 0 };
  int        m_abort_point{ 0 };
  int        m_num_threads{ 0 };

  using atomic_int_array = std::unique_ptr<std::atomic<int>[]>;
  std::atomic<int>            m_current_phase;
  std::atomic<int>            m_num_tasks_done;
  atomic_int_array            m_phase_tasks_done;

  std::vector<const char*>    m_labels{};
  std::vector<double>         m_times{};

  unsigned int                m_num_messages{};
  double                      m_message_volume{};

  const Output              * m_output_port;

};


class TaskRunner {

  public:

    TaskRunner() = default;

    TaskRunner( ThreadedTaskScheduler* scheduler )
      : m_scheduler{ scheduler }
    {}

    ~TaskRunner() {};

    void run() const;


  private:

    ThreadedTaskScheduler*  m_scheduler{ nullptr };

    TaskRunner( const TaskRunner & )            = delete;
    TaskRunner& operator=( const TaskRunner & ) = delete;
    TaskRunner( TaskRunner &&)                  = delete;
    TaskRunner& operator=( TaskRunner && )      = delete;

    friend class ThreadedTaskScheduler;

}; // class ThreadedTaskScheduler

} // namespace Uintah

#endif  // CCA_COMPONENTS_SCHEDULERS_THREADEDTASKSCHEDULER_H
