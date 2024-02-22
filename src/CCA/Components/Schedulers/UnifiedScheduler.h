/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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

#ifndef CCA_COMPONENTS_SCHEDULERS_UNIFIEDSCHEDULER_H
#define CCA_COMPONENTS_SCHEDULERS_UNIFIEDSCHEDULER_H

#include <CCA/Components/Schedulers/MPIScheduler.h>

#include <sci_defs/gpu_defs.h>

#include <map>
#include <queue>
#include <string>
#include <vector>

namespace Uintah {

class DetailedTask;
class DetailedTasks;
class UnifiedSchedulerWorker;

/**************************************

CLASS
   UnifiedScheduler


GENERAL INFORMATION
   UnifiedScheduler.h

   Qingyu Meng, Alan Humphrey, Brad Peterson
   Scientific Computing and Imaging Institute
   University of Utah

KEYWORDS
   Task Scheduler, Multi-threaded MPI, CPU

DESCRIPTION
   A multi-threaded scheduler that uses a combination of MPI +
   std::thread. Dynamic scheduling with non-deterministic,
   out-of-order execution of tasks at runtime. One MPI rank per
   multi-core node.  threads (std::thread) are pinned to individual
   CPU cores where these tasks are executed.

   Uses a decentralized model wherein all threads can access task
   queues, processes there own MPI sends and recvs, with shared access
   to the DataWarehouse.

   Uintah task scheduler to support, schedule and execute solely CPU
   tasks.

WARNING
   This scheduler is still EXPERIMENTAL and undergoing extensive
   development, not all tasks/components are thread-safe yet.

   Requires MPI_THREAD_MULTIPLE support.

****************************************/


class UnifiedScheduler : public MPIScheduler  {

  public:

    UnifiedScheduler( const ProcessorGroup * myworld,
                      UnifiedScheduler * parentScheduler = nullptr );

    virtual ~UnifiedScheduler();

    virtual void problemSetup( const ProblemSpecP & prob_spec,
                               const MaterialManagerP & materialManager );

    virtual SchedulerP createSubScheduler();

    virtual void execute( int tgnum = 0, int iteration = 0 );

    virtual bool useInternalDeps() { return !m_is_copy_data_timestep; }

    void runTask( DetailedTask * dtask, int iteration, int thread_id, CallBackEvent event );

    void runTasks( int thread_id );

    // timing statistics for Uintah infrastructure overhead
    enum ThreadStatsEnum {
        WaitTime
      , LocalTID
      , Affinity
      , NumTasks
      , NumPatches
    };

    VectorInfoMapper< ThreadStatsEnum, double > m_thread_info;

    static std::string myRankThread();

    friend class UnifiedSchedulerWorker;

  private:

    // eliminate copy, assignment and move
    UnifiedScheduler( const UnifiedScheduler & )            = delete;
    UnifiedScheduler& operator=( const UnifiedScheduler & ) = delete;
    UnifiedScheduler( UnifiedScheduler && )                 = delete;
    UnifiedScheduler& operator=( UnifiedScheduler && )      = delete;

    void markTaskConsumed( int & numTasksDone, int & currphase, int numPhases, DetailedTask * dtask );

    static void init_threads( UnifiedScheduler * scheduler, int num_threads );

    // thread shared data, needs lock protection when accessed
    std::vector<int>             m_phase_tasks;
    std::vector<int>             m_phase_tasks_done;
    std::vector<DetailedTask*>   m_phase_sync_task;
    std::vector<int>             m_histogram;
    DetailedTasks              * m_detailed_tasks{nullptr};

    QueueAlg m_task_queue_alg{MostMessages};
    int      m_curr_iteration{0};
    int      m_num_tasks_done{0};
    int      m_num_tasks{0};
    int      m_curr_phase{0};
    int      m_num_phases{0};
    bool     m_abort{false};
    int      m_abort_point{0};
};


class UnifiedSchedulerWorker {

public:

  UnifiedSchedulerWorker( UnifiedScheduler * scheduler, int tid, int affinity );

  void run();

  const double getWaitTime() const;
  const int    getLocalTID() const;
  const int    getAffinity() const;

  void   startWaitTime();
  void   stopWaitTime();
  void   resetWaitTime();

  friend class UnifiedScheduler;

private:

  UnifiedScheduler * m_scheduler{nullptr};
  int                m_rank{-1};
  int                m_tid{-1};
  int                m_affinity{-1};

  Timers::Simple     m_wait_timer{};
  double             m_wait_time{0.0};
};

} // namespace Uintah

#endif // CCA_COMPONENTS_SCHEDULERS_UNIFIEDSCHEDULER_H
