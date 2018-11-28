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

#ifndef CCA_COMPONENTS_SCHEDULERS_KOKKOSOPENMPSCHEDULER_H
#define CCA_COMPONENTS_SCHEDULERS_KOKKOSOPENMPSCHEDULER_H

#include <CCA/Components/Schedulers/MPIScheduler.h>

#include <atomic>
#include <string>
#include <vector>

namespace Uintah {

class Task;
class DetailedTask;

/**************************************

CLASS
   KokkosOpenMPScheduler
   

GENERAL INFORMATION
   KokkosOpenMPScheduler.h

   Alan Humphrey, John Holmen
   Scientific Computing and Imaging Institute
   University of Utah

   
KEYWORDS
   Task Scheduler, Multi-threaded MPI, OpenMP

DESCRIPTION
   A multi-threaded scheduler that uses a combination of MPI + OpenMP. This relies
   on the Kokkos OpenMP backend to manage OpenMP partitioning. Each partition has a
   "master" thread which runs a given functor.

   OpenMP 4.0 is required for this scheduler.

   This scheduler is designed primarily for use on the many-core Intel Xeon Phi
   architecture, specifically KNL and KNH, but should also work well on any
   multi-core CPU node with hyper-threading
  
WARNING
   This scheduler is EXPERIMENTAL and undergoing extensive development.
   Not all tasks/components are Kokkos-enabled and/or thread-safe
   
   Requires MPI_THREAD_MULTIPLE support.
  
****************************************/


class KokkosOpenMPScheduler : public MPIScheduler  {

  public:

    KokkosOpenMPScheduler( const ProcessorGroup  * myworld,
                           KokkosOpenMPScheduler * parentScheduler = nullptr );

    virtual ~KokkosOpenMPScheduler(){};

    virtual void problemSetup( const ProblemSpecP & prob_spec, const MaterialManagerP & materialManager );

    virtual SchedulerP createSubScheduler();

    virtual void execute( int tgnum = 0, int iteration = 0 );

    virtual bool useInternalDeps() { return !m_is_copy_data_timestep; }

    void runTasks();

    static std::string myRankThread();


  private:

    // eliminate copy, assignment and move
    KokkosOpenMPScheduler( const KokkosOpenMPScheduler & )            = delete;
    KokkosOpenMPScheduler& operator=( const KokkosOpenMPScheduler & ) = delete;
    KokkosOpenMPScheduler( KokkosOpenMPScheduler && )                 = delete;
    KokkosOpenMPScheduler& operator=( KokkosOpenMPScheduler && )      = delete;

    void markTaskConsumed( volatile int * numTasksDone, int & currphase, int numPhases, DetailedTask * dtask );

    // thread shared data, needs lock protection when accessed
    std::vector<int>             m_phase_tasks;
    std::vector<int>             m_phase_tasks_done;
    std::vector<DetailedTask*>   m_phase_sync_task;
    std::vector<int>             m_histogram;
    DetailedTasks              * m_detailed_tasks{nullptr};

    QueueAlg m_task_queue_alg{MostMessages};

    std::atomic<int>  m_curr_iteration{0};

    int               m_num_tasks{0};
    int               m_curr_phase{0};
    int               m_num_phases{0};
    int               m_abort_point{0};
    bool              m_abort{false};

    // OMP-specific
    int               m_num_partitions{0};
    int               m_threads_per_partition{0};

};

} // namespace Uintah
   
#endif // CCA_COMPONENTS_SCHEDULERS_KOKKOSOPENMPSCHEDULER_H
