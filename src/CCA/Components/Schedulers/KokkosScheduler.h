/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef CCA_COMPONENTS_SCHEDULERS_KOKKOSSCHEDULER_H
#define CCA_COMPONENTS_SCHEDULERS_KOKKOSSCHEDULER_H

#include <CCA/Components/Schedulers/MPIScheduler.h>

#include <string>
#include <vector>

namespace Uintah {

class Task;
class DetailedTask;

/**************************************

CLASS
   KokkosScheduler


GENERAL INFORMATION
   KokkosScheduler.h

   Alan Humphrey, John Holmen, Brad Peterson, Damodar Sahasrabudhe
   Scientific Computing and Imaging Institute
   University of Utah


KEYWORDS
   Task Scheduler, Multi-threaded MPI, Kokkos, OpenMP, CPU, GPU

DESCRIPTION A multi-threaded scheduler that uses a combination of MPI
   + OpenMP and offers support for Kokkos-enabled CPU and GPU
   tasks. This relies on the Kokkos OpenMP back-end to manage OpenMP
   partitioning. Each partition has a "master" thread which runs a
   given functor.

   OpenMP 4.0 is required for this scheduler.

   This scheduler is designed primarily to support scheduling and
   executing a mixture of Kokkos-enabled tasks across CPU and GPU.

WARNING
   This scheduler is EXPERIMENTAL and undergoing extensive development.
   Not all tasks/components are Kokkos-enabled and/or thread-safe

   Requires MPI_THREAD_MULTIPLE support.

****************************************/


class KokkosScheduler : public MPIScheduler  {

  public:

    KokkosScheduler( const ProcessorGroup * myworld,
                     KokkosScheduler * parentScheduler = nullptr );

    virtual ~KokkosScheduler();

    virtual void problemSetup( const ProblemSpecP & prob_spec,
                               const MaterialManagerP & materialManager );

    virtual SchedulerP createSubScheduler();

    virtual void execute( int tgnum = 0, int iteration = 0 );

    virtual bool useInternalDeps() { return !m_is_copy_data_timestep; }

    void runTasks( int thread_id );

    static std::string myRankThread();

    static int verifyAnyGpuActive();  // used only to check if this
                                      // Uintah build can communicate
                                      // with a GPU.  This function
                                      // exits the program

  private:

    // eliminate copy, assignment and move
    KokkosScheduler( const KokkosScheduler & )            = delete;
    KokkosScheduler& operator=( const KokkosScheduler & ) = delete;
    KokkosScheduler( KokkosScheduler && )                 = delete;
    KokkosScheduler& operator=( KokkosScheduler && )      = delete;

    void markTaskConsumed( int & numTasksDone, int & currphase, int numPhases, DetailedTask * dtask );

    void runTask( DetailedTask * dtask , int iteration , int thread_id , CallBackEvent event );

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
    int      m_abort_point{0};
    bool     m_abort{false};
};

} // namespace Uintah

#endif // CCA_COMPONENTS_SCHEDULERS_KOKKOSSCHEDULER_H
