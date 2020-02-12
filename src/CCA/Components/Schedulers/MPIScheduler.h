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

#ifndef CCA_COMPONENTS_SCHEDULERS_MPISCHEDULER_H
#define CCA_COMPONENTS_SCHEDULERS_MPISCHEDULER_H

#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Components/Schedulers/DetailedTask.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouseP.h>
#include <CCA/Ports/DataWarehouseP.h>

#include <Core/Parallel/CommunicationList.hpp>
#include <Core/Util/InfoMapper.h>
#include <Core/Util/Timers/Timers.hpp>

#include <fstream>
#include <vector>

namespace Uintah {

/**************************************

CLASS
   MPIScheduler


GENERAL INFORMATION

   MPIScheduler.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


KEYWORDS
   MPI Scheduler

DESCRIPTION
   Static task ordering and deterministic execution with MPI. One MPI rank per CPU core.

****************************************/

class MPIScheduler : public SchedulerCommon {

  public:

    MPIScheduler( const ProcessorGroup* myworld, MPIScheduler* parentScheduler = 0 );

    virtual ~MPIScheduler();

    virtual void problemSetup( const ProblemSpecP& prob_spec, const MaterialManagerP& materialManager );

    virtual void execute( int tgnum = 0, int iteration = 0 );

    virtual SchedulerP createSubScheduler();

    virtual void processMPIRecvs( int test_type );

            void postMPISends( DetailedTask* dtask, int iteration );

            void postMPIRecvs( DetailedTask* dtask, bool only_old_recvs, int abort_point, int iteration );

            void runTask( DetailedTask* dtask, int iteration );

    virtual void runReductionTask( DetailedTask* dtask );

    void compile() {
      m_num_messages   = 0;
      m_message_volume = 0;
      SchedulerCommon::compile();
    }

    // Performs the reduction task. (In threaded, Unified scheduler, a single worker thread will execute this.)
    virtual void initiateReduction( DetailedTask* dtask );

    void computeNetRuntimeStats();

    // timing statistics for Uintah infrastructure overhead
    enum TimingStatsEnum {
        TotalSend = 0
      , TotalRecv
      , TotalTest
      , TotalWait
      , TotalReduce
      , TotalTask
    };
    
    enum {
        TEST
      , WAIT_ONCE
      , WAIT_ALL
    };

    ReductionInfoMapper< TimingStatsEnum, double > m_mpi_info;
    MapInfoMapper< std::string, TaskStatsEnum, double > m_task_info;

    MPIScheduler* m_parent_scheduler{nullptr};

  protected:

    virtual void initiateTask( DetailedTask * dtask, bool only_old_recvs, int abort_point, int iteration );

    virtual void verifyChecksum();

    void emitTime( const char* label, double time );

    void outputTimingStats( const char* label );

    CommRequestPool             m_sends{};
    CommRequestPool             m_recvs{};

    std::vector<const char*>    m_labels;
    std::vector<double>         m_times;

    std::ofstream               m_max_stats;
    std::ofstream               m_avg_stats;

    std::atomic<unsigned int>   m_num_messages{0};
    double                      m_message_volume{0.0};

    Timers::Simple              m_exec_timer;
  
  private:

    // eliminate copy, assignment and move
    MPIScheduler( const MPIScheduler & )            = delete;
    MPIScheduler& operator=( const MPIScheduler & ) = delete;
    MPIScheduler( MPIScheduler && )                 = delete;
    MPIScheduler& operator=( MPIScheduler && )      = delete;
};

} // End namespace Uintah

#endif // End CCA_COMPONENTS_SCHEDULERS_MPISCHEDULER_H
