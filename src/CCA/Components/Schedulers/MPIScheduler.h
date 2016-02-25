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

#ifndef CCA_COMPONENTS_SCHEDULERS_MPISCHEDULER_H
#define CCA_COMPONENTS_SCHEDULERS_MPISCHEDULER_H

#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Components/Schedulers/MessageLog.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouseP.h>
#include <CCA/Ports/DataWarehouseP.h>

#include <Core/Parallel/PackBufferInfo.h>
#include <Core/Grid/Task.h>
#include <Core/Lockfree/Lockfree_Pool.hpp>
#include <Core/Parallel/CommunicationList.h>
#include <Core/Parallel/BufferInfo.h>
#include <Core/Util/InfoMapper.h>
#include <Core/Util/Timers/Timers.hpp>

#include <chrono>
#include <fstream>
#include <map>
#include <mutex>
#include <vector>

namespace Uintah {

namespace {

DebugStream mpi_stats("MPIStats", false);

} // namespace

class Task;

using clock_type = std::chrono::high_resolution_clock;
using nanoseconds = std::chrono::nanoseconds;

/**************************************

CLASS
   MPIScheduler

   Short description...

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

    MPIScheduler( const ProcessorGroup* myworld, const Output* oport, MPIScheduler* parentScheduler = 0 );

    virtual ~MPIScheduler();

    virtual void problemSetup( const ProblemSpecP& prob_spec, SimulationStateP& state );

    virtual void execute( int tgnum = 0, int iteration = 0 );

    virtual SchedulerP createSubScheduler();

    virtual void processMPIRecvs( int how_much );

            void postMPISends( DetailedTask* task, int iteration, int thread_id = 0 );

            void postMPIRecvs( DetailedTask* task, bool only_old_recvs, int abort_point, int iteration );

            void runTask( DetailedTask* task, int iteration, int thread_id = 0 );

    virtual void runReductionTask( DetailedTask* task );

    // get the processor group this scheduler is executing with (only valid during execute())
    const ProcessorGroup* getProcessorGroup() { return d_myworld; }

    void compile() {
      numMessages_   = 0;
      messageVolume_ = 0;
      SchedulerCommon::compile();
    }

    void printMPIStats() {
      if (mpi_stats.active()) {
        unsigned int total_messages;
        double total_volume;

        unsigned int max_messages;
        double max_volume;

        // do SUM and MAX reduction for numMessages and messageVolume
        MPI_Reduce(&numMessages_  , &total_messages, 1, MPI_UNSIGNED,MPI_SUM, 0, d_myworld->getComm());
        MPI_Reduce(&messageVolume_, &total_volume  , 1, MPI_DOUBLE,MPI_SUM  , 0, d_myworld->getComm());
        MPI_Reduce(&numMessages_  , &max_messages  , 1, MPI_UNSIGNED,MPI_MAX, 0, d_myworld->getComm());
        MPI_Reduce(&messageVolume_, &max_volume    , 1, MPI_DOUBLE,MPI_MAX  , 0, d_myworld->getComm());

        if( d_myworld->myrank() == 0 ) {
          mpi_stats << "MPIStats: Num Messages (avg): " << total_messages/(float)d_myworld->size() << " (max):" << max_messages << std::endl;
          mpi_stats << "MPIStats: Message Volume (avg): " << total_volume/(float)d_myworld->size() << " (max):" << max_volume << std::endl;
        }
      }
    }

    // timing statistics to test the mpi functionality
    enum TimingStat
    {
      TotalReduce = 0,
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

    ReductionInfoMapper< TimingStat, double > mpi_info_;

    void computeNetRunTimeStats(InfoMapper< SimulationState::RunTimeStat, double >& runTimeStats);

    MPIScheduler *  parentScheduler_;

    // Performs the reduction task. (In threaded schedulers, a single worker thread will execute this.)
    virtual void initiateReduction( DetailedTask* task );

    enum {
      TEST,
      WAIT_ONCE,
      WAIT_ALL
    };

  protected:

    virtual void initiateTask( DetailedTask* task, bool only_old_recvs, int abort_point, int iteration );

    virtual void verifyChecksum();

    void emitTime( const char* label );

    void emitTime( const char* label, double time );

    void emitNetMPIStats();

    void reduceRestartFlag( int task_graph_num  );

    void outputTimingStats( const char* label );

    MessageLog                  log;
    const Output              * oport_;

    SendCommList                m_send_list;
    RecvCommList                m_recv_list;

    std::vector<const char*>    d_labels;
    std::vector<double>         d_times;

    std::ofstream               timingStats;
    std::ofstream               maxStats;
    std::ofstream               avgStats;

    unsigned int                numMessages_;
    double                      messageVolume_;

    //-------------------------------------------------------------------------
    // The following locks are for multi-threaded schedulers that derive from MPIScheduler
    //   This eliminates miles of unnecessarily redundant code in threaded schedulers
    //-------------------------------------------------------------------------
    std::mutex      dlbLock;                // load balancer lock
    std::mutex      waittimesLock;          // MPI wait times lock

    // Timers for MPI stats
    Timers::Simple  m_last_exec_timer{};
    Timers::Simple  m_task_exec_timer{};
    Timers::Simple  m_mpi_send_timer{};
    Timers::Simple  m_total_send_timer{};
    Timers::Simple  m_mpi_recv_timer{};
    Timers::Simple  m_total_recv_timer{};
    Timers::Simple  m_mpi_test_timer{};
    Timers::Simple  m_mpi_wait_timer{};
    Timers::Simple  m_mpi_reduce_timer{};

  private:

    // disable copy, assignment, and move
    MPIScheduler( const MPIScheduler & )            = delete;
    MPIScheduler& operator=( const MPIScheduler & ) = delete;
    MPIScheduler( MPIScheduler &&)                  = delete;
    MPIScheduler& operator=( MPIScheduler && )      = delete;

};

} // End namespace Uintah

#endif // End CCA_COMPONENTS_SCHEDULERS_MPISCHEDULER_H
