/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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
#include <CCA/Components/Schedulers/CommRecMPI.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouseP.h>
#include <CCA/Ports/DataWarehouseP.h>

#include <Core/Parallel/PackBufferInfo.h>
#include <Core/Grid/Task.h>
#include <Core/Parallel/BufferInfo.h>

#include <vector>
#include <map>
#include <fstream>

namespace Uintah {

static DebugStream mpi_stats("MPIStats", false);

class Task;

struct mpi_timing_info_s {
  double totalreduce;
  double totalsend;
  double totalrecv;
  double totaltask;
  double totalreducempi;
  double totalsendmpi;
  double totalrecvmpi;
  double totaltestmpi;
  double totalwaitmpi;
};

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
    
    void compile(const LevelSet* levelSet = NULL) {
      numMessages_   = 0;
      messageVolume_ = 0;
      SchedulerCommon::compile(levelSet);
    }

    void printMPIStats() {
      if (mpi_stats.active()) {
        unsigned int total_messages;
        double total_volume;

        unsigned int max_messages;
        double max_volume;

        // do SUM and MAX reduction for numMessages and messageVolume
        MPI_Reduce(&numMessages_,&total_messages,1,MPI_UNSIGNED,MPI_SUM,0,d_myworld->getComm());
        MPI_Reduce(&messageVolume_,&total_volume,1,MPI_DOUBLE,MPI_SUM,0,d_myworld->getComm());
        MPI_Reduce(&numMessages_,&max_messages,1,MPI_UNSIGNED,MPI_MAX,0,d_myworld->getComm());
        MPI_Reduce(&messageVolume_,&max_volume,1,MPI_DOUBLE,MPI_MAX,0,d_myworld->getComm());

        if( d_myworld->myrank() == 0 ) {
          mpi_stats << "MPIStats: Num Messages (avg): " << total_messages/(float)d_myworld->size() << " (max):" << max_messages << std::endl;
          mpi_stats << "MPIStats: Message Volume (avg): " << total_volume/(float)d_myworld->size() << " (max):" << max_volume << std::endl;
        }
      }
    }

    mpi_timing_info_s   mpi_info_;
    MPIScheduler*       parentScheduler_;

    // Performs the reduction task. (In threaded schdeulers, a single worker thread will execute this.)
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

    void outputTimingStats( const char* label );

    MessageLog                  log;
    const Output*               oport_;
    CommRecMPI                  sends_[MAX_THREADS];
    CommRecMPI                  recvs_;

    double                      d_lasttime;
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
    // multiple reader, single writer lock (pthread_rwlock_t wrapper)
    mutable CrowdMonitor        recvLock;               // CommRecMPI recvs lock
    mutable CrowdMonitor        sendLock;               // CommRecMPI sends lock
    Mutex                       dlbLock;                // load balancer lock
    Mutex                       waittimesLock;          // MPI wait times lock

  private:

    // Disable copy and assignment
    MPIScheduler( const MPIScheduler& );
    MPIScheduler& operator=( const MPIScheduler& );
};

} // End namespace Uintah
   
#endif // End CCA_COMPONENTS_SCHEDULERS_MPISCHEDULER_H
