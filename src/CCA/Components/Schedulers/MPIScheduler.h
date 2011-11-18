/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef UINTAH_HOMEBREW_MPISCHEDULER_H
#define UINTAH_HOMEBREW_MPISCHEDULER_H

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

static DebugStream mpi_stats("MPIStats",false);

using std::vector;
using std::ofstream;

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
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Scheduler_Brain_Damaged

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class MPIScheduler : public SchedulerCommon {
  public:
    MPIScheduler(const ProcessorGroup* myworld, Output* oport, MPIScheduler* parentScheduler = 0);
    virtual ~MPIScheduler();
      
    virtual void problemSetup(const ProblemSpecP& prob_spec, SimulationStateP& state);
      
    //////////
    // Insert Documentation Here:
    virtual void execute(int tgnum = 0, int iteration = 0);

    virtual SchedulerP createSubScheduler();
    
    void postMPIRecvs( DetailedTask* task, bool only_old_recvs, int abort_point, int iteration);

    enum { TEST, WAIT_ONCE, WAIT_ALL};

    void processMPIRecvs(int how_much);    

    void postMPISends( DetailedTask* task, int iteration );

    void runTask( DetailedTask* task, int iteration );
    void runReductionTask( DetailedTask* task );        

    void addToSendList(const MPI_Request& request, int bytes, AfterCommunicationHandler* buf, const std::string& var);

    // get the processor group executing with (only valid during execute())
    const ProcessorGroup* getProcessorGroup()
    { return d_myworld; }
    
    void compile()
    {
      numMessages_=0;
      messageVolume_=0;
      SchedulerCommon::compile();
    }

    void printMPIStats()
    {
      if(mpi_stats.active())
      {
        unsigned int total_messages;
        double total_volume;

        unsigned int max_messages;
        double max_volume;

        MPI_Reduce(&numMessages_,&total_messages,1,MPI_UNSIGNED,MPI_SUM,0,d_myworld->getComm());
        MPI_Reduce(&messageVolume_,&total_volume,1,MPI_DOUBLE,MPI_SUM,0,d_myworld->getComm());
        
        MPI_Reduce(&numMessages_,&max_messages,1,MPI_UNSIGNED,MPI_MAX,0,d_myworld->getComm());
        MPI_Reduce(&messageVolume_,&max_volume,1,MPI_DOUBLE,MPI_MAX,0,d_myworld->getComm());

        if(d_myworld->myrank()==0)
        {
          mpi_stats << "MPIStats: Num Messages (avg): " << total_messages/(float)d_myworld->size() << " (max):" << max_messages << endl;
          mpi_stats << "MPIStats: Message Volume (avg): " << total_volume/(float)d_myworld->size() << " (max):" << max_volume << endl;
        }
      }
    }
    mpi_timing_info_s  mpi_info_;
    MPIScheduler*      parentScheduler;

    // Performs the reduction task. (In Mixed, gives the task to a thread.)    
    virtual void initiateReduction( DetailedTask          * task );    

  protected:
    // Runs the task. (In Mixed, gives the task to a thread.)
    virtual void initiateTask( DetailedTask          * task,
			       bool only_old_recvs, int abort_point, int iteration);


    // Waits until all tasks have finished.  In the MPI Scheduler,
    // this is basically a nop, for the mixed, it talks to the ThreadPool 
    // and waits until the threadpool in empty (ie: all tasks done.)
    virtual void wait_till_all_done();
    
    virtual void verifyChecksum();
    MessageLog  log;
    Output*     oport_;
    CommRecMPI  sends_;
    CommRecMPI  recvs_;

    double                   d_lasttime;
    std::vector<const char*> d_labels;
    std::vector<double>      d_times;
    ofstream                 timingStats, avgStats, maxStats;

    void emitTime(const char* label);
    void emitTime(const char* label, double time);

    unsigned int numMessages_;
    double messageVolume_;

  private:
    MPIScheduler(const MPIScheduler&);
    MPIScheduler& operator=(const MPIScheduler&);
  };

} // End namespace Uintah
   
#endif
