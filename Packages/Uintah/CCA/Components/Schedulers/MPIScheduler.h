#ifndef UINTAH_HOMEBREW_MPISCHEDULER_H
#define UINTAH_HOMEBREW_MPISCHEDULER_H

#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MessageLog.h>
#include <Packages/Uintah/CCA/Components/Schedulers/Relocate.h>
#include <Packages/Uintah/CCA/Components/Schedulers/CommRecMPI.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouseP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Parallel/PackBufferInfo.h>
 
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Parallel/BufferInfo.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <map>
#include <fstream>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using std::vector;
using std::ofstream;

class Task;
class SendState;

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
  long long totalcommflops;
  long long totalexecflops;
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
    MessageLog log;
  public:
    MPIScheduler(const ProcessorGroup* myworld, Output* oport, MPIScheduler* parentScheduler = 0);
    virtual ~MPIScheduler();
      
    virtual void problemSetup(const ProblemSpecP& prob_spec,
                              SimulationStateP& state);
      
    //////////
    // Insert Documentation Here:
    virtual void execute(int tgnum = 0, int iteration = 0);

    virtual SchedulerP createSubScheduler();
      
    //////////
    // Insert Documentation Here:
    virtual void scheduleParticleRelocation(const LevelP& level,
					    const VarLabel* old_posLabel,
					    const vector<vector<const VarLabel*> >& old_labels,
					    const VarLabel* new_posLabel,
					    const vector<vector<const VarLabel*> >& new_labels,
					    const VarLabel* particleIDLabel,
					    const MaterialSet* matls);
    
    void postMPIRecvs( DetailedTask* task, CommRecMPI& recvs,
		       list<DependencyBatch*>& externalRecvs,
		       bool only_old_recvs, int abort_point, int iteration);
    void processMPIRecvs( DetailedTask* task, CommRecMPI& recvs,
		       list<DependencyBatch*>& externalRecvs );    

    void postMPISends( DetailedTask* task, int iteration );

    void runTask( DetailedTask* task, int iteration );
    void runReductionTask( DetailedTask* task );        

    // get the processor group executing with (only valid during execute())
    const ProcessorGroup* getProcessorGroup()
    { return d_myworld; }
    virtual const MaterialSet* getMaterialSet() const {return reloc_.getMaterialSet();}
  protected:
    // Runs the task. (In Mixed, gives the task to a thread.)
    virtual void initiateTask( DetailedTask          * task,
			       bool only_old_recvs, int abort_point, int iteration);

    // Performs the reduction task. (In Mixed, gives the task to a thread.)    
    virtual void initiateReduction( DetailedTask          * task );    

    // Waits until all tasks have finished.  In the MPI Scheduler,
    // this is basically a nop, for the mixed, it talks to the ThreadPool 
    // and waits until the threadpool in empty (ie: all tasks done.)
    virtual void wait_till_all_done();

  private:
    MPIScheduler(const MPIScheduler&);
    MPIScheduler& operator=(const MPIScheduler&);
    
    virtual void verifyChecksum();

    MPIScheduler* parentScheduler;
    Output*       oport_;
    mpi_timing_info_s     mpi_info_;
    CommRecMPI            sends_;
    SendState*            ss_;
    SendState*            rs_;
    
    MPIRelocate      reloc_;
    double           d_lasttime;
    vector<char*>    d_labels;
    vector<double>   d_times;
    ofstream         timingStats, avgStats, maxStats;


    void emitTime(char* label);
    void emitTime(char* label, double time);
  };

} // End namespace Uintah
   
#endif
