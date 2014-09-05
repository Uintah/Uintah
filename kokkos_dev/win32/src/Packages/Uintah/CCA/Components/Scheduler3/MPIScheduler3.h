#ifndef UINTAH_HOMEBREW_MPISCHEDULER3_H
#define UINTAH_HOMEBREW_MPISCHEDULER3_H

#include <Packages/Uintah/CCA/Components/Scheduler3/Scheduler3Common.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MessageLog.h>
#include <Packages/Uintah/CCA/Components/Schedulers/Relocate.h>
#include <Packages/Uintah/CCA/Components/Schedulers/CommRecMPI.h>
#include <Packages/Uintah/CCA/Components/Scheduler3/PatchBasedDataWarehouse3P.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Parallel/PackBufferInfo.h>
 
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Parallel/BufferInfo.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <map>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using std::vector;

class Task;
 class DetailedTask3;
class SendState;

struct mpi_timing_info_s3 {
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

  class MPIScheduler3 : public Scheduler3Common {
    MessageLog log;
  public:
    MPIScheduler3(const ProcessorGroup* myworld, Output* oport, MPIScheduler3* parentScheduler = 0);
    virtual ~MPIScheduler3();
      
    virtual void problemSetup(const ProblemSpecP& prob_spec);
      
    //////////
    // Insert Documentation Here:
    virtual void execute();

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
    
    void postMPIRecvs( DetailedTask3* task, CommRecMPI& recvs,
		       list<DependencyBatch*>& externalRecvs,
		       bool only_old_recvs, int abort_point);
    void processMPIRecvs( DetailedTask3* task, CommRecMPI& recvs,
		       list<DependencyBatch*>& externalRecvs );    

    void postMPISends( DetailedTask3* task );

    void runTask( DetailedTask3* task );
    void runReductionTask( DetailedTask3* task );        

    // get the processor group executing with (only valid during execute())
    const ProcessorGroup* getProcessorGroup()
    { return d_myworld; }
    virtual const MaterialSet* getMaterialSet() const {return reloc_.getMaterialSet();}
  protected:
    virtual void actuallyCompile();
    
    // Runs the task. (In Mixed, gives the task to a thread.)
    virtual void initiateTask( DetailedTask3          * task,
			       bool only_old_recvs, int abort_point);

    // Performs the reduction task. (In Mixed, gives the task to a thread.)    
    virtual void initiateReduction( DetailedTask3          * task );    

    // Waits until all tasks have finished.  In the MPI Scheduler,
    // this is basically a nop, for the mixed, it talks to the ThreadPool 
    // and waits until the threadpool in empty (ie: all tasks done.)
    virtual void wait_till_all_done();

  private:
    MPIScheduler3(const MPIScheduler3&);
    MPIScheduler3& operator=(const MPIScheduler3&);
    
    virtual void verifyChecksum();

    MPIScheduler3* parentScheduler;
    mpi_timing_info_s3     mpi_info_;
    CommRecMPI            sends_;
    SendState*            ss_;
    SendState*            rs_;
    
    MPIRelocate      reloc_;
    const VarLabel * reloc_new_posLabel_;
    double           d_lasttime;
    vector<char*>    d_labels;
    vector<double>   d_times;
    bool             d_logTimes;

    void emitTime(char* label);
    void emitTime(char* label, double time);
  };

} // End namespace Uintah
   
#endif
