#ifndef UINTAH_HOMEBREW_MPISCHEDULER_H
#define UINTAH_HOMEBREW_MPISCHEDULER_H

#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MessageLog.h>
#include <Packages/Uintah/CCA/Components/Schedulers/Relocate.h>
#include <Packages/Uintah/CCA/Components/Schedulers/CommRecMPI.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouseP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/PackBufferInfo.h>
#include <Packages/Uintah/Core/Grid/TaskProduct.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/BufferInfo.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <map>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using std::vector;

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
      
    virtual void problemSetup(const ProblemSpecP& prob_spec);
      
    //////////
    // Insert Documentation Here:
    virtual void execute( const ProcessorGroup * pg);

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
		       list<DependencyBatch*>& externalRecvs );
    void processMPIRecvs( DetailedTask* task, CommRecMPI& recvs,
		       list<DependencyBatch*>& externalRecvs );    

    void postMPISends( DetailedTask* task );

    void runTask( DetailedTask* task );
    void runReductionTask( DetailedTask* task );        

    // get the processor group executing with (only valid during execute())
    const ProcessorGroup* getProcessorGroup()
    { return pg_; }
    virtual const MaterialSet* getMaterialSet(){return reloc_.getMaterialSet();}
  protected:
    virtual void actuallyCompile( const ProcessorGroup * pg );
    
    // Runs the task. (In Mixed, gives the task to a thread.)
    virtual void initiateTask( DetailedTask          * task );

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
    const ProcessorGroup* pg_;    
    mpi_timing_info_s     mpi_info_;
    CommRecMPI            sends_;
    SendState*            ss_;
    
    MPIRelocate      reloc_;
    const VarLabel * reloc_new_posLabel_;
    double           d_lasttime;
    vector<char*>    d_labels;
    vector<double>   d_times;

    void emitTime(char* label);
    void emitTime(char* label, double time);
  };

} // End namespace Uintah
   
#endif
