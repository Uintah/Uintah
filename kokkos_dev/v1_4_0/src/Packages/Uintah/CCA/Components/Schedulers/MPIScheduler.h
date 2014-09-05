#ifndef UINTAH_HOMEBREW_MPISCHEDULER_H
#define UINTAH_HOMEBREW_MPISCHEDULER_H

#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MessageLog.h>
#include <Packages/Uintah/CCA/Components/Schedulers/Relocate.h>
#include <Packages/Uintah/CCA/Components/Schedulers/CommRecMPI.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/PackBufferInfo.h>
#include <Packages/Uintah/Core/Grid/TaskProduct.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/BufferInfo.h>
#include <vector>
#include <map>

using std::vector;

namespace Uintah {

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
    MPIScheduler(const ProcessorGroup* myworld, Output* oport);
    virtual ~MPIScheduler();
      
    virtual void problemSetup(const ProblemSpecP& prob_spec);
      
    //////////
    // Insert Documentation Here:
    virtual void compile( const ProcessorGroup * pc, bool init_timestep);
    virtual void execute( const ProcessorGroup * pc);
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
    
    static void recvMPIData( const ProcessorGroup  * pg,
			     DetailedTask          * task, 
			     mpi_timing_info_s     & mpi_info,
			     OnDemandDataWarehouse * dws[2] );

    static void sendMPIData( const ProcessorGroup  * pg,
			     DetailedTask          * task,
			     mpi_timing_info_s     & mpi_info,
			     SendRecord            & sends,
			     SendState             & ss,
			     OnDemandDataWarehouse * dws[2],
			     const VarLabel        * reloc_label );

  protected:
    // Runs the task. (In Mixed, gives the task to a thread.)
    virtual void initiateTask( const ProcessorGroup  * pg,
			       DetailedTask          * task,
			       mpi_timing_info_s     & mpi_info,
			       SendRecord            & sends,
			       SendState             & ss,
			       OnDemandDataWarehouse * dws[2],
			       const VarLabel        * reloc_label );

    // Waits until all tasks have finished.  In the MPI Scheduler,
    // this is basically a nop, for the mixed, it talks to the ThreadPool 
    // and waits until the threadpool in empty (ie: all tasks done.)
    virtual void wait_till_all_done();

  private:
    MPIScheduler(const MPIScheduler&);
    MPIScheduler& operator=(const MPIScheduler&);
    
    virtual void verifyChecksum();

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
