
#ifndef UINTAH_HOMEBREW_MIXEDSCHEDULER_H
#define UINTAH_HOMEBREW_MIXEDSCHEDULER_H

#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MessageLog.h>
#include <Packages/Uintah/CCA/Components/Schedulers/ThreadPool.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/TaskProduct.h>
#include <Packages/Uintah/Core/Grid/Task.h>

#include <vector>
#include <mpi.h>

using std::vector;

namespace Uintah {

  class OnDemandDataWarehouse;
  class Task;

/**************************************

CLASS
   MixedScheduler
   
   Implements a mixed MPI/Threads version of the scheduler.

GENERAL INFORMATION

   MixedScheduler.h

   J. Davison de St. Germain
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Scheduler MPI Thread

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class MixedScheduler : public SchedulerCommon {
      struct SGArgs {
	 vector<int> dest;
	 vector<int> tags;
      };
      SGArgs sgargs; // THIS IS UGLY - Steve
      MessageLog log;
   public:
      MixedScheduler(const ProcessorGroup* myworld, Output* oport);
      virtual ~MixedScheduler();
      
      virtual void problemSetup(const ProblemSpecP& prob_spec);
      
      //////////
      // Insert Documentation Here:
     virtual void compile( const ProcessorGroup * pc );
     virtual void execute( const ProcessorGroup * pc );
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleParticleRelocation(const LevelP& level,
					      const VarLabel* old_posLabel,
					      const vector<vector<const VarLabel*> >& old_labels,
					      const VarLabel* new_posLabel,
					      const vector<vector<const VarLabel*> >& new_labels,
					      const MaterialSet* matls);


   private:
      void scatterParticles(const ProcessorGroup*,
			    const Patch* patch,
			    DataWarehouseP& old_dw,
			    DataWarehouseP& new_dw);
      void gatherParticles(const ProcessorGroup*,
			   const Patch* patch,
			   DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw);
      void displayTaskGraph( vector<Task*> & graph );

      void verifyChecksum( vector<Task*> & tasks, int me );

      void sendParticleSets( vector<Task*> & tasks, int me );
      void recvParticleSets( vector<Task*> & tasks, int me );

      void sendInitialData( vector<Task*> & tasks, int me );
      // Receive the data into the "old_dw"...
      void recvInitialData( vector<Task*> & tasks, DataWarehouseP & old_dw,
			    int me );

      // Creates the list of dependencies that each tasks needs before
      // it can be run.  Also creates a map from a dependency to the
      // task that it satisfies.
      void createDepencyList( DataWarehouseP & old_dw,
			      vector<Task*>  & tasks,
			      int              me );

      void makeAllRecvRequests( vector<Task*>       & tasks, 
				int                   me,
				DataWarehouseP      & old_dw,
				DataWarehouseP      & new_dw );

      // Removes the dependencies that are satisfied by "comps" computes
      // from the taskToDeps list.  Sends data to other processes if
      // necessary (if sendData is true).  List of sends returned in "send_ids".
#if 0
     // sparker
      void dependenciesSatisfied( const Task::compType & comps,
				  int                               me,
				  vector<MPI_Request>             & send_ids,
				  bool                              sendData = true );
      void dependencySatisfied( const Task::Dependency * comp,
				int                      me,
				vector<MPI_Request>    & send_ids,
				bool                     sendData = true );
#endif

      const VarLabel* reloc_old_posLabel;
      vector<vector<const VarLabel*> > reloc_old_labels;
      const VarLabel* reloc_new_posLabel;
      vector<vector<const VarLabel*> > reloc_new_labels;
      const MaterialSet* reloc_matls;
      const VarLabel* scatterGatherVariable;

      MixedScheduler(const MixedScheduler&);
      MixedScheduler& operator=(const MixedScheduler&);

      ThreadPool * d_threadPool;
   };
} // End namespace Uintah
   
#endif
