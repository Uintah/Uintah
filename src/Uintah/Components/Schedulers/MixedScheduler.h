#ifndef UINTAH_HOMEBREW_MIXEDSCHEDULER_H
#define UINTAH_HOMEBREW_MIXEDSCHEDULER_H

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Components/Schedulers/TaskGraph.h>
#include <Uintah/Components/Schedulers/MessageLog.h>
#include <Uintah/Components/Schedulers/ThreadPool.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/TaskProduct.h>
#include <Uintah/Grid/Task.h>

#include <vector>
#include <mpi.h>

using std::vector;

namespace Uintah {
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

   class MixedScheduler : public UintahParallelComponent, public Scheduler {
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
      virtual void initialize();
      
      //////////
      // Insert Documentation Here:
      virtual void execute( const ProcessorGroup * pc,
			          DataWarehouseP   & old_dwp,
			          DataWarehouseP   & dwp );
      
      //////////
      // Insert Documentation Here:
      virtual void addTask(Task* t);

      //////////
      // Insert Documentation Here:
      virtual DataWarehouseP createDataWarehouse( DataWarehouseP& parent);
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleParticleRelocation(const LevelP& level,
					      DataWarehouseP& old_dw,
					      DataWarehouseP& new_dw,
					      const VarLabel* old_posLabel,
					      const vector<vector<const VarLabel*> >& old_labels,
					      const VarLabel* new_posLabel,
					      const vector<vector<const VarLabel*> >& new_labels,
					      int numMatls);


       virtual LoadBalancer* getLoadBalancer();
       virtual void releaseLoadBalancer();
       
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
      void dependenciesSatisfied( const vector<Task::Dependency*> & comps,
				  int                               me,
				  vector<MPI_Request>             & send_ids,
				  bool                              sendData = true );
      void dependencySatisfied( const Task::Dependency * comp,
				int                      me,
				vector<MPI_Request>    & send_ids,
				bool                     sendData = true );

      const VarLabel* reloc_old_posLabel;
      vector<vector<const VarLabel*> > reloc_old_labels;
      const VarLabel* reloc_new_posLabel;
      vector<vector<const VarLabel*> > reloc_new_labels;
      int reloc_numMatls;
      const VarLabel* scatterGatherVariable;

      MixedScheduler(const MixedScheduler&);
      MixedScheduler& operator=(const MixedScheduler&);

      TaskGraph    d_graph;
      int          d_generation;

      ThreadPool * d_threadPool;
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/09/28 02:15:51  dav
// updates due to not sending 0 particles
//
// Revision 1.2  2000/09/27 01:47:47  dav
// added missing #endif
//
// Revision 1.1  2000/09/26 18:50:26  dav
// Initial commit.  These files are derived from Steve's MPIScheduler,
// and thus have a lot in common.  Perhaps in the future, the common
// routines should be moved into a common location.
//
//

#endif
