#ifndef UINTAH_HOMEBREW_BRAINDAMAGEDSCHEDULER_H
#define UINTAH_HOMEBREW_BRAINDAMAGEDSCHEDULER_H

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/TaskProduct.h>
#include <Uintah/Grid/Task.h>
#include <vector>
#include <map>

namespace SCICore {
    namespace Thread {
	class SimpleReducer;
	class ThreadPool;
    }
}

namespace Uintah {
   class Task;
   class ProcessorContext;
/**************************************

CLASS
   BrainDamagedScheduler
   
   Short description...

GENERAL INFORMATION

   BrainDamagedScheduler.h

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

   class BrainDamagedScheduler : public UintahParallelComponent, public Scheduler {
   public:
      BrainDamagedScheduler( int MpiRank, int MpiProcesses );
      virtual ~BrainDamagedScheduler();
      
      //////////
      // Insert Documentation Here:
      virtual void initialize();
      
      //////////
      // Insert Documentation Here:
      virtual void execute( const ProcessorContext * pc,
			          DataWarehouseP   & dwp );
      
      //////////
      // Insert Documentation Here:
      virtual void addTarget(const VarLabel*);
      
      //////////
      // Insert Documentation Here:
      virtual void addTask(Task* t);
      
      //////////
      // Insert Documentation Here:
      virtual DataWarehouseP createDataWarehouse( int generation );
      
      //////////
      // Insert Documentation Here:
      void setNumThreads(int numThreads);
      
   private:
      BrainDamagedScheduler(const BrainDamagedScheduler&);
      BrainDamagedScheduler& operator=(const BrainDamagedScheduler&);

      struct TaskRecord {
	 TaskRecord(Task*);
	 ~TaskRecord();

	 Task*                    task;
	 bool visited;
      };

      //////////
      // Insert Documentation Here:
      bool allDependenciesCompleted(TaskRecord* task) const;

      void performTask(TaskRecord* task, const ProcessorContext * pc) const;

      //////////
      // Insert Documentation Here:
      void setupTaskConnections();
      
      //////////
      // Insert Documentation Here:
      void runThreadedTask(int, TaskRecord*, const ProcessorContext*,
			   SCICore::Thread::SimpleReducer*);
      
      //////////
      // Insert Documentation Here:
      void dumpDependencies();

      SCICore::Thread::SimpleReducer* d_reducer;

      std::vector<TaskRecord*>        d_tasks;
      std::vector<const VarLabel*>    d_targets;

      SCICore::Thread::ThreadPool*    d_pool;
      int                             d_numThreads;
      std::map<TaskProduct, TaskRecord*>   d_allcomps;
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.11  2000/05/19 18:35:10  jehall
// - Added code to dump the task dependencies to a file, which can be made
//   into a pretty dependency graph.
//
// Revision 1.10  2000/05/11 20:10:19  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.9  2000/05/07 06:02:07  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.8  2000/05/05 06:42:43  dav
// Added some _hopefully_ good code mods as I work to get the MPI stuff to work.
//
// Revision 1.7  2000/04/26 06:48:32  sparker
// Streamlined namespaces
//
// Revision 1.6  2000/04/20 18:56:26  sparker
// Updates to MPM
//
// Revision 1.5  2000/04/19 21:20:02  dav
// more MPI stuff
//
// Revision 1.4  2000/04/19 05:26:10  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.3  2000/03/17 01:03:16  dav
// Added some cocoon stuff, fixed some namespace stuff, etc
//
//

#endif
