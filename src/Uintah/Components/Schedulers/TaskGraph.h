#ifndef UINTAH_HOMEBREW_TaskGraph_H
#define UINTAH_HOMEBREW_TaskGraph_H

#include <Uintah/Grid/TaskProduct.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/Patch.h>
#include <vector>
#include <list>
#include <map>

namespace Uintah {

using std::map;
using std::vector;
using std::list;

struct TaskData {

  const Task * task;

  TaskData() : task( 0 ) {}
  TaskData( const Task * task ) : task( task ) {}

  // Used for STL "map" ordering:
  bool operator()( const TaskData & t1, const TaskData & t2 ) const {
    return t1.task < t2.task;
  }

  bool operator<( const TaskData & t ) const {
    return task < t.task;
  }

  bool operator==( const TaskData & t ) const {
    if( t.task->getName() == task->getName() ) {
      if( t.task->getPatch() && task->getPatch() )
	return ( t.task->getPatch()->getID() == task->getPatch()->getID() );
      else if( ( t.task->getPatch() && !task->getPatch() ) ||
	       ( !t.task->getPatch() && task->getPatch() ) )
	return false;
      else
	return true;
    } else {
      return false;
    }
  }
};

struct DependData {

  const Task::Dependency * dep;

  DependData() : dep( 0 ) {}
  DependData( const Task::Dependency * dep ) : dep( dep ){}

  bool operator()( const DependData & d1, const DependData & d2 ) const;
  bool operator< ( const DependData & d ) const;
  bool operator==( const DependData & d1 ) const;
};

/**************************************

CLASS
   TaskGraph
   
   Short description...

GENERAL INFORMATION

   TaskGraph.h

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

   class TaskGraph {
   public:
      TaskGraph();
      virtual ~TaskGraph();
      
      //////////
      // Insert Documentation Here:
      virtual void initialize();
      
      //////////
      // Insert Documentation Here:
      virtual void addTask(Task* t);

      //////////
      // Insert Documentation Here:
      virtual void topologicalSort(vector<Task*>& tasks);

      //////////
      // Used for the MixedScheduler, this routine has the side effect
      // (just like the topological sort) of adding the reduction tasks.
      // However, this routine leaves the tasks in the order they were
      // added, so that reduction tasks are hit in the correct order
      // by each MPI process.
      void nullSort( vector<Task*>& tasks );
      
      //////////
      // Insert Documentation Here:
      bool allDependenciesCompleted(Task* task) const;

      void getRequiresForComputes(const Task::Dependency* comp,
				  vector<const Task::Dependency*>& reqs);

      const Task::Dependency* getComputesForRequires(const Task::Dependency* req);

      int getNumTasks() const;
      Task* getTask(int i);
      void assignSerialNumbers();

      ////////// 
      // Assigns unique id numbers to each dependency based on name,
      // material index, and patch.  In other words, even though a
      // number of tasks depend on the same data, they create there
      // own copy of the dependency data.  This routine determines
      // that the dependencies are actually the same, and gives them
      // the same id number.
      void assignUniqueSerialNumbers();

      vector<Task*>& getTasks() {
	 return d_tasks;
      }

      int getMaxSerialNumber() const {
	 return d_maxSerial;
      }

      // Makes and returns a map that maps strings to VarLabels of
      // that name and a list of material indices for which that
      // variable is valid (at least according to d_allcomps).
      typedef map< string, pair< const VarLabel*, list<int> > >
              VarLabelMaterialMap;
      VarLabelMaterialMap* makeVarLabelMaterialMap();
   private:
      TaskGraph(const TaskGraph&);
      TaskGraph& operator=(const TaskGraph&);

      //////////
      // Insert Documentation Here:
      void setupTaskConnections();

      void processTask(Task* task, vector<Task*>& sortedTasks) const;
      
      vector<Task*>        d_tasks;

      typedef map<TaskProduct, const Task::Dependency*> actype;
      actype d_allcomps;

      typedef multimap<TaskProduct, const Task::Dependency*> artype;
      artype d_allreqs;
      int d_maxSerial;
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.7  2000/12/10 09:06:12  sparker
// Merge from csafe_risky1
//
// Revision 1.6  2000/12/06 23:54:26  witzel
// Added makeVarLabelMaterialMap method
//
// Revision 1.5.4.2  2000/10/17 01:01:09  sparker
// Added optimization of getRequiresForComputes
//
// Revision 1.5.4.1  2000/10/10 05:28:04  sparker
// Added support for NullScheduler (used for profiling taskgraph overhead)
//
// Revision 1.5  2000/09/27 02:14:12  dav
// Added support for mixed model
//
// Revision 1.4  2000/07/27 22:39:47  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.3  2000/07/25 20:59:28  jehall
// - Simplified taskgraph output implementation
// - Sort taskgraph edges; makes critical path algorithm eastier
//
// Revision 1.2  2000/07/19 21:47:59  jehall
// - Changed task graph output to XML format for future extensibility
// - Added statistical information about tasks to task graph output
//
// Revision 1.1  2000/06/17 07:04:56  sparker
// Implemented initial load balancer modules
// Use ProcessorGroup
// Implemented TaskGraph - to contain the common scheduling stuff
//
//

#endif
