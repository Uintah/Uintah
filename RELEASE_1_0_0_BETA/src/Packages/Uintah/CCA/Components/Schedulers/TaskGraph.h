#ifndef UINTAH_HOMEBREW_TaskGraph_H
#define UINTAH_HOMEBREW_TaskGraph_H

#include <Packages/Uintah/Core/Grid/TaskProduct.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <vector>
#include <list>
#include <map>

namespace Uintah {

using namespace std;

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

      // Note: This just returns one of the computes for a given requires.
      // Some reduction variables may have several computes and this will
      // only return one of them.
      const Task::Dependency* getComputesForRequires(const Task::Dependency* req);

      // Get all of the requires needed from the old data warehouse
      // (carried forward).
      const vector<const Task::Dependency*>& getInitialRequires()
      { return d_initreqs; }

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

      // Makes and returns a map that associates VarLabel names with
      // the materials the variable is computed for.
      typedef map< string, list<int> > VarLabelMaterialMap;
      VarLabelMaterialMap* makeVarLabelMaterialMap();
   private:
      TaskGraph(const TaskGraph&);
      TaskGraph& operator=(const TaskGraph&);

      //////////
      // Insert Documentation Here:
      void setupTaskConnections();

      void processTask(Task* task, vector<Task*>& sortedTasks) const;
      
      vector<Task*>        d_tasks;

      typedef multimap<TaskProduct, const Task::Dependency*> actype;
      actype d_allcomps;
      
      typedef multimap<TaskProduct, const Task::Dependency*> artype;
      artype d_allreqs;

      // data required from old data warehouse
      vector<const Task::Dependency*> d_initreqs;
      
      int d_maxSerial;
   };

} // End namespace Uintah

#endif
