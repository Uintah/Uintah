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
  class DetailedTask;
  class DetailedTasks;
  class LoadBalancer;

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
      ~TaskGraph();
      
      //////////
      // Insert Documentation Here:
      void initialize();
      
      //////////
      // Insert Documentation Here:
      void addTask(Task* t, const PatchSet* patchset,
			   const MaterialSet* matlset);

      //////////
      // Insert Documentation Here:
      void topologicalSort(vector<Task*>& tasks);

     DetailedTasks* createDetailedTasks(const ProcessorGroup* pg);
     void createDetailedDependencies(DetailedTasks*, LoadBalancer* lb,
				     const ProcessorGroup* pg);

      //////////
      // Used for the MixedScheduler, this routine has the side effect
      // (just like the topological sort) of adding the reduction tasks.
      // However, this routine leaves the tasks in the order they were
      // added, so that reduction tasks are hit in the correct order
      // by each MPI process.
      void nullSort( vector<Task*>& tasks );
      
      // Get all of the requires needed from the old data warehouse
      // (carried forward).
      const vector<const Task::Dependency*>& getInitialRequires()
      { return d_initreqs; }

      int getNumTasks() const;
      Task* getTask(int i);

      ////////// 
      // Assigns unique id numbers to each dependency based on name,
      // material index, and patch.  In other words, even though a
      // number of tasks depend on the same data, they create there
      // own copy of the dependency data.  This routine determines
      // that the dependencies are actually the same, and gives them
      // the same id number.
      void assignUniqueMessageTags();

      vector<Task*>& getTasks() {
	 return d_tasks;
      }

      // Makes and returns a map that associates VarLabel names with
      // the materials the variable is computed for.
      typedef map< string, list<int> > VarLabelMaterialMap;
      VarLabelMaterialMap* makeVarLabelMaterialMap();
   private:
     void createDetailedTask(DetailedTasks* tasks, Task* task,
			     const PatchSubset* patches,
			     const MaterialSubset* matls);
     int findVariableLocation(LoadBalancer* lb, const ProcessorGroup* pg,
			      Task::Dependency* req,
			      const Patch* patch, int matl);

      TaskGraph(const TaskGraph&);
      TaskGraph& operator=(const TaskGraph&);

     bool overlaps(Task::Dependency* comp, Task::Dependency* req) const;

      //////////
      // Insert Documentation Here:
      void setupTaskConnections();

      void processTask(Task* task, vector<Task*>& sortedTasks) const;
      
      vector<Task*>        d_tasks;
     vector<Task::Edge*> edges;

      // data required from old data warehouse
      vector<const Task::Dependency*> d_initreqs;
      
   };

} // End namespace Uintah

#endif
