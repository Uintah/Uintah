#ifndef UINTAH_HOMEBREW_TaskGraph3_H
#define UINTAH_HOMEBREW_TaskGraph3_H

 
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  using namespace std;
  class DetailedTask3;
  class DetailedTasks3;
  class LoadBalancer;

/**************************************

CLASS
   TaskGraph
   
   During the TaskGraph compilation, the task graph does its work in
   two phases.  The first is the call to TaskGraph::createDetailedTasks()
   which adds edges between the tasks' requires and computes and orders the
   tasks for execution.

   Here is a function call tree for this phase:
   createDetailedTasks
     topologicalSort
       setupTaskConnections
         addDependencyEdges
       processTask
         processDependencies
     LoadBalancer::createNeighborhood (this stores the patches that border
                                       patches on the current processor)

   The second phase begins with a call to 
   TaskGraph::createDetailedDependencies, which sets up the data that need
   to be communicated between processors when the taskgraph executes.

   Here is a function call tree for this phase:
   createDetailedDependencies (public)
     remembercomps
     createDetailedDependencies (private)
       DetailedTasks::possiblyCreateDependency or Task::addInternalDependency

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
  class CompTable3;
  class Scheduler3Common;

  // this is so we can keep task independent of taskgraph
  struct GraphSortInfo3 {
    GraphSortInfo3() { visited = false; sorted = false; }
    bool visited;
    bool sorted;
  };

  typedef map<Task*, GraphSortInfo3> GraphSortInfo3Map;

   class TaskGraph3 {
   public:
     TaskGraph3(Scheduler3Common* sc, const ProcessorGroup* pg);
     ~TaskGraph3();

     /// Clears the TaskGraph and deletes all tasks.
     void initialize();
     
     /// Adds a task to the task graph.  If the task is empty, it 
     /// deletes it.  Also, as each task is added, it updates the list
     /// of vars that are required from the old DW
     void addTask(Task* t, const PatchSet* patchset,
                  const MaterialSet* matlset);
     
     /// sets up the task connections and puts them in a sorted order.
     /// Calls setupTaskConnections, which has the side effect of creating
     /// reduction tasks for tasks that compute reduction variables.  
     /// calls processTask on each task to sort them.
     void topologicalSort(vector<Task*>& tasks);
     

     /// Sorts the tasks, and makes DetailedTask's out of them, 
     /// and loads them into a new DetailedTasks object. (There is one 
     /// DetailedTask for each PatchSubset and MaterialSubset in a Task, 
     /// where a Task may have many PatchSubsets and MaterialSubsets.).
     /// Sorts using topologicalSort.
     DetailedTasks3* createDetailedTasks( LoadBalancer* lb,
                                          bool useInternalDeps );

     /// This will go through the detailed tasks and create the 
     /// dependencies need to communicate data across separate
     /// processors.  Calls the private createDetailedDependencies
     /// for each task as a helper.
     void createDetailedDependencies(DetailedTasks3*, LoadBalancer* lb);

     /// Connects the tasks, but does not sort them.
     /// Used for the MixedScheduler, this routine has the side effect
     /// (just like the topological sort) of adding the reduction tasks.
     /// However, this routine leaves the tasks in the order they were
     /// added, so that reduction tasks are hit in the correct order
     /// by each MPI process.
     void nullSort( vector<Task*>& tasks );
     
     /// Get all of the requires needed from the old data warehouse
     /// (carried forward).
     const vector<const Task::Dependency*>& getInitialRequires() const
       { return d_initRequires; }

     /// Set the requires need from the old data warehouse.
     const set<const VarLabel*, VarLabel::Compare>&
       getInitialRequiredVars() const
       { return d_initRequiredVars; }

     int getNumTasks() const;
     Task* getTask(int i);

     /// Assigns unique id numbers to each dependency based on name,
     /// material index, and patch.  In other words, even though a
     /// number of tasks depend on the same data, they create there
     /// own copy of the dependency data.  This routine determines
     /// that the dependencies are actually the same, and gives them
     /// the same id number.
     void assignUniqueMessageTags();
     
     vector<Task*>& getTasks() {
       return d_tasks;
     }

     /// Makes and returns a map that associates VarLabel names with
     /// the materials the variable is computed for.
     typedef map< string, list<int> > VarLabelMaterialMap;
     VarLabelMaterialMap* makeVarLabelMaterialMap();
   private:
     typedef multimap<const VarLabel*, Task::Dependency*, VarLabel::Compare>
       CompMap;

     /// Helper function for processTasks, processing the dependencies
     /// for the given task in the dependency list whose head is req.
     /// Will call processTask (recursively, as this is a helper for 
     /// processTask) for each dependent task.
     void processDependencies(Task* task, Task::Dependency* req,
			      vector<Task*>& sortedTasks,
                              GraphSortInfo3Map& sortinfo) const;
     
     /// Helper function for setupTaskConnections, adding dependency edges
     /// for the given task for each of the require (or modify) depencies in
     /// the list whose head is req.  If modifies is true then each found
     /// compute will be replaced by its modifying dependency on the CompMap.
     void addDependencyEdges(Task* task, GraphSortInfo3Map& sortinfo, 
                             Task::Dependency* req, CompMap& comps,
			     bool modifies);

     /// Used by (the public) createDetailedDependencies to store comps
     /// in a ComputeTable (See TaskGraph.cc).
     void remembercomps(DetailedTask3* task, Task::Dependency* comp,
			CompTable3& ct);

     /// This is the "detailed" version of addDependencyEdges.  It does for
     /// the public createDetailedDependencies member function essentially
     /// what addDependencyEdges does for setupTaskConnections.  This will
     /// set up the data dependencies that need to be communicated between
     /// processors.
     void createDetailedDependencies(DetailedTasks3* dt, LoadBalancer* lb,
				     DetailedTask3* task, Task::Dependency* req,
				     CompTable3& ct, bool modifies);
     
     /// Makes a DetailedTask from task with given PatchSubset and 
     /// MaterialSubset.
     void createDetailedTask(DetailedTasks3* tasks, Task* task,
			     const PatchSubset* patches,
			     const MaterialSubset* matls);
     
     /// find the processor that a variable (req) is on given patch and 
     /// material.
     int findVariableLocation(LoadBalancer* lb, Task::Dependency* req,
			      const Patch* patch, int matl);

     TaskGraph3(const TaskGraph3&);
     TaskGraph3& operator=(const TaskGraph3&);

     bool overlaps(Task::Dependency* comp, Task::Dependency* req) const;

     /// Adds edges in the TaskGraph3 between requires/modifies and their
     /// associated computes.  Uses addDependencyEdges as a helper
     void setupTaskConnections(GraphSortInfo3Map& sortinfo);

     /// Called for each task, this "sorts" the taskgraph.  
     /// This sorts in topological order by calling processDependency
     /// (which checks for cycles in the graph), which then recursively
     /// calls processTask for each dependentTask.  After this process is
     /// finished, then the task is added at the end of sortedTasks.
     void processTask(Task* task, vector<Task*>& sortedTasks,
                      GraphSortInfo3Map& sortinfo) const;
      
     vector<Task*>        d_tasks;
     vector<Task::Edge*> edges;
     Scheduler3Common* sc;
     const ProcessorGroup* d_myworld;


      // data required from old data warehouse
     set<const VarLabel*, VarLabel::Compare> d_initRequiredVars;
     vector<const Task::Dependency*> d_initRequires;

     typedef map<const VarLabel*, DetailedTask3*, VarLabel::Compare>
     ReductionTasksMap;
     ReductionTasksMap d_reductionTasks;
   };

} // End namespace Uintah

#endif
