#ifndef UINTAH_HOMEBREW_TaskGraph_H
#define UINTAH_HOMEBREW_TaskGraph_H

 
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  using namespace std;
  class DetailedTask;
  class DetailedTasks;
  class Patch;
  class LoadBalancer;

/**************************************

CLASS
   TaskGraph
   
   During the TaskGraph compilation, the task graph does its work in
   the createDetailedTasks function.  The first portion is to sort the tasks,
   by add edges between computing tasks and requiring tasks,
   and the second is to create detailed tasks and dependendices.

   Here is a function call tree for this phase:
   createDetailedTasks
     topologicalSort
       setupTaskConnections
         addDependencyEdges
       processTask
         processDependencies
     LoadBalancer::createNeighborhood (this stores the patches that border
                                       patches on the current processor)

   Detailed Task portion: divides up tasks into smaller pieces, and
   sets up the data that need
   to be communicated between processors when the taskgraph executes.

     createDetailedTask (for each task, patch subset, matl subset)

     createDetailedDependencies (public)
       remembercomps
       createDetailedDependencies (private)
         DetailedTasks::possiblyCreateDependency or Task::addInternalDependency

   Then at the and:
     DetailedTasks::compureLocalTasks

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
  class CompTable;
  class SchedulerCommon;

  // this is so we can keep task independent of taskgraph
  struct GraphSortInfo {
    GraphSortInfo() { visited = false; sorted = false; }
    bool visited;
    bool sorted;
  };

  typedef map<Task*, GraphSortInfo> GraphSortInfoMap;

   class TaskGraph {
   public:
     TaskGraph(SchedulerCommon* sc, const ProcessorGroup* pg, Scheduler::tgType type);
     ~TaskGraph();

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
     DetailedTasks* createDetailedTasks( bool useInternalDeps, const GridP& grid,
                                         const GridP& oldGrid);

     inline DetailedTasks* getDetailedTasks() { return dts_; }

     inline Scheduler::tgType getType() { return type_; }

     /// This will go through the detailed tasks and create the 
     /// dependencies need to communicate data across separate
     /// processors.  Calls the private createDetailedDependencies
     /// for each task as a helper.
     void createDetailedDependencies();

     /// Connects the tasks, but does not sort them.
     /// Used for the MixedScheduler, this routine has the side effect
     /// (just like the topological sort) of adding the reduction tasks.
     /// However, this routine leaves the tasks in the order they were
     /// added, so that reduction tasks are hit in the correct order
     /// by each MPI process.
     void nullSort( vector<Task*>& tasks );
     
     /// Set the requires need from the old data warehouse.
     const set<const VarLabel*, VarLabel::Compare>&
       getInitialRequiredVars() const
       { return d_initRequiredVars; }

     int getNumTasks() const;
     Task* getTask(int i);

     void remapTaskDWs(int dwmap[]);

     /// Assigns unique id numbers to each dependency based on name,
     /// material index, and patch.  In other words, even though a
     /// number of tasks depend on the same data, they create there
     /// own copy of the dependency data.  This routine determines
     /// that the dependencies are actually the same, and gives them
     /// the same id number.
     void assignUniqueMessageTags();

     /// sets the iteration of the current taskgraph in a multi-TG environment
     /// starting with 0
     void setIteration(int iter) {currentIteration = iter;}
     
     vector<Task*>& getTasks() {
       return d_tasks;
     }

     /// Makes and returns a map that associates VarLabel names with
     /// the materials the variable is computed for.
     typedef map< string, list<int> > VarLabelMaterialMap;
     void makeVarLabelMaterialMap(VarLabelMaterialMap* result);
   private:
     typedef multimap<const VarLabel*, Task::Dependency*, VarLabel::Compare>
       CompMap;

     /// Helper function for processTasks, processing the dependencies
     /// for the given task in the dependency list whose head is req.
     /// Will call processTask (recursively, as this is a helper for 
     /// processTask) for each dependent task.
     void processDependencies(Task* task, Task::Dependency* req,
			      vector<Task*>& sortedTasks,
                              GraphSortInfoMap& sortinfo) const;
     
     /// Helper function for setupTaskConnections, adding dependency edges
     /// for the given task for each of the require (or modify) depencies in
     /// the list whose head is req.  If modifies is true then each found
     /// compute will be replaced by its modifying dependency on the CompMap.
     void addDependencyEdges(Task* task, GraphSortInfoMap& sortinfo, 
                             Task::Dependency* req, CompMap& comps,
			     bool modifies);

     /// Used by (the public) createDetailedDependencies to store comps
     /// in a ComputeTable (See TaskGraph.cc).
     void remembercomps(DetailedTask* task, Task::Dependency* comp,
			CompTable& ct);

     /// This is the "detailed" version of addDependencyEdges.  It does for
     /// the public createDetailedDependencies member function essentially
     /// what addDependencyEdges does for setupTaskConnections.  This will
     /// set up the data dependencies that need to be communicated between
     /// processors.
     void createDetailedDependencies(DetailedTask* task, Task::Dependency* req, 
                                     CompTable& ct, bool modifies);
     
     /// Makes a DetailedTask from task with given PatchSubset and 
     /// MaterialSubset.
     void createDetailedTask(Task* task,
			     const PatchSubset* patches,
			     const MaterialSubset* matls);
     
     /// find the processor that a variable (req) is on given patch and 
     /// material.
     int findVariableLocation(Task::Dependency* req,
			      const Patch* patch, int matl);

     TaskGraph(const TaskGraph&);
     TaskGraph& operator=(const TaskGraph&);

     bool overlaps(Task::Dependency* comp, Task::Dependency* req) const;

     /// Adds edges in the TaskGraph between requires/modifies and their
     /// associated computes.  Uses addDependencyEdges as a helper
     void setupTaskConnections(GraphSortInfoMap& sortinfo);

     /// Called for each task, this "sorts" the taskgraph.  
     /// This sorts in topological order by calling processDependency
     /// (which checks for cycles in the graph), which then recursively
     /// calls processTask for each dependentTask.  After this process is
     /// finished, then the task is added at the end of sortedTasks.
     void processTask(Task* task, vector<Task*>& sortedTasks,
                      GraphSortInfoMap& sortinfo) const;
      
     vector<Task*>        d_tasks;
     vector<Task::Edge*> edges;

     SchedulerCommon* sc;
     LoadBalancer* lb;
     const ProcessorGroup* d_myworld;
     Scheduler::tgType type_;
     DetailedTasks         * dts_;

     // how many times this taskgraph has executed this timestep
     int currentIteration;

      // data required from old data warehouse
     set<const VarLabel*, VarLabel::Compare> d_initRequiredVars;

     typedef map<const VarLabel*, DetailedTask*, VarLabel::Compare>
     ReductionTasksMap;
     ReductionTasksMap d_reductionTasks;
   };

} // End namespace Uintah

#endif
