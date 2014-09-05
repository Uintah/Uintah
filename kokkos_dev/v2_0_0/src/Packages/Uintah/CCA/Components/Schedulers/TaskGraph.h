#ifndef UINTAH_HOMEBREW_TaskGraph_H
#define UINTAH_HOMEBREW_TaskGraph_H

#include <Packages/Uintah/Core/Grid/TaskProduct.h>
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
   class CompTable;
  class SchedulerCommon;
  
   class TaskGraph {
   public:
      TaskGraph(SchedulerCommon* sc);
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

     DetailedTasks* createDetailedTasks( const ProcessorGroup* pg,
					 LoadBalancer* lb,
					 bool useInternalDeps );
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
      const vector<const Task::Dependency*>& getInitialRequires() const
      { return d_initRequires; }

      const set<const VarLabel*, VarLabel::Compare>&
      getInitialRequiredVars() const
      { return d_initRequiredVars; }

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
     typedef multimap<const VarLabel*, Task::Dependency*, VarLabel::Compare>
       CompMap;

     // Helper function for proccessTasks, processing the dependencies
     // for the given task in the dependency list whose head is req.
     void processDependencies(Task* task, Task::Dependency* req,
			      vector<Task*>& sortedTasks) const;
     
     // Helper function for setupTaskConnections, adding dependency edges
     // for the given task for each of the require (or modify) depencies in
     // the list whose head is req.  If modifies is true then each found
     // compute will be replaced by its modifying dependency on the CompMap.
     void addDependencyEdges(Task* task, Task::Dependency* req, CompMap& comps,
			     bool modifies);

     void remembercomps(DetailedTask* task, Task::Dependency* comp,
			const ProcessorGroup* pg, CompTable& ct);

     // This is the "detailed" version of addDependencyEdges.  It does for
     // the public createDetailedDependencies member function essentially
     // what addDependencyEdges does for setupTaskConnections.
     void createDetailedDependencies(DetailedTasks* dt, LoadBalancer* lb,
				     const ProcessorGroup* pg,
				     DetailedTask* task, Task::Dependency* req,
				     CompTable& ct, bool modifies);
     
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
     SchedulerCommon* sc;

      // data required from old data warehouse
     set<const VarLabel*, VarLabel::Compare> d_initRequiredVars;
     vector<const Task::Dependency*> d_initRequires;

     typedef map<const VarLabel*, DetailedTask*, VarLabel::Compare>
     ReductionTasksMap;
     ReductionTasksMap d_reductionTasks;
   };

} // End namespace Uintah

#endif
