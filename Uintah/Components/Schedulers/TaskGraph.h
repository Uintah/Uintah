#ifndef UINTAH_HOMEBREW_TaskGraph_H
#define UINTAH_HOMEBREW_TaskGraph_H

#include <Uintah/Grid/TaskProduct.h>
#include <Uintah/Grid/Task.h>
#include <vector>
#include <map>

namespace Uintah {
   class Task;
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
      // Insert Documentation Here:
      bool allDependenciesCompleted(Task* task) const;

      //////////
      // Insert Documentation Here:
      void dumpDependencies();

      int getNumTasks() const;
      Task* getTask(int i);

   private:
      TaskGraph(const TaskGraph&);
      TaskGraph& operator=(const TaskGraph&);

      //////////
      // Insert Documentation Here:
      void setupTaskConnections();

      void processTask(Task* task, vector<Task*>& sortedTasks) const;
      
      std::vector<Task*>        d_tasks;

      std::map<TaskProduct, Task*>   d_allcomps;
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/06/17 07:04:56  sparker
// Implemented initial load balancer modules
// Use ProcessorGroup
// Implemented TaskGraph - to contain the common scheduling stuff
//
//

#endif
