/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef CCA_COMPONENTS_SCHEDULERS_TASKGRAPH_H
#define CCA_COMPONENTS_SCHEDULERS_TASKGRAPH_H

#include <CCA/Ports/Scheduler.h>

#include <Core/Containers/FastHashTable.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Task.h>

#include <list>
#include <map>
#include <memory>
#include <vector>

namespace Uintah {

  class DetailedTask;
  class DetailedTasks;
  class Patch;
  class LoadBalancerPort;

/**************************************

CLASS
   TaskGraph
   
   During the TaskGraph compilation, the task graph does its work in
   the createDetailedTasks function.  The first portion is to sort the tasks,
   by adding edges between computing tasks and requiring tasks,
   and the second is to create detailed tasks and dependencies

   Here is a function call tree for this phase:
   createDetailedTasks
     nullsort (formerly was topologicalSort)
     LoadBalancer::createNeighborhood (this stores the patches that border
                                       patches on the current processor)

   Detailed Task portion: divides up tasks into smaller pieces, and sets up the
   data that need to be communicated between processors when the taskgraph executes.

     createDetailedTask (for each task, patch subset, matl subset)

     createDetailedDependencies (public)
       remembercomps
       createDetailedDependencies (private)
         DetailedTasks::possiblyCreateDependency or Task::addInternalDependency

   Then at the and:
     DetailedTasks::computeLocalTasks

GENERAL INFORMATION

   TaskGraph.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   TaskGraph, CompTable


DESCRIPTION

  
****************************************/

class CompTable;
class SchedulerCommon;

class CompTable {

  struct Data {

    unsigned int string_hash( const char * p )
    {
      unsigned int sum = 0;
      while (*p) {
        sum = sum * 7 + (unsigned char)*p++;
      }
      return sum;
    }

    Data(       DetailedTask     * dtask
        ,       Task::Dependency * comp
        , const Patch            * patch
        , int                      matl
        )
      : m_dtask(dtask)
      , m_comp(comp)
      , m_patch(patch)
      , m_matl(matl)
    {
      m_hash = (unsigned int)(((unsigned int)comp->mapDataWarehouse() << 3) ^ (string_hash(comp->m_var->getName().c_str())) ^ matl);
      if (patch) {
        m_hash ^= (unsigned int)(patch->getID() << 4);
      }
    }

    ~Data(){}

    bool operator==(const Data& c)
    {
      return m_matl == c.m_matl && m_patch == c.m_patch && m_comp->m_reduction_level == c.m_comp->m_reduction_level &&
             m_comp->mapDataWarehouse() == c.m_comp->mapDataWarehouse() && m_comp->m_var->equals(c.m_comp->m_var);
    }

    Data             * m_next{nullptr};
    DetailedTask     * m_dtask;
    Task::Dependency * m_comp;
    const Patch      * m_patch;
    int                m_matl;
    unsigned int       m_hash;
  };

  FastHashTable<Data> m_data;

  void insert( Data * data );

public:

  CompTable(){};

  ~CompTable(){};

  void remembercomp(       DetailedTask     * dtask
                   ,       Task::Dependency * comp
                   , const PatchSubset      * patches
                   , const MaterialSubset   * matls
                   , const ProcessorGroup   * pg
                   );

  bool findcomp(       Task::Dependency  * req
               , const Patch             * patch
               ,       int                 matlIndex
               ,       DetailedTask     *& dtask
               ,       Task::Dependency *& comp
               , const ProcessorGroup    * pg
               );

  bool findReductionComps(       Task::Dependency           * req
                         , const Patch                      * patch
                         ,       int                          matlIndex
                         ,       std::vector<DetailedTask*> & dt
                         , const ProcessorGroup             * pg
                         );

private:

  void remembercomp(       Data           * newData
                   , const ProcessorGroup * pg
                   );
}; // class CompTable


class TaskGraph {

  public:

    TaskGraph(       SchedulerCommon   * sched
		     , SimulationStateP& state
		     , const ProcessorGroup    * proc_group
             ,       Scheduler::tgType   tg_type
             ,       int                 index
             );

    ~TaskGraph();

    /// Clears the TaskGraph and deletes all tasks.
    void initialize();

    /// Adds a task to the task graph.  If the task is empty, it
    /// deletes it.  Also, as each task is added, it updates the list
    /// of vars that are required from the old DW
    void addTask(       std::shared_ptr<Task>   task
                , const PatchSet              * patchset
                , const MaterialSet           * matlset
                );

    /// Sorts the tasks, and makes DetailedTask's out of them,
    /// and loads them into a new DetailedTasks object. (There is one
    /// DetailedTask for each PatchSubset and MaterialSubset in a Task,
    /// where a Task may have many PatchSubsets and MaterialSubsets.).
    /// Sorts using nullSort.
    DetailedTasks* createDetailedTasks(       bool            useInternalDeps
                                      ,       DetailedTasks * first
                                      , const GridP         & grid
                                      , const GridP         & oldGrid
                                      , const bool            hasDistalReqs = false
                                      );

    inline DetailedTasks* getDetailedTasks()
    {
      return m_detailed_tasks;
    }

    inline Scheduler::tgType getType() const
    {
      return m_type;
    }

    inline int getIndex()
    {
      return m_index;
    }

    /// This will go through the detailed tasks and create the
    /// dependencies needed to communicate data across separate
    /// processors.  Calls the private createDetailedDependencies
    /// for each task as a helper.
    void createDetailedDependencies();

    /// Connects the tasks, but does not sort them.
    /// Used for the UnifiedScheduler, this routine has the side effect
    /// (just like the topological sort) of adding the reduction tasks.
    /// However, this routine leaves the tasks in the order they were
    /// added, so that reduction tasks are hit in the correct order
    /// by each MPI process.
    void nullSort( std::vector<Task*> & tasks );

    int getNumTasks() const;

    Task* getTask( int i );

    void remapTaskDWs( int dwmap[] );

    /// Assigns unique id numbers to each dependency based on name,
    /// material index, and patch.  In other words, even though a
    /// number of tasks depend on the same data, they create there
    /// own copy of the dependency data.  This routine determines
    /// that the dependencies are actually the same, and gives them
    /// the same id number.
    void assignUniqueMessageTags();

    /// sets the iteration of the current taskgraph in a multi-TG environment starting with 0
    void setIteration( int iter )
    {
      m_current_iteration = iter;
    }

    inline int getNumTaskPhases()
    {
      return m_num_task_phases;
    }

    std::vector<std::shared_ptr<Task> > & getTasks()
    {
      return m_tasks;
    }

    /// Makes and returns a map that associates VarLabel names with
    /// the materials the variable is computed for.
    using VarLabelMaterialMap = std::map<std::string, std::list<int> >;
    void makeVarLabelMaterialMap( VarLabelMaterialMap * result );


  private:

    // eliminate copy, assignment and move
    TaskGraph( const TaskGraph & )            = delete;
    TaskGraph& operator=( const TaskGraph & ) = delete;
    TaskGraph( TaskGraph && )                 = delete;
    TaskGraph& operator=( TaskGraph && )      = delete;

    /// Used by (the public) createDetailedDependencies to store comps
    /// in a ComputeTable (See TaskGraph.cc).
    void remembercomps( DetailedTask     * task
                      , Task::Dependency * comp
                      , CompTable        & ct
                      );

    /// This is the "detailed" version of addDependencyEdges (removed).  It does for
    /// the public createDetailedDependencies member function essentially
    /// what addDependencyEdges (removed) did for setupTaskConnections.
    /// This will set up the data dependencies that need to be communicated
    /// between processors.
    void createDetailedDependencies( DetailedTask     * dtask
                                   , Task::Dependency * req
                                   , CompTable        & ct
                                   , bool               modifies
                                   );

    /// Makes a DetailedTask from task with given PatchSubset and MaterialSubset.
    void createDetailedTask(       Task           * task
                           , const PatchSubset    * patches
                           , const MaterialSubset * matls
                           );

    /// Find the processor that a variable (req) is on given patch and material.
    int findVariableLocation(       Task::Dependency * req
                            , const Patch            * patch
                            ,       int                matl
                            ,       int                iteration
                            );

    SchedulerCommon      * m_scheduler;
    LoadBalancerPort     * m_load_balancer;
    const ProcessorGroup * m_proc_group;
    Scheduler::tgType      m_type;
    DetailedTasks        * m_detailed_tasks{nullptr};

    // how many times this taskgraph has executed this timestep
    int m_current_iteration{0};

    // how many task phases this taskgraph has been through
    int m_num_task_phases{0};

    int m_index{-1};

    std::vector<std::shared_ptr<Task> > m_tasks;

    std::map<const VarLabel*, DetailedTask*, VarLabel::Compare>  m_reduction_tasks;

    struct LabelLevel {
      LabelLevel(const std::string& key, const int level) : key(key), level(level) {}
      std::string key;
      int level;
      bool operator<(const LabelLevel& rhs) const {
        if (this->level < rhs.level) {
          return true;
        } else if ((this->level == rhs.level) && (this->key < rhs.key)) {
          return true;
        }
        return false;
      }
    };

    std::map<LabelLevel, int> max_ghost_for_varlabelmap;

}; // class TaskGraph

}  // namespace Uintah

#endif // CCA_COMPONENTS_SCHEDULERS_TASKGRAPH_H
