/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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
  class LoadBalancer;

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
    DetailedTask     * m_dtask{nullptr};
    Task::Dependency * m_comp{nullptr};
    const Patch      * m_patch{nullptr};
    int                m_matl{};
    unsigned int       m_hash{};
  };

  FastHashTable<Data> m_data{};

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
    DetailedTasks* createDetailedTasks(       bool    useInternalDeps
                                      , const GridP & grid
                                      , const GridP & oldGrid
                                      , const bool    hasDistalReqs = false
                                      );

    void overrideGhostCells(const std::vector<Task*> &sorted_tasks);	//DS: 01042020: fix for OnDemandDW race condition

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
    /// This routine has the side effect (just like the topological sort)
    /// of adding the reduction tasks. However, this routine leaves the
    /// tasks in the order they were added, so that reduction tasks are
    /// hit in the correct order by each MPI process.
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

    inline int getNumTaskPhases() const
    {
      return m_num_task_phases;
    }

    std::vector<std::shared_ptr<Task> > & getTasks()
    {
      return m_tasks;
    }

    inline bool getDistalRequires() const
    {
      return m_has_distal_requires;
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

    struct LabelLevel {
      LabelLevel( const std::string & key
                , const int           level
                )
      : m_key(key)
      , m_level(level)
      {}

      std::string m_key{};
      int         m_level{};

      bool operator<( const LabelLevel& rhs ) const
      {
        if (this->m_level < rhs.m_level) {
          return true;
        }
        else if ((this->m_level == rhs.m_level) && (this->m_key < rhs.m_key)) {
          return true;
        }
        return false;
      }
    };

    std::map<LabelLevel, int> max_ghost_for_varlabelmap{};

    SchedulerCommon      * m_scheduler{nullptr};
    LoadBalancer         * m_load_balancer{nullptr};
    const ProcessorGroup * m_proc_group{nullptr};
    Scheduler::tgType      m_type{};
    DetailedTasks        * m_detailed_tasks{nullptr};

    // how many times this taskgraph has executed this timestep
    int m_current_iteration{0};

    // how many task phases this taskgraph has been through
    int m_num_task_phases{0};

    int m_index{-1};

    // does this TG contain requires with halo > MAX_HALO_DEPTH
    bool m_has_distal_requires{false};

    std::vector<std::shared_ptr<Task> > m_tasks{};


    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //
    //  Archived code for topological sort functionality. Please leave this here - APH, 04/05/19
    //
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  public:

    //______________________________________________________________________
    // This is so we can keep tasks independent of the task graph
    struct GraphSortInfo {

        GraphSortInfo()
          : m_visited{false}
          , m_sorted{false}
        {}

        bool m_visited;
        bool m_sorted;
    };

    using CompMap          = std::multimap<const VarLabel*, Task::Dependency*>;
    using GraphSortInfoMap = std::map<Task*, GraphSortInfo>;
    using ReductionTasksMap         = std::map<VarLabelMatl<Level>, Task*>;


  private:

    /// Helper function for setupTaskConnections, adding dependency edges
    /// for the given task for each of the require (or modify) depencies in
    /// the list whose head is req.  If modifies is true then each found
    /// compute will be replaced by its modifying dependency on the CompMap.
    void addDependencyEdges( Task              * task
                           , GraphSortInfoMap  & sortinfo
                           , Task::Dependency  * req
                           , CompMap           & comps
                           , ReductionTasksMap & reductionTasks
                           , bool                modifies
                           );

    bool overlaps( const Task::Dependency * comp
                 , const Task::Dependency * req
                 ) const;

    /// Helper function for processTasks, processing the dependencies
    /// for the given task in the dependency list whose head is req.
    /// Will call processTask (recursively, as this is a helper for
    /// processTask) for each dependent task.
    void processDependencies( Task               * task
                            , Task::Dependency   * req
                            , std::vector<Task*> & sortedTasks
                            , GraphSortInfoMap   & sortinfo
                            ) const;

    /// Called for each task, this "sorts" the taskgraph.
    /// This sorts in topological order by calling processDependency
    /// (which checks for cycles in the graph), which then recursively
    /// calls processTask for each dependentTask.  After this process is
    /// finished, then the task is added at the end of sortedTasks.
    void processTask( Task               * task
                    , std::vector<Task*> & sortedTasks
                    , GraphSortInfoMap   & sortinfo
                    ) const;

    /// Adds edges in the TaskGraph between requires/modifies and their
    /// associated computes.  Uses addDependencyEdges as a helper
    void setupTaskConnections( GraphSortInfoMap & sortinfo );

    /// sets up the task connections and puts them in a sorted order.
    /// Calls setupTaskConnections, which has the side effect of creating
    /// reduction tasks for tasks that compute reduction variables.
    /// calls processTask on each task to sort them.
    void topologicalSort( std::vector<Task*>& tasks );

    std::vector<Task::Edge*>            m_edges;

}; // class TaskGraph

}  // namespace Uintah

#endif // CCA_COMPONENTS_SCHEDULERS_TASKGRAPH_H
