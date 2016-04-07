/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#include <CCA/Components/Schedulers/TaskGraph.h>

#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/LoadBalancer.h>

#include <Core/Containers/FastHashTable.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/TypeMismatchException.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/ProgressiveWarning.h>

#include <sci_defs/config_defs.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

#include <unistd.h>

using namespace Uintah;

namespace {

static DebugStream tgdbg(       "TaskGraph",         false);
static DebugStream tgphasedbg(  "TaskGraphPhase",    false);
static DebugStream detaileddbg( "TaskGraphDetailed", false);
static DebugStream compdbg(     "FindComp",          false);

}


//______________________________________________________________________
//
TaskGraph::TaskGraph(       SchedulerCommon   * sched
                    , const ProcessorGroup    * pg
                    ,       Scheduler::tgType   type
                    )
  : m_sched{ sched }
  , m_proc_group{ pg }
  , m_tg_type{ type }
  , m_detailed_tasks{ 0 }
  , m_current_iteration{ 0 }
  , m_num_task_phases{ 0 }
{
  m_load_balancer = dynamic_cast<LoadBalancer*>(sched->getPort("load balancer"));
}

//______________________________________________________________________
//
TaskGraph::~TaskGraph()
{
  initialize(); // Frees all of the memory...
}

//______________________________________________________________________
//
void
TaskGraph::initialize()
{
  if (m_detailed_tasks) {
    delete m_detailed_tasks;
  }

  for (std::vector<Task*>::iterator iter = m_tasks.begin(); iter != m_tasks.end(); iter++) {
    delete *iter;
  }

  for (std::vector<Task::Edge*>::iterator iter = m_edges.begin(); iter != m_edges.end(); iter++) {
    delete *iter;
  }

  m_tasks.clear();
  m_num_task_phases = 0;

  m_edges.clear();
  m_current_iteration = 0;
}


//______________________________________________________________________
//
void
TaskGraph::processTask( Task               * task
                      , std::vector<Task*> & sortedTasks
                      , GraphSortInfoMap   & sortinfo
                      ) const
{
  if (detaileddbg.active()) {
    detaileddbg << m_proc_group->myrank() << " Looking at task: " << task->getName() << "\n";
  }

  GraphSortInfo& gsi = sortinfo.find(task)->second;
  // we throw an exception before calling processTask if this task has already been visited
  gsi.m_visited = true;

  processDependencies(task, task->getRequires(), sortedTasks, sortinfo);
  processDependencies(task, task->getModifies(), sortedTasks, sortinfo);

  // All prerequisites are done - add this task to the list
  sortedTasks.push_back(task);
  gsi.m_sorted = true;

  if (detaileddbg.active()) {
    detaileddbg << m_proc_group->myrank() << " Sorted task: " << task->getName() << "\n";
  }
}  // end processTask()


//______________________________________________________________________
//

void
TaskGraph::processDependencies( Task               * task
                              , Task::Dependency   * req
                              , std::vector<Task*> & sortedTasks
                              , GraphSortInfoMap   & sortinfo
                              ) const
{
  for (; req != 0; req = req->next) {
    if (detaileddbg.active()) {
      detaileddbg << m_proc_group->myrank() << " processDependencies for req: " << *req << "\n";
    }
    if (req->whichdw == Task::NewDW) {
      Task::Edge* edge = req->comp_head;
      for (; edge != 0; edge = edge->compNext) {
        Task* vtask = edge->comp->task;
        GraphSortInfo& gsi = sortinfo.find(vtask)->second;
        if (!gsi.m_sorted) {
          try {
            // this try-catch mechanism will serve to print out the entire TG cycle
            if (gsi.m_visited) {
              std::cout << m_proc_group->myrank() << " Cycle detected in task graph\n";
              SCI_THROW(InternalError("Cycle detected in task graph", __FILE__, __LINE__));
            }

            // recursively process the dependencies of the computing task
            processTask(vtask, sortedTasks, sortinfo);
          } catch (InternalError& e) {
            std::cout << m_proc_group->myrank() << "   Task " << task->getName() << " requires " << req->var->getName() << " from "
                      << vtask->getName() << "\n";

            throw;
          }
        }
      }
    }
  }
}

//______________________________________________________________________
//

void
TaskGraph::addTask(       Task        * task
                  , const PatchSet    * patchset
                  , const MaterialSet * matlset
                  )
{
  task->setSets(patchset, matlset);
  if ((patchset && patchset->totalsize() == 0) || (matlset && matlset->totalsize() == 0)) {
    delete task;
    if (detaileddbg.active()) {
      detaileddbg << m_proc_group->myrank() << " Killing empty task: " << *task << "\n";
    }
  } else {
    m_tasks.push_back(task);

    // debugging Code
    if (tgdbg.active()) {
      tgdbg << m_proc_group->myrank() << " Adding task: ";
      task->displayAll(tgdbg);
    }
  }
}

//______________________________________________________________________
//

DetailedTask *
TaskGraph::createDetailedTask(       Task           * task
                             , const PatchSubset    * patches
                             , const MaterialSubset * matls
                             )
{
  DetailedTask* dt = new DetailedTask(task, patches, matls, m_detailed_tasks);

  m_detailed_tasks->add(dt);
  return dt;
}

//______________________________________________________________________
//
namespace {

struct SimpleDependency
{
  enum Type {
    Computes,
    Modifies,
    Requires,
  };

  friend std::ostream & operator<<(std::ostream & out, Type const & t)
  {
         if (t == Computes) out << "Computes";
    else if (t == Modifies) out << "Modifies";
    else                    out << "Requires";

    return out;
  }

  Type                   m_type{};
  int                    m_idx{-1};
  const VarLabel       * m_var{nullptr};
  int                    m_level{-1};
  int                    m_dw{-1};
  std::vector<int>       m_materials{};

  SimpleDependency( const Task::Dependency * dep )
    : m_type{ dep->deptype == Task::Requires ? Requires :
              dep->deptype == Task::Modifies ? Modifies : Computes}
    , m_idx{ dep->task->index() }
    , m_var{ dep->var }
    , m_dw{ dep->whichdw }
  {
    // get the materials
    if (dep->matls) {
      for (int mat : dep->matls->getVector()) {
        m_materials.push_back(mat);
      }
    } else {
      for (auto const& mats : dep->task->getMaterialSet()->getVector() ) {
        for (int mat : mats->getVector()) {
          m_materials.push_back(mat);
        }
      }
    }
    if (m_materials.empty()) { m_materials.push_back(-1); }
    std::sort(m_materials.begin(), m_materials.end());

    //TODO get level

    // get task type
    auto task_type = dep->task->getType();
    if (task_type == Task::Reduction || task_type == Task::OncePerProc) {
      m_level = dep->reductionLevel ? dep->reductionLevel->getIndex() : -1;
    } else if (dep->deptype != Task::Requires) {
      // modifies and computes level is the task level
      m_level = getLevel(dep->task->getPatchSet())->getIndex();
    } else {
      auto level = dep->patches ? getLevel(dep->patches) : getLevel(dep->task->getPatchSet());
      int offset = dep->patches_dom == Task::CoarseLevel ? -dep->level_offset : dep->level_offset;
      m_level = level->getRelativeLevel(offset)->getIndex();
    }

  }

  SimpleDependency() = default;
  SimpleDependency( const SimpleDependency & ) = default;
  SimpleDependency( SimpleDependency && ) = default;

  SimpleDependency & operator=( const SimpleDependency & ) = default;
  SimpleDependency & operator=( SimpleDependency && ) = default;

  bool materials_overlap( SimpleDependency const & rhs ) const
  {

    if (m_materials[0] == -1 || rhs.m_materials[0] == -1) return true;

    auto find_m = [&rhs](int m) {
      return std::find( rhs.m_materials.begin(), rhs.m_materials.end(), m ) != rhs.m_materials.end();
    };

    return std::any_of( m_materials.begin(), m_materials.end()
                      , find_m
                      );
  };

  bool new_dw() const
  {
    return    m_dw == Task::NewDW
           || m_dw == Task::CoarseNewDW
           || m_dw == Task::ParentNewDW;
  }

  bool is_prerequisite( SimpleDependency const & rhs ) const
  {
    // not a new data warehouse
    if (!new_dw() || !rhs.new_dw()) return false;

    // different data warehouses
    // TODO - may not need this
//    if (m_dw != rhs.m_dw) return false;

    // not same variable
    if (m_var != rhs.m_var) return false;

    // requires are not ordered
    if (m_type == Requires && rhs.m_type == Requires) return false;

    // materials do not overlap
    if ( !materials_overlap(rhs) ) return false;

    // levels do not overlap
    if (m_level != rhs.m_level ) return false;

    // computes before modifies or requires
    if ( m_type == Computes && rhs.m_type != Computes ) return true;

    // modifies before requires
    if ( m_type == Modifies && rhs.m_type == Requires ) return true;

    // order multiple computes
    if ( m_type == Computes && rhs.m_type == Computes ) {
      if ( m_var->allowsMultipleComputes() ) return m_idx < rhs.m_idx;
      throw std::runtime_error("Multiple computes detected");
    }

    // TODO - may need to consider task type, e.g. OPP, reduction
    if ( m_type == Modifies && rhs.m_type == Modifies ) return m_idx < rhs.m_idx;

    return false;
  }
};

std::vector< std::pair<int,int> >
get_edges( std::vector<Task *> const & tasks )
{
  using DependencyVector = std::vector< SimpleDependency >;
  using DependencyIterator = DependencyVector::const_iterator;

  auto print_dependencies = [&](DependencyIterator b, const DependencyIterator e) {
    if (b != e) {
      std::cout << "Task[" << b->m_idx << "]: "  << tasks[b->m_idx]->getName() << std::endl;
      for (; b != e; ++b) {
        auto d = *b;
        std::cout << "  :  " << d.m_type
          << "  :  " << d.m_var->getName()
          << "  :  level[ " << d.m_level
          << " ]  :  dw[ " << ( d.new_dw() ? "NewDW" : "OldDW" );
        std::cout << " ]  :  materials[ ";
        for (auto m : d.m_materials) {
          std::cout << m << ",";
        }
        std::cout << "\b ] " << std::endl;
      }
    }
  };

  auto filter_dependencies = [&](DependencyVector & vec) {
    print_dependencies(vec.begin(), vec.end());
    // TODO compare dependencies
    auto compare = []( const SimpleDependency & a, const SimpleDependency &b )->bool {
      return a.m_dw < b.m_dw ? true  :
             a.m_dw > b.m_dw ? false :
             a.m_idx < b.m_idx ? true  :
             a.m_idx > b.m_idx ? false :
             a.m_var < b.m_var ? true  :
             a.m_var > b.m_var ? false :
             a.m_level < b.m_level ? true :
             a.m_level > b.m_level ? false :
             a.m_materials < b.m_materials;
    };
    auto compare_equal = [&compare]( const SimpleDependency & a, const SimpleDependency &b )->bool {
      return !compare(a,b) && !compare(b,a);
    };
    std::sort( vec.begin(), vec.end(), compare );

    auto end_itr = std::unique( vec.begin(), vec.end(), compare_equal);

    if ( end_itr != vec.end() ) {
      std::sort( end_itr, vec.end(), compare );

      // update the type
      // *dup_itr Modifies --> *itr Modifies
      // *dup_itr Computes && *itr Requires --> Computes
      {
        auto itr = vec.begin();
        const auto itr_end = end_itr;
        auto dup_itr = end_itr;
        const auto dup_end = vec.end();

        while (itr != itr_end && dup_itr != dup_end) {
          auto & dep = *itr;
          const auto & dup = *dup_itr;
          if (compare_equal(dep,dup)) {
            if (dup.m_type == SimpleDependency::Modifies) {
              dep.m_type = SimpleDependency::Modifies;
            } else if (dup.m_type == SimpleDependency::Computes && dep.m_type == SimpleDependency::Requires) {
              dep.m_type = SimpleDependency::Computes;
            }
            ++dup_itr;
          } else if (compare(dep, dup)) {
            ++itr;
          } else {
            ++dup_itr;
          }
        }
      }
      vec.erase(end_itr, vec.end());
    }
    print_dependencies(vec.begin(), vec.end());
  };

  auto get_dependencies = [&filter_dependencies](Task * t)->DependencyVector {

    DependencyVector vec;

    Task::Dependency * head = t->getModifies();
    while (head) {
      vec.emplace_back(head);
      head = head->next;
    }
    head = t->getComputes();
    while (head) {
      vec.emplace_back(head);
      head = head->next;
    }
    head = t->getRequires();
    while (head) {
      vec.emplace_back(head);
      head = head->next;
    }
    filter_dependencies( vec );
    return vec;
  };

  // get the dependencies for each task
  std::vector< DependencyVector > dependencies(tasks.size());
  for (auto task : tasks) {
    dependencies[task->index()] = get_dependencies(task);
  }

  auto is_prerequisite_task = [&](Task * a, Task * b)->bool {

    const auto & a_dependencies = dependencies[a->index()];
    const auto & b_dependencies = dependencies[b->index()];

    for (auto a_dep : a_dependencies) {
    for (auto b_dep : b_dependencies) {
      if ( a_dep.is_prerequisite( b_dep ) ) return true;
    }}

    return false;
  };

  // pair<parentIndex,childIndex>
  using result_type = std::vector< std::pair<int,int> >;
  result_type result;

  for (auto a : tasks) {
  for (auto b : tasks) {
    if (a == b) continue;

    if (is_prerequisite_task( a, b)) {
      if (is_prerequisite_task(b,a)) {
        throw std::runtime_error("Circular dependency");
      }
      result.emplace_back(a->index(),b->index());
    }
  }}

  return result;
}


} // namespace

DetailedTasks*
TaskGraph::createDetailedTasks(       bool            useInternalDeps
                              ,       DetailedTasks * first
                              , const GridP         & grid
                              , const GridP         & oldGrid
                              )
{
  // set the index on each task
  {
    int i = 0;
    for (auto task : m_tasks) {
      task->set_index(i++);
    }
  }

  auto is_sync_task = [](const Task * t)->bool {
    const bool result = t->getType() == Task::Reduction || t->usesMPI() || t->getType() == Task::OncePerProc;
    return result;
  };


  auto backward_edges = get_edges(m_tasks);

  using edge_vector = decltype(backward_edges);
  using pair_type = edge_vector::value_type;


  auto sort_by_second = []( edge_vector & edges ) {
    auto compare_less = [](const pair_type & a, const pair_type &b)->bool {
            return a.second < b.second ? true  :
                   a.second > b.second ? false :
                   a.first  < b.first;
    };
    std::sort( edges.begin(), edges.end(), compare_less );
    auto itr = std::unique( edges.begin(), edges.end());
    edges.erase(itr, edges.end());
  };

  sort_by_second( backward_edges );


  using edge_iterator = edge_vector::const_iterator;
  using range_type = std::pair< edge_iterator, edge_iterator >;

  using pair_type = decltype(backward_edges)::value_type;

  using range_vector = std::vector< range_type >;

  auto get_backward_range = [this]( edge_vector const& edges )->range_vector {
    range_vector backward_range;
    backward_range.reserve( m_tasks.size() );

    // get the edges for each task
    for (const auto task : m_tasks) {
      {
        auto lower = std::lower_bound( edges.begin(), edges.end(), task->index()
            , [](const pair_type & p, int i)->bool { return p.second < i; });
        auto upper = std::upper_bound( lower, edges.end(), task->index()
            , [](int i, const pair_type & p)->bool { return i < p.second; });
        backward_range.emplace_back(lower, upper);
      }
    }
    return backward_range;
  };

  auto backward_range = get_backward_range( backward_edges );


  // order tasks -- necessary to add fake edges between sync tasks
  std::vector<Task *> sorted_tasks;
  {
    const size_t n = m_tasks.size();

    std::vector<int> marked( n, 0);
    size_t visited = 0;

    std::vector<Task *> tasks( m_tasks );

    //stable sort by sync tasks
    std::stable_sort( tasks.begin(), tasks.end(),
               [&](const Task * a, const Task * b)->bool {
                  return is_sync_task(a) < is_sync_task(b);
               }
             );

    // push back all empty ranges
    for (auto task : tasks) {
      const int i = task->index();
      const auto & r = backward_range[i];
      if (r.second - r.first == 0) {
        sorted_tasks.push_back( task );
        task->setSortedOrder(visited++);
        marked[i] = true;
      }
    }
    while (visited < n) {
      for (auto task : tasks) {
        const int i = task->index();
        if (marked[i]) continue;
        bool parents_marked = true;
        for ( range_type r = backward_range[i]; parents_marked && r.first != r.second; ++r.first ) {
          const auto & edge = *r.first;
          parents_marked = marked[edge.first];
        }
        if (parents_marked) {
          sorted_tasks.push_back( task );
          task->setSortedOrder(visited++);
          marked[i] = true;
        }
      }
    }
  }

  // create fake edges between sync tasks
  {
    auto first = std::find_if( sorted_tasks.begin(), sorted_tasks.end(), is_sync_task );
    if (first != sorted_tasks.end()) {
      auto second = std::find_if( first+1, sorted_tasks.end(), is_sync_task );
      while (second != sorted_tasks.end()) {
        backward_edges.emplace_back( (*first)->index(), (*second)->index() );
        first = second;
        second = std::find_if( first+1, sorted_tasks.end(), is_sync_task );
      }
    }
  }

  // re-sort
  sort_by_second( backward_edges );

  // recompute the ranges for each task
  backward_range = get_backward_range( backward_edges );

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "MODIFIED Backward Ranges: " << std::endl;
  // print ranges
  for (const auto task : sorted_tasks) {
    std::cout << task->getSortedOrder() << ": " << task->getName()
              << ",  IS_SYNC? " << is_sync_task(task)
              << std::endl;
    for (range_type r = backward_range[task->index()]; r.first != r.second; ++r.first) {
      auto edge = *r.first;
      auto ptask = m_tasks[edge.first];
      std::cout << "  <--- " << ptask->getSortedOrder() << ": " << ptask->getName()
                << ",  IS_SYNC? " << is_sync_task(ptask)
                << std::endl;
    }
  }
  std::cout << std::endl;
  std::cout << std::endl;


  ASSERT(grid != 0);
  m_load_balancer->createNeighborhood(grid, oldGrid);

  const std::set<int> neighborhood_procs = m_load_balancer->getNeighborhoodProcessors();
  m_detailed_tasks = new DetailedTasks(m_sched, m_proc_group, first, this, neighborhood_procs, useInternalDeps);

  std::vector< std::vector<DetailedTask * > > dtasks( m_tasks.size() );

  for (auto task : sorted_tasks ) {
    const int i = task->index();
    const PatchSet* ps = task->getPatchSet();
    const MaterialSet* ms = task->getMaterialSet();
    if (ps && ms) {
      //only create OncePerProc tasks and output tasks once on each processor.
      if (task->getType() == Task::OncePerProc) {
        //only schedule this task on processors in the neighborhood
        for (std::set<int>::iterator p = neighborhood_procs.begin(); p != neighborhood_procs.end(); p++) {
          const PatchSubset* pss = ps->getSubset(*p);
          for (int m = 0; m < ms->size(); m++) {
            const MaterialSubset* mss = ms->getSubset(m);
            dtasks[i].push_back(createDetailedTask(task, pss, mss));
          }
        }
      } else if (task->getType() == Task::Output) {
        //compute subset that involves this rank
        int subset = (m_proc_group->myrank() / m_load_balancer->getNthProc()) * m_load_balancer->getNthProc();

        //only schedule output task for the subset involving our rank
        const PatchSubset* pss = ps->getSubset(subset);

        //don't schedule if there are no patches
        if (pss->size() > 0) {
          for (int m = 0; m < ms->size(); m++) {
            const MaterialSubset* mss = ms->getSubset(m);
            dtasks[i].push_back(createDetailedTask(task, pss, mss));
          }
        }

      } else {
        for (int p = 0; p < ps->size(); p++) {
          const PatchSubset* pss = ps->getSubset(p);
          // don't make tasks that are not in our neighborhood or tasks that do not have patches
          if (m_load_balancer->inNeighborhood(pss) && pss->size() > 0) {
            for (int m = 0; m < ms->size(); m++) {
              const MaterialSubset* mss = ms->getSubset(m);
              dtasks[i].push_back(createDetailedTask(task, pss, mss));
            }
          }
        }
      }
    } else if (!ps && !ms) {
      dtasks[i].push_back(createDetailedTask(task, 0, 0));
    } else if (!ps) {
      SCI_THROW(InternalError("Task has MaterialSet, but no PatchSet", __FILE__, __LINE__));
    } else {
      SCI_THROW(InternalError("Task has PatchSet, but no MaterialSet", __FILE__, __LINE__));
    }
  }

  // this can happen if a processor has no patches (which may happen at the beginning of some AMR runs)
  //  if(dts_->numTasks() == 0)
  //    cerr << "WARNING: Compiling scheduler with no tasks\n";

  m_load_balancer->assignResources(*m_detailed_tasks);

  // use this, even on a single processor, if for nothing else than to get scrub counts
  bool doDetailed = true;
  if (doDetailed) {
    createDetailedDependencies();
    if (m_detailed_tasks->getExtraCommunication() > 0 && m_proc_group->myrank() == 0) {
      std::cout << m_proc_group->myrank() << "  Warning: Extra communication.  This taskgraph on this rank overcommunicates about "
        << m_detailed_tasks->getExtraCommunication() << " cells\n";
    }
  }

  // TODO Fix logic
  // add internal dependencies to force correct ordering of computes, modifies, requires, and sync tasks
  for (auto task : m_tasks) {
    const int i = task->index();
    // get the backward edges for the current task
    for (range_type r = backward_range[i]; r.first != r.second; ++r.first) {
      // r is a pair< edge_iterator, edge_iterator >
      const int j = r.first->first;
      if (i == j ) continue;
      // for (auto child : dtasks[i]) {
        // for (auto parent : dtasks[j]) {
         // auto itr = parent->m_internal_dependents.find(child);
         // if ( itr != parent->m_internal_dependents.end() ) {
            //std::cout << "ADD parent " << parent->getName() << std::endl;
            //std::cout << "ADD child  " << child->getName() << std::endl;
            //std::cout << std::endl;

            //child->m_internal_dependencies.emplace_back( parent, child, nullptr, 0 );
            //parent->m_internal_dependents[child] = &child->m_internal_dependencies.back();
            //child->m_num_pending_internal_dependencies.fetch_add(1, std::memory_order_relaxed);
         // } else { std::cout << "Already exists" << std::endl; }
       // }
     // }
    }
  }

  if (m_proc_group->size() > 1) {
    m_detailed_tasks->assignMessageTags(m_proc_group->myrank());
  }

  m_detailed_tasks->computeLocalTasks(m_proc_group->myrank());
  m_detailed_tasks->makeDWKeyDatabase();

  return m_detailed_tasks;
} // end TaskGraph::createDetailedTasks



//______________________________________________________________________
//
namespace Uintah {

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

        Data(       DetailedTask     * task
            ,       Task::Dependency * comp
            , const Patch            * patch
            ,       int                matl
            )
          : next{ nullptr }
          , m_task{ task }
          , m_computes{ comp }
          , m_patch{ patch }
          , m_matl{ matl }
          , hash{ 0 }
        {
          hash = (unsigned int)(((unsigned int)comp->mapDataWarehouse() << 3) ^ (string_hash(comp->var->getName().c_str())) ^ matl);
          if (patch) {
            hash ^= (unsigned int)(patch->getID() << 4);
          }
        }

        ~Data() {}

        bool operator==(const Data& c)
        {
          return m_matl == c.m_matl && m_patch == c.m_patch && m_computes->reductionLevel == c.m_computes->reductionLevel
                 && m_computes->mapDataWarehouse() == c.m_computes->mapDataWarehouse() && m_computes->var->equals(c.m_computes->var);
        }


              Data             * next;
              DetailedTask     * m_task;
              Task::Dependency * m_computes;
        const Patch            * m_patch;
              int                m_matl;
              unsigned int       hash;
    };


    public:

      CompTable(){};

      ~CompTable(){};

      void remembercomp(       DetailedTask     * task
                       ,       Task::Dependency * comp
                       , const PatchSubset      * patches
                       , const MaterialSubset   * matls
                       , const ProcessorGroup   * pg
                       );

    bool findcomp(       Task::Dependency  * req
                 , const Patch             * patch
                 ,       int                 matlIndex
                 ,       DetailedTask     *& dt
                 ,       Task::Dependency *& comp
                 , const ProcessorGroup    * pg
                 );

    bool findReductionComps(       Task::Dependency            * req
                           ,  const Patch                      * patch
                           ,        int                          matlIndex
                           ,        std::vector<DetailedTask*> & dt
                           ,  const ProcessorGroup             * pg
                           );

    private:

      void remembercomp( Data * newData, const ProcessorGroup * pg );

      void insert( Data* d );

      FastHashTable<Data> data;
  };

}

//______________________________________________________________________
//
void
CompTable::remembercomp( Data * newData, const ProcessorGroup * pg )
{
  if (detaileddbg.active()) {
    detaileddbg << pg->myrank() << " remembercomp: " << *newData->m_computes << ", matl=" << newData->m_matl;
    if (newData->m_patch) {
      detaileddbg << ", patch=" << *newData->m_patch;
    }
    detaileddbg << "\n";
  }

  // can't have two computes for the same variable (need modifies)
  if (newData->m_computes->deptype != Task::Modifies && !newData->m_computes->var->typeDescription()->isReductionVariable()) {
    if (data.lookup(newData)) {
      std::cout << "Multiple compute found:\n";
      std::cout << "  matl: " << newData->m_matl << "\n";
      if (newData->m_patch) {
        std::cout << "  patch: " << *newData->m_patch << "\n";
      }
      std::cout << "  " << *newData->m_computes << "\n";
      std::cout << "  " << *newData->m_task << "\n\n";
      std::cout << "  It was originally computed by the following task(s):\n";
      for (Data* old = data.lookup(newData); old != 0; old = data.nextMatch(newData, old)) {
        std::cout << "  " << *old->m_task << std::endl;
        //old->comp->task->displayAll(cout);
      }
      SCI_THROW(InternalError("Multiple computes for variable: "+newData->m_computes->var->getName(), __FILE__, __LINE__));
    }
  }
  data.insert(newData);
}

//______________________________________________________________________
//
void
CompTable::remembercomp(       DetailedTask     * task
                       ,       Task::Dependency * comp
                       , const PatchSubset      * patches
                       , const MaterialSubset   * matls
                       , const ProcessorGroup   * pg
                       )
{
  if (patches && matls) {
    for (int p = 0; p < patches->size(); p++) {
      const Patch* patch = patches->get(p);
      for (int m = 0; m < matls->size(); m++) {
        int matl = matls->get(m);
        Data* newData = new Data(task, comp, patch, matl);
        remembercomp(newData, pg);
      }
    }
  } else if (matls) {
    for (int m = 0; m < matls->size(); m++) {
      int matl = matls->get(m);
      Data* newData = new Data(task, comp, 0, matl);
      remembercomp(newData, pg);
    }
  } else if (patches) {
    for (int p = 0; p < patches->size(); p++) {
      const Patch* patch = patches->get(p);
      Data* newData = new Data(task, comp, patch, 0);
      remembercomp(newData, pg);
    }
  } else {
    Data* newData = new Data(task, comp, 0, 0);
    remembercomp(newData, pg);
  }
}

//______________________________________________________________________
//
bool
CompTable::findcomp(       Task::Dependency  * req
                   , const Patch             * patch
                   ,       int                 matlIndex
                   ,       DetailedTask     *& dt
                   ,       Task::Dependency *& comp
                   , const ProcessorGroup    * pg
                   )
{
  if (compdbg.active()) {
    compdbg << pg->myrank() << "        Finding comp of req: " << *req << " for task: " << *req->task << "/" << "\n";
  }
  Data key(0, req, patch, matlIndex);
  Data* result = 0;
  for (Data* p = data.lookup(&key); p != 0; p = data.nextMatch(&key, p)) {
    if (compdbg.active()) {
      compdbg << pg->myrank() << "          Examining comp from: " << p->m_computes->task->getName() << ", order="
              << p->m_computes->task->getSortedOrder() << "\n";
    }

    ASSERT(!result || p->m_computes->task->getSortedOrder() != result->m_computes->task->getSortedOrder());

    if (p->m_computes->task->getSortedOrder() < req->task->getSortedOrder()) {
      if (!result || p->m_computes->task->getSortedOrder() > result->m_computes->task->getSortedOrder()) {
        if (compdbg.active()) {
          compdbg << pg->myrank() << "          New best is comp from: " << p->m_computes->task->getName() << ", order="
                  << p->m_computes->task->getSortedOrder() << "\n";
        }
        result = p;
      }
    }
  }
  if (result) {
    if (compdbg.active()) {
      compdbg << pg->myrank() << "          Found comp at: " << result->m_computes->task->getName() << ", order="
              << result->m_computes->task->getSortedOrder() << "\n";
    }
    dt = result->m_task;
    comp = result->m_computes;
    return true;
  } else {
    return false;
  }
}

//______________________________________________________________________
//
bool
CompTable::findReductionComps(       Task::Dependency           * req
                             , const Patch                      * patch
                             ,       int                          matlIndex
                             ,       std::vector<DetailedTask*> & creators
                             , const ProcessorGroup             * pg
                             )
{
  // reduction variables for each level can be computed by several tasks (once per patch)
  // return the list of all tasks nearest the req

  Data key(0, req, patch, matlIndex);
  int bestSortedOrder = -1;
  for (Data* p = data.lookup(&key); p != 0; p = data.nextMatch(&key, p)) {
    if (detaileddbg.active()) {
      detaileddbg << pg->myrank() << "          Examining comp from: " << p->m_computes->task->getName() << ", order="
                  << p->m_computes->task->getSortedOrder() << " (" << req->task->getName() << " order: "
                  << req->task->getSortedOrder() << "\n";
    }

    if (p->m_computes->task->getSortedOrder() < req->task->getSortedOrder() && p->m_computes->task->getSortedOrder()
        >= bestSortedOrder) {
      if (p->m_computes->task->getSortedOrder() > bestSortedOrder) {
        creators.clear();
        bestSortedOrder = p->m_computes->task->getSortedOrder();
        detaileddbg << pg->myrank() << "          New Best Sorted Order: " << bestSortedOrder << "!\n";
      }
      if (detaileddbg.active()) {
        detaileddbg << pg->myrank() << "          Adding comp from: " << p->m_computes->task->getName() << ", order="
                    << p->m_computes->task->getSortedOrder() << "\n";
      }
      creators.push_back(p->m_task);
    }
  }
  return creators.size() > 0;
}

//______________________________________________________________________
//
void
TaskGraph::createDetailedDependencies()
{
  // TODO this logic needs to change to use the task graph

  // Collect all of the computes
  CompTable ct;
  for (int i = 0; i < m_detailed_tasks->numTasks(); i++) {
    DetailedTask* task = m_detailed_tasks->getTask(i);

    remembercomps(task, task->m_task->getComputes(), ct);
    remembercomps(task, task->m_task->getModifies(), ct);
  }

  // Assign task phase number based on the reduction tasks so a mixed thread/mpi
  // scheduler won't have out of order reduction problems.
  int currphase = 0;
  int currcomm = 0;
  for (int i = 0; i < m_detailed_tasks->numTasks(); i++) {
    DetailedTask* task = m_detailed_tasks->getTask(i);
    task->m_task->d_phase = currphase;
    if (task->m_task->getType() == Task::Reduction) {
      task->m_task->d_comm = currcomm;
      currcomm++;
      currphase++;
    } else if (task->m_task->usesMPI() || task->m_task->getType() == Task::OncePerProc) {
      currphase++;
    }
  }
  m_proc_group->setgComm(currcomm);
  m_num_task_phases = currphase + 1;

  // Go through the modifies/requires and create data dependencies as appropriate
  for (int i = 0; i < m_detailed_tasks->numTasks(); i++) {
    DetailedTask* task = m_detailed_tasks->getTask(i);

    createDetailedDependencies(task, task->m_task->getRequires(), ct, false);

    createDetailedDependencies(task, task->m_task->getModifies(), ct, true);
  }
}

//______________________________________________________________________
//
void
TaskGraph::remembercomps( DetailedTask     * task
                        , Task::Dependency * comp
                        , CompTable        & ct
                        )
{
  //calling getPatchesUnderDomain can get expensive on large processors.  Thus we
  //cache results and use them on the next call.  This works well because comps
  //are added in order and they share the same patches under the domain
  const PatchSubset *cached_task_patches = 0, *cached_comp_patches = 0;
  constHandle<PatchSubset> cached_patches;

  for (; comp != 0; comp = comp->next) {
    if (comp->var->typeDescription()->isReductionVariable()) {
      //if(task->getTask()->getType() == Task::Reduction || comp->deptype == Task::Modifies) {
      // this is either the task computing the var, modifying it, or the reduction itself
      ct.remembercomp(task, comp, 0, comp->matls, m_proc_group);
    } else {
      // Normal tasks
      constHandle<PatchSubset> patches;

      //if the patch pointer on both the dep and the task have not changed then use the
      //cached result
      if (task->m_patches == cached_task_patches && comp->patches == cached_comp_patches) {
        patches = cached_patches;
      } else {
        //compute the intersection
        patches = comp->getPatchesUnderDomain(task->m_patches);
        //cache the result for the next iteration
        cached_patches = patches;
        cached_task_patches = task->m_patches;
        cached_comp_patches = comp->patches;
      }
      constHandle<MaterialSubset> matls = comp->getMaterialsUnderDomain(task->m_matls);
      if (!patches->empty() && !matls->empty()) {
        ct.remembercomp(task, comp, patches.get_rep(), matls.get_rep(), m_proc_group);
      }
    }
  }
}

//______________________________________________________________________
//
void
TaskGraph::remapTaskDWs( int dwmap[] )
{
  // the point of this function is for using the multiple taskgraphs.
  // When you execute a taskgraph a subsequent time, you must rearrange the DWs
  // to point to the next point-in-time's DWs.
  int levelmin = 999;
  for (unsigned i = 0; i < m_tasks.size(); i++) {
    m_tasks[i]->setMapping(dwmap);

    // for the Int timesteps, we have tasks on multiple levels.
    // we need to adjust based on which level they are on, but first
    // we need to find the coarsest level.  The NewDW is relative to the coarsest
    // level executing in this taskgraph.
    if (m_tg_type == Scheduler::IntermediateTaskGraph && (m_tasks[i]->getType() != Task::Output
        && m_tasks[i]->getType() != Task::OncePerProc)) {
      if (m_tasks[i]->getType() == Task::OncePerProc || m_tasks[i]->getType() == Task::Output) {
        levelmin = 0;
        continue;
      }

      const PatchSet* ps = m_tasks[i]->getPatchSet();
      if (!ps) {
        continue;
      }
      const Level* l = getLevel(ps);
      levelmin = Min(levelmin, l->getIndex());
    }
  }
  if (detaileddbg.active()) {
    detaileddbg << m_proc_group->myrank() << " Basic mapping " << "Old " << dwmap[Task::OldDW] << " New " << dwmap[Task::NewDW]
                << " CO " << dwmap[Task::CoarseOldDW] << " CN " << dwmap[Task::CoarseNewDW] << " levelmin " << levelmin
                << std::endl;
  }

  if (m_tg_type == Scheduler::IntermediateTaskGraph) {
    // fix the CoarseNewDW for finer levels.  The CoarseOld will only matter
    // on the level it was originally mapped, so leave it as it is
    dwmap[Task::CoarseNewDW] = dwmap[Task::NewDW];
    for (unsigned i = 0; i < m_tasks.size(); i++) {
      if (m_tasks[i]->getType() != Task::Output && m_tasks[i]->getType() != Task::OncePerProc) {
        const PatchSet* ps = m_tasks[i]->getPatchSet();
        if (!ps) {
          continue;
        }
        if (getLevel(ps)->getIndex() > levelmin) {
          m_tasks[i]->setMapping(dwmap);
          if (detaileddbg.active()) {
            detaileddbg << m_tasks[i]->getName() << " mapping " << "Old " << dwmap[Task::OldDW] << " New " << dwmap[Task::NewDW]
                        << " CO " << dwmap[Task::CoarseOldDW] << " CN " << dwmap[Task::CoarseNewDW] << " (levelmin=" << levelmin
                        << ")" << std::endl;
          }
        }
      }
    }
  }
}

//______________________________________________________________________
//
void
TaskGraph::createDetailedDependencies( DetailedTask     * task
                                     , Task::Dependency * req
                                     , CompTable        & ct
                                     , bool               modifies
                                     )
{
  int me = m_proc_group->myrank();

  for (; req != nullptr; req = req->next) {

    //if(req->var->typeDescription()->isReductionVariable())
    //  continue;

    if (m_sched->isOldDW(req->mapDataWarehouse()) && !m_sched->isNewDW(req->mapDataWarehouse() + 1)) {
      continue;
    }

    constHandle<PatchSubset> patches = req->getPatchesUnderDomain(task->m_patches);
    if (req->var->typeDescription()->isReductionVariable() && m_sched->isNewDW(req->mapDataWarehouse())) {
      // make sure newdw reduction variable requires link up to the reduction tasks.
      patches = nullptr;
    }
    constHandle<MaterialSubset> matls = req->getMaterialsUnderDomain(task->m_matls);

    // this section is just to find the low and the high of the patch that will use the other
    // level's data.  Otherwise, we have to use the entire set of patches (and ghost patches if
    // applicable) that lay above/beneath this patch.

    const Patch* origPatch = nullptr;
    IntVector otherLevelLow, otherLevelHigh;
    if (req->patches_dom == Task::CoarseLevel || req->patches_dom == Task::FineLevel) {
      // the requires should have been done with Task::CoarseLevel or FineLevel, with null patches
      // and the task->patches should be size one (so we don't have to worry about overlapping regions)
      origPatch = task->m_patches->get(0);
      ASSERT(req->patches == NULL);
      ASSERT(task->m_patches->size() == 1);
      ASSERT(req->level_offset > 0);
      const Level* origLevel = origPatch->getLevel();
      if (req->patches_dom == Task::CoarseLevel) {
        // change the ghost cells to reflect coarse level
        LevelP nextLevel = origPatch->getLevelP();
        int levelOffset = req->level_offset;
        IntVector ratio = origPatch->getLevel()->getRefinementRatio();
        while (--levelOffset) {
          nextLevel = nextLevel->getCoarserLevel();
          ratio = ratio * nextLevel->getRefinementRatio();
        }
        int ngc = req->numGhostCells * Max(Max(ratio.x(), ratio.y()), ratio.z());
        IntVector ghost(ngc, ngc, ngc);

        // manually set it, can't use computeVariableExtents since there might not be
        // a neighbor fine patch, and it would throw it off.
        otherLevelLow = origPatch->getExtraCellLowIndex() - ghost;
        otherLevelHigh = origPatch->getExtraCellHighIndex() + ghost;

        otherLevelLow = origLevel->mapCellToCoarser(otherLevelLow, req->level_offset);
        otherLevelHigh = origLevel->mapCellToCoarser(otherLevelHigh, req->level_offset) + ratio - IntVector(1, 1, 1);
      } else {
        origPatch->computeVariableExtents(req->var->typeDescription()->getType(), req->var->getBoundaryLayer(), req->gtype,
                                          req->numGhostCells, otherLevelLow, otherLevelHigh);

        otherLevelLow = origLevel->mapCellToFiner(otherLevelLow);
        otherLevelHigh = origLevel->mapCellToFiner(otherLevelHigh);
      }
    }

    if (patches && !patches->empty() && matls && !matls->empty()) {
      if (req->var->typeDescription()->isReductionVariable()) {
        continue;
      }
      for (int i = 0; i < patches->size(); i++) {
        const Patch* patch = patches->get(i);

        //only allocate once
        static Patch::selectType neighbors;
        neighbors.resize(0);

        IntVector low, high;

        Patch::VariableBasis basis = Patch::translateTypeToBasis(req->var->typeDescription()->getType(), false);

        patch->computeVariableExtents(req->var->typeDescription()->getType(), req->var->getBoundaryLayer(), req->gtype,
                                      req->numGhostCells, low, high);

        if (req->patches_dom == Task::CoarseLevel || req->patches_dom == Task::FineLevel) {
          // make sure the bounds of the dep are limited to the original patch's (see above)
          // also limit to current patch, as patches already loops over all patches
          IntVector origlow = low, orighigh = high;
          if (req->patches_dom == Task::FineLevel) {
            // don't coarsen the extra cells
            low = patch->getLowIndex(basis);
            high = patch->getHighIndex(basis);
          } else {
            low = Max(low, otherLevelLow);
            high = Min(high, otherLevelHigh);
          }

          if (high.x() <= low.x() || high.y() <= low.y() || high.z() <= low.z()) {
            continue;
          }

          // don't need to selectPatches.  Just use the current patch, as we're
          // already looping over our required patches.
          neighbors.push_back(patch);
        } else {
          origPatch = patch;
          if (req->numGhostCells > 0) {
            patch->getLevel()->selectPatches(low, high, neighbors);
          } else {
            neighbors.push_back(patch);
          }
        }
        ASSERT(std::is_sorted(neighbors.begin(), neighbors.end(), Patch::Compare()));
        if (detaileddbg.active()) {
          detaileddbg << m_proc_group->myrank() << "    Creating dependency on " << neighbors.size() << " neighbors\n";
          detaileddbg << m_proc_group->myrank() << "      Low=" << low << ", high=" << high << ", var=" << req->var->getName()
                      << "\n";
        }

        for (int i = 0; i < neighbors.size(); i++) {
          const Patch* neighbor = neighbors[i];

          //if neighbor is not in my neighborhood just continue as its dependencies are not important to this processor
          if (!m_load_balancer->inNeighborhood(neighbor->getRealPatch())) {
            continue;
          }

          static Patch::selectType fromNeighbors;
          fromNeighbors.resize(0);

          IntVector l = Max(neighbor->getExtraLowIndex(basis, req->var->getBoundaryLayer()), low);
          IntVector h = Min(neighbor->getExtraHighIndex(basis, req->var->getBoundaryLayer()), high);
          if (neighbor->isVirtual()) {
            l -= neighbor->getVirtualOffset();
            h -= neighbor->getVirtualOffset();
            neighbor = neighbor->getRealPatch();
          }
          if (req->patches_dom == Task::OtherGridDomain) {
            // this is when we are copying data between two grids (currently between timesteps)
            // the grid assigned to the old dw should be the old grid.
            // This should really only impact things required from the OldDW.
            LevelP fromLevel = m_sched->get_dw(0)->getGrid()->getLevel(patch->getLevel()->getIndex());
            fromLevel->selectPatches(Max(neighbor->getExtraLowIndex(basis, req->var->getBoundaryLayer()), l),
                                     Min(neighbor->getExtraHighIndex(basis, req->var->getBoundaryLayer()), h), fromNeighbors);
          } else {
            fromNeighbors.push_back(neighbor);
          }

          for (int j = 0; j < fromNeighbors.size(); j++) {
            const Patch* fromNeighbor = fromNeighbors[j];

            //only add the requirements both fromNeighbor is in my neighborhood
            if (!m_load_balancer->inNeighborhood(fromNeighbor)) {
              continue;
            }

            IntVector from_l;
            IntVector from_h;

            if (req->patches_dom == Task::OtherGridDomain && fromNeighbor->getLevel()->getIndex() > 0) {
              // DON'T send extra cells (unless they're on the domain boundary)
              from_l = Max(fromNeighbor->getLowIndexWithDomainLayer(basis), l);
              from_h = Min(fromNeighbor->getHighIndexWithDomainLayer(basis), h);
            } else {
              // TODO - APH This intersection should not be needed, but let's clean this up if not
              //from_l = Max(fromNeighbor->getExtraLowIndex(basis, req->var->getBoundaryLayer()), l);
              //from_h = Min(fromNeighbor->getExtraHighIndex(basis, req->var->getBoundaryLayer()), h);
              from_l = l;
              from_h = h;
              //verify in debug mode that the intersection is unneeded
              ASSERT(Max(fromNeighbor->getExtraLowIndex(basis, req->var->getBoundaryLayer()), l) == l);
              ASSERT(Min(fromNeighbor->getExtraHighIndex(basis, req->var->getBoundaryLayer()), h) == h);
            }
            if (patch->getLevel()->getIndex() > 0 && patch != fromNeighbor && req->patches_dom == Task::ThisLevel) {
              // cull annoying overlapping AMR patch dependencies
              patch->cullIntersection(basis, req->var->getBoundaryLayer(), fromNeighbor, from_l, from_h);
              if (from_l == from_h) {
                continue;
              }
            }

            for (int m = 0; m < matls->size(); m++) {
              int matl = matls->get(m);

              // creator is the task that performs the original compute.
              // If the require is for the OldDW, then it will be a send old
              // data task
              DetailedTask* creator = 0;
              Task::Dependency* comp = 0;

              // look in old dw or in old TG.  Legal to modify across TG boundaries
              int proc = -1;
              if (m_sched->isOldDW(req->mapDataWarehouse())) {
                ASSERT(!modifies);
                proc = findVariableLocation(req, fromNeighbor, matl, 0);
                creator = m_detailed_tasks->getOldDWSendTask(proc);
                comp = 0;
              } else {
                if (!ct.findcomp(req, neighbor, matl, creator, comp, m_proc_group)) {
                  if (m_tg_type == Scheduler::IntermediateTaskGraph && req->lookInOldTG) {
                    // same stuff as above - but do the check for findcomp first, as this is a "if you don't find it here, assign it
                    // from the old TG" dependency
                    proc = findVariableLocation(req, fromNeighbor, matl, 0);
                    creator = m_detailed_tasks->getOldDWSendTask(proc);
                    comp = 0;
                  } else {

                    //if neither the patch or the neighbor are on this processor then the computing task doesn't exist so just continue
                    if (m_load_balancer->getPatchwiseProcessorAssignment(patch) != m_proc_group->myrank() && m_load_balancer->getPatchwiseProcessorAssignment(
                        neighbor)
                                                                                                             != m_proc_group->myrank()) {
                      continue;
                    }

                    std::cout << "Failure finding " << *req << " for " << *task << "\n";
                    if (creator) {
                      std::cout << "creator=" << *creator << "\n";
                    }
                    std::cout << "neighbor=" << *fromNeighbor << ", matl=" << matl << "\n";
                    std::cout << "me=" << me << "\n";
                    //WAIT_FOR_DEBUGGER();
                    SCI_THROW(InternalError("Failed to find comp for dep!", __FILE__, __LINE__));
                  }
                }
              }

              if (modifies && comp) {  // comp means NOT send-old-data tasks

                // find the tasks that up to this point require the variable
                // that we are modifying (i.e., the ones that use the computed
                // variable before we modify it), and put a dependency between
                // those tasks and this tasks
                // i.e., the task that requires data computed by a task on this processor
                // needs to finish its task before this task, which modifies the data
                // computed by the same task
                std::list<DetailedTask*> requireBeforeModifiedTasks;
                creator->findRequiringTasks(req->var, requireBeforeModifiedTasks);

                std::list<DetailedTask*>::iterator reqTaskIter;
                for (reqTaskIter = requireBeforeModifiedTasks.begin(); reqTaskIter != requireBeforeModifiedTasks.end();
                    ++reqTaskIter) {
                  DetailedTask* prevReqTask = *reqTaskIter;
                  if (prevReqTask == task) {
                    continue;
                  }
                  if (prevReqTask->m_task == task->m_task) {
                    if (!task->m_task->getHasSubScheduler()) {
                      std::ostringstream message;
                      message << " WARNING - task (" << task->getName()
                              << ") requires with Ghost cells *and* modifies and may not be correct" << std::endl;
                      static ProgressiveWarning warn(message.str(), 10);
                      warn.invoke();
                      if (detaileddbg.active()) {
                        detaileddbg << m_proc_group->myrank() << " Task that requires with ghost cells and modifies\n";
                        detaileddbg << m_proc_group->myrank() << " RGM: var: " << *req->var << " compute: " << *creator << " mod "
                                    << *task << " PRT " << *prevReqTask << " " << from_l << " " << from_h << "\n";
                      }
                    }
                  } else {
                    // dep requires what is to be modified before it is to be
                    // modified so create a dependency between them so the
                    // modifying won't conflict with the previous require.
                    if (detaileddbg.active()) {
                      detaileddbg << m_proc_group->myrank() << "       Requires to modifies dependency from "
                                  << prevReqTask->getName() << " to " << task->getName() << " (created by " << creator->getName()
                                  << ")\n";
                    }
                    if (creator->getPatches() && creator->getPatches()->size() > 1) {
                      // if the creator works on many patches, then don't create links between patches that don't touch
                      const PatchSubset* psub = task->getPatches();
                      const PatchSubset* req_sub = prevReqTask->getPatches();
                      if (psub->size() == 1 && req_sub->size() == 1) {
                        const Patch* p = psub->get(0);
                        const Patch* req_patch = req_sub->get(0);
                        Patch::selectType n;
                        IntVector low, high;

                        req_patch->computeVariableExtents(req->var->typeDescription()->getType(), req->var->getBoundaryLayer(),
                                                          Ghost::AroundCells, 2, low, high);

                        req_patch->getLevel()->selectPatches(low, high, n);
                        bool found = false;
                        for (int i = 0; i < n.size(); i++) {
                          if (n[i]->getID() == p->getID()) {
                            found = true;
                            break;
                          }
                        }
                        if (!found) {
                          continue;
                        }
                      }
                    }
                    m_detailed_tasks->possiblyCreateDependency(prevReqTask, 0, 0, task, req, 0, matl, from_l, from_h,
                                                               DetailedDependency::Always);
                  }
                }
              }

              DetailedDependency::CommCondition cond = DetailedDependency::Always;
              if (proc != -1 && req->patches_dom != Task::OtherGridDomain) {
                // for OldDW tasks - see comment in class DetailedDep by CommCondition
                int subsequentProc = findVariableLocation(req, fromNeighbor, matl, 1);
                if (subsequentProc != proc) {
                  cond = DetailedDependency::FirstIteration;  // change outer cond from always to first-only
                  DetailedTask* subsequentCreator = m_detailed_tasks->getOldDWSendTask(subsequentProc);
                  m_detailed_tasks->possiblyCreateDependency(subsequentCreator, comp, fromNeighbor, task, req, patch, matl, from_l,
                                                             from_h, DetailedDependency::SubsequentIterations);
                  detaileddbg << m_proc_group->myrank() << "   Adding condition reqs for " << *req->var << " task : " << *creator
                              << "  to " << *task << "\n";
                }
              }
              m_detailed_tasks->possiblyCreateDependency(creator, comp, fromNeighbor, task, req, patch, matl, from_l, from_h, cond);
            }
          }
        }
      }
    } else if (!patches && matls && !matls->empty()) {
      // requiring reduction variables
      for (int m = 0; m < matls->size(); m++) {
        int matl = matls->get(m);
        static std::vector<DetailedTask*> creators;
        creators.resize(0);

        // TODO - FIXME: figure out what this does/did - APH (02/15/16)
#if 0
        if (m_tg_type == Scheduler::IntermediateTaskGraph && req->lookInOldTG && m_sched->isNewDW(req->mapDataWarehouse())) {
          continue;  // will we need to fix for mixed scheduling?
        }
#endif
        ct.findReductionComps(req, 0, matl, creators, m_proc_group);
        // if the size is 0, that's fine.  It means that there are more procs than patches on this level,
        // so the reducer will pick a benign value that won't affect the reduction

        ASSERTRANGE(task->getAssignedResourceIndex(), 0, m_proc_group->size());
        for (unsigned i = 0; i < creators.size(); i++) {
          DetailedTask* creator = creators[i];
          if (task->getAssignedResourceIndex() == creator->getAssignedResourceIndex() && task->getAssignedResourceIndex() == me) {
            task->addInternalDependency(creator, req->var);
          }
        }
      }
    } else if (patches
        && patches->empty()
        && (req->patches_dom == Task::FineLevel || task->getTask()->getType() == Task::OncePerProc
            || task->getTask()->getType() == Task::Output || task->getTask()->getName() == "SchedulerCommon::copyDataToNewGrid")) {
      // this is a either coarsen task where there aren't any fine patches, or a PerProcessor task where
      // there aren't any patches on this processor.  Perfectly legal, so do nothing

      // another case is the copy-data-to-new-grid task, which will wither compute or modify to every patch
      // but not both.  So it will yell at you for the detailed task's patches not intersecting with the
      // computes or modifies... (maybe there's a better way) - bryan
    } else {
      std::ostringstream desc;
      desc << "TaskGraph::createDetailedDependencies, task dependency not supported without patches and materials"
           << " \n Trying to require or modify " << *req << " in Task " << task->getTask()->getName() << "\n\n";
      if (task->m_matls) {
        desc << "task materials:" << *task->m_matls << "\n";
      } else {
        desc << "no task materials\n";
      }
      if (req->matls) {
        desc << "req materials: " << *req->matls << "\n";
      } else {
        desc << "no req materials\n";
        desc << "domain materials: " << *matls.get_rep() << "\n";
      }
      if (task->m_patches) {
        desc << "task patches:" << *task->m_patches << "\n";
      } else {
        desc << "no task patches\n";
      }
      if (req->patches) {
        desc << "req patches: " << *req->patches << "\n";
      } else {
        desc << "no req patches\n";
      }

      desc << "domain patches: " << *patches.get_rep() << "\n";
      SCI_THROW(InternalError(desc.str(), __FILE__, __LINE__));
    }
  }
}

//______________________________________________________________________
//
int
TaskGraph::findVariableLocation(       Task::Dependency * req
                               , const Patch            * patch
                               ,       int                matl
                               ,       int                iteration
                               )
{
  // This needs to be improved, especially for re-distribution on
  // restart from checkpoint.
  int proc;
  if ((req->task->mapDataWarehouse(Task::ParentNewDW) != -1 && req->whichdw != Task::ParentOldDW) || iteration > 0
      || (req->lookInOldTG && m_tg_type == Scheduler::IntermediateTaskGraph)) {
    // Provide some accommodation for Dynamic load balancers and sub schedulers.  We need to
    // treat the requirement like a "old" dw req but it needs to be found on the current processor.
    // Same goes for successive executions of the same TG.
    proc = m_load_balancer->getPatchwiseProcessorAssignment(patch);
  } else {
    proc = m_load_balancer->getOldProcessorAssignment(patch);
  }
  return proc;
}

//______________________________________________________________________
//
int
TaskGraph::getNumTasks() const
{
  return static_cast<int>(m_tasks.size());
}

//______________________________________________________________________
//
Task*
TaskGraph::getTask( int idx )
{
  return m_tasks[idx];
}

//______________________________________________________________________
//
void
TaskGraph::makeVarLabelMaterialMap( Scheduler::VarLabelMaterialMap * result )
{
  for (int i = 0; i < (int)m_tasks.size(); i++) {
    Task* task = m_tasks[i];
    for (Task::Dependency* comp = task->getComputes(); comp != 0; comp = comp->next) {
      // assume all patches will compute the same labels on the same materials
      const VarLabel* label = comp->var;
      std::list<int>& matls = (*result)[label->getName()];
      const MaterialSubset* msubset = comp->matls;
      if (msubset) {
        for (int mm = 0; mm < msubset->size(); mm++) {
          matls.push_back(msubset->get(mm));
        }
      } else if (label->typeDescription()->getType() == TypeDescription::ReductionVariable) {
        // Default to material -1 (global)
        matls.push_back(-1);
      } else {
        const MaterialSet* ms = task->getMaterialSet();
        for (int m = 0; m < ms->size(); m++) {
          const MaterialSubset* msubset = ms->getSubset(m);
          for (int mm = 0; mm < msubset->size(); mm++) {
            matls.push_back(msubset->get(mm));
          }
        }
      }
    }
  }
}
