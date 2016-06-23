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
#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/LoadBalancer.h>

#include <Core/Containers/FastHashTable.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/TypeMismatchException.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/ProgressiveWarning.h>

#include <sci_defs/config_defs.h>
#include <sci_algorithm.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <cstring>
#include <sstream>
#include <unistd.h>

using namespace Uintah;

static DebugStream tgdbg(       "TaskGraph",         false);
static DebugStream tgphasedbg(  "TaskGraphPhase",    false);
static DebugStream detaileddbg( "TaskGraphDetailed", false);
static DebugStream compdbg(     "FindComp",          false);

//______________________________________________________________________
//
TaskGraph::TaskGraph(SchedulerCommon* sc,
                     const ProcessorGroup* pg,
                     Scheduler::tgType type)
    : sc(sc),
      d_myworld(pg),
      type_(type),
      dts_(nullptr),
      currentIteration(0),
      d_numtaskphases(0)
{
  lb = dynamic_cast<LoadBalancer*>(sc->getPort("load balancer"));
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
  if (dts_) {
    delete dts_;
  }

  for (std::vector<Task*>::iterator iter = d_tasks.begin(); iter != d_tasks.end(); iter++) {
    delete *iter;
  }

  for (std::vector<Task::Edge*>::iterator iter = edges.begin(); iter != edges.end(); iter++) {
    delete *iter;
  }

  d_tasks.clear();
  d_numtaskphases = 0;

  edges.clear();
  currentIteration = 0;
}

//______________________________________________________________________
//
bool
TaskGraph::overlaps( const Task::Dependency* comp, const Task::Dependency* req ) const
{
  constHandle<PatchSubset> saveHandle2;
  const PatchSubset* ps1 = comp->m_patches;
  if (!ps1) {
    if (!comp->m_task->getPatchSet()) {
      return false;
    }
    ps1 = comp->m_task->getPatchSet()->getUnion();
    if (comp->m_patches_dom == Task::CoarseLevel || comp->m_patches_dom == Task::FineLevel) {
      SCI_THROW(InternalError("Should not compute onto another level!", __FILE__, __LINE__));
      // This may not be a big deal if it were needed, but I didn't
      // think that it should be allowed - Steve
      // saveHandle1 = comp->getPatchesUnderDomain(ps1);
      // ps1 = saveHandle1.get_rep();
    }
  }

  const PatchSubset* ps2 = req->m_patches;
  if (!ps2) {
    if (!req->m_task->getPatchSet()) {
      return false;
    }
    ps2 = req->m_task->getPatchSet()->getUnion();
    if (req->m_patches_dom == Task::CoarseLevel || req->m_patches_dom == Task::FineLevel) {
      saveHandle2 = req->getPatchesUnderDomain(ps2);
      ps2 = saveHandle2.get_rep();
    }
  }

  if (!PatchSubset::overlaps(ps1, ps2)) {  // && !(ps1->size() == 0 && (!req->patches || ps2->size() == 0) && comp->task->getType() == Task::OncePerProc))
    return false;
  }

  const MaterialSubset* ms1 = comp->m_matls;
  if (!ms1) {
    if (!comp->m_task->getMaterialSet()) {
      return false;
    }
    ms1 = comp->m_task->getMaterialSet()->getUnion();
  }
  const MaterialSubset* ms2 = req->m_matls;
  if (!ms2) {
    if (!req->m_task->getMaterialSet()) {
      return false;
    }
    ms2 = req->m_task->getMaterialSet()->getUnion();
  }
  if (!MaterialSubset::overlaps(ms1, ms2)) {
    return false;
  }
  return true;
}

//______________________________________________________________________
//
// setupTaskConnections also adds Reduction Tasks to the graph...
void
TaskGraph::setupTaskConnections( GraphSortInfoMap& sortinfo )
{
  std::vector<Task*>::iterator iter;
  // Initialize variables on the tasks
  for (iter = d_tasks.begin(); iter != d_tasks.end(); iter++) {
    sortinfo[*iter] = GraphSortInfo();
  }

  if (edges.size() > 0) {
    return; // already been done
  }

  // Look for all of the reduction variables - we must treat those special.  Create a fake task that performs the reduction
  // While we are at it, ensure that we aren't producing anything into an "old" data warehouse
  ReductionTasksMap reductionTasks;
  for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
    Task* task = *iter;
    if (task->isReductionTask()) {
      continue; // already a reduction task so skip it
    }

    for (Task::Dependency* comp = task->getComputes(); comp != nullptr; comp = comp->m_next) {
      if(sc->isOldDW(comp->mapDataWarehouse())) {
        if (detaileddbg.active()) {
          detaileddbg << d_myworld->myrank() << " which = " << comp->m_whichdw << ", mapped to " << comp->mapDataWarehouse() << "\n";
        }
        SCI_THROW(InternalError("Variable produced in old datawarehouse: " +comp->m_var->getName(), __FILE__, __LINE__));
      } else if(comp->m_var->typeDescription()->isReductionVariable()){
        int levelidx = comp->m_reduction_level ? comp->m_reduction_level->getIndex() : -1;
        // Look up this variable in the reductionTasks map
        int dw = comp->mapDataWarehouse();
        // for reduction var allows multi computes such as delT
        // do not generate reduction task each time it computes,
        // instead computes it in a system wide reduction task
        if (comp->m_var->allowsMultipleComputes()) {
          if (detaileddbg.active()) {
            detaileddbg << d_myworld->myrank() << " Skipping Reduction task for variable: " << comp->m_var->getName() << " on level "
                        << levelidx << ", DW " << dw << "\n";
          }
          continue;
        }
        ASSERT(comp->m_patches == nullptr);

        // use the dw as a 'material', just for the sake of looking it up.
        // it should only differentiate on AMR W-cycle graphs...
        VarLabelMatl<Level> key(comp->m_var, dw, comp->m_reduction_level);
        const MaterialSet* ms = task->getMaterialSet();
        const Level* level = comp->m_reduction_level;

        ReductionTasksMap::iterator it=reductionTasks.find(key);
        if(it == reductionTasks.end()) {
          // No reduction task yet, create one
          if (detaileddbg.active()) {
            detaileddbg << d_myworld->myrank() << " creating Reduction task for variable: " << comp->m_var->getName() << " on level "
                << levelidx << ", DW " << dw << "\n";
          }
          std::ostringstream taskname;
          taskname << "Reduction: " << comp->m_var->getName() << ", level: " << levelidx << ", dw: " << dw;
          Task* newtask = scinew Task(taskname.str(), Task::Reduction);

          sortinfo[newtask] = GraphSortInfo();

          int dwmap[Task::TotalDWs];
          for (int i = 0; i < Task::TotalDWs; i++) {
            dwmap[i] = Task::InvalidDW;
          }
          dwmap[Task::OldDW] = Task::NoDW;
          dwmap[Task::NewDW] = dw;
          newtask->setMapping(dwmap);

          // compute and require for all patches but some set of materials
          // (maybe global material, but not necessarily)
          if (comp->m_matls != nullptr) {
            // TODO APH - figure this out and clean up - 06/15/16
            //newtask->computes(comp->var, level, comp->matls, Task::OutOfDomain);
            //newtask->requires(Task::NewDW, comp->var, level, comp->matls, Task::OutOfDomain);
            newtask->modifies(comp->m_var, level, comp->m_matls, Task::OutOfDomain);
          }
          else {
            for (int m = 0; m < ms->size(); m++) {
              // TODO APH - figure this out and clean up - 01/31/15
              //newtask->computes(comp->var, level, ms->getSubset(m), Task::OutOfDomain);
              //newtask->requires(Task::NewDW, comp->var, level, ms->getSubset(m), Task::OutOfDomain);
              newtask->modifies(comp->m_var, level, ms->getSubset(m), Task::OutOfDomain);
            }
          }
          reductionTasks[key]=newtask;
          it = reductionTasks.find(key);
        }
      }
    }
  }

  // Add the new reduction tasks to the list of tasks
  for(ReductionTasksMap::iterator it = reductionTasks.begin(); it != reductionTasks.end(); it++) {
    addTask(it->second, 0, 0);
  }

  // Gather the comps for the tasks into a map
  CompMap comps;
  for (iter = d_tasks.begin(); iter != d_tasks.end(); iter++) {
    Task* task = *iter;
    if (detaileddbg.active()) {
      detaileddbg << d_myworld->myrank() << " Gathering comps from task: " << *task << "\n";
    }
    for (Task::Dependency* comp = task->getComputes(); comp != nullptr; comp = comp->m_next) {
      comps.insert(std::make_pair(comp->m_var, comp));
      if (detaileddbg.active()) {
        detaileddbg << d_myworld->myrank() << "   Added comp for: " << *comp << "\n";
      }
    }
  }

  // Connect the tasks where the requires/modifies match a comp.
  // Also, updates the comp map with each modify and doing this in task order
  // so future modifies/requires find the modified var.  Also do a type check
  for (iter = d_tasks.begin(); iter != d_tasks.end(); iter++) {
    Task* task = *iter;
    if (detaileddbg.active()) {
      detaileddbg << d_myworld->myrank() << "   Looking at dependencies for task: " << *task << "\n";
    }
    addDependencyEdges(task, sortinfo, task->getRequires(), comps, reductionTasks, false);
    addDependencyEdges(task, sortinfo, task->getModifies(), comps, reductionTasks, true);
    // Used here just to warn if a modifies comes before its computes
    // in the order that tasks were added to the graph.
    sortinfo.find(task)->second.visited = true;
    task->m_all_child_tasks.clear();
    if (detaileddbg.active()) {
      std::cout << d_myworld->myrank() << "   Looking at dependencies for task: " << *task << "child task num="
           << task->m_child_tasks.size() << std::endl;
    }
  }
  
  //count the all child tasks
  int nd_task = d_tasks.size();
  while (nd_task > 0) {
    nd_task = d_tasks.size();
    for (iter = d_tasks.begin(); iter != d_tasks.end(); iter++) {
      Task* task = *iter;
      if (task->m_all_child_tasks.size() == 0) {
        if (task->m_child_tasks.size() == 0) {     // leaf task, add itself to the set
          task->m_all_child_tasks.insert(task);
          break;
        }
        std::set<Task*>::iterator it;
        for (it = task->m_child_tasks.begin(); it != task->m_child_tasks.end(); it++) {
          if ((*it)->m_all_child_tasks.size() > 0) {
            task->m_all_child_tasks.insert((*it)->m_all_child_tasks.begin(), (*it)->m_all_child_tasks.end());
            task->m_all_child_tasks.insert(*it);
          }
          // if child didn't finish computing allChildTasks
          else {
            task->m_all_child_tasks.clear();
            break;
          }
        }
      }
      else {
        nd_task--;
      }
    }
  }

  // Initialize variables on the tasks
  GraphSortInfoMap::iterator sort_iter;
  for (sort_iter = sortinfo.begin(); sort_iter != sortinfo.end(); sort_iter++) {
    sort_iter->second.visited = false;
    sort_iter->second.sorted  = false;
  }
} // end setupTaskConnections()

//______________________________________________________________________
//
void
TaskGraph::addDependencyEdges( Task*              task,
                               GraphSortInfoMap&  sortinfo,
                               Task::Dependency*  req,
                               CompMap&           comps,
                               ReductionTasksMap& reductionTasks,
                               bool               modifies )
{
  for(; req != nullptr; req=req->m_next){
    if (detaileddbg.active()) {
      detaileddbg << d_myworld->myrank() << "     Checking edge for req: " << *req << ", task: " << *req->m_task << ", domain: "
                  << req->m_patches_dom << "\n";
    }
    if(req->m_whichdw==Task::NewDW) {
      // If DW is finalized, we assume that we already have it,
      // or that we will get it sent to us.  Otherwise, we set
      // up an edge to connect this req to a comp

      std::pair<CompMap::iterator, CompMap::iterator> iters = comps.equal_range(static_cast<const Uintah::VarLabel*>(req->m_var));
      int count = 0;
      for (CompMap::iterator compiter = iters.first; compiter != iters.second; ++compiter) {

        if (req->m_var->typeDescription() != compiter->first->typeDescription()) {
          SCI_THROW(TypeMismatchException("Type mismatch for variable: "+req->m_var->getName(), __FILE__, __LINE__));
        }

        // determine if we need to add a dependency edge
        bool add = false;
        bool requiresReductionTask = false;
        if (detaileddbg.active()) {
          detaileddbg << d_myworld->myrank() << "  Checking edge from comp: " << *compiter->second << ", task: "
                      << *compiter->second->m_task << ", domain: " << compiter->second->m_patches_dom << "\n";
        }
        if (req->mapDataWarehouse() == compiter->second->mapDataWarehouse()) {
          if (req->m_var->typeDescription()->isReductionVariable()) {
            // Match the level first 
            if (compiter->second->m_reduction_level == req->m_reduction_level) {
              add = true;
            }
            // with reduction variables, you can modify them up to the Reduction Task, which also modifies
            // those who don't modify will get the reduced value.
            if (!modifies && !req->m_var->allowsMultipleComputes()) {
              requiresReductionTask = true;
            }
          }
          else if (overlaps(compiter->second, req)) {
            add = true;
          }
        }

        if (!add) {
          if (detaileddbg.active()) {
            detaileddbg << d_myworld->myrank() << "       did NOT create dependency\n";
          }
        }
        else {
          Task::Dependency* comp;
          if (requiresReductionTask) {
            VarLabelMatl<Level> key(req->m_var, req->mapDataWarehouse(), req->m_reduction_level);
            Task* redTask = reductionTasks[key];
            ASSERT(redTask != nullptr);
            // reduction tasks should have exactly 1 require, and it should be a modify
            // assign the requiring task's require to it
            comp = redTask->getModifies();
            ASSERT(comp != nullptr);
            detaileddbg << "  Using Reduction task: " << *redTask << std::endl;
          }
          else {
            comp = compiter->second;
          }

          if (modifies) {
            // Add dependency edges to each task that requires the data
            // before it is modified.
            for (Task::Edge* otherEdge = comp->m_req_head; otherEdge != nullptr; otherEdge = otherEdge->m_req_next) {
              Task::Dependency* priorReq = const_cast<Task::Dependency*>(otherEdge->m_req);
              if (priorReq != req) {
                ASSERT(priorReq->m_var->equals(req->m_var));
                if (priorReq->m_task != task) {
                  Task::Edge* edge = scinew Task::Edge(priorReq, req);
                  edges.push_back(edge);
                  req->addComp(edge);
                  priorReq->addReq(edge);
                  if (detaileddbg.active()) {
                    detaileddbg << d_myworld->myrank() << " Creating edge from task: " << *priorReq->m_task << " to task: "
                                << *req->m_task << "\n";
                    detaileddbg << d_myworld->myrank() << " Prior Req=" << *priorReq << "\n";
                    detaileddbg << d_myworld->myrank() << " Modify=" << *req << "\n";
                  }
                }
              }
            }
          }

          // add the edge between the require/modify and compute
          Task::Edge* edge = scinew Task::Edge(comp, req);
          edges.push_back(edge);
          req->addComp(edge);
          comp->addReq(edge);

          if (!sortinfo.find(edge->m_comp->m_task)->second.visited && !edge->m_comp->m_task->isReductionTask()) {
            std::cout << "\nWARNING: The task, '" << task->getName() << "', that ";
            if (modifies) {
              std::cout << "modifies '";
            }
            else {
              std::cout << "requires '";
            }
            std::cout << req->m_var->getName() << "' was added before the computing task";
            std::cout << ", '" << edge->m_comp->m_task->getName() << "'\n";
            std::cout << "  Required/modified by: " << *task << "\n";
            std::cout << "  req: " << *req << "\n";
            std::cout << "  Computed by: " << *edge->m_comp->m_task << "\n";
            std::cout << "  comp: " << *comp << "\n";
            std::cout << std::endl;
          }
          count++;
          task->m_child_tasks.insert(comp->m_task);
          if (detaileddbg.active()) {
            detaileddbg << d_myworld->myrank() << "       Creating edge from task: " << *comp->m_task << " to task: " << *task << "\n";
            detaileddbg << d_myworld->myrank() << "         Req=" << *req << "\n";
            detaileddbg << d_myworld->myrank() << "         Comp=" << *comp << "\n";
          }
        }
      }

      // if we cannot find the required variable, throw an exception
      if (count == 0 && (!req->m_matls || req->m_matls->size() > 0) && (!req->m_patches || req->m_patches->size() > 0)
          && !(req->m_look_in_old_tg && type_ == Scheduler::IntermediateTaskGraph)) {
        // if this is an Intermediate TG and the requested data is done from another TG,
        // we need to look in this TG first, but don't worry if you don't find it

        std::cout << "ERROR: Cannot find the task that computes the variable (" << req->m_var->getName() << ")\n";

        std::cout << "The task (" << task->getName() << ") is requesting data from:\n";
        std::cout << "  Level:           " << getLevel(task->getPatchSet())->getIndex() << "\n";
        std::cout << "  Task:PatchSet    " << *(task->getPatchSet()) << "\n";
        std::cout << "  Task:MaterialSet " << *(task->getMaterialSet()) << "\n \n";

        std::cout << "The variable (" << req->m_var->getName() << ") is requiring data from:\n";

        if (req->m_patches) {
          std::cout << "  Level: " << getLevel(req->m_patches)->getIndex() << "\n";
          std::cout << "  Patches': " << *(req->m_patches) << "\n";
        }
        else {
          std::cout << "  Patches:  All \n";
        }

        if (req->m_matls) {
          std::cout << "  Materials: " << *(req->m_matls) << "\n";
        }
        else {
          std::cout << "  Materials:  All \n";
        }

        std::cout << "\nTask Details:\n";
        task->display(std::cout);
        std::cout << "\nRequirement Details:\n" << *req << "\n";

        SCI_THROW(InternalError("Scheduler could not find  production for variable: "+req->m_var->getName()+", required for task: "+task->getName(), __FILE__, __LINE__));
      }
      
      if (modifies) {
        // not just requires, but modifies, so the comps map must be
        // updated so future modifies or requires will link to this one.
        comps.insert(std::make_pair(req->m_var, req));
        if (detaileddbg.active()) {
          detaileddbg << d_myworld->myrank() << " Added modified comp for: " << *req << "\n";
        }
      }
    }
  }
}

//______________________________________________________________________
//
void TaskGraph::processTask(Task* task,
                            std::vector<Task*>& sortedTasks,
                            GraphSortInfoMap& sortinfo) const
{
  if (detaileddbg.active()) {
    detaileddbg << d_myworld->myrank() << " Looking at task: " << task->getName() << "\n";
  }

  GraphSortInfo& gsi = sortinfo.find(task)->second;
  // we throw an exception before calling processTask if this task has already been visited
  gsi.visited = true;

  processDependencies(task, task->getRequires(), sortedTasks, sortinfo);
  processDependencies(task, task->getModifies(), sortedTasks, sortinfo);

  // All prerequisites are done - add this task to the list
  sortedTasks.push_back(task);
  gsi.sorted = true;

  if (detaileddbg.active()) {
    detaileddbg << d_myworld->myrank() << " Sorted task: " << task->getName() << "\n";
  }
}  // end processTask()


//______________________________________________________________________
//
void TaskGraph::processDependencies(Task* task,
                                    Task::Dependency* req,
                                    std::vector<Task*>& sortedTasks,
                                    GraphSortInfoMap& sortinfo) const

{
  for (; req != nullptr; req = req->m_next) {
    if (detaileddbg.active()) {
      detaileddbg << d_myworld->myrank() << " processDependencies for req: " << *req << "\n";
    }
    if (req->m_whichdw == Task::NewDW) {
      Task::Edge* edge = req->m_comp_head;
      for (; edge != nullptr; edge = edge->m_comp_next) {
        Task* vtask = edge->m_comp->m_task;
        GraphSortInfo& gsi = sortinfo.find(vtask)->second;
        if (!gsi.sorted) {
          try {
            // this try-catch mechanism will serve to print out the entire TG cycle
            if (gsi.visited) {
              std::cout << d_myworld->myrank() << " Cycle detected in task graph\n";
              SCI_THROW(InternalError("Cycle detected in task graph", __FILE__, __LINE__));
            }

            // recursively process the dependencies of the computing task
            processTask(vtask, sortedTasks, sortinfo);
          }
          catch (InternalError& e) {
            std::cout << d_myworld->myrank() << "   Task " << task->getName() << " requires " << req->m_var->getName() << " from "
                 << vtask->getName() << std::endl;

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
TaskGraph::nullSort( std::vector<Task*>& tasks )
{
  std::vector<Task*>::iterator iter;

  // No longer going to sort them... let the UnifiedScheduler (threaded) take care
  // of calling the tasks when all dependencies are satisfied.
  // Sorting the tasks causes problem because now tasks (actually task
  // groups) run in different orders on different MPI processes.
  int n = 0;
  for (iter = d_tasks.begin(); iter != d_tasks.end(); iter++) {
    // For all reduction tasks filtering out the one that is not in ReductionTasksMap 
    if ((*iter)->getType() == Task::Reduction) {
      for (SchedulerCommon::ReductionTasksMap::iterator it = sc->reductionTasks.begin(); it != sc->reductionTasks.end(); it++) {
        if ((*iter) == it->second) {
          (*iter)->setSortedOrder(n++);
          tasks.push_back(*iter);
          break;
        }
      }
    }
    else {
      (*iter)->setSortedOrder(n++);
      tasks.push_back(*iter);
    }
  }
}

//______________________________________________________________________
//
void
TaskGraph::topologicalSort( std::vector<Task*>& sortedTasks )
{
  GraphSortInfoMap sortinfo;

  setupTaskConnections(sortinfo);

  for( std::vector<Task*>::iterator iter = d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
    Task* task = *iter;
    if (!sortinfo.find(task)->second.sorted) {
      processTask(task, sortedTasks, sortinfo);
    }
  }
  int n = 0;
  for( std::vector<Task*>::iterator iter = sortedTasks.begin(); iter != sortedTasks.end(); iter++ ) {
    (*iter)->setSortedOrder(n++);
  }
}

//______________________________________________________________________
//

void
TaskGraph::addTask(       Task*        task,
                    const PatchSet*    patchset,
                    const MaterialSet* matlset )
{
  task->setSets(patchset, matlset);
  if ((patchset && patchset->totalsize() == 0) || (matlset && matlset->totalsize() == 0)) {
    delete task;
    if (detaileddbg.active()) {
      detaileddbg << d_myworld->myrank() << " Killing empty task: " << *task << "\n";
    }
  }
  else {
    d_tasks.push_back(task);

    // debugging Code
    if (tgdbg.active()) {
      tgdbg << d_myworld->myrank() << " Adding task: ";
      task->displayAll(tgdbg);
    }
    
#if 0
    // This snippet will find all the tasks that require a label
    for (Task::Dependency* m_req = m_task->getRequires(); m_req != 0; m_req = m_req->m_next) {
      const VarLabel* label = m_req->m_var;
      string name = label->getName();
      if (name == "p.size") {
        cout << "\n" << Parallel::getMPIRank() << "This Task Requires label p.size" << endl;
        m_task->display(cout);
      }
    }
#endif
  }
}

//______________________________________________________________________
//

void
TaskGraph::createDetailedTask(       Task*           task,
                               const PatchSubset*    patches,
                               const MaterialSubset* matls )
{
  DetailedTask* dt = scinew DetailedTask(task, patches, matls, dts_);

  if (task->getType() == Task::Reduction) {
    Task::Dependency* req = task->getModifies();
    // reduction tasks should have exactly 1 require, and it should be a modify
    ASSERT(req != nullptr);
    d_reductionTasks[req->m_var] = dt;
  }

  dts_->add(dt);
}

//______________________________________________________________________
//

DetailedTasks*
TaskGraph::createDetailedTasks(       bool           useInternalDeps,
                                      DetailedTasks* first,
                                const GridP&         grid,
                                const GridP&         oldGrid )
{
  std::vector<Task*> sorted_tasks;

  // TODO plz leave this commented line alone, APH 01/07/15
  //topologicalSort(sorted_tasks);
  nullSort(sorted_tasks);

  d_reductionTasks.clear();

  ASSERT(grid != nullptr);
  lb->createNeighborhood(grid, oldGrid);

  const std::set<int> neighborhood_procs=lb->getNeighborhoodProcessors();
  dts_ = scinew DetailedTasks(sc, d_myworld, first, this, neighborhood_procs, useInternalDeps );
  
  for (int i = 0; i < (int)sorted_tasks.size(); i++) {

    Task* task = sorted_tasks[i];
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
            createDetailedTask(task, pss, mss);
          }
        }
      }
      else if (task->getType() == Task::Output) {
        //compute subset that involves this rank
        int subset = (d_myworld->myrank() / lb->getNthProc()) * lb->getNthProc();

        //only schedule output task for the subset involving our rank
        const PatchSubset* pss = ps->getSubset(subset);

        //don't schedule if there are no patches
        if (pss->size() > 0) {
          for (int m = 0; m < ms->size(); m++) {
            const MaterialSubset* mss = ms->getSubset(m);
            createDetailedTask(task, pss, mss);
          }
        }

      }
      else {
        for (int p = 0; p < ps->size(); p++) {
          const PatchSubset* pss = ps->getSubset(p);
          // don't make tasks that are not in our neighborhood or tasks that do not have patches
          if (lb->inNeighborhood(pss) && pss->size() > 0) {
            for (int m = 0; m < ms->size(); m++) {
              const MaterialSubset* mss = ms->getSubset(m);
              createDetailedTask(task, pss, mss);
            }
          }
        }
      }
    }
    else if (!ps && !ms) {
      createDetailedTask(task, nullptr, nullptr);
    }
    else if (!ps) {
      SCI_THROW(InternalError("Task has MaterialSet, but no PatchSet", __FILE__, __LINE__));
    }
    else {
      SCI_THROW(InternalError("Task has PatchSet, but no MaterialSet", __FILE__, __LINE__));
    }
  }
  
  
// this can happen if a processor has no patches (which may happen at the beginning of some AMR runs)
//  if(dts_->numTasks() == 0)
//    cerr << "WARNING: Compiling scheduler with no tasks\n";

  lb->assignResources(*dts_);

  // use this, even on a single processor, if for nothing else than to get scrub counts
  bool doDetailed = Parallel::usingMPI() || useInternalDeps || grid->numLevels() > 1;
  if (doDetailed) {
    createDetailedDependencies();
    if (dts_->getExtraCommunication() > 0 && d_myworld->myrank() == 0) {
      std::cout << d_myworld->myrank() << "  Warning: Extra communication.  This taskgraph on this rank overcommunicates about "
           << dts_->getExtraCommunication() << " cells\n";
    }
  }

  if (d_myworld->size() > 1) {
    dts_->assignMessageTags(d_myworld->myrank());
  }

  dts_->computeLocalTasks(d_myworld->myrank());
  dts_->makeDWKeyDatabase();

  if (!doDetailed) {
    // the createDetailedDependencies will take care of scrub counts, otherwise do it here.
    dts_->createScrubCounts();
  }

  return dts_;
} // end TaskGraph::createDetailedTasks

//______________________________________________________________________
//
namespace Uintah {

class CompTable {

  struct Data {
    Data* next;
    DetailedTask* task;
    Task::Dependency* comp;
    const Patch* patch;
    int matl;
    unsigned int hash;

    unsigned int string_hash(const char* p)
    {
      unsigned int sum = 0;
      while (*p) {
        sum = sum * 7 + (unsigned char)*p++;
      }
      return sum;
    }

    Data(DetailedTask* task,
         Task::Dependency* comp,
         const Patch* patch,
         int matl)
        : task(task),
          comp(comp),
          patch(patch),
          matl(matl)
    {
      hash = (unsigned int)(((unsigned int)comp->mapDataWarehouse() << 3) ^ (string_hash(comp->m_var->getName().c_str())) ^ matl);
      if (patch) {
        hash ^= (unsigned int)(patch->getID() << 4);
      }
    }

    ~Data() {}

    bool operator==(const Data& c)
    {
      return matl == c.matl && patch == c.patch && comp->m_reduction_level == c.comp->m_reduction_level
             && comp->mapDataWarehouse() == c.comp->mapDataWarehouse() && comp->m_var->equals(c.comp->m_var);
    }
  };

  FastHashTable<Data> data;

  void insert(Data* d);

public:

  CompTable();

  ~CompTable();

  void remembercomp(DetailedTask* task,
                    Task::Dependency* comp,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    const ProcessorGroup* pg);

  bool findcomp(Task::Dependency* req,
                const Patch* patch,
                int matlIndex,
                DetailedTask*& dt,
                Task::Dependency*& comp,
                const ProcessorGroup* pg);

  bool findReductionComps(Task::Dependency* req,
                          const Patch* patch,
                          int matlIndex,
                          std::vector<DetailedTask*>& dt,
                          const ProcessorGroup* pg);

private:

  void remembercomp(Data* newData,
                    const ProcessorGroup* pg);
};

}

CompTable::CompTable(){}

CompTable::~CompTable(){}

//______________________________________________________________________
//
void
CompTable::remembercomp( Data* newData, const ProcessorGroup* pg )
{
  if (detaileddbg.active()) {
    detaileddbg << pg->myrank() << " remembercomp: " << *newData->comp << ", matl=" << newData->matl;
    if (newData->patch) {
      detaileddbg << ", patch=" << *newData->patch;
    }
    detaileddbg << "\n";
  }

  // can't have two computes for the same variable (need modifies)
  if (newData->comp->m_dep_type != Task::Modifies && !newData->comp->m_var->typeDescription()->isReductionVariable()) {
    if (data.lookup(newData)) {
      std::cout << "Multiple compute found:\n";
      std::cout << "  matl: " << newData->matl << "\n";
      if (newData->patch) {
        std::cout << "  patch: " << *newData->patch << "\n";
      }
      std::cout << "  " << *newData->comp << "\n";
      std::cout << "  " << *newData->task << "\n\n";
      std::cout << "  It was originally computed by the following task(s):\n";
      for (Data* old = data.lookup(newData); old != nullptr; old = data.nextMatch(newData, old)) {
        std::cout << "  " << *old->task << std::endl;
        //old->comp->task->displayAll(cout);
      }
      SCI_THROW(InternalError("Multiple computes for variable: "+newData->comp->m_var->getName(), __FILE__, __LINE__));
    }
  }
  data.insert(newData);
}

//______________________________________________________________________
//
void
CompTable::remembercomp(       DetailedTask*     task,
                               Task::Dependency* comp,
                         const PatchSubset*      patches,
                         const MaterialSubset*   matls,
                         const ProcessorGroup*   pg )
{
  if (patches && matls) {
    for (int p = 0; p < patches->size(); p++) {
      const Patch* patch = patches->get(p);
      for (int m = 0; m < matls->size(); m++) {
        int matl = matls->get(m);
        Data* newData = scinew Data(task, comp, patch, matl);
        remembercomp(newData, pg);
      }
    }
  }
  else if (matls) {
    for (int m = 0; m < matls->size(); m++) {
      int matl = matls->get(m);
      Data* newData = scinew Data(task, comp, 0, matl);
      remembercomp(newData, pg);
    }
  }
  else if (patches) {
    for (int p = 0; p < patches->size(); p++) {
      const Patch* patch = patches->get(p);
      Data* newData = scinew Data(task, comp, patch, 0);
      remembercomp(newData, pg);
    }
  }
  else {
    Data* newData = scinew Data(task, comp, nullptr, 0);
    remembercomp(newData, pg);
  }
}

bool
CompTable::findcomp(       Task::Dependency*  req,
                     const Patch*             patch,
                           int                matlIndex,
                           DetailedTask*&     dt,
                           Task::Dependency*& comp,
                     const ProcessorGroup*    pg )
{
  if (compdbg.active()) {
    compdbg << pg->myrank() << "        Finding comp of req: " << *req << " for task: " << *req->m_task << "/" << "\n";
  }
  Data key(nullptr, req, patch, matlIndex);
  Data* result = nullptr;
  for (Data* p = data.lookup(&key); p != nullptr; p = data.nextMatch(&key, p)) {
    if (compdbg.active()) {
      compdbg << pg->myrank() << "          Examining comp from: " << p->comp->m_task->getName() << ", order="
              << p->comp->m_task->getSortedOrder() << "\n";
    }

    ASSERT(!result || p->comp->m_task->getSortedOrder() != result->comp->m_task->getSortedOrder());

    if (p->comp->m_task->getSortedOrder() < req->m_task->getSortedOrder()) {
      if (!result || p->comp->m_task->getSortedOrder() > result->comp->m_task->getSortedOrder()) {
        if (compdbg.active()) {
          compdbg << pg->myrank() << "          New best is comp from: " << p->comp->m_task->getName() << ", order="
                  << p->comp->m_task->getSortedOrder() << "\n";
        }
        result = p;
      }
    }
  }
  if (result) {
    if (compdbg.active()) {
      compdbg << pg->myrank() << "          Found comp at: " << result->comp->m_task->getName() << ", order="
              << result->comp->m_task->getSortedOrder() << "\n";
    }
    dt = result->task;
    comp = result->comp;
    return true;
  }
  else {
    return false;
  }
}

//______________________________________________________________________
//
bool
CompTable::findReductionComps(       Task::Dependency*      req,
                               const Patch*                 patch,
                                     int                    matlIndex,
                                     std::vector<DetailedTask*>& creators,
                               const ProcessorGroup*        pg )
{
  // reduction variables for each level can be computed by several tasks (once per patch)
  // return the list of all tasks nearest the req

  Data key(nullptr, req, patch, matlIndex);
  int bestSortedOrder = -1;
  for (Data* p = data.lookup(&key); p != nullptr; p = data.nextMatch(&key, p)) {
    if (detaileddbg.active()) {
      detaileddbg << pg->myrank() << "          Examining comp from: " << p->comp->m_task->getName() << ", order="
                  << p->comp->m_task->getSortedOrder() << " (" << req->m_task->getName() << " order: " << req->m_task->getSortedOrder()
                  << "\n";
    }

    if (p->comp->m_task->getSortedOrder() < req->m_task->getSortedOrder() && p->comp->m_task->getSortedOrder() >= bestSortedOrder) {
      if (p->comp->m_task->getSortedOrder() > bestSortedOrder) {
        creators.clear();
        bestSortedOrder = p->comp->m_task->getSortedOrder();
        detaileddbg << pg->myrank() << "          New Best Sorted Order: " << bestSortedOrder << "!\n";
      }
      if (detaileddbg.active()) {
        detaileddbg << pg->myrank() << "          Adding comp from: " << p->comp->m_task->getName() << ", order="
                    << p->comp->m_task->getSortedOrder() << "\n";
      }
      creators.push_back(p->task);
    }
  }
  return creators.size() > 0;
}

//______________________________________________________________________
//
void
TaskGraph::createDetailedDependencies()
{
  // Collect all of the computes
  CompTable ct;
  for (int i = 0; i < dts_->numTasks(); i++) {
    DetailedTask* task = dts_->getTask(i);

    if (detaileddbg.active()) {
      detaileddbg << d_myworld->myrank() << " createDetailedDependencies (collect comps) for:\n";
      task->task->displayAll(detaileddbg);
    }

    remembercomps(task, task->task->getComputes(), ct);
    remembercomps(task, task->task->getModifies(), ct);
  }

  // Assign task phase number based on the reduction tasks so a mixed thread/mpi
  // scheduler won't have out of order reduction problems.
  int currphase = 0;
  int currcomm = 0;
  for (int i = 0; i < dts_->numTasks(); i++) {
    DetailedTask* task = dts_->getTask(i);
    task->task->m_phase = currphase;
    if (tgphasedbg.active()) {
      tgphasedbg << "Rank-" << d_myworld->myrank() << " Task: " << *task << " phase: " << currphase << "\n";
    }
    if (task->task->getType() == Task::Reduction) {
      task->task->m_comm = currcomm;
      currcomm++;
      currphase++;
    }
    else if (task->task->usesMPI()) {
      currphase++;
    }
  }
  d_myworld->setgComm(currcomm);
  d_numtaskphases = currphase + 1;

  // Go through the modifies/requires and create data dependencies as appropriate
  for (int i = 0; i < dts_->numTasks(); i++) {
    DetailedTask* task = dts_->getTask(i);

    if (detaileddbg.active() && (task->task->getRequires() != nullptr)) {
      detaileddbg << d_myworld->myrank() << " Looking at requires of detailed task: " << *task << "\n";
    }

    createDetailedDependencies(task, task->task->getRequires(), ct, false);

    if (detaileddbg.active() && (task->task->getModifies() != nullptr)) {
      detaileddbg << d_myworld->myrank() << " Looking at modifies of detailed task: " << *task << "\n";
    }

    createDetailedDependencies(task, task->task->getModifies(), ct, true);
  }

  if (detaileddbg.active()) {
    detaileddbg << d_myworld->myrank() << " Done creating detailed tasks\n";
  }
}

//______________________________________________________________________
//
void
TaskGraph::remembercomps( DetailedTask*     task,
                          Task::Dependency* comp,
                          CompTable&        ct )
{
  //calling getPatchesUnderDomain can get expensive on large processors.  Thus we 
  //cache results and use them on the next call.  This works well because comps
  //are added in order and they share the same patches under the domain
  const PatchSubset * cached_task_patches = nullptr;
  const PatchSubset * cached_comp_patches = nullptr;
  constHandle<PatchSubset> cached_patches;

  for (; comp != nullptr; comp = comp->m_next) {
    if (comp->m_var->typeDescription()->isReductionVariable()) {
      //if(task->getTask()->getType() == Task::Reduction || comp->deptype == Task::Modifies) {
      // this is either the task computing the var, modifying it, or the reduction itself
      ct.remembercomp(task, comp, 0, comp->m_matls, d_myworld);
    }
    else {
      // Normal tasks
      constHandle<PatchSubset> patches;

      //if the patch pointer on both the dep and the task have not changed then use the 
      //cached result
      if (task->patches == cached_task_patches && comp->m_patches == cached_comp_patches) {
        patches = cached_patches;
      }
      else {
        //compute the intersection 
        patches = comp->getPatchesUnderDomain(task->patches);
        //cache the result for the next iteration
        cached_patches = patches;
        cached_task_patches = task->patches;
        cached_comp_patches = comp->m_patches;
      }
      constHandle<MaterialSubset> matls = comp->getMaterialsUnderDomain(task->matls);
      if (!patches->empty() && !matls->empty()) {
        ct.remembercomp(task, comp, patches.get_rep(), matls.get_rep(), d_myworld);
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
  for (unsigned i = 0; i < d_tasks.size(); i++) {
    d_tasks[i]->setMapping(dwmap);

    // for the Int timesteps, we have tasks on multiple levels.  
    // we need to adjust based on which level they are on, but first 
    // we need to find the coarsest level.  The NewDW is relative to the coarsest
    // level executing in this taskgraph.
    if (type_ == Scheduler::IntermediateTaskGraph && (d_tasks[i]->getType() != Task::Output && d_tasks[i]->getType() != Task::OncePerProc)) {
      if (d_tasks[i]->getType() == Task::OncePerProc || d_tasks[i]->getType() == Task::Output) {
        levelmin = 0;
        continue;
      }

      const PatchSet* ps = d_tasks[i]->getPatchSet();
      if (!ps) {
        continue;
      }
      const Level* l = getLevel(ps);
      levelmin = Min(levelmin, l->getIndex());
    }
  }
  if (detaileddbg.active()) {
    detaileddbg << d_myworld->myrank() << " Basic mapping " << "Old " << dwmap[Task::OldDW] << " New " << dwmap[Task::NewDW]
                << " CO " << dwmap[Task::CoarseOldDW] << " CN " << dwmap[Task::CoarseNewDW] << " levelmin " << levelmin << std::endl;
  }

  if (type_ == Scheduler::IntermediateTaskGraph) {
    // fix the CoarseNewDW for finer levels.  The CoarseOld will only matter
    // on the level it was originally mapped, so leave it as it is
    dwmap[Task::CoarseNewDW] = dwmap[Task::NewDW];
    for (unsigned i = 0; i < d_tasks.size(); i++) {
      if (d_tasks[i]->getType() != Task::Output && d_tasks[i]->getType() != Task::OncePerProc) {
        const PatchSet* ps = d_tasks[i]->getPatchSet();
        if (!ps) {
          continue;
        }
        if (getLevel(ps)->getIndex() > levelmin) {
          d_tasks[i]->setMapping(dwmap);
          if (detaileddbg.active()) {
            detaileddbg << d_tasks[i]->getName() << " mapping " << "Old " << dwmap[Task::OldDW] << " New " << dwmap[Task::NewDW]
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
TaskGraph::createDetailedDependencies( DetailedTask*     task,
                                       Task::Dependency* req,
                                       CompTable&        ct,
                                       bool              modifies )
{
  int me = d_myworld->myrank();

  for( ; req != nullptr; req = req->m_next) {
    
    //if(req->var->typeDescription()->isReductionVariable())
    //  continue;

    if(sc->isOldDW(req->mapDataWarehouse()) && !sc->isNewDW(req->mapDataWarehouse()+1)) {
      continue;
    }
    
    if(detaileddbg.active()) {
      detaileddbg << d_myworld->myrank() << "  req: " << *req << "\n";
    }

    constHandle<PatchSubset> patches = req->getPatchesUnderDomain(task->patches);
    if (req->m_var->typeDescription()->isReductionVariable() && sc->isNewDW(req->mapDataWarehouse())) {
      // make sure newdw reduction variable requires link up to the reduction tasks.
      patches = nullptr;
    }
    constHandle<MaterialSubset> matls = req->getMaterialsUnderDomain(task->matls);

    // this section is just to find the low and the high of the patch that will use the other
    // level's data.  Otherwise, we have to use the entire set of patches (and ghost patches if 
    // applicable) that lay above/beneath this patch.

    const Patch* origPatch = nullptr;
    IntVector otherLevelLow, otherLevelHigh;
    if (req->m_patches_dom == Task::CoarseLevel || req->m_patches_dom == Task::FineLevel) {
      // the requires should have been done with Task::CoarseLevel or FineLevel, with null patches
      // and the task->patches should be size one (so we don't have to worry about overlapping regions)
      origPatch = task->patches->get(0);
      ASSERT(req->m_patches == nullptr);
      ASSERT(task->patches->size() == 1);
      ASSERT(req->m_level_offset > 0);
      const Level* origLevel = origPatch->getLevel();
      if (req->m_patches_dom == Task::CoarseLevel) {
        // change the ghost cells to reflect coarse level
        LevelP nextLevel = origPatch->getLevelP();
        int levelOffset = req->m_level_offset;
        IntVector ratio = origPatch->getLevel()->getRefinementRatio();
        while (--levelOffset) {
          nextLevel = nextLevel->getCoarserLevel();
          ratio = ratio * nextLevel->getRefinementRatio();
        }
        int ngc = req->m_num_ghost_cells * Max(Max(ratio.x(), ratio.y()), ratio.z());
        IntVector ghost(ngc, ngc, ngc);

        // manually set it, can't use computeVariableExtents since there might not be
        // a neighbor fine patch, and it would throw it off.  
        otherLevelLow = origPatch->getExtraCellLowIndex() - ghost;
        otherLevelHigh = origPatch->getExtraCellHighIndex() + ghost;

        otherLevelLow = origLevel->mapCellToCoarser(otherLevelLow, req->m_level_offset);
        otherLevelHigh = origLevel->mapCellToCoarser(otherLevelHigh, req->m_level_offset) + ratio - IntVector(1, 1, 1);
      }
      else {
        origPatch->computeVariableExtents(req->m_var->typeDescription()->getType(), req->m_var->getBoundaryLayer(), req->m_gtype,
                                          req->m_num_ghost_cells, otherLevelLow, otherLevelHigh);

        otherLevelLow = origLevel->mapCellToFiner(otherLevelLow);
        otherLevelHigh = origLevel->mapCellToFiner(otherLevelHigh);
      }
    }

    if (patches && !patches->empty() && matls && !matls->empty()) {
      if (req->m_var->typeDescription()->isReductionVariable()) {
        continue;
      }
      for (int i = 0; i < patches->size(); i++) {
        const Patch* patch = patches->get(i);

        //only allocate once
        static Patch::selectType neighbors;
        neighbors.resize(0);

        IntVector low, high;

        Patch::VariableBasis basis = Patch::translateTypeToBasis(req->m_var->typeDescription()->getType(), false);

        patch->computeVariableExtents(req->m_var->typeDescription()->getType(), req->m_var->getBoundaryLayer(), req->m_gtype,
                                      req->m_num_ghost_cells, low, high);

        if (req->m_patches_dom == Task::CoarseLevel || req->m_patches_dom == Task::FineLevel) {
          // make sure the bounds of the dep are limited to the original patch's (see above)
          // also limit to current patch, as patches already loops over all patches
          IntVector origlow = low, orighigh = high;
          if (req->m_patches_dom == Task::FineLevel) {
            // don't coarsen the extra cells
            low = patch->getLowIndex(basis);
            high = patch->getHighIndex(basis);
          }
          else {
            low = Max(low, otherLevelLow);
            high = Min(high, otherLevelHigh);
          }

          if (high.x() <= low.x() || high.y() <= low.y() || high.z() <= low.z()) {
            continue;
          }

          // don't need to selectPatches.  Just use the current patch, as we're
          // already looping over our required patches.
          neighbors.push_back(patch);
        }
        else {
          origPatch = patch;
          if (req->m_num_ghost_cells > 0) {
            patch->getLevel()->selectPatches(low, high, neighbors);
          }
          else {
            neighbors.push_back(patch);
          }
        }
        ASSERT(std::is_sorted(neighbors.begin(), neighbors.end(), Patch::Compare()));
        if (detaileddbg.active()) {
          detaileddbg << d_myworld->myrank() << "    Creating dependency on " << neighbors.size() << " neighbors\n";
          detaileddbg << d_myworld->myrank() << "      Low=" << low << ", high=" << high << ", var=" << req->m_var->getName()
                      << "\n";
        }

        for (int i = 0; i < neighbors.size(); i++) {
          const Patch* neighbor = neighbors[i];

          //if neighbor is not in my neighborhood just continue as its dependencies are not important to this processor
          if (!lb->inNeighborhood(neighbor->getRealPatch())) {
            continue;
          }

          static Patch::selectType fromNeighbors;
          fromNeighbors.resize(0);

          IntVector l = Max(neighbor->getExtraLowIndex(basis, req->m_var->getBoundaryLayer()), low);
          IntVector h = Min(neighbor->getExtraHighIndex(basis, req->m_var->getBoundaryLayer()), high);
          if (neighbor->isVirtual()) {
            l -= neighbor->getVirtualOffset();
            h -= neighbor->getVirtualOffset();
            neighbor = neighbor->getRealPatch();
          }
          if (req->m_patches_dom == Task::OtherGridDomain) {
            // this is when we are copying data between two grids (currently between timesteps)
            // the grid assigned to the old dw should be the old grid.
            // This should really only impact things required from the OldDW.
            LevelP fromLevel = sc->get_dw(0)->getGrid()->getLevel(patch->getLevel()->getIndex());
            fromLevel->selectPatches(Max(neighbor->getExtraLowIndex(basis, req->m_var->getBoundaryLayer()), l),
                                     Min(neighbor->getExtraHighIndex(basis, req->m_var->getBoundaryLayer()), h), fromNeighbors);
          }
          else {
            fromNeighbors.push_back(neighbor);
          }

          for (int j = 0; j < fromNeighbors.size(); j++) {
            const Patch* fromNeighbor = fromNeighbors[j];

            //only add the requirements both fromNeighbor is in my neighborhood
            if (!lb->inNeighborhood(fromNeighbor)) {
              continue;
            }

            IntVector from_l;
            IntVector from_h;

            if (req->m_patches_dom == Task::OtherGridDomain && fromNeighbor->getLevel()->getIndex() > 0) {
              // DON'T send extra cells (unless they're on the domain boundary)
              from_l = Max(fromNeighbor->getLowIndexWithDomainLayer(basis), l);
              from_h = Min(fromNeighbor->getHighIndexWithDomainLayer(basis), h);
            }
            else {
              // TODO - APH This intersection should not be needed, but let's clean this up if not - 06/15/16
              //from_l = Max(fromNeighbor->getExtraLowIndex(basis, req->var->getBoundaryLayer()), l);
              //from_h = Min(fromNeighbor->getExtraHighIndex(basis, req->var->getBoundaryLayer()), h);
              from_l = l;
              from_h = h;
              //verify in debug mode that the intersection is unneeded
              ASSERT(Max(fromNeighbor->getExtraLowIndex(basis, req->m_var->getBoundaryLayer()), l) == l);
              ASSERT(Min(fromNeighbor->getExtraHighIndex(basis, req->m_var->getBoundaryLayer()), h) == h);
            }
            if (patch->getLevel()->getIndex() > 0 && patch != fromNeighbor && req->m_patches_dom == Task::ThisLevel) {
              // cull annoying overlapping AMR patch dependencies
              patch->cullIntersection(basis, req->m_var->getBoundaryLayer(), fromNeighbor, from_l, from_h);
              if (from_l == from_h) {
                continue;
              }
            }

            for (int m = 0; m < matls->size(); m++) {
              int matl = matls->get(m);

              // creator is the task that performs the original compute.
              // If the require is for the OldDW, then it will be a send old
              // data task
              DetailedTask* creator  = nullptr;
              Task::Dependency* comp = nullptr;

              // look in old dw or in old TG.  Legal to modify across TG boundaries
              int proc = -1;
              if (sc->isOldDW(req->mapDataWarehouse())) {
                ASSERT(!modifies);
                proc = findVariableLocation(req, fromNeighbor, matl, 0);
                creator = dts_->getOldDWSendTask(proc);
                comp = nullptr;
              }
              else {
                if (!ct.findcomp(req, neighbor, matl, creator, comp, d_myworld)) {
                  if (type_ == Scheduler::IntermediateTaskGraph && req->m_look_in_old_tg) {
                    // same stuff as above - but do the check for findcomp first, as this is a "if you don't find it here, assign it
                    // from the old TG" dependency
                    proc = findVariableLocation(req, fromNeighbor, matl, 0);
                    creator = dts_->getOldDWSendTask(proc);
                    comp = nullptr;
                  }
                  else {

                    //if neither the patch or the neighbor are on this processor then the computing task doesn't exist so just continue
                    if (lb->getPatchwiseProcessorAssignment(patch) != d_myworld->myrank() &&
                        lb->getPatchwiseProcessorAssignment(neighbor) != d_myworld->myrank()) {
                      continue;
                    }

                    std::cout << "Failure finding " << *req << " for " << *task << std::endl;
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
                creator->findRequiringTasks(req->m_var, requireBeforeModifiedTasks);

                std::list<DetailedTask*>::iterator reqTaskIter;
                for (reqTaskIter = requireBeforeModifiedTasks.begin(); reqTaskIter != requireBeforeModifiedTasks.end(); ++reqTaskIter) {
                  DetailedTask* prevReqTask = *reqTaskIter;
                  if (prevReqTask == task) {
                    continue;
                  }
                  if (prevReqTask->task == task->task) {
                    if (!task->task->getHasSubScheduler()) {
                      std::ostringstream message;
                      message << " WARNING - task (" << task->getName()
                              << ") requires with Ghost cells *and* modifies and may not be correct" << std::endl;
                      static ProgressiveWarning warn(message.str(), 10);
                      warn.invoke();
                      if (detaileddbg.active()) {
                        detaileddbg << d_myworld->myrank() << " Task that requires with ghost cells and modifies\n";
                        detaileddbg << d_myworld->myrank() << " RGM: var: " << *req->m_var << " compute: " << *creator << " mod "
                                    << *task << " PRT " << *prevReqTask << " " << from_l << " " << from_h << "\n";
                      }
                    }
                  }
                  else {
                    // dep requires what is to be modified before it is to be
                    // modified so create a dependency between them so the
                    // modifying won't conflict with the previous require.
                    if (detaileddbg.active()) {
                      detaileddbg << d_myworld->myrank() << "       Requires to modifies dependency from " << prevReqTask->getName()
                                  << " to " << task->getName() << " (created by " << creator->getName() << ")\n";
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

                        req_patch->computeVariableExtents(req->m_var->typeDescription()->getType(), req->m_var->getBoundaryLayer(),
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
                    dts_->possiblyCreateDependency(prevReqTask, nullptr, 0, task, req, 0, matl, from_l, from_h, DetailedDep::Always);
                  }
                }
              }

              DetailedDep::CommCondition cond = DetailedDep::Always;
              if (proc != -1 && req->m_patches_dom != Task::OtherGridDomain) {
                // for OldDW tasks - see comment in class DetailedDep by CommCondition
                int subsequentProc = findVariableLocation(req, fromNeighbor, matl, 1);
                if (subsequentProc != proc) {
                  cond = DetailedDep::FirstIteration;  // change outer cond from always to first-only
                  DetailedTask* subsequentCreator = dts_->getOldDWSendTask(subsequentProc);
                  dts_->possiblyCreateDependency(subsequentCreator, comp, fromNeighbor,
                      task, req, patch,
                      matl, from_l, from_h, DetailedDep::SubsequentIterations);
                  detaileddbg << d_myworld->myrank() << "   Adding condition reqs for " << *req->m_var << " task : " << *creator
                              << "  to " << *task << "\n";
                }
              }
              dts_->possiblyCreateDependency(creator, comp, fromNeighbor,
                  task, req, patch,
                  matl, from_l, from_h, cond);
            }
          }
        }
      }
    }
    else if (!patches && matls && !matls->empty()) {
      // requiring reduction variables
      for (int m = 0; m < matls->size(); m++) {
        int matl = matls->get(m);
        static std::vector<DetailedTask*> creators;
        creators.resize(0);

        // TODO APH - figure this out (06/15/16)
#if 0
        if (type_ == Scheduler::IntermediateTaskGraph && m_req->m_look_in_old_tg && sc->isNewDW(m_req->mapDataWarehouse())) {
          continue;  // will we need to fix for mixed scheduling?
        }
#endif
        ct.findReductionComps(req, 0, matl, creators, d_myworld);
        // if the size is 0, that's fine.  It means that there are more procs than patches on this level,
        // so the reducer will pick a benign value that won't affect the reduction

        ASSERTRANGE(task->getAssignedResourceIndex(), 0, d_myworld->size());
        for (unsigned i = 0; i < creators.size(); i++) {
          DetailedTask* creator = creators[i];
          if (task->getAssignedResourceIndex() == creator->getAssignedResourceIndex() && task->getAssignedResourceIndex() == me) {
            task->addInternalDependency(creator, req->m_var);
            detaileddbg << d_myworld->myrank() << "   Created reduction dependency between " << *task << " and " << *creator
                        << "\n";
          }
        }
      }
    }
    else if (patches && patches->empty() && (req->m_patches_dom == Task::FineLevel || task->getTask()->getType() == Task::OncePerProc
            || task->getTask()->getType() == Task::Output || task->getTask()->getName() == "SchedulerCommon::copyDataToNewGrid")) {
      // this is a either coarsen task where there aren't any fine patches, or a PerProcessor task where
      // there aren't any patches on this processor.  Perfectly legal, so do nothing

      // another case is the copy-data-to-new-grid task, which will wither compute or modify to every patch
      // but not both.  So it will yell at you for the detailed task's patches not intersecting with the 
      // computes or modifies... (maybe there's a better way) - bryan
    }
    else {
      std::ostringstream desc;
      desc << "TaskGraph::createDetailedDependencies, task dependency not supported without patches and materials"
           << " \n Trying to require or modify " << *req << " in Task " << task->getTask()->getName() << "\n\n";
      if (task->matls) {
        desc << "task materials:" << *task->matls << "\n";
      }
      else {
        desc << "no task materials\n";
      }
      if (req->m_matls) {
        desc << "req materials: " << *req->m_matls << "\n";
      }
      else {
        desc << "no req materials\n";
        desc << "domain materials: " << *matls.get_rep() << "\n";
      }
      if (task->patches) {
        desc << "task patches:" << *task->patches << "\n";
      }
      else {
        desc << "no task patches\n";
      }
      if (req->m_patches) {
        desc << "req patches: " << *req->m_patches << "\n";
      }
      else {
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
TaskGraph::findVariableLocation(       Task::Dependency* req,
                                 const Patch*            patch,
                                       int               matl,
                                       int               iteration )
{
  // This needs to be improved, especially for re-distribution on
  // restart from checkpoint.
  int proc;
  if( ( req->m_task->mapDataWarehouse(Task::ParentNewDW) != -1 && req->m_whichdw != Task::ParentOldDW ) ||
      iteration > 0 ||
      ( req->m_look_in_old_tg && type_ == Scheduler::IntermediateTaskGraph ) ) {
    // Provide some accommodation for Dynamic load balancers and sub schedulers.  We need to
    // treat the requirement like a "old" dw req but it needs to be found on the current processor.
    // Same goes for successive executions of the same TG.
    proc = lb->getPatchwiseProcessorAssignment( patch );
  }
  else {
    proc = lb->getOldProcessorAssignment( patch );
  }
  return proc;
}

//______________________________________________________________________
//
int
TaskGraph::getNumTasks() const
{
  return static_cast<int>(d_tasks.size());
}

//______________________________________________________________________
//
Task*
TaskGraph::getTask( int idx )
{
  return d_tasks[idx];
}

//______________________________________________________________________
//
void
TaskGraph::makeVarLabelMaterialMap( Scheduler::VarLabelMaterialMap* result )
{
  for (int i = 0; i < (int)d_tasks.size(); i++) {
    Task* task = d_tasks[i];
    for (Task::Dependency* comp = task->getComputes(); comp != nullptr; comp = comp->m_next) {
      // assume all patches will compute the same labels on the same materials
      const VarLabel* label = comp->m_var;
      std::list<int>& matls = (*result)[label->getName()];
      const MaterialSubset* msubset = comp->m_matls;
      if (msubset) {
        for (int mm = 0; mm < msubset->size(); mm++) {
          matls.push_back(msubset->get(mm));
        }
      }
      else if (label->typeDescription()->getType() == TypeDescription::ReductionVariable) {
        // Default to material -1 (global)
        matls.push_back(-1);
      }
      else {
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
