#include <Packages/Uintah/CCA/Components/Schedulers/TaskGraph.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabelLevelDW.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Util/ProgressiveWarning.h>

#include <Core/Containers/FastHashTable.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>

#include <sci_defs/config_defs.h>
#include <sci_algorithm.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <unistd.h>

using namespace Uintah;

using namespace SCIRun;
using namespace std;

static DebugStream dbg("TaskGraph", false);
static DebugStream compdbg("FindComp", false);

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern Mutex cerrLock;
extern DebugStream mixedDebug;

#define DAV_DEBUG 0


TaskGraph::TaskGraph(SchedulerCommon* sc, const ProcessorGroup* pg)
  : sc(sc), d_myworld(pg)
{
}

TaskGraph::~TaskGraph()
{
  initialize(); // Frees all of the memory...
}

void
TaskGraph::initialize()
{
  for(vector<Task*>::iterator iter=d_tasks.begin(); iter != d_tasks.end(); iter++ )
    delete *iter;

  for(vector<Task::Edge*>::iterator iter = edges.begin(); iter != edges.end(); iter++)
    delete *iter;

  d_tasks.clear();

  d_initRequires.clear();
  d_initRequiredVars.clear();
  edges.clear();
}

bool
TaskGraph::overlaps(Task::Dependency* comp, Task::Dependency* req) const
{
  constHandle<PatchSubset> saveHandle2;
  const PatchSubset* ps1 = comp->patches;
  if(!ps1){
    if(!comp->task->getPatchSet())
      return false;
    ps1 = comp->task->getPatchSet()->getUnion();
    if(comp->patches_dom == Task::CoarseLevel
       || comp->patches_dom == Task::FineLevel) {
      SCI_THROW(InternalError("Should not compute onto another level!", __FILE__, __LINE__));
      // This may not be a big deal if it were needed, but I didn't
      // think that it should be allowed - Steve
      // saveHandle1 = comp->getPatchesUnderDomain(ps1);
      // ps1 = saveHandle1.get_rep();
    }
  }

  const PatchSubset* ps2 = req->patches;
  if(!ps2){
    if(!req->task->getPatchSet())
      return false;
    ps2 = req->task->getPatchSet()->getUnion();
    if(req->patches_dom == Task::CoarseLevel
       || req->patches_dom == Task::FineLevel){
      saveHandle2 = req->getPatchesUnderDomain(ps2);
      ps2 = saveHandle2.get_rep();
    }
  }

  if(!PatchSubset::overlaps(ps1, ps2)) // && !(ps1->size() == 0 && (!req->patches || ps2->size() == 0) && comp->task->getType() == Task::OncePerProc))
    return false;

  const MaterialSubset* ms1 = comp->matls;
  if(!ms1){
    if(!comp->task->getMaterialSet())
      return false;
    ms1 = comp->task->getMaterialSet()->getUnion();
  }
  const MaterialSubset* ms2 = req->matls;
  if(!ms2){
    if(!req->task->getMaterialSet())
      return false;
    ms2 = req->task->getMaterialSet()->getUnion();
  }
  if(!MaterialSubset::overlaps(ms1, ms2))
    return false;
  return true;
}

// setupTaskConnections also adds Reduction Tasks to the graph...
void
TaskGraph::setupTaskConnections(GraphSortInfoMap& sortinfo)
{
  vector<Task*>::iterator iter;
  // Initialize variables on the tasks
  for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
    sortinfo[*iter] = GraphSortInfo();
  }    
  if (edges.size() > 0) {
    return; // already been done
  }

  // Look for all of the reduction variables - we must treat those
  // special.  Create a fake task that performs the reduction
  // While we are at it, ensure that we aren't producing anything
  // into an "old" data warehouse
  typedef map<VarLabelLevelDW, Task*> ReductionTasksMap;
  ReductionTasksMap reductionTasks;
  for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
    Task* task = *iter;
    if (task->isReductionTask())
      continue; // already a reduction task so skip it

    for(Task::Dependency* comp = task->getComputes();
	comp != 0; comp=comp->next){
      if(sc->isOldDW(comp->mapDataWarehouse())){
	if (dbg.active())
          dbg << d_myworld->myrank() << " which = " << comp->whichdw << ", mapped to " << comp->mapDataWarehouse() << '\n';
	SCI_THROW(InternalError("Variable produced in old datawarehouse: "
				+comp->var->getName(), __FILE__, __LINE__));
      } else if(comp->var->typeDescription()->isReductionVariable()){
	ASSERT(comp->patches == 0);
	// Look up this variable in the reductionTasks map
	int dw = comp->mapDataWarehouse();
	VarLabelLevelDW key(comp->var, comp->reductionLevel, dw);
	const MaterialSet* ms = task->getMaterialSet();
	const Level* level = comp->reductionLevel;

	ReductionTasksMap::iterator it=reductionTasks.find(key);
	if(it == reductionTasks.end()){
	  // No reduction task yet, create one
	  int levelidx = comp->reductionLevel?comp->reductionLevel->getIndex():-1;
          if (dbg.active())
            dbg << d_myworld->myrank() << " creating Reduction task for variable: " 
                << comp->var->getName() << " on level " << levelidx 
                << ", DW " << dw << '\n';
	  ostringstream taskname;
	  taskname << "Reduction: " << comp->var->getName() 
		   << ", level " << levelidx << ", dw " << dw;
	  Task* newtask = scinew Task(taskname.str(), Task::Reduction);

          sortinfo[newtask] = GraphSortInfo();

	  int dwmap[Task::TotalDWs];
	  for(int i=0;i<Task::TotalDWs;i++)
	    dwmap[i]=Task::InvalidDW;
	  dwmap[Task::OldDW] = Task::NoDW;
	  dwmap[Task::NewDW] = dw;
	  newtask->setMapping(dwmap);

	  // compute for all patches but some set of materials
	  // (maybe global material, but not necessarily)
	  if (comp->matls != 0)
	    newtask->computes(comp->var, level, comp->matls, Task::OutOfDomain);
	  else {
	    for(int m=0;m<ms->size();m++)
	      newtask->computes(comp->var, level, ms->getSubset(m), Task::OutOfDomain);
	  }
	  reductionTasks[key]=newtask;
	  it = reductionTasks.find(key);
	}

	// Make the reduction task require its variable for the appropriate
	// patch and materials subset(s).
	Task* task = it->second;
	if(comp->matls){
	  task->requires(Task::NewDW, comp->var, level, comp->matls, Task::OutOfDomain);
	} else {
	  for(int m=0;m<ms->size();m++){
	    const MaterialSubset* mss = ms->getSubset(m);
	    task->requires(Task::NewDW, comp->var, level, mss);
	  }
	}
      }
    }
  }

  // Add the new reduction tasks to the list of tasks
  for(ReductionTasksMap::iterator it = reductionTasks.begin();
      it != reductionTasks.end(); it++){
    addTask(it->second, 0, 0);
  }

  // Gather the comps for the tasks into a map
  CompMap comps;
  for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
    Task* task = *iter;
    if (dbg.active())
      dbg << d_myworld->myrank() << " Gathering comps from task: " << *task << '\n';
    for(Task::Dependency* comp = task->getComputes();
	comp != 0; comp=comp->next){
      comps.insert(make_pair(comp->var, comp));
      if (dbg.active())
        dbg << d_myworld->myrank() << "   Added comp for: " << *comp << '\n';
    }
  }

  // Connect the tasks where the requires/modifies match a comp.
  // Also, updates the comp map with each modify and doing this in task order
  // so future modifies/requires find the modified var.  Also do a type check
  for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
    Task* task = *iter;
    if(dbg.active())
      dbg << d_myworld->myrank() << "   Looking at dependencies for task: " << *task << '\n';
    addDependencyEdges(task, sortinfo, task->getRequires(), comps, false);
    addDependencyEdges(task, sortinfo, task->getModifies(), comps, true);
    // Used here just to warn if a modifies comes before its computes
    // in the order that tasks were added to the graph.
    sortinfo.find(task)->second.visited = true;
  }

  // Initialize variables on the tasks
  GraphSortInfoMap::iterator sort_iter;
  for( sort_iter=sortinfo.begin(); sort_iter != sortinfo.end(); sort_iter++ ) {
    sort_iter->second.visited=false;
    sort_iter->second.sorted=false;
  }
} // end setupTaskConnections()

void TaskGraph::addDependencyEdges(Task* task, GraphSortInfoMap& sortinfo,
                                   Task::Dependency* req,
                                   CompMap& comps, bool modifies)
{
  for(; req != 0; req=req->next){
    if (dbg.active())
      dbg << d_myworld->myrank() << "     Checking edge for req: " << *req << ", task: " << *req->task << ", domain: " << req->patches_dom << '\n';
    if(sc->isNewDW(req->mapDataWarehouse())){
      // If DW is finalized, we assume that we already have it,
      // or that we will get it sent to us.  Otherwise, we set
      // up an edge to connect this req to a comp
      pair<CompMap::iterator,CompMap::iterator> iters 
	= comps.equal_range(req->var);
      int count=0;
      for(CompMap::iterator compiter = iters.first;
	  compiter != iters.second; ++compiter){

	if(req->var->typeDescription() != compiter->first->typeDescription())
	  SCI_THROW(TypeMismatchException("Type mismatch for variable: "+req->var->getName(), __FILE__, __LINE__));

        // determine if we need to add a dependency edge
	bool add=false;
	if (dbg.active())
          dbg << d_myworld->myrank() << "       Checking edge from comp: " << *compiter->second << ", task: " << *compiter->second->task << ", domain: " << compiter->second->patches_dom << '\n';
	if(req->mapDataWarehouse() == compiter->second->mapDataWarehouse()){
	  if(task->isReductionTask()){
            // Make sure that we get the comp from the reduction task
	    // Add if the level matches, but do not create a dependency on self
	    if(compiter->second->reductionLevel == req->reductionLevel
	       && compiter->second->task != task)
	      add=true;
	  } else if(req->var->typeDescription()->isReductionVariable()){
	    if(compiter->second->task->isReductionTask())
	      add=true;
          } else if(overlaps(compiter->second, req))
	    add=true;
	}
	if(!add)
          if (dbg.active())
            dbg << d_myworld->myrank() << "       did NOT create dependency\n";
	if(add){
	  Task::Dependency* comp = compiter->second;
	  
	  if (modifies) {
	    // Add dependency edges to each task that requires the data
	    // before it is modified.
	    for (Task::Edge* otherEdge = comp->req_head; otherEdge != 0;
		 otherEdge = otherEdge->reqNext) {
	      Task::Dependency* priorReq =
		const_cast<Task::Dependency*>(otherEdge->req);
	      if (priorReq != req) {
		ASSERT(priorReq->var->equals(req->var));
		if (priorReq->task != task) {		
		  Task::Edge* edge = scinew Task::Edge(priorReq, req);
		  edges.push_back(edge);
		  req->addComp(edge);
		  priorReq->addReq(edge);
		  if(dbg.active()){
		    dbg << d_myworld->myrank() << " Creating edge from task: " << *priorReq->task << " to task: " << *req->task << '\n';
		    dbg << d_myworld->myrank() << " Prior Req=" << *priorReq << '\n';
		    dbg << d_myworld->myrank() << " Modify=" << *req << '\n';
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
	  
	  if (!sortinfo.find(edge->comp->task)->second.visited &&
	      !edge->comp->task->isReductionTask()) {
	    cout << "\nWARNING: A task, '" << task->getName() << "', that ";
	    if (modifies)
	      cout << "modifies '";
	    else
	      cout << "requires '";
	    cout << req->var->getName() << "' was added before computing task";
	    cout << ", '" << edge->comp->task->getName() << "'\n";
	    cout << "  Required/modified by: " << *task << '\n';
	    cout << "  req: " << *req << '\n';
	    cout << "  Computed by: " << *edge->comp->task << '\n';
	    cout << "  comp: " << *comp << '\n';
	    cout << "\n";
	  }
	  count++;
	  if(dbg.active()){
	    dbg << d_myworld->myrank() << "       Creating edge from task: " << *comp->task << " to task: " << *task << '\n';
	    dbg << d_myworld->myrank() << "         Req=" << *req << '\n';
	    dbg << d_myworld->myrank() << "         Comp=" << *comp << '\n';
	  }
	}
      }

      // if we cannot find the required variable, throw an exception
      if(count == 0 && (!req->matls || req->matls->size() > 0) 
	 && (!req->patches || req->patches->size() > 0)){
	if(req->patches){
	  cout << req->patches->size() << " Patches: ";
	  for(int i=0;i<req->patches->size();i++)
	    cout << req->patches->get(i)->getID() << " ";
	  cout << '\n';
	} else if(req->reductionLevel) {
	  cout << "On level " << req->reductionLevel->getIndex() << '\n';
	} else if(task->getPatchSet()){
	  cout << "Patches from task: ";
	  const PatchSet* patches = task->getPatchSet();
	  for(int i=0;i<patches->size();i++){
	    const PatchSubset* pat=patches->getSubset(i);
	    for(int i=0;i<pat->size();i++)
	      cout << pat->get(i)->getID() << " ";
	    cout << " ";
	  }
	  cout << '\n';
	} else {
	  cout << "On global level\n";
	}
	if(req->matls){
	  cout << req->matls->size() << " Matls: ";
	  for(int i=0;i<req->matls->size();i++)
	    cout << req->matls->get(i) << " ";
	  cout << '\n';
	} else if(task->getMaterialSet()){
	  cout << "Matls from task: ";
	  const MaterialSet* matls = task->getMaterialSet();
	  for(int i=0;i<matls->size();i++){
	    const MaterialSubset* mat = matls->getSubset(i);
	    for(int i=0;i<mat->size();i++)
	      cout << mat->get(i) << " ";
	    cout << " ";
	  }
	  cout << '\n';
	} else {
	  cout << "No matls\n";
	}
	SCI_THROW(InternalError("Scheduler could not find specific production for variable: "+req->var->getName()+", required for task: "+task->getName(), __FILE__, __LINE__));
      }
      
      if (modifies) {
	// not just requires, but modifies, so the comps map must be
	// updated so future modifies or requires will link to this one.
	comps.insert(make_pair(req->var, req));
	if (dbg.active())
          dbg << d_myworld->myrank() << " Added modified comp for: " << *req << '\n';
      }
    }
  }
}

void
TaskGraph::processTask(Task* task, vector<Task*>& sortedTasks,
                       GraphSortInfoMap& sortinfo) const
{
  if(dbg.active())
    dbg << d_myworld->myrank() << " Looking at task: " << task->getName() << '\n';

  GraphSortInfo& gsi = sortinfo.find(task)->second;
  if(gsi.visited){
    ostringstream error;
    error << "Cycle detected in task graph: already did\n\t"
	  << task->getName();
    error << "\n";
    SCI_THROW(InternalError(error.str(), __FILE__, __LINE__));
  }

  gsi.visited = true;
   
  processDependencies(task, task->getRequires(), sortedTasks, sortinfo);
  processDependencies(task, task->getModifies(), sortedTasks, sortinfo);

  // All prerequisites are done - add this task to the list
  sortedTasks.push_back(task);
  gsi.sorted=true;
  if(dbg.active())
    dbg << d_myworld->myrank() << " Sorted task: " << task->getName() << '\n';
} // end processTask()


void TaskGraph::processDependencies(Task* task, Task::Dependency* req,
				    vector<Task*>& sortedTasks,
                                    GraphSortInfoMap& sortinfo) const

{
  for(; req != 0; req=req->next){
    if (dbg.active())
      dbg << d_myworld->myrank() << " processDependencies for req: " << *req << '\n';
    if(sc->isNewDW(req->mapDataWarehouse())){
      Task::Edge* edge = req->comp_head;
      for (;edge != 0; edge = edge->compNext){
	Task* vtask = edge->comp->task;
        GraphSortInfo& gsi = sortinfo.find(vtask)->second;
	if(!gsi.sorted){
	  if(gsi.visited){
	    ostringstream error;
	    error << "Cycle detected in task graph: trying to do\n\t"
		  << task->getName();
	    error << "\nbut already did:\n\t"
		  << vtask->getName();
	    error << ",\nwhile looking for variable: \n\t" 
		  << req->var->getName();
	    error << "\n";
	    SCI_THROW(InternalError(error.str(), __FILE__, __LINE__));
	  }
	  processTask(vtask, sortedTasks, sortinfo);
	}
      }
    }
  }
}

void
TaskGraph::nullSort( vector<Task*>& tasks )
{
  GraphSortInfoMap sortinfo;
  // setupTaskConnections also creates the reduction tasks...
  setupTaskConnections(sortinfo);

  vector<Task*>::iterator iter;

  // No longer going to sort them... let the MixedScheduler take care
  // of calling the tasks when all dependencies are satisfied.
  // Sorting the tasks causes problem because now tasks (actually task
  // groups) run in different orders on different MPI processes.

  for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
    tasks.push_back( *iter );
  }
}

void
TaskGraph::topologicalSort(vector<Task*>& sortedTasks)
{
  GraphSortInfoMap sortinfo;

  setupTaskConnections(sortinfo);

  for(vector<Task*>::iterator iter=d_tasks.begin();
      iter != d_tasks.end(); iter++ ) {
    Task* task = *iter;
    if(!sortinfo.find(task)->second.sorted){
      processTask(task, sortedTasks, sortinfo);
    }
  }
  int n=0;
  for(vector<Task*>::iterator iter = sortedTasks.begin();
      iter != sortedTasks.end(); iter++) {
    (*iter)->setSortedOrder(n++);
  }
}

void
TaskGraph::addTask(Task* task, const PatchSet* patchset,
		   const MaterialSet* matlset)
{
  task->setSets(patchset, matlset);
  if((patchset && patchset->totalsize() == 0)
     || (matlset && matlset->totalsize() == 0)){
    delete task;
    if(dbg.active())
      dbg << d_myworld->myrank() << " Killing empty task: " << *task << "\n";
  } else {
    d_tasks.push_back(task);
    if(dbg.active()) {
      dbg << d_myworld->myrank() << " Adding task:\n";
      task->displayAll( dbg );
    }
  }

  //  maintain d_initRequires and d_initRequiredVars
  for(Task::Dependency* req = task->getRequires(); req != 0; req=req->next){
    if(sc->isOldDW(req->mapDataWarehouse())) {
      d_initRequires.push_back(req);
      d_initRequiredVars.insert(req->var);
    }
  }
}

void
TaskGraph::createDetailedTask(DetailedTasks* tasks, Task* task,
			      const PatchSubset* patches,
			      const MaterialSubset* matls)
{
  DetailedTask* dt = scinew DetailedTask(task, patches, matls, tasks);

  if (task->getType() == Task::Reduction) {
    Task::Dependency* req = task->getRequires();
    // reduction tasks should have at least 1 requires (and they
    // should all be for the same variable).
    ASSERT(req != 0); 
    d_reductionTasks[req->var] = dt;
  }
		     
  tasks->add(dt);
}

DetailedTasks*
TaskGraph::createDetailedTasks( LoadBalancer* lb, bool useInternalDeps )
{
  vector<Task*> sorted_tasks;
  topologicalSort(sorted_tasks);

  d_reductionTasks.clear();

  // WARNING - this just grabs ONE grid.
  // more careful.
  GridP grid;
  for(int i=0;grid == 0 && i<(int)sorted_tasks.size();i++){
    Task* task = sorted_tasks[i];
    const PatchSet* ps = task->getPatchSet();
    if(ps && task->getMaterialSet()){
      for(int p=0;grid == 0 && p<ps->size();p++){
	const PatchSubset* pss = ps->getSubset(p);
	for(int s=0;grid == 0 && s<pss->size();s++){
	  const Patch* patch = pss->get(s);
	  grid = patch->getLevel()->getGrid();
	}
      }
    }
  }

  ASSERT(grid != 0);
  lb->createNeighborhood(grid);

  DetailedTasks* dt = scinew DetailedTasks(sc, d_myworld, this, useInternalDeps );
  for(int i=0;i<(int)sorted_tasks.size();i++){
    Task* task = sorted_tasks[i];
    const PatchSet* ps = task->getPatchSet();
    const MaterialSet* ms = task->getMaterialSet();
    if(ps && ms){
      for(int p=0;p<ps->size();p++){
	const PatchSubset* pss = ps->getSubset(p);
	for(int m=0;m<ms->size();m++){
	  const MaterialSubset* mss = ms->getSubset(m);
	  if(lb->inNeighborhood(pss, mss) || 
             (task->getType() == Task::Output 
              && d_myworld->myrank() % lb->getNthProc() == 0) ||
             task->getType() == Task::OncePerProc)
                                                                

	    createDetailedTask(dt, task, pss, mss);
	}
      }
    } else if(!ps && !ms){
      createDetailedTask(dt, task, 0, 0);
    } else if(!ps){
      SCI_THROW(InternalError("Task has MaterialSet, but no PatchSet", __FILE__, __LINE__));
    } else {
      SCI_THROW(InternalError("Task has PatchSet, but no MaterialSet", __FILE__, __LINE__));
    }
  }
  return dt;
}

namespace Uintah {
  
class CompTable {
  struct Data {
    Data* next;
    DetailedTask* task;
    Task::Dependency* comp;
    const Patch* patch;
    int matl;
    unsigned int hash;

    unsigned int string_hash(const char* p) {
      unsigned int sum=0;
      while(*p)
	sum = sum*7 + (unsigned char)*p++;
      return sum;
    }

    Data(DetailedTask* task, Task::Dependency* comp,
	 const Patch* patch, int matl)
      : task(task), comp(comp), patch(patch), matl(matl)
    {
      hash=(unsigned int)(((unsigned int)comp->mapDataWarehouse()<<3)
			  ^(string_hash(comp->var->getName().c_str()))
			  ^matl);
      if(patch)
	hash ^= (unsigned int)(patch->getID()<<4);
    }
    ~Data()
    {
    }
    bool operator==(const Data& c) {
      return matl == c.matl && patch == c.patch &&
	comp->reductionLevel == c.comp->reductionLevel &&
	comp->mapDataWarehouse() == c.comp->mapDataWarehouse() &&
	comp->var->equals(c.comp->var);
    }
  };
  FastHashTable<Data> data;
  void insert(Data* d);
 public:
  CompTable();
  ~CompTable();
  void remembercomp(DetailedTask* task, Task::Dependency* comp,
		    const PatchSubset* patches, const MaterialSubset* matls,
                    const ProcessorGroup* pg);
  bool findcomp(Task::Dependency* req, const Patch* patch, int matlIndex,
		DetailedTask*& dt, Task::Dependency*& comp,
                const ProcessorGroup* pg);
private:
  void remembercomp(Data* newData, const ProcessorGroup* pg);
};

}

CompTable::CompTable()
{
}

CompTable::~CompTable()
{
}

void CompTable::remembercomp(Data* newData, const ProcessorGroup* pg)
{
  if(dbg.active()){
    dbg << pg->myrank() << " remembercomp: " << *newData->comp << ", matl=" << newData->matl;
    if(newData->patch)
      dbg << ", patch=" << *newData->patch;
    dbg << '\n';
  }

  // can't have two computes for the same variable (need modifies)
  if(newData->comp->deptype != Task::Modifies){
    if(data.lookup(newData)){
      cout << "Multiple compute found:\n";
      cout << "matl: " << newData->matl << "\n";
      cout << "patch: " << *newData->patch << "\n";
      cout << *newData->comp << "\n";
      cout << *newData->task << "\n";
      cout << "It was originally computed by the following task(s):\n";
      for(Data* old = data.lookup(newData); old != 0; old = data.nextMatch(newData, old)){
        cout << *old->task << endl;
	old->comp->task->displayAll(cout);
      }
      SCI_THROW(InternalError("Multiple computes for variable: "+newData->comp->var->getName(), __FILE__, __LINE__));
    }
  }
  data.insert(newData);
}

void CompTable::remembercomp(DetailedTask* task, Task::Dependency* comp,
			     const PatchSubset* patches, 
                             const MaterialSubset* matls,
                             const ProcessorGroup* pg)
{
  if(patches && matls){
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      for(int m=0;m<matls->size();m++){
	int matl = matls->get(m);
	Data* newData = new Data(task, comp, patch, matl);
	remembercomp(newData, pg);
      }
    }
  } 
  else if (matls) {
    for(int m=0;m<matls->size();m++){
      int matl = matls->get(m);
      Data* newData = new Data(task, comp, 0, matl);      
      remembercomp(newData, pg);
    }
  }
  else if (patches) {
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      Data* newData = new Data(task, comp, patch, 0);
      remembercomp(newData, pg);
    }
  }
  else {
    Data* newData = new Data(task, comp, 0, 0);
    remembercomp(newData, pg);
  }
}

bool CompTable::findcomp(Task::Dependency* req, const Patch* patch,
			 int matlIndex, DetailedTask*& dt,
			 Task::Dependency*& comp, const ProcessorGroup *pg)
{
  if (compdbg.active())
    compdbg << pg->myrank() << "        Finding comp of req: " << *req << " for task: " << *req->task << "/" << '\n';
  Data key(0, req, patch, matlIndex);
  Data* result = 0;
  for(Data* p = data.lookup(&key); p != 0; p = data.nextMatch(&key, p)){
    if (compdbg.active())
      compdbg << pg->myrank() << "          Examining comp from: " << p->comp->task->getName() << ", order=" << p->comp->task->getSortedOrder() << '\n';
    ASSERT(!result || p->comp->task->getSortedOrder() != result->comp->task->getSortedOrder());
    if(p->comp->task->getSortedOrder() < req->task->getSortedOrder()){
      if(!result || p->comp->task->getSortedOrder() > result->comp->task->getSortedOrder()){
	if (compdbg.active())
          compdbg << pg->myrank() << "          New best is comp from: " << p->comp->task->getName() << ", order=" << p->comp->task->getSortedOrder() << '\n';
	result = p;
      }
    }
  }
  if(result){
    if (compdbg.active())
      compdbg << pg->myrank() << "          Found comp at: " << result->comp->task->getName() << ", order=" << result->comp->task->getSortedOrder() << '\n';
    dt=result->task;
    comp=result->comp;
    return true;
  } else {
    return false;
  }
}

void
TaskGraph::createDetailedDependencies(DetailedTasks* dt, LoadBalancer* lb)
{
  // Collect all of the comps
  CompTable ct;
  for(int i=0;i<dt->numTasks();i++){
    DetailedTask* task = dt->getTask(i);

    if( dbg.active() ) {
      dbg << d_myworld->myrank() << " createDetailedDependencies (collect comps) for:\n";
      task->task->displayAll( dbg );
    }

    remembercomps(task, task->task->getComputes(), ct);
    remembercomps(task, task->task->getModifies(), ct);
  }

  // Put internal links between the reduction tasks so a mixed thread/mpi
  // scheduler won't have out of order reduction problems.
  DetailedTask* lastReductionTask = 0;
  for(int i=0;i<dt->numTasks();i++){
    DetailedTask* task = dt->getTask(i);
    if (task->task->getType() == Task::Reduction) {
      if (lastReductionTask != 0)
	task->addInternalDependency(lastReductionTask,
				    task->task->getComputes()->var);
      lastReductionTask = task;
    }
  }

  // Go through the modifies/requires and 
  // create data dependencies as appropriate
  for(int i=0;i<dt->numTasks();i++){
    DetailedTask* task = dt->getTask(i);

    if(task->task->getType() == Task::Reduction) {
      // Reduction tasks were dealt with above (adding internal dependencies
      // to the tasks computing the reduction variables.
      continue; 
    }

    if(dbg.active() && (task->task->getRequires() != 0))
      dbg << d_myworld->myrank() << " Looking at requires of detailed task: " << *task << '\n';

    createDetailedDependencies(dt, lb, task, task->task->getRequires(),
                               ct, false);

    if(dbg.active() && (task->task->getModifies() != 0))
      dbg << d_myworld->myrank() << " Looking at modifies of detailed task: " << *task << '\n';

    createDetailedDependencies(dt, lb, task, task->task->getModifies(),
			       ct, true);
  }

  if(dbg.active())
    dbg << d_myworld->myrank() << " Done creating detailed tasks\n";
}

void TaskGraph::remembercomps(DetailedTask* task, Task::Dependency* comp,
			      CompTable& ct)
{
  int me = d_myworld->myrank();
  
  for(;comp != 0; comp = comp->next){
    if (comp->var->typeDescription()->isReductionVariable()){
      if(task->getTask()->getType() == Task::Reduction) {
	// Reduction task
	ct.remembercomp(task, comp, 0, comp->matls, d_myworld);
      } else {
	// create internal dependencies to reduction tasks from any task
	// computing the reduction
	DetailedTask* reductionTask = d_reductionTasks[comp->var];
	ASSERTRANGE(reductionTask->getAssignedResourceIndex(), 0, d_myworld->size());
	ASSERTRANGE(task->getAssignedResourceIndex(), 0, d_myworld->size());
	if (reductionTask->getAssignedResourceIndex() == 
	    task->getAssignedResourceIndex() &&
	    task->getAssignedResourceIndex() == me) {
	  // the tasks are on the same processor, so add an internal dependency
	  reductionTask->addInternalDependency(task, comp->var);
	}
      }
    } else {
      // Normal tasks
      constHandle<PatchSubset> patches =
	comp->getPatchesUnderDomain(task->patches);
      constHandle<MaterialSubset> matls =
	comp->getMaterialsUnderDomain(task->matls);
      if(!patches->empty() && !matls->empty()) {
        ct.remembercomp(task, comp, patches.get_rep(), matls.get_rep(),
                        d_myworld);
      }
    }
  }
}

void
TaskGraph::createDetailedDependencies(DetailedTasks* dt,
				      LoadBalancer* lb,
				      DetailedTask* task,
				      Task::Dependency* req, CompTable& ct,
				      bool modifies)
{
  int me = d_myworld->myrank();
  for( ; req != 0; req = req->next){
    if(dbg.active())
      dbg << d_myworld->myrank() << "  req: " << *req << '\n';

    if(sc->isOldDW(req->mapDataWarehouse()) && !sc->isNewDW(req->mapDataWarehouse()+1))
      continue;

    constHandle<PatchSubset> patches =
      req->getPatchesUnderDomain(task->patches);
    if (req->var->typeDescription()->isReductionVariable() &&
	sc->isNewDW(req->mapDataWarehouse())){
      // make sure newdw reduction variable requires link up to the
      // reduction tasks.
      patches = 0;
    }
    constHandle<MaterialSubset> matls =
      req->getMaterialsUnderDomain(task->matls);

    // this section is just to find the low and the high of the patch that will use the other
    // level's data.  Otherwise, we have to use the entire set of patches (and ghost patches if 
    // applicable) that lay above/beneath this patch.

    IntVector otherLevelLow, otherLevelHigh;
    if (req->patches_dom == Task::CoarseLevel || req->patches_dom == Task::FineLevel) {
      // the requires should have been done with Task::CoarseLevel or FineLevel, with null patches
      // and the task->patches should be size one (so we don't have to worry about overlapping regions)
      ASSERT(req->patches == NULL);
      ASSERT(task->patches->size() == 1);
      const Patch* origPatch = task->patches->get(0);
      const Level* origLevel = origPatch->getLevel();
      if (req->patches_dom == Task::CoarseLevel) {
        // change the ghost cells to reflect coarse level
        IntVector ratio = origPatch->getLevel()->getRefinementRatio();
        int ngc = req->numGhostCells * Max(Max(ratio.x(), ratio.y()), ratio.z());
        IntVector ghost(ngc,ngc,ngc);

        // manually set it, can't use computeVariableExtents since there might not be
        // a neighbor fine patch, and it would throw it off.  
        otherLevelLow = origPatch->getLowIndex() - ghost;
        otherLevelHigh = origPatch->getHighIndex() + ghost;

        otherLevelLow = origLevel->mapCellToCoarser(otherLevelLow);
        otherLevelHigh = origLevel->mapCellToCoarser(otherLevelHigh) + 
          ratio - IntVector(1,1,1);
      }
      else {
        origPatch->computeVariableExtents(req->var->typeDescription()->getType(),
                                          req->var->getBoundaryLayer(),
                                          req->gtype, req->numGhostCells,
                                          otherLevelLow, otherLevelHigh);

        otherLevelLow = origLevel->mapCellToFiner(otherLevelLow);
        otherLevelHigh = origLevel->mapCellToFiner(otherLevelHigh);
      }
    }

    if(patches && !patches->empty() && matls && !matls->empty()){
      for(int i=0;i<patches->size();i++){
	const Patch* patch = patches->get(i);
	Patch::selectType neighbors;
	IntVector low, high;

	Patch::VariableBasis basis = Patch::translateTypeToBasis(req->var->typeDescription()->getType(),
								 false);

	patch->computeVariableExtents(req->var->typeDescription()->getType(),
				      req->var->getBoundaryLayer(),
				      req->gtype, req->numGhostCells,
				      low, high);

        if (req->patches_dom == Task::CoarseLevel || req->patches_dom == Task::FineLevel) {
          // make sure the bounds of the dep are limited to the original patch's (see above)
          // also limit to current patch, as patches already loops over all patches
          IntVector origlow = low, orighigh = high;
          low = Max(low, otherLevelLow);
          if (req->patches_dom == Task::FineLevel) {
            // don't coarsen the extra cells
            low = patch->getInteriorLowIndex(basis);
            high = patch->getInteriorHighIndex(basis);
          }
          high = Min(high, otherLevelHigh);

          if (high.x() <= low.x() || high.y() <= low.y() || high.z() <= low.z())
            continue;
          
          // don't need to selectPatches, patches is already all the patches we'll need.
          neighbors.push_back(patch);
        }
        else {
          patch->getLevel()->selectPatches(low, high, neighbors);
        }
	ASSERT(is_sorted(neighbors.begin(), neighbors.end(),
			 Patch::Compare()));
	if(dbg.active()){
	  dbg << d_myworld->myrank() << "    Creating dependency on " << neighbors.size() << " neighbors\n";
	  dbg << d_myworld->myrank() << "      Low=" << low << ", high=" << high << ", var=" << req->var->getName() << '\n';
	}
        
	for(int i=0;i<neighbors.size();i++){
	  const Patch* neighbor=neighbors[i];
          Patch::selectType fromNeighbors;
	  IntVector l = Max(neighbor->getLowIndex(basis, req->var->getBoundaryLayer()), low);
	  IntVector h = Min(neighbor->getHighIndex(basis, req->var->getBoundaryLayer()), high);
	  if (neighbor->isVirtual()) {
	    l -= neighbor->getVirtualOffset();
	    h -= neighbor->getVirtualOffset();	    
	    neighbor = neighbor->getRealPatch();
	  }
          if (req->patches_dom == Task::OtherGridDomain) {
            // this is when we are copying data between two grids (currently between timesteps)
            // the grid assigned to the old dw should be the old grid.
            // This should really only impact things required from the OldDW.
            LevelP fromLevel = sc->get_dw(0)->getGrid()->getLevel(patch->getLevel()->getIndex());
            fromLevel->selectPatches(Max(neighbor->getLowIndex(basis, req->var->getBoundaryLayer()), l),
                                     Min(neighbor->getHighIndex(basis, req->var->getBoundaryLayer()), h),
                                     fromNeighbors);
          }
          else
            fromNeighbors.push_back(neighbor);

          for (int j = 0; j < fromNeighbors.size(); j++) {
            const Patch* fromNeighbor = fromNeighbors[j];

	    if(!(lb->inNeighborhood(neighbor) || lb->inNeighborhood(fromNeighbor)))
              continue;
            IntVector from_l;
            IntVector from_h;

            if (req->patches_dom == Task::OtherGridDomain && fromNeighbor->getLevel()->getIndex() > 0) {
              // DON'T send extra cells (unless they're on the domain boundary)
              from_l = Max(fromNeighbor->getInteriorLowIndexWithBoundary(basis), l);
              from_h = Min(fromNeighbor->getInteriorHighIndexWithBoundary(basis), h);
            }
            else {
              from_l = Max(fromNeighbor->getLowIndex(basis, req->var->getBoundaryLayer()), l);
              from_h = Min(fromNeighbor->getHighIndex(basis, req->var->getBoundaryLayer()), h);
            }

            if (patch->getLevel()->getIndex() > 0 && patch != fromNeighbor && req->patches_dom == Task::NormalDomain) {
              // cull annoying overlapping AMR patch dependencies
              patch->cullIntersection(basis, req->var->getBoundaryLayer(), fromNeighbor, from_l, from_h);
              if (from_l == from_h) {
                continue;
              }
            }

	    for(int m=0;m<matls->size();m++){
	      int matl = matls->get(m);

	      // creator is the task that performs the original compute.
	      // If the require is for the OldDW, then it will be a send old
	      // data task
	      DetailedTask* creator = 0;
	      Task::Dependency* comp = 0;
              //              cout << d_myworld->myrank() << "  asdf var: " << *req->var << " patch " << fromNeighbor->getID() << " DW " << req->whichdw << " " << req->mapDataWarehouse() << " ISOLD " << sc->isOldDW(req->mapDataWarehouse()) << endl;
	      if(sc->isOldDW(req->mapDataWarehouse())){
	        ASSERT(!modifies);
	        int proc = findVariableLocation(lb, req, fromNeighbor, matl);
	        creator = dt->getOldDWSendTask(proc);
	        comp=0;
	      } else {
	        if (!ct.findcomp(req, neighbor, matl, creator, comp, d_myworld)){
		  cout << "Failure finding " << *req << " for " << *task
		      << "\n";
                  if (creator)
                    cout << "creator=" << *creator << '\n';
		  cout << "neighbor=" << *neighbor << ", matl=" << matl << '\n';
		  cout << "me=" << me << '\n';
		  SCI_THROW(InternalError("Failed to find comp for dep!", __FILE__, __LINE__));
	        }
	      }
	      if (modifies) {
	        // find the tasks that up to this point require the variable
	        // that we are modifying (i.e., the ones that use the computed
	        // variable before we modify it), and put a dependency between
	        // those tasks and this tasks
                // i.e., the task that requires data computed by a task on this processor
                // needs to finish its task before this task, which modifies the data
                // computed by the same task
	        list<DetailedTask*> requireBeforeModifiedTasks;
	        creator->findRequiringTasks(req->var,
					    requireBeforeModifiedTasks);

	        list<DetailedTask*>::iterator reqTaskIter;
	        for (reqTaskIter = requireBeforeModifiedTasks.begin();
		    reqTaskIter != requireBeforeModifiedTasks.end();
		    ++reqTaskIter) {
		  DetailedTask* prevReqTask = *reqTaskIter;
		  if(prevReqTask->task == task->task){
		    if(!task->task->getHasSubScheduler()) {
                      ostringstream message;
                      message << " WARNING - task ("<< task->getName() << ") requires with Ghost cells *and* modifies and may not be correct" << endl;
                      static ProgressiveWarning warn(message.str(),10);
                      warn.invoke();
                      if (dbg.active())
                        dbg << d_myworld->myrank() << " Task that requires with ghost cells and modifies\n";
                      // cout <<  d_myworld->myrank() << " RGM: var: " << *req->var << " compute: " 
                      //      << *creator << " mod " << *task << " PRT " << *prevReqTask << " " << from_l << " " << from_h << endl;
                    }
		  } else if(prevReqTask != task){
		    // dep requires what is to be modified before it is to be
		    // modified so create a dependency between them so the
		    // modifying won't conflist with the previous require.
		    if (dbg.active()) {
		      dbg << d_myworld->myrank() << "       Requires to modifies dependency from "
			  << prevReqTask->getName()
                           << " to " << task->getName() << " (created by " << creator->getName() << ")\n";
                    }
                    if (creator->getPatches()->size() > 1) {
                      // if the creator works on many patches, then don't create links between patches that don't touch
                      const PatchSubset* psub = task->getPatches();
                      const PatchSubset* req_sub = prevReqTask->getPatches();
                      if (psub->size() == 1 && req_sub->size() == 1) {
                        const Patch* p = psub->get(0);
                        const Patch* req_patch = req_sub->get(0);
                        Patch::selectType n;
                        IntVector low, high;
                        
                        Patch::VariableBasis basis = Patch::translateTypeToBasis(req->var->typeDescription()->getType(),
                                                                                 false);
                        
                        req_patch->computeVariableExtents(req->var->typeDescription()->getType(),
                                                          req->var->getBoundaryLayer(),
                                                          Ghost::AroundCells, 2,
                                                          low, high);

                        req_patch->getLevel()->selectPatches(low, high, n);
                        bool found = false;
                        for (int i = 0; i < n.size(); i++) {
                          if (n[i]->getID() == p->getID()) {
                            found = true;
                            break;
                          }
                        }
                        if (!found)
                          continue;
                      }
                    }
		    dt->possiblyCreateDependency(prevReqTask, 0, 0, task, req, 0,
					        matl, from_l, from_h);
		  }
	        }
	      }

	      dt->possiblyCreateDependency(creator, comp, fromNeighbor,
					  task, req, fromNeighbor,
					  matl, from_l, from_h);
	    }
	  }
        }
      }
    }
    else if (!patches && matls && !matls->empty()) {
      // requiring reduction variables
      for (int m=0;m<matls->size();m++){
	int matl = matls->get(m);
	DetailedTask* creator;
	Task::Dependency* comp = 0;
	if(!ct.findcomp(req, 0, matl, creator, comp, d_myworld)){
	  cout << "Failure finding " << *req << " for " 
	       << task->getTask()->getName() << "\n"; 
	  SCI_THROW(InternalError("Failed to find comp for dep!", __FILE__, __LINE__));
	}
	ASSERTRANGE(task->getAssignedResourceIndex(), 0, d_myworld->size());
	if(task->getAssignedResourceIndex() ==
	   creator->getAssignedResourceIndex() &&
	   task->getAssignedResourceIndex() == me) {
	  task->addInternalDependency(creator, req->var);
	}
      }
    } 
    else if (patches && patches->empty() && 
             (req->patches_dom == Task::FineLevel || task->getTask()->getType() == Task::OncePerProc ||
              strcmp(task->getTask()->getName(), "SchedulerCommon::copyDataToNewGrid") == 0))
    {
      // this is a either coarsen task where there aren't any fine patches, or a PerProcessor task where
      // there aren't any patches on this processor.  Perfectly legal, so do nothing

      // another case is the copy-data-to-new-grid task, which will wither compute or modify to every patch
      // but not both.  So it will yell at you for the detailed task's patches not intersecting with the 
      // computes or modifies... (maybe there's a better way) - bryan
    }
    else {
      ostringstream desc;
      desc << "TaskGraph::createDetailedDependencies, task dependency not supported without patches and materials"
           << " \n Trying to require or modify " << *req << " in Task " << task->getTask()->getName()<<"\n\n";
      if (task->matls)
        desc << "task materials:" << *task->matls << "\n";
      else
        desc << "no task materials\n";
      if (req->matls)
        desc << "req materials: " << *req->matls << "\n";
      else
        desc << "no req materials\n";
      desc << "domain materials: " << *matls.get_rep() << "\n";
      if (task->patches)
        desc << "task patches:" << *task->patches << "\n";
      else
        desc << "no task patches\n";
      if (req->patches)
        desc << "req patches: " << *req->patches << "\n";
      else
        desc << "no req patches\n";
      desc << "domain patches: " << *patches.get_rep() << "\n";
      SCI_THROW(InternalError(desc.str(), __FILE__, __LINE__)); 
    }
  }
}

int TaskGraph::findVariableLocation(LoadBalancer* lb,
				    Task::Dependency* req,
				    const Patch* patch, int matl)
{
  // This needs to be improved, especially for re-distribution on
  // restart from checkpoint.
  int proc;
  if (req->task->mapDataWarehouse(Task::ParentNewDW) != -1 && req->whichdw != Task::ParentOldDW) {
    // provide some accomodation for Dynamic load balancers and sub schedulers.  We need to
    // treat the requirement like a "old" dw req but it needs to be found on the current processor
    proc = lb->getPatchwiseProcessorAssignment(patch);
  }
  else
    proc = lb->getOldProcessorAssignment(req->var, patch, matl);
  return proc;
}

int TaskGraph::getNumTasks() const
{
  return (int)d_tasks.size();
}

Task* TaskGraph::getTask(int idx)
{
  return d_tasks[idx];
}

TaskGraph::VarLabelMaterialMap* TaskGraph::makeVarLabelMaterialMap()
{
  VarLabelMaterialMap* result = scinew VarLabelMaterialMap;

  for(int i=0;i<(int)d_tasks.size();i++){
    Task* task = d_tasks[i];
    for(Task::Dependency* comp = task->getComputes();
	comp != 0; comp=comp->next){
      // assume all patches will compute the same labels on the same
      // materials
      const VarLabel* label = comp->var;
      list<int>& matls = (*result)[label->getName()];
      const MaterialSubset* msubset = comp->matls;
      if(msubset){
	for(int mm=0;mm<msubset->size();mm++){
	  matls.push_back(msubset->get(mm));
	}
      } else if(label->typeDescription()->getType() == TypeDescription::ReductionVariable) {
	// Default to material -1 (global)
	matls.push_back(-1);
      } else {
	const MaterialSet* ms = task->getMaterialSet();
	for(int m=0;m<ms->size();m++){
	  const MaterialSubset* msubset = ms->getSubset(m);
	  for(int mm=0;mm<msubset->size();mm++){
	    matls.push_back(msubset->get(mm));
	  }
	}
      }
    }
  }
  return result;
}
