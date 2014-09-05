#include <Packages/Uintah/CCA/Components/Schedulers/TaskGraph.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabelLevelDW.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>

#include <Core/Containers/FastHashTable.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/NotFinished.h>

#include <sci_algorithm.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <unistd.h>

using namespace Uintah;

using namespace SCIRun;
using std::cerr;
using std::is_sorted;

static DebugStream dbg("TaskGraph", false);

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
       || comp->patches_dom == Task::FineLevel){
      SCI_THROW(InternalError("Should not compute onto another level!"));
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
  if(!PatchSubset::overlaps(ps1, ps2))
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
TaskGraph::setupTaskConnections()
{
  vector<Task*>::iterator iter;
  // Initialize variables on the tasks
  for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
    Task* task = *iter;
    task->visited=false;
    task->sorted=false;
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
				+comp->var->getName()));
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
    addDependencyEdges(task, task->getRequires(), comps, false);
    addDependencyEdges(task, task->getModifies(), comps, true);
    // Used here just to warn if a modifies comes before its computes
    // in the order that tasks were added to the graph.
    task->visited = true;
  }

  // Initialize variables on the tasks
  for( iter=d_tasks.begin(); iter != d_tasks.end(); iter++ ) {
    Task* task = *iter;
    task->visited=false;
    task->sorted=false;
  }
} // end setupTaskConnections()

void TaskGraph::addDependencyEdges(Task* task, Task::Dependency* req,
				  CompMap& comps, bool modifies)
{
  for(; req != 0; req=req->next){
    if (dbg.active())
      dbg << d_myworld->myrank() << "     Looking at req: " << *req << '\n';
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
	  SCI_THROW(TypeMismatchException("Type mismatch for variable: "+req->var->getName()));

        // determine if we need to add a dependency edge
	bool add=false;
	if (dbg.active())
          dbg << d_myworld->myrank() << "       Checking edge from task: " << *compiter->second->task << " to task: " << *req->task << '\n';
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
	  
	  if (!edge->comp->task->visited &&
	      !edge->comp->task->isReductionTask()) {
	    cerr << "\nWARNING: A task, '" << task->getName() << "', that ";
	    if (modifies)
	      cerr << "modifies '";
	    else
	      cerr << "requires '";
	    cerr << req->var->getName() << "' was added before computing task";
	    cerr << ", '" << edge->comp->task->getName() << "'\n";
	    cerr << "  Required/modified by: " << *task << '\n';
	    cerr << "  req: " << *req << '\n';
	    cerr << "  Computed by: " << *edge->comp->task << '\n';
	    cerr << "  comp: " << *comp << '\n';
	    cerr << "\n";
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
	  cerr << req->patches->size() << " Patches: ";
	  for(int i=0;i<req->patches->size();i++)
	    cerr << req->patches->get(i)->getID() << " ";
	  cerr << '\n';
	} else if(req->reductionLevel) {
	  cerr << "On level " << req->reductionLevel->getIndex() << '\n';
	} else if(task->getPatchSet()){
	  cerr << "Patches from task: ";
	  const PatchSet* patches = task->getPatchSet();
	  for(int i=0;i<patches->size();i++){
	    const PatchSubset* pat=patches->getSubset(i);
	    for(int i=0;i<pat->size();i++)
	      cerr << pat->get(i)->getID() << " ";
	    cerr << " ";
	  }
	  cerr << '\n';
	} else {
	  cerr << "On global level\n";
	}
	if(req->matls){
	  cerr << req->matls->size() << " Matls: ";
	  for(int i=0;i<req->matls->size();i++)
	    cerr << req->matls->get(i) << " ";
	  cerr << '\n';
	} else if(task->getMaterialSet()){
	  cerr << "Matls from task: ";
	  const MaterialSet* matls = task->getMaterialSet();
	  for(int i=0;i<matls->size();i++){
	    const MaterialSubset* mat = matls->getSubset(i);
	    for(int i=0;i<mat->size();i++)
	      cerr << mat->get(i) << " ";
	    cerr << " ";
	  }
	  cerr << '\n';
	} else {
	  cerr << "No matls\n";
	}
	SCI_THROW(InternalError("Scheduler could not find specific production for variable: "+req->var->getName()+", required for task: "+task->getName()));
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
TaskGraph::processTask(Task* task, vector<Task*>& sortedTasks) const
{
  if(dbg.active())
    dbg << d_myworld->myrank() << " Looking at task: " << task->getName() << '\n';

  if(task->visited){
    ostringstream error;
    error << "Cycle detected in task graph: already did\n\t"
	  << task->getName();
    error << "\n";
    SCI_THROW(InternalError(error.str()));
  }

  task->visited=true;
   
  processDependencies(task, task->getRequires(), sortedTasks);
  processDependencies(task, task->getModifies(), sortedTasks);

  // All prerequisites are done - add this task to the list
  sortedTasks.push_back(task);
  task->sorted=true;
  if(dbg.active())
    dbg << d_myworld->myrank() << " Sorted task: " << task->getName() << '\n';
} // end processTask()


void TaskGraph::processDependencies(Task* task, Task::Dependency* req,
				    vector<Task*>& sortedTasks) const

{
  for(; req != 0; req=req->next){
    if (dbg.active())
      dbg << d_myworld->myrank() << " processDependencies for req: " << *req << '\n';
    if(sc->isNewDW(req->mapDataWarehouse())){
      Task::Edge* edge = req->comp_head;
      for (;edge != 0; edge = edge->compNext){
	Task* vtask = edge->comp->task;
	if(!vtask->sorted){
	  if(vtask->visited){
	    ostringstream error;
	    error << "Cycle detected in task graph: trying to do\n\t"
		  << task->getName();
	    error << "\nbut already did:\n\t"
		  << vtask->getName();
	    error << ",\nwhile looking for variable: \n\t" 
		  << req->var->getName();
	    error << "\n";
	    SCI_THROW(InternalError(error.str()));
	  }
	  processTask(vtask, sortedTasks);
	}
      }
    }
  }
}

void
TaskGraph::nullSort( vector<Task*>& tasks )
{
  // setupTaskConnections also creates the reduction tasks...
  setupTaskConnections();

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
  setupTaskConnections();

  for(vector<Task*>::iterator iter=d_tasks.begin();
      iter != d_tasks.end(); iter++ ) {
    Task* task = *iter;
    if(!task->sorted){
      processTask(task, sortedTasks);
    }
  }
  int n=0;
  for(vector<Task*>::iterator iter = sortedTasks.begin();
      iter != sortedTasks.end(); iter++)
    (*iter)->sortedOrder = n++;
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
    const PatchSet* ps = task->patch_set;
    if(ps && task->matl_set){
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
  lb->createNeighborhood(grid, d_myworld, sc);

  DetailedTasks* dt = scinew DetailedTasks(sc, d_myworld, this, useInternalDeps );
  for(int i=0;i<(int)sorted_tasks.size();i++){
    Task* task = sorted_tasks[i];
    const PatchSet* ps = task->patch_set;
    const MaterialSet* ms = task->matl_set;
    if(ps && ms){
      for(int p=0;p<ps->size();p++){
	const PatchSubset* pss = ps->getSubset(p);
	for(int m=0;m<ms->size();m++){
	  const MaterialSubset* mss = ms->getSubset(m);
	  if(lb->inNeighborhood(pss, mss)) // can we move this comparison up
          	                  //(does mss determines neighborhood)? - bryan

	    createDetailedTask(dt, task, pss, mss);
	}
      }
    } else if(!ps && !ms){
      createDetailedTask(dt, task, 0, 0);
    } else if(!ps){
      SCI_THROW(InternalError("Task has MaterialSet, but no PatchSet"));
    } else {
      SCI_THROW(InternalError("Task has PatchSet, but no MaterialSet"));
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
      cerr << "Multiple compute found:\n";
      cerr << "matl: " << newData->matl << "\n";
      cerr << "patch: " << *newData->patch << "\n";
      cerr << *newData->comp << "\n";
      cerr << *newData->task << "\n";
      cerr << "It was originally computed by the following task(s):\n";
      for(Data* old = data.lookup(newData); old != 0; old = data.nextMatch(newData, old)){
	old->comp->task->displayAll(cerr);
      }
      SCI_THROW(InternalError("Multiple computes for variable: "+newData->comp->var->getName()));
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
  if (dbg.active())
    dbg << pg->myrank() << "         Finding comp of req: " << *req << " for task: " << *req->task << "/" << '\n';
  Data key(0, req, patch, matlIndex);
  Data* result = 0;
  for(Data* p = data.lookup(&key); p != 0; p = data.nextMatch(&key, p)){
    if (dbg.active())
      dbg << pg->myrank() << "         Examining comp from: " << p->comp->task->getName() << ", order=" << p->comp->task->getSortedOrder() << '\n';
    ASSERT(!result || p->comp->task->getSortedOrder() != result->comp->task->getSortedOrder());
    if(p->comp->task->getSortedOrder() < req->task->getSortedOrder()){
      if(!result || p->comp->task->getSortedOrder() > result->comp->task->getSortedOrder()){
	if (dbg.active())
          dbg << pg->myrank() << "         New best is comp from: " << p->comp->task->getName() << ", order=" << p->comp->task->getSortedOrder() << '\n';
	result = p;
      }
    }
  }
  if(result){
    if (dbg.active())
      dbg << pg->myrank() << "         Found comp at: " << result->comp->task->getName() << ", order=" << result->comp->task->getSortedOrder() << '\n';
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

    if(patches && !patches->empty() && matls && !matls->empty()){
      for(int i=0;i<patches->size();i++){
	const Patch* patch = patches->get(i);
	Patch::selectType neighbors;
	IntVector low, high;
	patch->computeVariableExtents(req->var->typeDescription()->getType(),
				      req->var->getBoundaryLayer(),
				      req->gtype, req->numGhostCells,
				      neighbors, low, high);
	ASSERT(is_sorted(neighbors.begin(), neighbors.end(),
			 Patch::Compare()));
	if(dbg.active()){
	  dbg << d_myworld->myrank() << "      Creating dependency on " << neighbors.size() << " neighbors\n";
	  dbg << d_myworld->myrank() << "        Low=" << low << ", high=" << high << ", var=" << req->var->getName() << '\n';
	}
	Patch::VariableBasis basis = Patch::translateTypeToBasis(req->var->typeDescription()->getType(),
								 false);
	for(int i=0;i<neighbors.size();i++){
	  const Patch* neighbor=neighbors[i];
	  IntVector l = Max(neighbor->getLowIndex(basis, req->var->getBoundaryLayer()), low);
	  IntVector h = Min(neighbor->getHighIndex(basis, req->var->getBoundaryLayer()), high);
	  if (neighbor->isVirtual()) {
	    l -= neighbor->getVirtualOffset();
	    h -= neighbor->getVirtualOffset();	    
	    neighbor = neighbor->getRealPatch();
	  }
	  if(!lb->inNeighborhood(neighbor))
	    continue;
	  for(int m=0;m<matls->size();m++){
	    if(sc->isOldDW(req->mapDataWarehouse()) && !sc->isNewDW(req->mapDataWarehouse()+1))
	      continue;
	    int matl = matls->get(m);

	    // creator is the task that performs the original compute.
	    // If the require is for the OldDW, then it will be a send old
	    // data task
	    DetailedTask* creator;
	    Task::Dependency* comp = 0;
	    if(sc->isOldDW(req->mapDataWarehouse())){
	      ASSERT(!modifies);
	      int proc = findVariableLocation(lb, req, neighbor, matl);
	      creator = dt->getOldDWSendTask(proc);
	      comp=0;
	    } else {
	      if (!ct.findcomp(req, neighbor, matl, creator, comp, d_myworld)){
		cerr << "Failure finding " << *req << " for " << *task
		     << "\n";
		cerr << "creator=" << *creator << '\n';
		cerr << "neighbor=" << *neighbor << '\n';
		cerr << "me=" << me << '\n';
		SCI_THROW(InternalError("Failed to find comp for dep!"));
	      }
	    }
	    if (modifies) {
	      // find the tasks that up to this point require the variable
	      // that we are modifying (i.e., the ones that use the computed
	      // variable before we modify it), and put a dependency between
	      // this task and those tasks
	      list<DetailedTask*> requireBeforeModifiedTasks;
	      creator->findRequiringTasks(req->var,
					  requireBeforeModifiedTasks);

	      list<DetailedTask*>::iterator reqTaskIter;
	      for (reqTaskIter = requireBeforeModifiedTasks.begin();
		   reqTaskIter != requireBeforeModifiedTasks.end();
		   ++reqTaskIter) {
		DetailedTask* prevReqTask = *reqTaskIter;
		if(prevReqTask->task == task->task){
		  if(!task->task->d_hasSubScheduler)
		    cerr << "\n\n\nWARNING - task that requires with Ghost cells *and* modifies may not be correct\n";
		} else if(prevReqTask != task){
		  // dep requires what is to be modified before it is to be
		  // modified so create a dependency between them so the
		  // modifying won't conflist with the previous require.
		  if (dbg.active()) {
		    dbg << d_myworld->myrank() << "       Requires to modifies dependency from "
			<< prevReqTask->getTask()->getName()
			<< " to " << task->getTask()->getName() << "\n";
		  }
		  dt->possiblyCreateDependency(prevReqTask, 0, 0, task, req, 0,
					       matl, l, h);
		}
	      }
	    }

	    dt->possiblyCreateDependency(creator, comp, neighbor,
					 task, req, patch,
					 matl, l, h);
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
	  cerr << "Failure finding " << *req << " for " 
	       << task->getTask()->getName() << "\n"; 
	  SCI_THROW(InternalError("Failed to find comp for dep!"));
	}
	ASSERTRANGE(task->getAssignedResourceIndex(), 0, d_myworld->size());
	if(task->getAssignedResourceIndex() ==
	   creator->getAssignedResourceIndex() &&
	   task->getAssignedResourceIndex() == me) {
	  task->addInternalDependency(creator, req->var);
	}
      }
    } else {
      ostringstream desc;
      desc << "TaskGraph::createDetailedDependencies, task dependency not supported without patches and materials"
           << " \n Trying to require or modify " << *req << " in Task " << task->getTask()->getName()<<"\n\n";
      SCI_THROW(InternalError(desc.str())); 
    }
  }
}

int TaskGraph::findVariableLocation(LoadBalancer* lb,
				    Task::Dependency* req,
				    const Patch* patch, int matl)
{
  // This needs to be improved, especially for re-distribution on
  // restart from checkpoint.
  int proc = lb->getOldProcessorAssignment(req->var, patch, matl, d_myworld);
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
