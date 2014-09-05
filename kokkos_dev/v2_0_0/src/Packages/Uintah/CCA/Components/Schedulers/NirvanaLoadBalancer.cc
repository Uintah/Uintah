

#include <Packages/Uintah/CCA/Components/Schedulers/NirvanaLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/NotFinished.h>

using namespace Uintah;

NirvanaLoadBalancer::NirvanaLoadBalancer(const ProcessorGroup* myworld,
					 const IntVector& layout)
   : UintahParallelComponent(myworld), layout(layout)
{
  npatches=0;
}

NirvanaLoadBalancer::~NirvanaLoadBalancer()
{
}

static int getproc(const IntVector& l, const IntVector& d,
		   const IntVector& layout,
		   int patches_per_processor, int processors_per_host)
{
  IntVector h = l/d;
  IntVector p = IntVector(l.x()%d.x(), l.y()%d.y(), l.z()%d.z());
  int host = h.x()*layout.y()*layout.z()+h.y()*layout.z()+h.z();
  int hostpatch = p.x()*d.y()*d.z()+p.y()*d.z()+p.z();
  int hostproc = hostpatch/patches_per_processor;
  int proc = hostproc+host*processors_per_host;
  return proc;
}

void NirvanaLoadBalancer::assignResources(DetailedTasks& graph,
					  const ProcessorGroup* group)
{
  int nTasks = graph.numTasks();
  const Level* level = 0;
  for(int i=0;i<nTasks && !level;i++){
    DetailedTask* task = graph.getTask(i);
    const PatchSubset* patches = task->getPatches();
    if(patches && patches->size()){
      const Patch* patch = patches->get(0);
      npatches = patch->getLevel()->numPatches();
      level = patch->getLevel();
    }
  }
  ASSERT(npatches != 0);
  ASSERT(level != 0);
  numhosts = layout.x()*layout.y()*layout.z();
  numProcs = group->size();
  processors_per_host = numProcs/numhosts;
  if(processors_per_host * numhosts != numProcs)
    SCI_THROW(InternalError("NirvanaLoadBalancer will not work with uneven numbers of processors per host"));
  patches_per_processor = npatches/numProcs;
  if(patches_per_processor * numProcs != npatches) {
    SCI_THROW(InternalError("NirvanaLoadBalancer will not work with uneven number of patches per processor"));
  }
  
  IntVector max(-1,-1,-1);
  for(Level::const_patchIterator iter = level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    IntVector l;
    if(!(*iter)->getLayoutHint(l))
      SCI_THROW(InternalError("NirvanaLoadBalancer requires layout hints"));
    max = Max(max, l);
  }
  max+=IntVector(1,1,1);
  d = max/layout;
  for(int i=0;i<nTasks;i++){
    DetailedTask* task = graph.getTask(i);
    const PatchSubset* patches = task->getPatches();
    if(patches && patches->size() > 0){
      const Patch* patch = patches->get(0);
      IntVector l;
      if(!patch->getLayoutHint(l))
	SCI_THROW(InternalError("NirvanaLoadBalancer requires layout hints"));
      int idx = getproc(l, d, layout, patches_per_processor, processors_per_host);
      task->assignResource(idx);
      for(int i=1;i<patches->size();i++){
	const Patch* p = patches->get(i);
	IntVector l;
	if(!p->getLayoutHint(l))
	  SCI_THROW(InternalError("NirvanaLoadBalancer requires layout hints"));
	int pidx = getproc(l, d, layout, patches_per_processor, processors_per_host);
	if(pidx != idx){
	  cerr << "WARNING: inconsistent task assignment in NirvanaLoadBalancer\n";
	}
      }
    } else {
      if( Parallel::usingMPI() && task->getTask()->isReductionTask() ){
	task->assignResource( Parallel::getRootProcessorGroup()->myrank() );
      } else if( task->getTask()->getType() == Task::InitialSend){
	// Already assigned, do nothing
	ASSERT(task->getAssignedResourceIndex() != -1);
      } else {
#if DAV_DEBUG
	cerr << "Task " << *task << " IS ASSIGNED TO PG 0!\n";
#endif
	task->assignResource(0);
      }
    }
  }
}

int NirvanaLoadBalancer::getPatchwiseProcessorAssignment(const Patch* patch,
							 const ProcessorGroup* group)
{
  if(npatches == 0){
    npatches = patch->getLevel()->numPatches();
    numhosts = layout.x()*layout.y()*layout.z();
    numProcs = group->size();
    processors_per_host = numProcs/numhosts;
    if(processors_per_host * numhosts != numProcs)
      SCI_THROW(InternalError("NirvanaLoadBalancer will not work with uneven numbers of processors per host"));
    patches_per_processor = npatches/numProcs;
    if(patches_per_processor * numProcs != npatches) {
      SCI_THROW(InternalError("NirvanaLoadBalancer will not work with uneven number of patches per processor"));
    }
    IntVector max(-1,-1,-1);
    const Level* level = patch->getLevel();
    for(Level::const_patchIterator iter = level->patchesBegin();
	iter != level->patchesEnd(); iter++){
      IntVector l;
      if(!(*iter)->getLayoutHint(l))
	SCI_THROW(InternalError("NirvanaLoadBalancer requires layout hints"));
      max = Max(max, l);
    }
    max+=IntVector(1,1,1);
    d = max/layout;
  }
  
  IntVector l;
  if(!patch->getLayoutHint(l))
    SCI_THROW(InternalError("NirvanaLoadBalancer requires layout hints"));
  int pidx = getproc(l, d, layout, patches_per_processor, processors_per_host);
  ASSERTRANGE(pidx, 0, group->size());
  return pidx;
}

const PatchSet*
NirvanaLoadBalancer::createPerProcessorPatchSet(const LevelP& level,
						const ProcessorGroup* world)
{
  PatchSet* patches = scinew PatchSet();
  patches->createEmptySubsets(world->size());
  for(Level::const_patchIterator iter = level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch = *iter;
    int proc = getPatchwiseProcessorAssignment(patch, world);
    ASSERTRANGE(proc, 0, world->size());
    PatchSubset* subset = patches->getSubset(proc);
    subset->add(patch);
  }
  patches->sortSubsets();  
  return patches;
}


void
NirvanaLoadBalancer::createNeighborhood(const GridP& grid,
					const ProcessorGroup* group,
					const Scheduler* /*sch*/)
{
  int me = group->myrank();
  // WARNING - this should be determined from the taskgraph? - Steve
  int maxGhost = 2;
  d_neighbors.clear();
  for(int l=0;l<grid->numLevels();l++){
    const LevelP& level = grid->getLevel(l);
    for(Level::const_patchIterator iter = level->patchesBegin();
	iter != level->patchesEnd(); iter++){
      const Patch* patch = *iter;
      if(getPatchwiseProcessorAssignment(patch, group) == me){
	Level::selectType n;
	IntVector lowIndex, highIndex;
	patch->computeVariableExtents(Patch::CellBased, IntVector(0,0,0),
				      Ghost::AroundCells, maxGhost, n,
				      lowIndex, highIndex);
	for(int i=0;i<(int)n.size();i++){
	  const Patch* neighbor = n[i];
	  if(d_neighbors.find(neighbor) == d_neighbors.end())
	    d_neighbors.insert(neighbor);
	}
      }
    }
  }
}

bool
NirvanaLoadBalancer::inNeighborhood(const PatchSubset* ps,
				    const MaterialSubset*)
{
  for(int i=0;i<ps->size();i++){
    const Patch* patch = ps->get(i);
    if(d_neighbors.find(patch) != d_neighbors.end())
      return true;
  }
  return false;
}

bool
NirvanaLoadBalancer::inNeighborhood(const Patch* patch)
{
  if(d_neighbors.find(patch) != d_neighbors.end())
    return true;
  else
    return false;
}
