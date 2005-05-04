

#include <Packages/Uintah/CCA/Components/LoadBalancers/NirvanaLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Components/Scheduler3/DetailedTasks3.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <Core/Util/FancyAssert.h>

using namespace Uintah;
using namespace std;

NirvanaLoadBalancer::NirvanaLoadBalancer(const ProcessorGroup* myworld,
					 const IntVector& layout)
   : LoadBalancerCommon(myworld), layout(layout)
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

void NirvanaLoadBalancer::assignResources(DetailedTasks& graph)
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
  numProcs = d_myworld->size();
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
	//	cerr << "Task " << *task << " IS ASSIGNED TO PG 0!\n";
#endif
	task->assignResource(0);
      }
    }
  }
}

void NirvanaLoadBalancer::assignResources(DetailedTasks3& graph)
{
  int nTasks = graph.numTasks();
  const Level* level = 0;
  for(int i=0;i<nTasks && !level;i++){
    DetailedTask3* task = graph.getTask(i);
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
  numProcs = d_myworld->size();
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
    DetailedTask3* task = graph.getTask(i);
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

int NirvanaLoadBalancer::getPatchwiseProcessorAssignment(const Patch* patch)
{
  if(npatches == 0){
    npatches = patch->getLevel()->numPatches();
    numhosts = layout.x()*layout.y()*layout.z();
    numProcs = d_myworld->size();
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
  ASSERTRANGE(pidx, 0, d_myworld->size());
  return pidx;
}

