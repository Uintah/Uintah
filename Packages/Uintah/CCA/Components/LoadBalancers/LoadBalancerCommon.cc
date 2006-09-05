
#include <Packages/Uintah/CCA/Components/LoadBalancers/LoadBalancerCommon.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/NotFinished.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Mutex.h>
#include <sci_values.h>
#include <sstream>

using namespace Uintah;

#ifdef _WIN32
#define SCISHARE __declspec(dllimport)
#else
#define SCISHARE
#endif
// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern SCISHARE SCIRun::Mutex       cerrLock;
DebugStream lbDebug( "LoadBalancer", false );
DebugStream neiDebug("Neighborhood", false );

LoadBalancerCommon::LoadBalancerCommon(const ProcessorGroup* myworld)
   : UintahParallelComponent(myworld)
{
}

LoadBalancerCommon::~LoadBalancerCommon()
{
}

void LoadBalancerCommon::assignResources(DetailedTasks& graph)
{
  int nTasks = graph.numTasks();

  if( lbDebug.active() ) {
    cerrLock.lock();
    lbDebug << "Assigning Tasks to Resources!\n";
    cerrLock.unlock();
  }

  for(int i=0;i<nTasks;i++){
    DetailedTask* task = graph.getTask(i);

    const PatchSubset* patches = task->getPatches();
    if(patches && patches->size() > 0 && task->getTask()->getType() != Task::OncePerProc){
      const Patch* patch = patches->get(0);

      int idx = getPatchwiseProcessorAssignment(patch);
      ASSERTRANGE(idx, 0, d_myworld->size());

      if (task->getTask()->getType() == Task::Output) {
        task->assignResource((idx/d_outputNthProc)*d_outputNthProc);
      }
      else {
        task->assignResource(idx);
      }

      if( lbDebug.active() ) {
	cerrLock.lock();
	lbDebug << "1) Task " << *(task->getTask()) << " put on resource "
		   << idx << "\n";
	cerrLock.unlock();
      }

      ostringstream ostr;
      ostr << patch->getID() << ':' << idx;

      for(int i=1;i<patches->size();i++){
	const Patch* p = patches->get(i);
	int pidx = getPatchwiseProcessorAssignment(p);
        ostr << ' ' << p->getID() << ';' << pidx;
	ASSERTRANGE(pidx, 0, d_myworld->size());
	if (pidx != idx && task->getTask()->getType() != Task::Output) {
	  cerrLock.lock();
	  cerr << d_myworld->myrank() << " WARNING: inconsistent task (" << task->getTask()->getName() 
	       << ") assignment (" << pidx << ", " << idx 
	       << ") in LoadBalancerCommon\n";
	  cerrLock.unlock();
       }
      }
    } else {
      if( Parallel::usingMPI() && task->getTask()->isReductionTask() ){
	task->assignResource( d_myworld->myrank() );

	if( lbDebug.active() ) {
	  cerrLock.lock();
	  lbDebug << "  Resource (for no patch task) is : " 
		     << d_myworld->myrank() << "\n";
	  cerrLock.unlock();
	}

      } else if( task->getTask()->getType() == Task::InitialSend){
	// Already assigned, do nothing
	ASSERT(task->getAssignedResourceIndex() != -1);
      } else if( task->getTask()->getType() == Task::OncePerProc) {
        // patch-less task, not execute-once, set to run on all procs
        // once per patch subset (empty or not)
        // at least one example is the multi-level (impAMRICE) pressureSolve
        for (int i = 0; i < task->getTask()->getPatchSet()->size(); i++)
        if (patches == task->getTask()->getPatchSet()->getSubset(i)) {
          task->assignResource(i);
        }
      } else {
	task->assignResource(0);
      }
    }
    if( lbDebug.active() ) {
      cerrLock.lock();
      //      lbDebug << "For Task: " << *task << "\n";
      cerrLock.unlock();
    }
  }

}

bool LoadBalancerCommon::possiblyDynamicallyReallocate(const GridP& grid, bool force)
{
  if (force) {
    // have it create a new patch set, and have the DLB version call this.
    // This is a good place to do it, as it is automatically called when the
    // grid changes.
    levelPerProcPatchSets.clear();
    outputPatchSets.clear();
    gridPerProcPatchSet = createPerProcessorPatchSet(grid);
    for (int i = 0; i < grid->numLevels(); i++) {
      levelPerProcPatchSets.push_back(createPerProcessorPatchSet(grid->getLevel(i)));
      outputPatchSets.push_back(createOutputPatchSet(grid->getLevel(i)));
    }
  }
  return false;
}

// Creates a PatchSet containing PatchSubsets for each processor for a
// single level.
const PatchSet*
LoadBalancerCommon::createPerProcessorPatchSet(const LevelP& level)
{
  PatchSet* patches = scinew PatchSet();
  patches->createEmptySubsets(d_myworld->size());
  for(Level::const_patchIterator iter = level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch = *iter;
    int proc = getPatchwiseProcessorAssignment(patch);
    ASSERTRANGE(proc, 0, d_myworld->size());
    PatchSubset* subset = patches->getSubset(proc);
    subset->add(patch);
  }
  patches->sortSubsets();  
  return patches;
}

// Creates a PatchSet containing PatchSubsets for each processor for an
// entire grid.
const PatchSet*
LoadBalancerCommon::createPerProcessorPatchSet(const GridP& grid)
{
  PatchSet* patches = scinew PatchSet();
  patches->createEmptySubsets(d_myworld->size());
  for (int i = 0; i < grid->numLevels(); i++) {
    const LevelP level = grid->getLevel(i);
    
    for(Level::const_patchIterator iter = level->patchesBegin();
        iter != level->patchesEnd(); iter++){
      const Patch* patch = *iter;
      int proc = getPatchwiseProcessorAssignment(patch);
      ASSERTRANGE(proc, 0, d_myworld->size());
      PatchSubset* subset = patches->getSubset(proc);
      subset->add(patch);
    }
  }
  patches->sortSubsets();  
  return patches;
}

const PatchSet* 
LoadBalancerCommon::createOutputPatchSet(const LevelP& level)
{
  if (d_outputNthProc == 1) {
    // assume the perProcessor set on the level was created first
    return levelPerProcPatchSets[level->getIndex()].get_rep();
  }
  else {
    PatchSet* patches = scinew PatchSet();
    patches->createEmptySubsets(d_myworld->size());
    for(Level::const_patchIterator iter = level->patchesBegin();
        iter != level->patchesEnd(); iter++){
      const Patch* patch = *iter;
      int proc = (getPatchwiseProcessorAssignment(patch) / d_outputNthProc) * d_outputNthProc;
      ASSERTRANGE(proc, 0, d_myworld->size());
      PatchSubset* subset = patches->getSubset(proc);
      subset->add(patch);
    }
    patches->sortSubsets();
    return patches;
  }
}


void
LoadBalancerCommon::createNeighborhood(const GridP& grid, const GridP& oldGrid)
{
  int me = d_myworld->myrank();
  // WARNING - this should be determined from the taskgraph? - Steve
  int maxGhost = 2;
  d_neighbors.clear();

  // go through all patches on all levels, and if the patchwise
  // processor assignment equals the current processor, then store the 
  // patch's neighbors in the load balancer array
  for(int l=0;l<grid->numLevels();l++){
    LevelP level = grid->getLevel(l);
    
    for(Level::const_patchIterator iter = level->patchesBegin();
        iter != level->patchesEnd(); iter++){
      const Patch* patch = *iter;
      
      // we need to check both where the patch is and where
        // it used to be (in the case of a dynamic reallocation)
      int proc = getPatchwiseProcessorAssignment(patch);
      int oldproc = getOldProcessorAssignment(NULL, patch, 0);
      
      // we also need to see if the output processor for patch is this proc,
      // in case it wouldn't otherwise have been in the neighborhood
      int outputproc = (proc / d_outputNthProc)*d_outputNthProc;
      
      if(proc == me || oldproc == me || outputproc == me) {
        // one for current level, coarse level, find level, old level
        // each call to level->selectPatches must be done with an empty patch set
        // or otherwise it will conflict with the sorted order of the cached patches
        Patch::selectType neighbor, coarse, fine, old;
        vector<Patch::selectType*> n; // shortcut to not have to use the same code 4 times
        n.push_back(&neighbor);
        IntVector ghost(maxGhost,maxGhost,maxGhost);
        
        IntVector low(patch->getLowIndex(Patch::CellBased, IntVector(0,0,0)));
        IntVector high(patch->getHighIndex(Patch::CellBased, IntVector(0,0,0)));
        level->selectPatches(low-ghost, high+ghost, neighbor);

        if (d_sharedState->isCopyDataTimestep() && proc == me) {
          if (oldGrid->numLevels() > l) {
            // on copy data timestep we need old patches that line up with this proc's patches,
            // get the other way around at the end
            const LevelP& oldLevel = oldGrid->getLevel(l);
            oldLevel->selectPatches(patch->getCellLowIndex()-ghost, patch->getCellHighIndex()+ghost, old);
            n.push_back(&old);
          }
        }
        
        // add amr stuff - so the patch will know about coarsening and refining
        if (l > 0 && (proc == me || (oldproc == me && !d_sharedState->isCopyDataTimestep()))) {
          const LevelP& coarseLevel = level->getCoarserLevel();
          IntVector ratio = level->getRefinementRatio();
          
          // we can require up to 1 ghost cell from a coarse patch
          int ngc = 1 * Max(Max(ratio.x(), ratio.y()), ratio.z());
          IntVector ghost(ngc,ngc,ngc);
          coarseLevel->selectPatches(level->mapCellToCoarser(low) - ghost, 
                                     level->mapCellToCoarser(high) + ghost, coarse);
          n.push_back(&coarse);
        }
        if (l < grid->numLevels()-1 && (proc == me || (oldproc == me && !d_sharedState->isCopyDataTimestep()))) {
          // we don't use ghost cells from fine patches
          IntVector ghost(1,1,1);
          const LevelP& fineLevel = level->getFinerLevel();
          fineLevel->selectPatches(level->mapCellToFiner(low)-ghost, 
                                   level->mapCellToFiner(high)+ghost, fine);
          n.push_back(&fine);
          
        }
        for (unsigned i = 0; i < n.size(); i++) {
          for(int j=0;j<(*n[i]).size();j++)
            d_neighbors.insert((*n[i])[j]->getRealPatch());
        }
      }
    }
  }

  if (d_sharedState->isCopyDataTimestep()) {
    IntVector ghost(maxGhost, maxGhost, maxGhost);
    // Regrid timestep postprocess 
    // 1)- go through the old grid and 
    //     find which patches used to be on this proc 
    for(int l=0;l<oldGrid->numLevels();l++){
      if (grid->numLevels() <= l)
        continue;
      LevelP oldLevel = oldGrid->getLevel(l);
      LevelP newLevel = grid->getLevel(l);
      
      for(Level::const_patchIterator iter = oldLevel->patchesBegin();
	  iter != oldLevel->patchesEnd(); iter++){
        const Patch* oldPatch = *iter;

        // we need to check both where the patch is and where
        // it used to be (in the case of a dynamic reallocation)
        int oldproc = getOldProcessorAssignment(NULL, oldPatch, 0);

        if (oldproc == me) {
          // don't get extra cells or ghost cells
	  Patch::selectType n;
          newLevel->selectPatches(oldPatch->getCellLowIndex()-ghost, oldPatch->getCellHighIndex()+ghost, n);
          d_neighbors.insert(oldPatch);

	  for(int i=0;i<(int)n.size();i++){
            d_neighbors.insert(n[i]->getRealPatch());
	  }
        }
      }
    }
  }
  if (neiDebug.active() && d_myworld->myrank() == 0)
    for (std::set<const Patch*>::iterator iter = d_neighbors.begin(); iter != d_neighbors.end(); iter++)
      cout << d_myworld->myrank() << "  Neighborhood: " << (*iter)->getID() << " Proc " << getPatchwiseProcessorAssignment(*iter) << endl;

}

bool
LoadBalancerCommon::inNeighborhood(const PatchSubset* ps,
				   const MaterialSubset*)
{
  for(int i=0;i<ps->size();i++){
    const Patch* patch = ps->get(i);
    if(d_neighbors.find(patch) != d_neighbors.end())
      return true;
  }
  // also count a subset with no patches
  return ps->size() == 0;
}

bool
LoadBalancerCommon::inNeighborhood(const Patch* patch)
{
  if(d_neighbors.find(patch) != d_neighbors.end())
    return true;
  else
    return false;
}

void
LoadBalancerCommon::problemSetup(ProblemSpecP& pspec, SimulationStateP& state)
{
  d_sharedState = state;
  d_scheduler = dynamic_cast<Scheduler*>(getPort("scheduler"));
  ProblemSpecP p = pspec->findBlock("LoadBalancer");
  d_outputNthProc = 1;
  
  if (p != 0) {
    p->getWithDefault("outputNthProc", d_outputNthProc, 1);
  }
}
