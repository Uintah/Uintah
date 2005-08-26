
#include <Packages/Uintah/CCA/Components/LoadBalancers/LoadBalancerCommon.h>
#include <Packages/Uintah/CCA/Components/LoadBalancers/ParticleLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Components/Scheduler3/DetailedTasks3.h>
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

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern Mutex cerrLock;
DebugStream lbDebug( "LoadBalancer", false );
DebugStream neiDebug("Neighborhood", false );
DebugStream clusterDebug("Clustering", false);

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
    if(patches && patches->size() > 0){
      const Patch* patch = patches->get(0);

      int idx = getPatchwiseProcessorAssignment(patch);
      ASSERTRANGE(idx, 0, d_myworld->size());

      if (task->getTask()->getType() == Task::Output) {
        task->assignResource((idx/d_outputNthProc)*d_outputNthProc);
      }
      else        
        task->assignResource(idx);

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
	if(pidx != idx){
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
      } else {
#if DAV_DEBUG
	//	cerr << "Task " << *task << " IS ASSIGNED TO PG 0!\n";
#endif
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

void LoadBalancerCommon::assignResources(DetailedTasks3& graph)
{
  int nTasks = graph.numTasks();

  if( lbDebug.active() ) {
    cerrLock.lock();
    lbDebug << "Assigning Tasks to Resources!\n";
    cerrLock.unlock();
  }

  for(int i=0;i<nTasks;i++){
    DetailedTask3* task = graph.getTask(i);

    const PatchSubset* patches = task->getPatches();
    if(patches && patches->size() > 0){
      const Patch* patch = patches->get(0);

      int idx = getPatchwiseProcessorAssignment(patch);
      ASSERTRANGE(idx, 0, d_myworld->size());

      if (task->getTask()->getType() == Task::Output) {
        task->assignResource((idx/d_outputNthProc)*d_outputNthProc);
      }
      else        
        task->assignResource(idx);

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
	if(pidx != idx){
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
      } else {
#if DAV_DEBUG
	cerr << "Task " << *task << " IS ASSIGNED TO PG 0!\n";
#endif
	task->assignResource(0);
      }
    }
  }
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

void
LoadBalancerCommon::createNeighborhood(const GridP& grid)
{
  int me = d_myworld->myrank();
  // WARNING - this should be determined from the taskgraph? - Steve
  int maxGhost = 2;
  d_neighbors.clear();

  if (d_myworld->myrank() == 0 && clusterDebug.active())
    clusterDebug << *(grid.get_rep()) << endl;

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


      // if this is a copy data (AMR) timestep, we don't really know if the send
      // patch will be in the neighborhood or not, so add it.   this won't be
      // an expensive taskgraph anyway.

      if(proc == me || oldproc == me || outputproc == me ||
         d_sharedState->isCopyDataTimestep()) {
	Patch::selectType n;
        IntVector lowGhost, highGhost;

        // don't use computeVariableExtents here - in certain cases it may not 
        // create a complete neighborhood.  
        Patch::getGhostOffsets(Patch::CellBased, Ghost::AroundCells,
                               maxGhost, lowGhost, highGhost);

        IntVector low(patch->getLowIndex(Patch::CellBased, IntVector(0,0,0)));
        IntVector high(patch->getHighIndex(Patch::CellBased, IntVector(0,0,0)));
        level->selectPatches(low-lowGhost, high+highGhost, n);

        // use only for the coarse-fine relationship
        IntVector lowIndex = low-lowGhost, highIndex = high+highGhost;

        // add amr stuff - so the patch will know about coarsening and refining
        if (l > 0) {
          const LevelP& coarseLevel = level->getCoarserLevel();
          coarseLevel->selectPatches(level->mapCellToCoarser(lowIndex), 
                                     level->mapCellToCoarser(highIndex), n);
        }
        if (l < grid->numLevels()-1) {
          const LevelP& fineLevel = level->getFinerLevel();
          fineLevel->selectPatches(level->mapCellToFiner(lowIndex), 
                                     level->mapCellToFiner(highIndex), n);
        }
	for(int i=0;i<(int)n.size();i++){
	  const Patch* neighbor = n[i]->getRealPatch();
	  if(d_neighbors.find(neighbor) == d_neighbors.end())
	    d_neighbors.insert(neighbor);
	}
        if (d_myworld->myrank() == 0 && clusterDebug.active()) {
          clusterDebug << patch->getID() << " ";
          for (int i = 0; i < n.size(); i++)
            if (n[i]->getID() >= 0)
              clusterDebug << n[i]->getID() << " ";
          clusterDebug << endl;
        }
      }
    }
  }


  if (d_myworld->myrank() == 0 && clusterDebug.active()) {

    for(int l=0;l<grid->numLevels();l++){
      LevelP level = grid->getLevel(l);
      
      for(Level::const_patchIterator iter = level->patchesBegin();
          iter != level->patchesEnd(); iter++){
        const Patch* patch = *iter;
        //clusterDebug << " Patch " << patch->getID() - low_patch << ": proc " <<getPatchwiseProcessorAssignment(patch) << endl;
      }
    }
  }
  if (neiDebug.active())
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
  return false;
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
  string dynamicAlgo;
  double interval = 0;
  double cellFactor = .1;
  int timestepInterval = 0;
  d_outputNthProc = 1;
  double threshold = 0.0;
  bool spaceCurve = false;
  
  if (p != 0) {
    p->getWithDefault("outputNthProc", d_outputNthProc, 1);
    if(!p->get("timestepInterval", timestepInterval))
      timestepInterval = 0;
    if (timestepInterval != 0 && !p->get("interval", interval))
      interval = 0.0; // default
    p->getWithDefault("dynamicAlgorithm", dynamicAlgo, "static");
    p->getWithDefault("cellFactor", cellFactor, .1);
    p->getWithDefault("gainThreshold", threshold, 0.0);
    p->getWithDefault("doSpaceCurve", spaceCurve, false);
  }

  setDynamicAlgorithm(dynamicAlgo, interval, timestepInterval, cellFactor, spaceCurve, threshold);
}
