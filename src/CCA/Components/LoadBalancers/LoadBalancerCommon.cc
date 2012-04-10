/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/

#include <TauProfilerForSCIRun.h>
#include <CCA/Components/LoadBalancers/LoadBalancerCommon.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/NotFinished.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Mutex.h>
#include <sci_values.h>
#include <sstream>

using namespace Uintah;
using namespace std;

#undef UINTAHSHARE
#if defined(_WIN32) && !defined(BUILD_UINTAH_STATIC)
#define UINTAHSHARE __declspec(dllimport)
#else
#define UINTAHSHARE
#endif
// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern UINTAHSHARE SCIRun::Mutex       cerrLock;
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
  TAU_PROFILE("LoadBalancerCommon::assignResources()", " ", TAU_USER);
  int nTasks = graph.numTasks();

  if( lbDebug.active() ) {
    cerrLock.lock();
    lbDebug << d_myworld->myrank() << " Assigning Tasks to Resources! (" << nTasks << " tasks)\n";
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
        task->assignResource(getOutputProc(patch));
      }
      else {
        task->assignResource(idx);
      }

      if( lbDebug.active() ) {
        cerrLock.lock();
        lbDebug << d_myworld->myrank() << " Task " << *(task->getTask()) << " put on resource "
          << idx << "\n";
        cerrLock.unlock();
      }
#if SCI_ASSERTION_LEVEL>0
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
#endif
    } else {
      if( Parallel::usingMPI() && task->getTask()->isReductionTask() ){
        task->assignResource( d_myworld->myrank() );

        if( lbDebug.active() ) {
          cerrLock.lock();
          lbDebug << d_myworld->myrank() << "  Resource (for no patch task) " << *task->getTask() << " is : " 
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
            lbDebug << d_myworld->myrank() << " OncePerProc Task " << *(task->getTask()) << " put on resource "
              << i << "\n";
          }
      } else {
        lbDebug << d_myworld->myrank() << " Unknown-type Task " << *(task->getTask()) << " put on resource "
          << 0 << "\n";
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

bool LoadBalancerCommon::possiblyDynamicallyReallocate(const GridP& grid, int state)
{
  if (state != check) {
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
      int proc = (static_cast<long long>(getPatchwiseProcessorAssignment(patch)) / static_cast<long long>(d_outputNthProc)) * d_outputNthProc;
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
  // Now maxGhost is from taskgraph 
  int maxGhost = d_scheduler->getMaxGhost();
  int maxLevelOffset = d_scheduler->getMaxLevelOffset();
  d_neighbors.clear();
  d_neighborProcessors.clear();
  
  //this processor should always be in the neighbhood
  d_neighborProcessors.insert(d_myworld->myrank());
 
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
      int outputproc = (static_cast<long long>(proc) / static_cast<long long>(d_outputNthProc))*d_outputNthProc;

      if(proc == me || oldproc == me || outputproc == me) {
        // one for current level, coarse level, find level, old level
        // each call to level->selectPatches must be done with an empty patch set
        // or otherwise it will conflict with the sorted order of the cached patches
        Patch::selectType neighbor;

        IntVector ghost(maxGhost,maxGhost,maxGhost);

        IntVector low(patch->getExtraLowIndex(Patch::CellBased, IntVector(0,0,0)));
        IntVector high(patch->getExtraHighIndex(Patch::CellBased, IntVector(0,0,0)));
        level->selectPatches(low-ghost, high+ghost, neighbor);
        for(int i=0;i<neighbor.size();i++) //add owning processors
        { 
          d_neighbors.insert(neighbor[i]->getRealPatch());
          int nproc=getPatchwiseProcessorAssignment(neighbor[i]);
          if(nproc>=0)
            d_neighborProcessors.insert(nproc);
          int oproc=getOldProcessorAssignment(0,neighbor[i],0);
          if(oproc>=0)
            d_neighborProcessors.insert(oproc);
        }
        if (d_sharedState->isCopyDataTimestep() && proc == me) {
          if (oldGrid->numLevels() > l) {
            // on copy data timestep we need old patches that line up with this proc's patches,
            // get the other way around at the end
            Patch::selectType old;
            const LevelP& oldLevel = oldGrid->getLevel(l);
            oldLevel->selectPatches(patch->getExtraCellLowIndex()-ghost, patch->getExtraCellHighIndex()+ghost, old);
            for(int i=0;i<old.size();i++) //add owning processors (they are the old owners)
            { 
              d_neighbors.insert(old[i]->getRealPatch());
              int nproc=getPatchwiseProcessorAssignment(old[i]);
              if(nproc>=0)
                d_neighborProcessors.insert(nproc);
              int oproc=getOldProcessorAssignment(0,old[i],0);
              if(oproc>=0)
                d_neighborProcessors.insert(oproc);
            }
          }
        }

        // add amr stuff - so the patch will know about coarsening and refining
        if (l > 0 && (proc == me || (oldproc == me && !d_sharedState->isCopyDataTimestep()))) {
          LevelP coarseLevel = level;
          IntVector ghost(maxGhost, maxGhost, maxGhost);
          for (int offset = 1; offset <= maxLevelOffset && coarseLevel->hasCoarserLevel(); ++offset) {
            ghost = ghost * coarseLevel->getRefinementRatio();
            coarseLevel = coarseLevel->getCoarserLevel();
            Patch::selectType coarse;

            coarseLevel->selectPatches(level->mapCellToCoarser(low - ghost, offset),
                    level->mapCellToCoarser(high + ghost, offset), coarse);
            for (int i = 0; i < coarse.size(); i++) //add owning processors
            {
              d_neighbors.insert(coarse[i]->getRealPatch());
              int nproc = getPatchwiseProcessorAssignment(coarse[i]);
              if (nproc >= 0)
                d_neighborProcessors.insert(nproc);
              int oproc = getOldProcessorAssignment(0, coarse[i], 0);
              if (oproc >= 0)
                d_neighborProcessors.insert(oproc);
            }
          }
        }
        if (l < grid->numLevels()-1 && (proc == me || (oldproc == me && !d_sharedState->isCopyDataTimestep()))) {
          IntVector ghost(maxGhost, maxGhost, maxGhost);
          const LevelP& fineLevel = level->getFinerLevel();
          Patch::selectType fine;
          fineLevel->selectPatches(level->mapCellToFiner(low-ghost), 
              level->mapCellToFiner(high+ghost), fine);
          for(int i=0;i<fine.size();i++) //add owning processors
          { 
            d_neighbors.insert(fine[i]->getRealPatch());
            int nproc=getPatchwiseProcessorAssignment(fine[i]);
            if(nproc>=0)
              d_neighborProcessors.insert(nproc);
            int oproc=getOldProcessorAssignment(0,fine[i],0);
            if(oproc>=0)
              d_neighborProcessors.insert(oproc);
          }
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
          newLevel->selectPatches(oldPatch->getExtraCellLowIndex()-ghost, oldPatch->getExtraCellHighIndex()+ghost, n);
          d_neighbors.insert(oldPatch);
          
          int nproc=getPatchwiseProcessorAssignment(oldPatch);
          if(nproc>=0)
            d_neighborProcessors.insert(nproc);
          int oproc=getOldProcessorAssignment(0,oldPatch,0);
          if(oproc>=0)
            d_neighborProcessors.insert(oproc);
          
          for(int i=0;i<(int)n.size();i++){
            d_neighbors.insert(n[i]->getRealPatch());
            int nproc=getPatchwiseProcessorAssignment(n[i]);
            if(nproc>=0)
              d_neighborProcessors.insert(nproc);
            int oproc=getOldProcessorAssignment(0,n[i],0);
            if(oproc>=0)
              d_neighborProcessors.insert(oproc);
          }
        }
      }
    }
  }
#if 0
  cout << d_myworld->myrank() << " np: ";
  for(set<int>::iterator iter=d_neighborProcessors.begin();iter!=d_neighborProcessors.end();iter++)
  {
    cout << *iter << " ";
  }
  cout << endl;
#endif

  if (neiDebug.active())
    for (std::set<const Patch*>::iterator iter = d_neighbors.begin(); iter != d_neighbors.end(); iter++)
      cout << d_myworld->myrank() << "  Neighborhood: " << (*iter)->getID() << " Proc " << getPatchwiseProcessorAssignment(*iter) << endl;

}

bool
LoadBalancerCommon::inNeighborhood(const PatchSubset* ps) 
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
LoadBalancerCommon::problemSetup(ProblemSpecP& pspec, GridP& grid, SimulationStateP& state)
{
  d_sharedState = state;
  d_scheduler = dynamic_cast<Scheduler*>(getPort("scheduler"));
  ProblemSpecP p = pspec->findBlock("LoadBalancer");
  d_outputNthProc = 1;
  
  if (p != 0) {
    p->getWithDefault("outputNthProc", d_outputNthProc, 1);
  }
}
