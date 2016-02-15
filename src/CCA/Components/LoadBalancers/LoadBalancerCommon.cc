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

#include <CCA/Components/LoadBalancers/LoadBalancerCommon.h>

#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/DataArchive/DataArchive.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/NotFinished.h>

#include <sci_values.h>
#include <sstream>

using namespace Uintah;
using namespace std;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern SCIRun::Mutex cerrLock;

DebugStream lbDebug( "LoadBalancer", false );
DebugStream neiDebug("Neighborhood", false );

// If defined, the space-filling curve will be computed in parallel,
// this may not be a good idea because the time to compute the
// space-filling curve is so small that it might not parallelize well.
#define SFC_PARALLEL

LoadBalancerCommon::LoadBalancerCommon( const ProcessorGroup * myworld ) :
  UintahParallelComponent( myworld ), d_sfc( myworld )
{
}

LoadBalancerCommon::~LoadBalancerCommon()
{
}
//______________________________________________________________________
//
void
LoadBalancerCommon::assignResources( DetailedTasks & graph )
{
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
        for(set<int>::iterator p=d_neighborProcessors.begin();p!=d_neighborProcessors.end();p++) {
          int i=(*p);
          if (patches == task->getTask()->getPatchSet()->getSubset(i)) {
            task->assignResource(i);
            
            if( lbDebug.active() ) {
              cerrLock.lock();
              lbDebug << d_myworld->myrank() << " OncePerProc Task " << *(task->getTask()) << " put on resource "
                       << i << "\n";
              cerrLock.unlock();
            }
          }
        }
      } else {
        if( lbDebug.active() ) {
          cerrLock.lock();
          lbDebug << d_myworld->myrank() << " Unknown-type Task " << *(task->getTask()) << " put on resource "
                  << 0 << "\n";
          cerrLock.unlock();
        }
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
//______________________________________________________________________
//
int
LoadBalancerCommon::getPatchwiseProcessorAssignment( const Patch * patch )
{
  // If on a copy-data timestep and we ask about an old patch, that could cause problems.
  if( d_sharedState->isCopyDataTimestep() && patch->getRealPatch()->getID() < d_assignmentBasePatch ) {
    return -patch->getID();
  }
 
  ASSERTRANGE( patch->getRealPatch()->getID(), d_assignmentBasePatch, d_assignmentBasePatch + (int) d_processorAssignment.size() );
  int proc = d_processorAssignment[ patch->getRealPatch()->getGridIndex() ];

  ASSERTRANGE(proc, 0, d_myworld->size());
  return proc;
}
//______________________________________________________________________
//
int
LoadBalancerCommon::getOldProcessorAssignment( const Patch * patch )
{
  // At one point, the var label was a parameter to this function, but it was not actually
  // passed in anywhere, so at least for now, this is commented out:
  //
  // if (var && var->typeDescription()->isReductionVariable()) {
  //    return d_myworld->myrank();
  // }

  // On an initial-regrid-timestep, this will get called from createNeighborhood
  // and can have a patch with a higher index than we have.
  if ((int)patch->getRealPatch()->getID() < d_oldAssignmentBasePatch || patch->getRealPatch()->getID() >= d_oldAssignmentBasePatch + (int)d_oldAssignment.size()) {
    return -9999;
  }
  
  if (patch->getGridIndex() >= (int) d_oldAssignment.size()) {
    return -999;
  }

  int proc = d_oldAssignment[patch->getRealPatch()->getGridIndex()];
  ASSERTRANGE( proc, 0, d_myworld->size() );
  return proc;
}
//______________________________________________________________________
//
void
LoadBalancerCommon::useSFC( const LevelP & level, int * order )
{
  vector<DistributedIndex> indices; //output
  vector<double> positions;

  //this should be removed when dimensions in shared state is done
  int dim=d_sharedState->getNumDims();
  int *dimensions=d_sharedState->getActiveDims();

  IntVector min_patch_size(INT_MAX,INT_MAX,INT_MAX);  

  // get the overall range in all dimensions from all patches
  IntVector high(INT_MIN,INT_MIN,INT_MIN);
  IntVector low(INT_MAX,INT_MAX,INT_MAX);
#ifdef SFC_PARALLEL 
  vector<int> originalPatchCount(d_myworld->size(),0); //store how many patches each patch has originally
#endif
  for (Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
    const Patch* patch = *iter;
   
    //calculate patchset bounds
    high = Max(high, patch->getCellHighIndex());
    low  = Min(low, patch->getCellLowIndex());
    
    //calculate minimum patch size
    IntVector size = patch->getCellHighIndex()-patch->getCellLowIndex();
    min_patch_size = min(min_patch_size,size);
    
    //create positions vector

#ifdef SFC_PARALLEL
    //place in long longs to avoid overflows with large numbers of patches and processors
    long long pindex = patch->getLevelIndex();
    long long num_patches = d_myworld->size();
    long long proc = (pindex*num_patches) /(long long)level->numPatches();

    ASSERTRANGE(proc,0,d_myworld->size());
    if(d_myworld->myrank()==(int)proc) {
      Vector point=(patch->getCellLowIndex()+patch->getCellHighIndex()).asVector()/2.0;
      for(int d=0;d<dim;d++) {
        positions.push_back(point[dimensions[d]]);
      }
    }
    originalPatchCount[proc]++;
#else
    Vector point=(patch->getCellLowIndex()+patch->getCellHighIndex()).asVector()/2.0;
    for(int d=0;d<dim;d++) {
      positions.push_back(point[dimensions[d]]);
    }
#endif
  }

#ifdef SFC_PARALLEL
  //compute patch starting locations
  vector<int> originalPatchStart(d_myworld->size(),0);
  for(int p=1;p<d_myworld->size();p++) {
    originalPatchStart[p]=originalPatchStart[p-1]+originalPatchCount[p-1];
  }
#endif

  // Patchset dimensions
  IntVector range = high-low;
  
  // Center of patchset
  Vector center = (high+low).asVector()/2.0;
 
  double r[3]     = {(double)range[dimensions[0]], (double)range[dimensions[1]], (double)range[dimensions[2]]};
  double c[3]     = {(double)center[dimensions[0]],(double)center[dimensions[1]], (double)center[dimensions[2]]};
  double delta[3] = {(double)min_patch_size[dimensions[0]], (double)min_patch_size[dimensions[1]], (double)min_patch_size[dimensions[2]]};

  // Create SFC
  d_sfc.SetDimensions(r);
  d_sfc.SetCenter(c);
  d_sfc.SetRefinementsByDelta(delta); 
  d_sfc.SetLocations(&positions);
  d_sfc.SetOutputVector(&indices);
  
#ifdef SFC_PARALLEL
  d_sfc.SetLocalSize(originalPatchCount[d_myworld->myrank()]);
  d_sfc.GenerateCurve();
#else
  d_sfc.SetLocalSize(level->numPatches());
  d_sfc.GenerateCurve(SERIAL);
#endif
  
#ifdef SFC_PARALLEL
  if( d_myworld->size() > 1 ) {
    vector<int> recvcounts(d_myworld->size(), 0);
    vector<int> displs(d_myworld->size(), 0);
    
    for (unsigned i = 0; i < recvcounts.size(); i++) {
      displs[i]=originalPatchStart[i]*sizeof(DistributedIndex);
      if( displs[i] < 0 ) {
        throw InternalError("Displacments < 0",__FILE__,__LINE__);
      }
      recvcounts[i]=originalPatchCount[i]*sizeof(DistributedIndex);
      if( recvcounts[i] < 0 ) {
        throw InternalError("Recvcounts < 0",__FILE__,__LINE__);
      }
    }

    vector<DistributedIndex> rbuf(level->numPatches());

    // Gather curve
    MPI_Allgatherv(&indices[0], recvcounts[d_myworld->myrank()], MPI_BYTE, &rbuf[0], &recvcounts[0], 
                   &displs[0], MPI_BYTE, d_myworld->getComm());

    indices.swap(rbuf);
  
  }

  // Convert distributed indices to normal indices.
  for(unsigned int i=0;i<indices.size();i++) {
    DistributedIndex di=indices[i];
    order[i]=originalPatchStart[di.p]+di.i;
  }
#else
  // Write order array
  for(unsigned int i=0;i<indices.size();i++) {
    order[i]=indices[i].i;
  }
#endif

#if 0
  cout << "SFC order: ";
  for (int i = 0; i < level->numPatches(); i++) {
    cout << order[i] << " ";
  }
  cout << endl;
#endif
#if 0
  if(d_myworld->myrank()==0) {
    cout << "Warning checking SFC correctness\n";
  }
  for (int i = 0; i < level->numPatches(); i++) {
    for (int j = i+1; j < level->numPatches(); j++) {
      if (order[i] == order[j]) 
      {
        cout << "Rank:" << d_myworld->myrank() <<  ":   ALERT!!!!!! index done twice: index " << i << " has the same value as index " << j << " " << order[i] << endl;
        throw InternalError("SFC unsuccessful", __FILE__, __LINE__);
      }
    }
  }
#endif
}
//______________________________________________________________________
//
void
LoadBalancerCommon::restartInitialize(       DataArchive  * archive,
                                       const int            time_index,
                                       const string       & ts_url,
                                       const GridP        & grid )
{
  // Here we need to grab the uda data to reassign patch data to the processor that will get the data.
  int num_patches = 0;
  const Patch* first_patch = *(grid->getLevel(0)->patchesBegin());
  int startingID = first_patch->getID();
  int prevNumProcs = 0;

  for( int l = 0; l < grid->numLevels(); l++ ) {
    const LevelP& level = grid->getLevel(l);
    num_patches += level->numPatches();
  }

  d_processorAssignment.resize(num_patches);
  d_assignmentBasePatch = startingID;
  for( unsigned int i = 0; i < d_processorAssignment.size(); i++ ) {
    d_processorAssignment[i]= -1;
  }

  if( archive->queryPatchwiseProcessor( first_patch, time_index ) != -1 ) {
    // for uda 1.1 - if proc is saved with the patches
    for( int l = 0; l < grid->numLevels(); l++ ) {
      const LevelP& level = grid->getLevel(l);
      for (Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
        d_processorAssignment[(*iter)->getID()-startingID] = archive->queryPatchwiseProcessor(*iter, time_index) % d_myworld->size();
      }
    }
  } // end queryPatchwiseProcessor
  else {
    // Before uda 1.1 - DELETED THIS CODE - we don't support pre 1.1 UDAs any more.
    throw InternalError( "LoadBalancerCommon::restartInitialize() - UDA too old...", __FILE__, __LINE__);
  }
  for (unsigned i = 0; i < d_processorAssignment.size(); i++) {
    if (d_processorAssignment[i] == -1) {
      cout << "index " << i << " == -1\n";
    }
    ASSERT(d_processorAssignment[i] != -1);
  }
  
  d_oldAssignment = d_processorAssignment;
  d_oldAssignmentBasePatch = d_assignmentBasePatch;

  if (prevNumProcs != d_myworld->size() || d_outputNthProc > 1) {
    if (d_myworld->myrank() == 0){
      lbDebug << "  Original run had " << prevNumProcs << ", this has " << d_myworld->size() << endl;
    }
    d_checkAfterRestart = true;
  }

  if (d_myworld->myrank() == 0) {
    lbDebug << d_myworld->myrank() << " check after restart: " << d_checkAfterRestart << "\n";
#if 0
    int startPatch = (int) (*grid->getLevel(0)->patchesBegin())->getID();
    if (lb.active()) {
      for (unsigned i = 0; i < d_processorAssignment.size(); i++) {
        lb <<d_myworld-> myrank() << " patch " << i << " (real " << i+startPatch << ") -> proc " 
           << d_processorAssignment[i] << " (old " << d_oldAssignment[i] << ") - " 
           << d_processorAssignment.size() << ' ' << d_oldAssignment.size() << "\n";
      }
    }
#endif
  }
} // end restartInitialize()
//______________________________________________________________________
//
bool
LoadBalancerCommon::possiblyDynamicallyReallocate( const GridP & grid, int state )
{
  if( state != check ) {
    // Have it create a new patch set, and have the DLB version call this.
    // This is a good place to do it, as it is automatically called when the
    // grid changes.
    d_levelPerProcPatchSets.clear();
    d_outputPatchSets.clear();
    d_gridPerProcPatchSet = createPerProcessorPatchSet( grid );

    for( int i = 0; i < grid->numLevels(); i++ ) {
      d_levelPerProcPatchSets.push_back( createPerProcessorPatchSet( grid->getLevel(i) ) );
      d_outputPatchSets.push_back( createOutputPatchSet( grid->getLevel(i) ) );
    }
  }
  return false;
}

//______________________________________________________________________
//
// Creates a PatchSet containing PatchSubsets for each processor for a single level.
const PatchSet*
LoadBalancerCommon::createPerProcessorPatchSet( const LevelP & level )
{
  PatchSet* patches = scinew PatchSet();
  patches->createEmptySubsets(d_myworld->size());
  
  for(Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++){
    const Patch* patch = *iter;
    int proc = getPatchwiseProcessorAssignment(patch);
    ASSERTRANGE(proc, 0, d_myworld->size());
    PatchSubset* subset = patches->getSubset(proc);
    subset->add(patch);
  }
  patches->sortSubsets();  
  return patches;
}

//______________________________________________________________________
//
// Creates a PatchSet containing PatchSubsets for each processor for an
// entire grid.
const PatchSet*
LoadBalancerCommon::createPerProcessorPatchSet( const GridP & grid )
{
  PatchSet* patches = scinew PatchSet();
  patches->createEmptySubsets(d_myworld->size());
  
  for( int i = 0; i < grid->numLevels(); i++ ) {
    const LevelP level = grid->getLevel(i);
    
    for( Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++ ) {
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

//______________________________________________________________________
//
const PatchSet* 
LoadBalancerCommon::createOutputPatchSet(const LevelP& level)
{
  if (d_outputNthProc == 1) {
    // assume the perProcessor set on the level was created first
    return d_levelPerProcPatchSets[level->getIndex()].get_rep();
  }
  else {
    PatchSet* patches = scinew PatchSet();
    patches->createEmptySubsets(d_myworld->size());
    
    for(Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++){
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

//______________________________________________________________________
//
void
LoadBalancerCommon::createNeighborhood(const GridP& grid, const GridP& oldGrid)
{
  int me = d_myworld->myrank();
  // TODO consider this old warning from Steve:
  //    WARNING - this should be determined from the taskgraph? - Steve

  // get the max level offset and max ghost cells to consider for neighborhood creation
  int maxGhost = d_scheduler->getMaxGhost();
  int maxLevelOffset = d_scheduler->getMaxLevelOffset();
    // TODO replace after Mira DDT problem is debugged (APH - 03/24/15)
//  const std::map<int, int>& maxGhostCells = d_scheduler->getMaxGhostCells();
//  const std::map<int, int>& maxLevelOffsets = d_scheduler->getMaxLevelOffsets();

  d_neighbors.clear();
  d_neighborProcessors.clear();
  
  //this processor should always be in the neighborhood
  d_neighborProcessors.insert(d_myworld->myrank());
 
  // go through all patches on all levels, and if the patch-wise
  // processor assignment equals the current processor, then store the 
  // patch's neighbors in the load balancer array
  for( int l = 0; l < grid->numLevels(); l++ ) {
    LevelP level = grid->getLevel(l);

    // TODO replace after Mira DDT problem is debugged (APH - 03/24/15)
//    // determine max ghost cells and max level offset for the current level
//    int maxGC = maxGhostCells.find(l)->second;
//    int maxOffset = maxLevelOffsets.find(l)->second;

    for(Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
      const Patch* patch = *iter;

      // we need to check both where the patch is and where
      // it used to be (in the case of a dynamic reallocation)
      int proc    = getPatchwiseProcessorAssignment( patch );
      int oldproc = getOldProcessorAssignment( patch );

      // we also need to see if the output processor for patch is this proc,
      // in case it wouldn't otherwise have been in the neighborhood
      int outputproc = (static_cast<long long>(proc) / static_cast<long long>(d_outputNthProc))*d_outputNthProc;

      if(proc == me || oldproc == me || outputproc == me) {
        // one for current level, coarse level, find level, old level
        // each call to level->selectPatches must be done with an empty patch set
        // or otherwise it will conflict with the sorted order of the cached patches
        Patch::selectType neighbor;

        // TODO replace after Mira DDT problem is debugged (APH - 03/24/15)
        IntVector ghost(maxGhost,maxGhost,maxGhost);
//        IntVector ghost(maxGC,maxGC,maxGC);

        IntVector low( patch->getExtraLowIndex( Patch::CellBased, IntVector(0,0,0)));
        IntVector high(patch->getExtraHighIndex(Patch::CellBased, IntVector(0,0,0)));
        level->selectPatches(low-ghost, high+ghost, neighbor);
        
        for (int i = 0; i < neighbor.size(); i++)  //add owning processors
            {
          d_neighbors.insert(neighbor[i]->getRealPatch());
          int nproc = getPatchwiseProcessorAssignment(neighbor[i]);
          if (nproc >= 0) {
            d_neighborProcessors.insert(nproc);
          }

          int oproc = getOldProcessorAssignment(neighbor[i]);
          if (oproc >= 0) {
            d_neighborProcessors.insert(oproc);
          }
        }
        
        if (d_sharedState->isCopyDataTimestep() && proc == me) {
          if (oldGrid->numLevels() > l) {
            // on copy data timestep we need old patches that line up with this proc's patches,
            // get the other way around at the end
            Patch::selectType old;
            const LevelP& oldLevel = oldGrid->getLevel(l);
            oldLevel->selectPatches(patch->getExtraCellLowIndex() - ghost, patch->getExtraCellHighIndex() + ghost, old);
            for (int i = 0; i < old.size(); i++)  //add owning processors (they are the old owners)
                {
              d_neighbors.insert(old[i]->getRealPatch());
              int nproc = getPatchwiseProcessorAssignment(old[i]);
              if (nproc >= 0) {
                d_neighborProcessors.insert(nproc);
              }
              int oproc = getOldProcessorAssignment(old[i]);
              if (oproc >= 0) {
                d_neighborProcessors.insert(oproc);
              }
            }
          }
        }

        // add AMR stuff - so the patch will know about coarsening and refining
        if (l > 0 && (proc == me || (oldproc == me && !d_sharedState->isCopyDataTimestep()))) {
          LevelP coarseLevel = level;

          // TODO replace after Mira DDT problem is debugged (APH - 03/24/15)
          IntVector ghost(maxGhost, maxGhost, maxGhost);
          for (int offset = 1; offset <= maxLevelOffset && coarseLevel->hasCoarserLevel(); ++offset) {
//          IntVector ghost(maxGC, maxGC, maxGC);
//          for (int offset = 1; offset <= maxOffset && coarseLevel->hasCoarserLevel(); ++offset) {          
            ghost = ghost * coarseLevel->getRefinementRatio();
            coarseLevel = coarseLevel->getCoarserLevel();
            Patch::selectType coarse;

            coarseLevel->selectPatches(level->mapCellToCoarser(low - ghost, offset), level->mapCellToCoarser(high + ghost, offset),
                                       coarse);
            for (int i = 0; i < coarse.size(); i++)  //add owning processors
                {
              d_neighbors.insert(coarse[i]->getRealPatch());

              int nproc = getPatchwiseProcessorAssignment(coarse[i]);
              if (nproc >= 0) {
                d_neighborProcessors.insert(nproc);
              }

              int oproc = getOldProcessorAssignment(coarse[i]);
              if (oproc >= 0) {
                d_neighborProcessors.insert(oproc);
              }
            }
          }
        }

        if (l < grid->numLevels() - 1 && (proc == me || (oldproc == me && !d_sharedState->isCopyDataTimestep()))) {

          // TODO replace after Mira DDT problem is debugged (APH - 03/24/15)
          IntVector ghost(maxGhost, maxGhost, maxGhost);
//          IntVector ghost(maxGC, maxGC, maxGC);
          const LevelP& fineLevel = level->getFinerLevel();
          Patch::selectType fine;
          fineLevel->selectPatches(level->mapCellToFiner(low - ghost), level->mapCellToFiner(high + ghost), fine);
          for (int i = 0; i < fine.size(); i++) {  //add owning processors
            d_neighbors.insert(fine[i]->getRealPatch());
            int nproc = getPatchwiseProcessorAssignment(fine[i]);
            if (nproc >= 0) {
              d_neighborProcessors.insert(nproc);
            }
            int oproc = getOldProcessorAssignment(fine[i]);
            if (oproc >= 0) {
              d_neighborProcessors.insert(oproc);
            }
          }
        }
      }
    }
  }

  if (d_sharedState->isCopyDataTimestep()) {
    // Regrid timestep postprocess 
    // 1)- go through the old grid and 
    //     find which patches used to be on this proc 
    for (int l = 0; l < oldGrid->numLevels(); l++) {
      if (grid->numLevels() <= l) {
        continue;
      }

      // TODO replace after Mira DDT problem is debugged (APH - 03/24/15)
      IntVector ghost(maxGhost, maxGhost, maxGhost);
//      int maxGC = maxGhostCells.find(l)->second;
//      IntVector ghost(maxGC, maxGC, maxGC);

      LevelP oldLevel = oldGrid->getLevel(l);
      LevelP newLevel = grid->getLevel(l);

      for (Level::const_patchIterator iter = oldLevel->patchesBegin(); iter != oldLevel->patchesEnd(); iter++) {
        const Patch* oldPatch = *iter;

        // we need to check both where the patch is and where
        // it used to be (in the case of a dynamic reallocation)
        int oldproc = getOldProcessorAssignment(oldPatch);

        if (oldproc == me) {
          // don't get extra cells or ghost cells
          Patch::selectType n;
          newLevel->selectPatches(oldPatch->getExtraCellLowIndex() - ghost, oldPatch->getExtraCellHighIndex() + ghost, n);
          d_neighbors.insert(oldPatch);

          int nproc = getPatchwiseProcessorAssignment(oldPatch);
          if (nproc >= 0) {
            d_neighborProcessors.insert(nproc);
          }

          int oproc = getOldProcessorAssignment(oldPatch);
          if (oproc >= 0) {
            d_neighborProcessors.insert(oproc);
          }

          for (int i = 0; i < (int)n.size(); i++) {
            d_neighbors.insert(n[i]->getRealPatch());

            int nproc = getPatchwiseProcessorAssignment(n[i]);
            if (nproc >= 0) {
              d_neighborProcessors.insert(nproc);
            }

            int oproc = getOldProcessorAssignment(n[i]);
            if (oproc >= 0) {
              d_neighborProcessors.insert(oproc);
            }
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

  if (neiDebug.active()) {
    for (std::set<const Patch*>::iterator iter = d_neighbors.begin(); iter != d_neighbors.end(); iter++) {
      cout << d_myworld->myrank() << "  Neighborhood: " << (*iter)->getID() << " Proc " << getPatchwiseProcessorAssignment(*iter)
           << endl;
    }
  }
} // end createNeighborhood()

//______________________________________________________________________
//
bool
LoadBalancerCommon::inNeighborhood(const PatchSubset* ps) 
{
  for (int i = 0; i < ps->size(); i++) {
    const Patch* patch = ps->get(i);
    if (d_neighbors.find(patch) != d_neighbors.end())
      return true;
  }
  // also count a subset with no patches
  return ps->size() == 0;
}

//______________________________________________________________________
//
bool
LoadBalancerCommon::inNeighborhood(const Patch* patch)
{
  if (d_neighbors.find(patch) != d_neighbors.end()) {
    return true;
  } else {
    return false;
  }
}

//______________________________________________________________________
//
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

//______________________________________________________________________
// Cost profiling functions
void
LoadBalancerCommon::addContribution( DetailedTask * task ,double cost )
{
  static bool warned = false;
  if( !warned ) {
//    proc0cout << "Warning: addContribution not implemented for LoadBalancerCommon.\n";
    warned = true;
  }
}

//______________________________________________________________________  
// Finalize the contributions (updates the weight, should be called once per timestep):
void
LoadBalancerCommon::finalizeContributions( const GridP & currentGrid )
{
  static bool warned = false;
  if( !warned ) {
//    proc0cout << "Warning: finalizeContributions not implemented for LoadBalancerCommon.\n";
    warned = true;
  }
}

//______________________________________________________________________
// Initializes the regions in the new level that are not in the old level.
void
LoadBalancerCommon::initializeWeights(const Grid* oldgrid, const Grid* newgrid)
{
  static bool warned = false;
  if( !warned ) {
//    proc0cout << "Warning: initializeWeights not implemented for LoadBalancerCommon.\n";
    warned = true;
  }
}

//______________________________________________________________________
// Resets the profiler counters to zero
void
LoadBalancerCommon::resetCostForecaster()
{
  static bool warned = false;
  if( !warned ) {
//    proc0cout << "Warning: resetCostForecaster not implemented for LoadBalancerCommon.\n";
    warned = true;
  }
}
