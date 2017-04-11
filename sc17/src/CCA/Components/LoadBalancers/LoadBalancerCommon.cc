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
#include <Core/Util/DebugStream.h>
#include <Core/Util/DOUT.hpp>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/NotFinished.h>

#include <sci_defs/visit_defs.h>

#include <cfloat>
#include <climits>
#include <mutex>
#include <sstream>

using namespace Uintah;

namespace {

Dout g_lb_dbg(           "LoadBalancer", false );
Dout g_neighborhood_dbg( "Neighborhood", false );

std::mutex g_pidx_mutex{};

}

// If defined, the space-filling curve will be computed in parallel,
// this may not be a good idea because the time to compute the
// space-filling curve is so small that it might not parallelize well.
#define SFC_PARALLEL


//______________________________________________________________________
//
LoadBalancerCommon::LoadBalancerCommon( const ProcessorGroup * myworld )
  : UintahParallelComponent( myworld )
  , m_sfc( myworld )
{
}

//______________________________________________________________________
//
LoadBalancerCommon::~LoadBalancerCommon()
{
}

//______________________________________________________________________
//
void
LoadBalancerCommon::assignResources( DetailedTasks & graph )
{
  int nTasks = graph.numTasks();

  DOUT(g_lb_dbg, d_myworld->myrank() << " Assigning Tasks to Resources! (" << nTasks << " tasks)");

  for(int i=0;i<nTasks;i++){
    DetailedTask* task = graph.getTask(i);

    const PatchSubset* patches = task->getPatches();
    if(patches && patches->size() > 0 && task->getTask()->getType() != Task::OncePerProc){
      const Patch* patch = patches->get(0);

      int idx = getPatchwiseProcessorAssignment(patch);
      ASSERTRANGE(idx, 0, d_myworld->size());

      if (task->getTask()->getType() == Task::Output) {
        task->assignResource( getOutputRank( patch ) );
      }
      else {
        task->assignResource(idx);
      }

      DOUT(g_lb_dbg, d_myworld->myrank() << " Task " << *(task->getTask()) << " put on resource " << idx);

#if SCI_ASSERTION_LEVEL > 0
      std::ostringstream ostr;
      ostr << patch->getID() << ':' << idx;

      for(int i=1;i<patches->size();i++){
        const Patch* p = patches->get(i);
        int pidx = getPatchwiseProcessorAssignment(p);
        ostr << ' ' << p->getID() << ';' << pidx;
        ASSERTRANGE(pidx, 0, d_myworld->size());
        
        if (pidx != idx && task->getTask()->getType() != Task::Output) {
          {
            std::lock_guard<std::mutex> lock(g_pidx_mutex);
            std::cerr << d_myworld->myrank() << " WARNING: inconsistent task (" << task->getTask()->getName()
                      << ") assignment (" << pidx << ", " << idx << ") in LoadBalancerCommon" << std::endl;;
          }
        }
      }
#endif
    } else {
      if( Parallel::usingMPI() && task->getTask()->isReductionTask() ){
        task->assignResource( d_myworld->myrank() );

        DOUT(g_lb_dbg, d_myworld->myrank() << "  Resource (for no patch task) " << *task->getTask() << " is : " << d_myworld->myrank());

      } else if( task->getTask()->getType() == Task::InitialSend){
        // Already assigned, do nothing
        ASSERT(task->getAssignedResourceIndex() != -1);
      } else if( task->getTask()->getType() == Task::OncePerProc) {
      
        // patch-less task, not execute-once, set to run on all procs
        // once per patch subset (empty or not)
        // at least one example is the multi-level (impAMRICE) pressureSolve
        for(std::set<int>::iterator p=m_neighborhood_processors.begin(); p != m_neighborhood_processors.end(); p++) {
          int i=(*p);
          if (patches == task->getTask()->getPatchSet()->getSubset(i)) {
            task->assignResource(i);
            DOUT(g_lb_dbg, d_myworld->myrank() << " OncePerProc Task " << *(task->getTask()) << " put on resource " << i);
          }
        }
      } else {
        task->assignResource(0);
        DOUT(g_lb_dbg, d_myworld->myrank() << " Unknown-type Task " << *(task->getTask()) << " put on resource " << 0);
      }
    }
  }
}

//______________________________________________________________________
//
int
LoadBalancerCommon::getPatchwiseProcessorAssignment( const Patch * patch )
{
  // If on a copy-data timestep and we ask about an old patch, that could cause problems.
  if( m_shared_state->isCopyDataTimestep() && patch->getRealPatch()->getID() < m_assignment_base_patch ) {
    return -patch->getID();
  }
 
  ASSERTRANGE( patch->getRealPatch()->getID(), m_assignment_base_patch, m_assignment_base_patch + static_cast<int>(m_processor_assignment.size()) );
  int rank = m_processor_assignment[ patch->getRealPatch()->getGridIndex() ];

  ASSERTRANGE( rank, 0, d_myworld->size() );
  return rank;
}

//______________________________________________________________________
//
int
LoadBalancerCommon::getOldProcessorAssignment( const Patch * patch )
{

  // On an initial-regrid-timestep, this will get called from createNeighborhood
  // and can have a patch with a higher index than we have.
  if ( static_cast<int>(patch->getRealPatch()->getID()) < m_old_assignment_base_patch ||
      patch->getRealPatch()->getID() >= m_old_assignment_base_patch + static_cast<int>(m_old_assignment.size()) ) {
    return -9999;
  }
  
  if (patch->getGridIndex() >= (int) m_old_assignment.size()) {
    return -999;
  }

  int proc = m_old_assignment[patch->getRealPatch()->getGridIndex()];
  ASSERTRANGE( proc, 0, d_myworld->size() );

  return proc;
}

//______________________________________________________________________
//
void
LoadBalancerCommon::useSFC( const LevelP & level, int * order )
{
  std::vector<DistributedIndex> indices; //output
  std::vector<double> positions;

  //this should be removed when dimensions in shared state is done
  int dim=m_shared_state->getNumDims();
  int *dimensions=m_shared_state->getActiveDims();

  IntVector min_patch_size(INT_MAX,INT_MAX,INT_MAX);  

  // get the overall range in all dimensions from all patches
  IntVector high(INT_MIN,INT_MIN,INT_MIN);
  IntVector low(INT_MAX,INT_MAX,INT_MAX);

#ifdef SFC_PARALLEL 
  std::vector<int> originalPatchCount(d_myworld->size(),0); //store how many patches each patch has originally
#endif

  for (Level::const_patch_iterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
    const Patch* patch = *iter;
   
    //calculate patchset bounds
    high = Max(high, patch->getCellHighIndex());
    low  = Min(low, patch->getCellLowIndex());
    
    //calculate minimum patch size
    IntVector size = patch->getCellHighIndex()-patch->getCellLowIndex();
    min_patch_size = std::min(min_patch_size,size);
    
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
  std::vector<int> originalPatchStart(d_myworld->size(),0);
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
  m_sfc.SetDimensions(r);
  m_sfc.SetCenter(c);
  m_sfc.SetRefinementsByDelta(delta); 
  m_sfc.SetLocations(&positions);
  m_sfc.SetOutputVector(&indices);
  
#ifdef SFC_PARALLEL
  m_sfc.SetLocalSize(originalPatchCount[d_myworld->myrank()]);
  m_sfc.GenerateCurve();
#else
  m_sfc.SetLocalSize(level->numPatches());
  m_sfc.GenerateCurve(SERIAL);
#endif
  
#ifdef SFC_PARALLEL
  if( d_myworld->size() > 1 ) {
    std::vector<int> recvcounts(d_myworld->size(), 0);
    std::vector<int> displs(d_myworld->size(), 0);
    
    for (unsigned i = 0; i < recvcounts.size(); i++) {
      displs[i]=originalPatchStart[i]*sizeof(DistributedIndex);
      if( displs[i] < 0 ) {
        throw InternalError("Displacements < 0",__FILE__,__LINE__);
      }
      recvcounts[i]=originalPatchCount[i]*sizeof(DistributedIndex);
      if( recvcounts[i] < 0 ) {
        throw InternalError("Recvcounts < 0",__FILE__,__LINE__);
      }
    }

    std::vector<DistributedIndex> rbuf(level->numPatches());

    // Gather curve
    Uintah::MPI::Allgatherv(&indices[0], recvcounts[d_myworld->myrank()], MPI_BYTE, &rbuf[0],
                            &recvcounts[0], &displs[0], MPI_BYTE, d_myworld->getComm());

    indices.swap(rbuf);
  
  }

  // Convert distributed indices to normal indices.
  for (unsigned int i = 0; i < indices.size(); i++) {
    DistributedIndex di = indices[i];
    order[i] = originalPatchStart[di.p] + di.i;
  }
#else
  // Write order array
  for(unsigned int i=0;i<indices.size();i++) {
    order[i]=indices[i].i;
  }
#endif

#if 0
  std::cout << "SFC order: ";
  for (int i = 0; i < level->numPatches(); i++) {
    std::cout << order[i] << " ";
  }
  std::cout << std::endl;
#endif

#if 0
  if(d_myworld->myrank()==0) {
    std::cout << "Warning checking SFC correctness\n";
  }
  for (int i = 0; i < level->numPatches(); i++) {
    for (int j = i+1; j < level->numPatches(); j++) {
      if (order[i] == order[j]) 
      {
        std::cout << "Rank:" << d_myworld->myrank() <<  ":   ALERT!!!!!! index done twice: index "
                  << i << " has the same value as index " << j << " " << order[i] << std::endl;
        throw InternalError("SFC unsuccessful", __FILE__, __LINE__);
      }
    }
  }
#endif
}

//______________________________________________________________________
//
void
LoadBalancerCommon::restartInitialize(       DataArchive  * archive
                                     , const int            time_index
                                     , const std::string  & ts_url
                                     , const GridP        & grid
                                     )
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

  m_processor_assignment.resize(num_patches);
  m_assignment_base_patch = startingID;
  for( unsigned int i = 0; i < m_processor_assignment.size(); i++ ) {
    m_processor_assignment[i]= -1;
  }

  if( archive->queryPatchwiseProcessor( first_patch, time_index ) != -1 ) {
    // for uda 1.1 - if proc is saved with the patches
    for( int l = 0; l < grid->numLevels(); l++ ) {
      const LevelP& level = grid->getLevel(l);
      for (Level::const_patch_iterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
        m_processor_assignment[(*iter)->getID()-startingID] = archive->queryPatchwiseProcessor(*iter, time_index) % d_myworld->size();
      }
    }
  } // end queryPatchwiseProcessor
  else {
    // Before uda 1.1 - DELETED THIS CODE - we don't support pre 1.1 UDAs any more.
    throw InternalError( "LoadBalancerCommon::restartInitialize() - UDA too old...", __FILE__, __LINE__);
  }
  for (unsigned i = 0; i < m_processor_assignment.size(); i++) {
    if (m_processor_assignment[i] == -1) {
      std::cout << "index " << i << " == -1\n";
    }
    ASSERT(m_processor_assignment[i] != -1);
  }
  
  m_old_assignment = m_processor_assignment;
  m_old_assignment_base_patch = m_assignment_base_patch;

  if (prevNumProcs != d_myworld->size() || m_output_Nth_proc > 1) {
    if (d_myworld->myrank() == 0){
      DOUT(g_lb_dbg, "  Original run had " << prevNumProcs << ", this has " << d_myworld->size());
    }
    m_check_after_restart = true;
  }

  if (d_myworld->myrank() == 0) {
    DOUT(g_lb_dbg, d_myworld->myrank() << " check after restart: " << m_check_after_restart);

#if 0
    int startPatch = (int) (*grid->getLevel(0)->patchesBegin())->getID();
    std::ostringstream message;
      for (unsigned i = 0; i < m_processor_assignment.size(); i++) {
        message << d_myworld-> myrank() << " patch " << i << " (real " << i+startPatch << ") -> proc "
                << m_processor_assignment[i] << " (old " << m_old_assignment[i] << ") - "
                << m_processor_assignment.size() << ' ' << m_old_assignment.size() << "\n";
      }
      DOUT(true, message.str();)
#endif
  }
} // end restartInitialize()

//______________________________________________________________________
//
bool
LoadBalancerCommon::possiblyDynamicallyReallocate( const GridP & grid, int state )
{
  if (state != check) {
    // Have it create a new patch set, and have the DLB version call this.
    // This is a good place to do it, as it is automatically called when the
    // grid changes.
    m_level_perproc_patchsets.clear();
    m_output_patchsets.clear();
    m_grid_perproc_patchsets = createPerProcessorPatchSet(grid);

    for (int i = 0; i < grid->numLevels(); i++) {
      m_level_perproc_patchsets.push_back(createPerProcessorPatchSet(grid->getLevel(i)));
      m_output_patchsets.push_back(createOutputPatchSet(grid->getLevel(i)));
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

  for (Level::const_patch_iterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
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

  for (int i = 0; i < grid->numLevels(); i++) {
    const LevelP level = grid->getLevel(i);

    for (Level::const_patch_iterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
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
LoadBalancerCommon::createOutputPatchSet( const LevelP & level )
{
  if (m_output_Nth_proc == 1) {
    // assume the perProcessor set on the level was created first
    return m_level_perproc_patchsets[level->getIndex()].get_rep();
  } else {
    PatchSet* patches = scinew PatchSet();
    patches->createEmptySubsets(d_myworld->size());

    for (Level::const_patch_iterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
      const Patch* patch = *iter;
      int proc = (static_cast<long long>(getPatchwiseProcessorAssignment(patch)) / static_cast<long long>(m_output_Nth_proc))
                  * m_output_Nth_proc;
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
LoadBalancerCommon::createNeighborhood( const GridP & grid, const GridP & oldGrid )
{
  int me = d_myworld->myrank();
  // TODO consider this old warning from Steve:
  //    WARNING - this should be determined from the taskgraph? - Steve

  m_neighbors.clear();
  m_neighborhood_processors.clear();
  
  //this processor should always be in the neighborhood
  m_neighborhood_processors.insert(d_myworld->myrank());
 
  // go through all patches on all levels, and if the patch-wise
  // processor assignment equals the current processor, then store the 
  // patch's neighbors in the load balancer array
  int num_levels = grid->numLevels();
  for( int l = 0; l < num_levels; l++ ) {
    LevelP level = grid->getLevel(l);

    // get the max ghost cells to consider for neighborhood creation
    int maxGhost = m_scheduler->getMaxGhost();

    for(Level::const_patch_iterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
      const Patch* patch = *iter;

      // we need to check both where the patch is and where
      // it used to be (in the case of a dynamic reallocation)
      int proc    = getPatchwiseProcessorAssignment( patch );
      int oldproc = getOldProcessorAssignment( patch );

      // we also need to see if the output processor for patch is this proc,
      // in case it wouldn't otherwise have been in the neighborhood
      int outputproc = (static_cast<long long>(proc) / static_cast<long long>(m_output_Nth_proc))*m_output_Nth_proc;

      if(proc == me || oldproc == me || outputproc == me) {
        // one for current level, coarse level, find level, old level
        // each call to level->selectPatches must be done with an empty patch set
        // or otherwise it will conflict with the sorted order of the cached patches
        Patch::selectType neighbor;

        IntVector ghost(maxGhost,maxGhost,maxGhost);

        IntVector low( patch->getExtraLowIndex( Patch::CellBased, IntVector(0,0,0)));
        IntVector high(patch->getExtraHighIndex(Patch::CellBased, IntVector(0,0,0)));
        level->selectPatches(low-ghost, high+ghost, neighbor);
        
        //add owning processors
        for (int i = 0; i < neighbor.size(); i++) {
          m_neighbors.insert(neighbor[i]->getRealPatch());
          int nproc = getPatchwiseProcessorAssignment(neighbor[i]);
          if (nproc >= 0) {
            m_neighborhood_processors.insert(nproc);
          }

          int oproc = getOldProcessorAssignment(neighbor[i]);
          if (oproc >= 0) {
            m_neighborhood_processors.insert(oproc);
          }
        }
        
        if (m_shared_state->isCopyDataTimestep() && proc == me) {
          if (oldGrid->numLevels() > l) {
            // on copy data timestep we need old patches that line up with this proc's patches,
            // get the other way around at the end
            Patch::selectType old;
            const LevelP& oldLevel = oldGrid->getLevel(l);
            oldLevel->selectPatches(patch->getExtraCellLowIndex() - ghost, patch->getExtraCellHighIndex() + ghost, old);
            for (int i = 0; i < old.size(); i++)  //add owning processors (they are the old owners)
                {
              m_neighbors.insert(old[i]->getRealPatch());
              int nproc = getPatchwiseProcessorAssignment(old[i]);
              if (nproc >= 0) {
                m_neighborhood_processors.insert(nproc);
              }
              int oproc = getOldProcessorAssignment(old[i]);
              if (oproc >= 0) {
                m_neighborhood_processors.insert(oproc);
              }
            }
          }
        }

        // add AMR stuff - so the patch will know about coarsening and refining
        if (l > 0 && (proc == me || (oldproc == me && !m_shared_state->isCopyDataTimestep()))) {
          LevelP coarseLevel = level;

          // get the max level offset and max ghost cells to consider for neighborhood creation
          int maxLevelOffset = m_scheduler->getMaxLevelOffset();
          IntVector ghost(maxGhost, maxGhost, maxGhost);
          for (int offset = 1; offset <= maxLevelOffset && coarseLevel->hasCoarserLevel(); ++offset) {
            ghost = ghost * coarseLevel->getRefinementRatio();
            coarseLevel = coarseLevel->getCoarserLevel();
            Patch::selectType coarse;

            coarseLevel->selectPatches(level->mapCellToCoarser(low - ghost, offset), level->mapCellToCoarser(high + ghost, offset),
                                       coarse);
            // add owning processors
            for (int i = 0; i < coarse.size(); i++) {
              m_neighbors.insert(coarse[i]->getRealPatch());

              int nproc = getPatchwiseProcessorAssignment(coarse[i]);
              if (nproc >= 0) {
                m_neighborhood_processors.insert(nproc);
              }

              int oproc = getOldProcessorAssignment(coarse[i]);
              if (oproc >= 0) {
                m_neighborhood_processors.insert(oproc);
              }
            }
          }
        }

        if (l < grid->numLevels() - 1 && (proc == me || (oldproc == me && !m_shared_state->isCopyDataTimestep()))) {

          IntVector ghost(maxGhost, maxGhost, maxGhost);
          const LevelP& fineLevel = level->getFinerLevel();
          Patch::selectType fine;
          fineLevel->selectPatches(level->mapCellToFiner(low - ghost), level->mapCellToFiner(high + ghost), fine);
          for (int i = 0; i < fine.size(); i++) {  //add owning processors
            m_neighbors.insert(fine[i]->getRealPatch());
            int nproc = getPatchwiseProcessorAssignment(fine[i]);
            if (nproc >= 0) {
              m_neighborhood_processors.insert(nproc);
            }
            int oproc = getOldProcessorAssignment(fine[i]);
            if (oproc >= 0) {
              m_neighborhood_processors.insert(oproc);
            }
          }
        }

      }
    }
  }

  if (m_shared_state->isCopyDataTimestep()) {
    // Regrid timestep postprocess 
    // 1)- go through the old grid and 
    //     find which patches used to be on this proc 
    for (int l = 0; l < oldGrid->numLevels(); l++) {

      if (grid->numLevels() <= l) {
        continue;
      }

      // NOTE: all other components use uniform ghost cells across levels, so RMCRT is a specific case.
      int maxGhost = m_scheduler->getMaxGhost();
      IntVector ghost(maxGhost, maxGhost, maxGhost);

      LevelP oldLevel = oldGrid->getLevel(l);
      LevelP newLevel = grid->getLevel(l);

      for (Level::const_patch_iterator iter = oldLevel->patchesBegin(); iter != oldLevel->patchesEnd(); iter++) {
        const Patch* oldPatch = *iter;

        // we need to check both where the patch is and where
        // it used to be (in the case of a dynamic reallocation)
        int oldproc = getOldProcessorAssignment(oldPatch);

        if (oldproc == me) {
          // don't get extra cells or ghost cells
          Patch::selectType n;
          newLevel->selectPatches(oldPatch->getExtraCellLowIndex() - ghost, oldPatch->getExtraCellHighIndex() + ghost, n);
          m_neighbors.insert(oldPatch);

          int nproc = getPatchwiseProcessorAssignment(oldPatch);
          if (nproc >= 0) {
            m_neighborhood_processors.insert(nproc);
          }

          int oproc = getOldProcessorAssignment(oldPatch);
          if (oproc >= 0) {
            m_neighborhood_processors.insert(oproc);
          }

          for (int i = 0; i < (int)n.size(); i++) {
            m_neighbors.insert(n[i]->getRealPatch());

            int nproc = getPatchwiseProcessorAssignment(n[i]);
            if (nproc >= 0) {
              m_neighborhood_processors.insert(nproc);
            }

            int oproc = getOldProcessorAssignment(n[i]);
            if (oproc >= 0) {
              m_neighborhood_processors.insert(oproc);
            }
          }
        }
      }
    }
  }

#if 0
  std::cout << d_myworld->myrank() << " np: ";
  for(std::set<int>::iterator iter=m_neighborhood_processors.begin();iter!=m_neighborhood_processors.end();iter++)
  {
    std::cout << *iter << " ";
  }
  std::cout << std::endl;
#endif

  if (g_neighborhood_dbg) {
    std::ostringstream message;
    for (std::set<const Patch*>::iterator iter = m_neighbors.begin(); iter != m_neighbors.end(); iter++) {
      message << "Rank-" << d_myworld->myrank() << "  Neighborhood: " << (*iter)->getID() << " Proc "
                << getPatchwiseProcessorAssignment(*iter) << std::endl;
    }
    DOUT(true, message.str());
  }

} // end createNeighborhood()

//______________________________________________________________________
//
bool
LoadBalancerCommon::inNeighborhood( const PatchSubset * ps )
{
  for (int i = 0; i < ps->size(); i++) {
    const Patch* patch = ps->get(i);
    if (m_neighbors.find(patch) != m_neighbors.end())
      return true;
  }
  // also count a subset with no patches
  return ps->size() == 0;
}

//______________________________________________________________________
//
bool
LoadBalancerCommon::inNeighborhood( const Patch * patch )
{
  if (m_neighbors.find(patch) != m_neighbors.end()) {
    return true;
  } else {
    return false;
  }
}

//______________________________________________________________________
//
void
LoadBalancerCommon::problemSetup( ProblemSpecP     & pspec
                                , GridP            & grid
                                , SimulationStateP & state
                                )
{
  m_shared_state = state;
  m_scheduler = dynamic_cast<Scheduler*>(getPort("scheduler"));
  ProblemSpecP p = pspec->findBlock("LoadBalancer");
  m_output_Nth_proc = 1;

  if (p != 0) {
    p->getWithDefault("outputNthProc", m_output_Nth_proc, 1);
  }

#ifdef HAVE_VISIT
  static bool initialized = false;

  // Running with VisIt so add in the variables that the user can
  // modify.
  if( m_shared_state->getVisIt() & !initialized ) {
    SimulationState::interactiveVar var;
    var.name     = "LoadBalancer-DoSpaceCurve";
    var.type     = Uintah::TypeDescription::bool_type;
    var.value    = (void *) &m_do_space_curve;
    var.modifiable = true;
    var.recompile  = false;
    var.modified   = false;
    m_shared_state->d_UPSVars.push_back( var );

//    m_shared_state->d_debugStreams.push_back( &g_lb_dbg  );
//    m_shared_state->d_debugStreams.push_back( &g_neighborhood_dbg );

    initialized = true;
  }
#endif
}

//______________________________________________________________________
// Cost profiling functions
void
LoadBalancerCommon::addContribution( DetailedTask * task ,double cost )
{
  static bool warned = false;
  if (!warned) {
    proc0cout << "Warning: addContribution not implemented for LoadBalancerCommon.\n";
    warned = true;
  }
}

//______________________________________________________________________  
// Finalize the contributions (updates the weight, should be called once per timestep):
void
LoadBalancerCommon::finalizeContributions( const GridP & currentGrid )
{
  static bool warned = false;
  if (!warned) {
    proc0cout << "Warning: finalizeContributions not implemented for LoadBalancerCommon.\n";
    warned = true;
  }
}

//______________________________________________________________________
// Initializes the regions in the new level that are not in the old level.
void
LoadBalancerCommon::initializeWeights( const Grid * oldgrid, const Grid * newgrid )
{
  static bool warned = false;
  if (!warned) {
    proc0cout << "Warning: initializeWeights not implemented for LoadBalancerCommon.\n";
    warned = true;
  }
}

//______________________________________________________________________
// Resets the profiler counters to zero
void
LoadBalancerCommon::resetCostForecaster()
{
  static bool warned = false;
  if (!warned) {
    proc0cout << "Warning: resetCostForecaster not implemented for LoadBalancerCommon.\n";
    warned = true;
  }
}
