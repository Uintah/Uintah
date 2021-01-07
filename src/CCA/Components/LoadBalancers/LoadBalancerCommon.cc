/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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
#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/DataArchive/DataArchive.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DOUT.hpp>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/NotFinished.h>

#include <sci_defs/visit_defs.h>

#include <cfloat>
#include <climits>
#include <iomanip>
#include <sstream>
#include <unordered_set>

using namespace Uintah;

namespace {

  Dout g_lb_dbg(                "LoadBalancer"     , "LoadBalancerCommon", "general info on LB patch assignment", false );
  Dout g_neighborhood_dbg(      "Neighborhood"     , "LoadBalancerCommon", "report processor neighborhood contents", false );
  Dout g_neighborhood_size_dbg( "NeighborhoodSize" , "LoadBalancerCommon", "report patch neighborhood sizes, local & distal", false );
  Dout g_patch_assignment(      "LBPatchAssignment", "LoadBalancerCommon", "report per-process patch assignment", false );

}

namespace Uintah {
  DebugStream g_profile_stats ("ProfileStats",   "LoadBalancerCommon", "", false );
  DebugStream g_profile_stats2("ProfileStats2",  "LoadBalancerCommon", "", false );
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
  , stats( "LBStats", "LoadBalancerCommon", "", false )
  , times( "LBTimes", "LoadBalancerCommon", "", false )
  , lbout( "LBOut",   "LoadBalancerCommon", "", false )
{
  m_activeDims[0] = m_activeDims[1] = m_activeDims[2] = 0;
}

//______________________________________________________________________
//
LoadBalancerCommon::~LoadBalancerCommon()
{
}

//______________________________________________________________________
//
void LoadBalancerCommon::getComponents()
{
  m_scheduler = dynamic_cast<Scheduler*>(getPort("scheduler"));

  if( !m_scheduler ) {
    throw InternalError("dynamic_cast of 'm_scheduler' failed!", __FILE__, __LINE__);
  }

  m_application = dynamic_cast<ApplicationInterface*>( getPort("application") );

  if( !m_application ) {
    throw InternalError("dynamic_cast of 'm_application' failed!", __FILE__, __LINE__);
  }
}

//______________________________________________________________________
//
void LoadBalancerCommon::releaseComponents()
{
  releasePort( "scheduler" );
  releasePort( "application" );

  m_scheduler   = nullptr;
  m_application = nullptr;

  m_materialManager = nullptr;
}

//______________________________________________________________________
//
void
LoadBalancerCommon::assignResources( DetailedTasks & graph )
{
  int nTasks = graph.numTasks();

  DOUT(g_lb_dbg, "Rank-" << d_myworld->myRank() << " Assigning Tasks to Resources! (" << nTasks << " tasks)");

  for (int i = 0; i < nTasks; i++) {
    DetailedTask* task = graph.getTask(i);

    const PatchSubset* patches = task->getPatches();
    if (patches && patches->size() > 0 && task->getTask()->getType() != Task::OncePerProc && task->getTask()->getType() != Task::Hypre) {
      const Patch* patch = patches->get(0);

      int idx = getPatchwiseProcessorAssignment(patch);
      ASSERTRANGE(idx, 0, d_myworld->nRanks());

      if (task->getTask()->getType() == Task::Output) {
        task->assignResource(getOutputRank(patch));
      }
      else {
        task->assignResource(idx);
      }

      DOUT(g_lb_dbg, "Rank-" << d_myworld->myRank() << " Task " << *(task->getTask()) << " put on resource " << idx);

#if SCI_ASSERTION_LEVEL > 0
      std::ostringstream ostr;
      ostr << patch->getID() << ':' << idx;

      for (int i = 1; i < patches->size(); i++) {
        const Patch* p = patches->get(i);
        int rank = getPatchwiseProcessorAssignment(p);
        ostr << ' ' << p->getID() << ';' << rank;
        ASSERTRANGE(rank, 0, d_myworld->nRanks());

        if (rank != idx && task->getTask()->getType() != Task::Output) {
          DOUT( true, "Rank-" << d_myworld->myRank() << " WARNING: inconsistent task (" << task->getTask()->getName()
                              << ") assignment (" << rank << ", " << idx << ") in LoadBalancerCommon");
        }
      }
#endif
    }
    else {
      if (task->getTask()->isReductionTask()) {
        task->assignResource(d_myworld->myRank());

        DOUT(g_lb_dbg,
             d_myworld->myRank() << "  Resource (for no patch task) " << *task->getTask() << " is : " << d_myworld->myRank());

      }
      else if (task->getTask()->getType() == Task::InitialSend) {
        // Already assigned, do nothing
        ASSERT(task->getAssignedResourceIndex() != -1);
      }
      else if (task->getTask()->getType() == Task::OncePerProc || task->getTask()->getType() == Task::Hypre) {

        // patch-less task, not execute-once, set to run on all procs
        // once per patch subset (empty or not)
        // at least one example is the multi-level (impAMRICE) pressureSolve
        for (std::unordered_set<int>::iterator p = m_local_neighbor_processes.begin(); p != m_local_neighbor_processes.end(); ++p) {
          int i = (*p);
          if (patches == task->getTask()->getPatchSet()->getSubset(i)) {
            task->assignResource(i);
            DOUT(g_lb_dbg, d_myworld->myRank() << " " << task->getTask()->getType() << " Task " << *(task->getTask()) << " put on resource " << i);
          }
        }
      }
      else {
        task->assignResource(0);
        DOUT(g_lb_dbg, d_myworld->myRank() << " Unknown-type Task " << *(task->getTask()) << " put on resource " << 0);
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
  if( m_scheduler->isCopyDataTimestep() && patch->getRealPatch()->getID() < m_assignment_base_patch ) {
    return -patch->getID();
  }
 
  ASSERTRANGE( patch->getRealPatch()->getID(), m_assignment_base_patch, m_assignment_base_patch + static_cast<int>(m_processor_assignment.size()) );
  int rank = m_processor_assignment[ patch->getRealPatch()->getGridIndex() ];

  ASSERTRANGE( rank, 0, d_myworld->nRanks() );
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
  ASSERTRANGE( proc, 0, d_myworld->nRanks() );

  return proc;
}

//______________________________________________________________________
//
void
LoadBalancerCommon::useSFC( const LevelP & level
                          ,       int    * order
                          )
{
  std::vector<DistributedIndex> indices; //output
  std::vector<double> positions;

  IntVector min_patch_size(INT_MAX,INT_MAX,INT_MAX);  

  // get the overall range in all dimensions from all patches
  IntVector high(INT_MIN,INT_MIN,INT_MIN);
  IntVector low(INT_MAX,INT_MAX,INT_MAX);

#ifdef SFC_PARALLEL 
  std::vector<int> originalPatchCount(d_myworld->nRanks(),0); //store how many patches each patch has originally
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
    long long num_patches = d_myworld->nRanks();
    long long proc = (pindex*num_patches) /(long long)level->numPatches();

    ASSERTRANGE(proc,0,d_myworld->nRanks());
    if(d_myworld->myRank()==(int)proc) {
      Vector point=(patch->getCellLowIndex()+patch->getCellHighIndex()).asVector()/2.0;
      for(int d=0;d<m_numDims;d++) {
        positions.push_back(point[m_activeDims[d]]);
      }
    }
    originalPatchCount[proc]++;
#else
    Vector point=(patch->getCellLowIndex()+patch->getCellHighIndex()).asVector()/2.0;
    for(int d=0;d<m_numDims;d++) {
      positions.push_back(point[m_activeDims[d]]);
    }
#endif
  }

#ifdef SFC_PARALLEL
  //compute patch starting locations
  std::vector<int> originalPatchStart(d_myworld->nRanks(),0);
  for(int p=1;p<d_myworld->nRanks();p++) {
    originalPatchStart[p]=originalPatchStart[p-1]+originalPatchCount[p-1];
  }
#endif

  // Patchset dimensions
  IntVector range = high-low;
  
  // Center of patchset
  Vector center = (high+low).asVector()/2.0;
 
  double r[3]     = {(double)range[m_activeDims[0]],
                     (double)range[m_activeDims[1]],
                     (double)range[m_activeDims[2]]};

  double c[3]     = {(double)center[m_activeDims[0]],
                     (double)center[m_activeDims[1]],
                     (double)center[m_activeDims[2]]};

  double delta[3] = {(double)min_patch_size[m_activeDims[0]],
                     (double)min_patch_size[m_activeDims[1]],
                     (double)min_patch_size[m_activeDims[2]]};

  // Create SFC
  m_sfc.SetDimensions(r);
  m_sfc.SetCenter(c);
  m_sfc.SetRefinementsByDelta(delta); 
  m_sfc.SetLocations(&positions);
  m_sfc.SetOutputVector(&indices);
  
#ifdef SFC_PARALLEL
  m_sfc.SetLocalSize(originalPatchCount[d_myworld->myRank()]);
  m_sfc.GenerateCurve();
#else
  m_sfc.SetLocalSize(level->numPatches());
  m_sfc.GenerateCurve(SERIAL);
#endif
  
#ifdef SFC_PARALLEL
  if( d_myworld->nRanks() > 1 ) {
    std::vector<int> recvcounts(d_myworld->nRanks(), 0);
    std::vector<int> displs(d_myworld->nRanks(), 0);
    
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
    Uintah::MPI::Allgatherv(&indices[0], recvcounts[d_myworld->myRank()], MPI_BYTE, &rbuf[0],
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
  if(d_myworld->myRank()==0) {
    std::cout << "Warning checking SFC correctness\n";
  }
  for (int i = 0; i < level->numPatches(); i++) {
    for (int j = i+1; j < level->numPatches(); j++) {
      if (order[i] == order[j]) 
      {
        std::cout << "Rank:" << d_myworld->myRank() <<  ":   ALERT!!!!!! index done twice: index "
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
LoadBalancerCommon::restartInitialize(       DataArchive * archive
                                     , const int           time_index
                                     , const std::string & ts_url
                                     , const GridP       & grid
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
        m_processor_assignment[(*iter)->getID()-startingID] = archive->queryPatchwiseProcessor(*iter, time_index) % d_myworld->nRanks();
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

  if (prevNumProcs != d_myworld->nRanks() || m_output_Nth_proc > 1) {
    if (d_myworld->myRank() == 0){
      DOUT(g_lb_dbg, "  Original run had " << prevNumProcs << ", this has " << d_myworld->nRanks());
    }
    m_check_after_restart = true;
  }

  if (d_myworld->myRank() == 0) {
    DOUT(g_lb_dbg, d_myworld->myRank() << " check after restart: " << m_check_after_restart);

#if 0
    int startPatch = (int) (*grid->getLevel(0)->patchesBegin())->getID();
    std::ostringstream message;
      for (unsigned i = 0; i < m_processor_assignment.size(); i++) {
        message << d_myworld-> myRank() << " patch " << i << " (real " << i+startPatch << ") -> proc "
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
LoadBalancerCommon::possiblyDynamicallyReallocate( const GridP & grid
                                                 ,       int     state
                                                 )
{
  if( state != LoadBalancer::CHECK_LB ) {
    // Have it create a new patch set, and have the DLB version call this.
    // This is a good place to do it, as it is automatically called when the grid changes.
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
  patches->createEmptySubsets(d_myworld->nRanks());

  for (Level::const_patch_iterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
    const Patch* patch = *iter;
    int proc = getPatchwiseProcessorAssignment(patch);
    ASSERTRANGE(proc, 0, d_myworld->nRanks());
    PatchSubset* subset = patches->getSubset(proc);
    subset->add(patch);
  }

  patches->sortSubsets();
  return patches;
}

//______________________________________________________________________
//
// Creates a PatchSet containing PatchSubsets for each processor for an entire grid.
const PatchSet*
LoadBalancerCommon::createPerProcessorPatchSet( const GridP & grid )
{
  PatchSet* patches = scinew PatchSet();
  patches->createEmptySubsets(d_myworld->nRanks());

  for (int i = 0; i < grid->numLevels(); i++) {
    const LevelP level = grid->getLevel(i);

    for (Level::const_patch_iterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
      const Patch* patch = *iter;
      int proc = getPatchwiseProcessorAssignment(patch);
      ASSERTRANGE(proc, 0, d_myworld->nRanks());
      PatchSubset* subset = patches->getSubset(proc);
      subset->add(patch);

      // DEBUG: report patch level assignment
      if (g_patch_assignment) {
        std::ostringstream mesg;
        mesg << "Patch: " << patch->getID() << " is on level: " << patch->getLevel()->getIndex();
        DOUT(d_myworld->myRank() == 0, mesg.str());
      }
    }
  }

  patches->sortSubsets();

  // DEBUG: report per-proc patch assignment
  if (g_patch_assignment) {
    const PatchSubset* my_patches = patches->getSubset(d_myworld->myRank());
    std::ostringstream mesg;
    mesg << "Rank-" << d_myworld->myRank() << " assigned patches: {";
    for (auto p = 0; p < my_patches->size(); p++) {
      mesg << (( p == 0 || p == my_patches->size()) ? " " : ", ") << my_patches->get(p)->getID();
    }
    mesg << " }";
    DOUT(true, mesg.str());
  }

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
    patches->createEmptySubsets(d_myworld->nRanks());

    for (Level::const_patch_iterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
      const Patch* patch = *iter;
      int proc = (static_cast<long long>(getPatchwiseProcessorAssignment(patch)) / static_cast<long long>(m_output_Nth_proc))
                  * m_output_Nth_proc;
      ASSERTRANGE(proc, 0, d_myworld->nRanks());
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
LoadBalancerCommon::createNeighborhoods( const GridP & grid
                                       , const GridP & oldGrid
                                       , const bool    hasDistalReqs /* = false */
                                       )
{
  int my_rank = d_myworld->myRank();

  m_local_neighbor_patches.clear();
  m_local_neighbor_processes.clear();
  m_distal_neighbor_patches.clear();
  m_distal_neighbor_processes.clear();
  
  // this processor should always be in all neighborhoods
  m_local_neighbor_processes.insert(my_rank);
  m_distal_neighbor_processes.insert(my_rank);

  // go through all patches on all levels, and if the patch-wise processor assignment equals
  // the current processor, then store the patch's neighbors in the load balancer array
  const auto num_levels = grid->numLevels();
  for (auto l = 0; l < num_levels; ++l) {
    LevelP level = grid->getLevel(l);

    for (Level::const_patch_iterator iter = level->patchesBegin(); iter != level->patchesEnd(); ++iter) {
      const Patch* patch = *iter;

      // we need to check both where the patch is and where
      // it used to be (in the case of a dynamic reallocation)
      const int proc    = getPatchwiseProcessorAssignment( patch );
      const int oldproc = getOldProcessorAssignment( patch );
      IntVector low( patch->getExtraLowIndex( Patch::CellBased, IntVector(0,0,0)));
      IntVector high(patch->getExtraHighIndex(Patch::CellBased, IntVector(0,0,0)));

      // we also need to see if the output processor for patch is this proc,
      // in case it wouldn't otherwise have been in the neighborhood
      int outputproc = (static_cast<long long>(proc) / static_cast<long long>(m_output_Nth_proc)) * m_output_Nth_proc;
      if (proc == my_rank || oldproc == my_rank || outputproc == my_rank) {

        // add owning processors (local)
        const int maxLocalGhost = m_scheduler->getMaxGhost();
        IntVector localGhost(maxLocalGhost, maxLocalGhost, maxLocalGhost);
        addPatchesAndProcsToNeighborhood(level.get_rep(),
                                         low-localGhost,
                                         high+localGhost,
                                         m_local_neighbor_patches,
                                         m_local_neighbor_processes);
        
        // add owning processors (distal)
        if (hasDistalReqs) {
          const int maxDistalGhost = m_scheduler->getMaxDistalGhost();
          IntVector distalGhost(maxDistalGhost, maxDistalGhost, maxDistalGhost);
          addPatchesAndProcsToNeighborhood(level.get_rep(),
                                           low-distalGhost,
                                           high+distalGhost,
                                           m_distal_neighbor_patches,
                                           m_distal_neighbor_processes);
        }

        if (m_scheduler->isCopyDataTimestep() && proc == my_rank) {
          if (oldGrid->numLevels() > l) {
            // on copy data timestep we need old patches that line up with this proc's patches,
            // get the other way around at the end
            Patch::selectType oldPatches;
            const LevelP& oldLevel = oldGrid->getLevel(l);
            oldLevel->selectPatches(patch->getExtraCellLowIndex() - localGhost, patch->getExtraCellHighIndex() + localGhost, oldPatches);
            // add owning processors (they are the old owners)
            const auto num_patches = oldPatches.size();
            for (auto i = 0u; i < num_patches; ++i) {
              m_local_neighbor_patches.insert(oldPatches[i]->getRealPatch());
              int nproc = getPatchwiseProcessorAssignment(oldPatches[i]);
              if (nproc >= 0) {
                m_local_neighbor_processes.insert(nproc);
              }
              int oproc = getOldProcessorAssignment(oldPatches[i]);
              if (oproc >= 0) {
                m_local_neighbor_processes.insert(oproc);
              }
            }
          }
        }

        //-----------------------------------------------------------------------------------------
        // add multi-level (AMR) stuff - so the patch will know about coarsening and refining
        //-----------------------------------------------------------------------------------------

        // First look down levels (coarser)
        if (l > 0 && (proc == my_rank || (oldproc == my_rank && !m_scheduler->isCopyDataTimestep()))) {
          LevelP coarseLevel = level;

          // get the max level offset and max ghost cells to consider for neighborhood creation
          int maxLevelOffset = m_scheduler->getMaxLevelOffset();
          for (auto offset = 1; offset <= maxLevelOffset && coarseLevel->hasCoarserLevel(); ++offset) {

            // add owning processors (local)
            const int maxLocalGhost = m_scheduler->getMaxGhost();
            IntVector localGhost(maxLocalGhost, maxLocalGhost, maxLocalGhost);
            localGhost  = localGhost * coarseLevel->getRefinementRatio();
            coarseLevel = coarseLevel->getCoarserLevel();
            addPatchesAndProcsToNeighborhood(coarseLevel.get_rep(),
                                             level->mapCellToCoarser(low - localGhost, offset),
                                             level->mapCellToCoarser(high + localGhost, offset),
                                             m_local_neighbor_patches,
                                             m_local_neighbor_processes);

            // add owning processors (distal)
            if (hasDistalReqs) {
              const int maxDistalGhost = m_scheduler->getMaxDistalGhost();
              IntVector distalGhost(maxDistalGhost, maxDistalGhost, maxDistalGhost);
              distalGhost = distalGhost * coarseLevel->getRefinementRatio();
              addPatchesAndProcsToNeighborhood(coarseLevel.get_rep(),
                                               level->mapCellToCoarser(low - distalGhost, offset),
                                               level->mapCellToCoarser(high + distalGhost, offset),
                                               m_distal_neighbor_patches,
                                               m_distal_neighbor_processes);
            }
          }
        }

        // Second look up a single level (finer)
        if (l < grid->numLevels() - 1 && (proc == my_rank || (oldproc == my_rank && !m_scheduler->isCopyDataTimestep()))) {

          const LevelP& fineLevel = level->getFinerLevel();
          Patch::selectType fine;
          fineLevel->selectPatches(level->mapCellToFiner(low - localGhost), level->mapCellToFiner(high + localGhost), fine);

          const auto num_fine_neighbors = fine.size();
          for (auto i = 0u; i < num_fine_neighbors; ++i) {  //add owning processors
            m_local_neighbor_patches.insert(fine[i]->getRealPatch());
            int nproc = getPatchwiseProcessorAssignment(fine[i]);
            if (nproc >= 0) {
              m_local_neighbor_processes.insert(nproc);
            }
            int oproc = getOldProcessorAssignment(fine[i]);
            if (oproc >= 0) {
              m_local_neighbor_processes.insert(oproc);
            }
          }
        }
      }
    }
  }

  if (m_scheduler->isCopyDataTimestep()) {
    // Regrid timestep postprocess: go through the old grid and find which patches used to be on this proc
    const auto num_levels = oldGrid->numLevels();
    for (auto l = 0; l < num_levels; ++l) {

      if (grid->numLevels() <= l) {
        continue;
      }

      // NOTE: all other components use uniform ghost cells across levels, global and non-uniform halos are specific cases.
      LevelP oldLevel = oldGrid->getLevel(l);
      LevelP newLevel = grid->getLevel(l);
      for (Level::const_patch_iterator iter = oldLevel->patchesBegin(); iter != oldLevel->patchesEnd(); ++iter) {
        const Patch* oldPatch = *iter;

        // we need to check both where the patch is and where
        // it used to be (in the case of a dynamic reallocation)
        int oldproc = getOldProcessorAssignment(oldPatch);

        if (oldproc == my_rank) {
          const int maxLocalGhost = m_scheduler->getMaxGhost();
          Patch::selectType neighborPatches;
          newLevel->selectPatches(oldPatch->getExtraCellLowIndex() - maxLocalGhost, oldPatch->getExtraCellHighIndex() + maxLocalGhost, neighborPatches);
          m_local_neighbor_patches.insert(oldPatch);

          int nproc = getPatchwiseProcessorAssignment(oldPatch);
          if (nproc >= 0) {
            m_local_neighbor_processes.insert(nproc);
          }
          int oproc = getOldProcessorAssignment(oldPatch);
          if (oproc >= 0) {
            m_local_neighbor_processes.insert(oproc);
          }

          for (size_t i = 0; i < neighborPatches.size(); ++i) {
            m_local_neighbor_patches.insert(neighborPatches[i]->getRealPatch());

            int nproc = getPatchwiseProcessorAssignment(neighborPatches[i]);
            if (nproc >= 0) {
              m_local_neighbor_processes.insert(nproc);
            }
            int oproc = getOldProcessorAssignment(neighborPatches[i]);
            if (oproc >= 0) {
              m_local_neighbor_processes.insert(oproc);
            }
          }
        }
      }
    }
  }

#if 0
  std::ostringstream message;
  message << "Rank-" << my_rank << " Neighborhood contains procs: ";
  for (auto iter = m_neighborhood_processors.begin(); iter != m_neighborhood_processors.end(); ++iter) {
    message << *iter << " ";
  }
  DOUT(true, message.str());
#endif

  if (g_neighborhood_dbg) {
    std::ostringstream message;
    message << "Rank-" << my_rank << " Neighborhood contains: ";
    for (auto iter = m_local_neighbor_patches.cbegin(); iter != m_local_neighbor_patches.cend(); ++iter) {
       message << "patch: " << (*iter)->getID() << " from proc " << getPatchwiseProcessorAssignment(*iter) << "\n";
    }
    DOUT(true, message.str());
  }

  if (g_neighborhood_size_dbg) {
    std::ostringstream message;
    message << "Rank-" << std::left << std::setw(5) << my_rank << "        m_neighbors size: " << std::setw(4) << m_local_neighbor_patches.size()  << "             m_neighbor_processes size: " << std::setw(4) << m_local_neighbor_processes.size() << "\n";
    message << "Rank-" << std::left << std::setw(5) << my_rank << " m_distal_neighbors size: " << std::setw(4) << m_distal_neighbor_patches.size() << "      m_distal_neighbor_processes size: " << std::setw(4) << m_distal_neighbor_processes.size();
    DOUT(true, message.str());
  }

} // end createNeighborhood()


//______________________________________________________________________
//
void
LoadBalancerCommon::addPatchesAndProcsToNeighborhood( const Level                            * const level
                                                    , const IntVector                        & low
                                                    , const IntVector                        & high
                                                    ,       std::unordered_set<const Patch*> & neighbors
                                                    ,       std::unordered_set<int>          & processors
                                                    )
{
  // each call to level->selectPatches must be done with an empty patch set
  // or otherwise it will conflict with the sorted order of the cached patches
  Patch::selectType neighborPatches;
  level->selectPatches(low, high, neighborPatches);

  for (auto i = 0u; i < neighborPatches.size(); ++i) {
    neighbors.insert(neighborPatches[i]->getRealPatch());
    int nproc = getPatchwiseProcessorAssignment(neighborPatches[i]);
    if (nproc >= 0) {
      processors.insert(nproc);
    }
    int oproc = getOldProcessorAssignment(neighborPatches[i]);
    if (oproc >= 0) {
      processors.insert(oproc);
    }
  }
}

//______________________________________________________________________
//
bool
LoadBalancerCommon::inNeighborhood( const PatchSubset * pss
                                  , const bool          hasDistalReqs /* = false */
                                  )
{
  // accept a subset with no patches as being inNeighborhood.
  if (pss->size() == 0) {
    return true;
  }

  bool found = false;
  int i = 0;

  while (!found && i < pss->size()) {
    const Patch* patch = pss->get(i);
    if (hasDistalReqs) {
      found = (m_distal_neighbor_patches.find(patch) != m_distal_neighbor_patches.end());
    }
    else {
      found = (m_local_neighbor_patches.find(patch) != m_local_neighbor_patches.end());
    }
    i++;
  }
  return found;
}

//______________________________________________________________________
//
bool
LoadBalancerCommon::inNeighborhood( const Patch * patch
                                  , const bool    hasDistalReqs /* = false */
                                  )
{
  if (hasDistalReqs) {
    return m_distal_neighbor_patches.find(patch) != m_distal_neighbor_patches.end();
  }
  else {
    return m_local_neighbor_patches.find(patch) != m_local_neighbor_patches.end();
  }
}

//______________________________________________________________________
//
void
LoadBalancerCommon::problemSetup(       ProblemSpecP     & pspec
                                ,       GridP            & grid
                                , const MaterialManagerP & materialManager
                                )
{
  m_materialManager = materialManager;

  ProblemSpecP p = pspec->findBlock("LoadBalancer");
  m_output_Nth_proc = 1;

  if (p != nullptr) {
    p->getWithDefault("outputNthProc", m_output_Nth_proc, 1);
  }

#ifdef HAVE_VISIT
  static bool initialized = false;

  // Running with VisIt so add in the variables that the user can
  // modify.
  if( m_application->getVisIt() && !initialized ) {
    ApplicationInterface::interactiveVar var;
    var.component  = "LoadBalancer";
    var.name       = "DoSpaceCurve";
    var.type       = Uintah::TypeDescription::bool_type;
    var.value      = (void *) &m_do_space_curve;
    var.range[0]   = 0;
    var.range[1]   = 1;
    var.modifiable = true;
    var.recompile  = false;
    var.modified   = false;
    m_application->getUPSVars().push_back( var );

    initialized = true;
  }
#endif
}

//__________________________________
//
void
LoadBalancerCommon::setDimensionality( bool x
                                     , bool y
                                     , bool z
                                     )
{
  m_numDims = 0;
  
  int currentDim = 0;
  bool args[3] = {x,y,z};

  for (int i = 0; i < 3; i++) {
    if (args[i]) {
      m_numDims++;
      m_activeDims[currentDim] = i;

      ++currentDim;
    }
  }
}

//______________________________________________________________________
// Cost profiling functions
void
LoadBalancerCommon::addContribution( DetailedTask * task
                                   , double         cost
                                   )
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
LoadBalancerCommon::initializeWeights( const Grid * oldgrid
                                     , const Grid * newgrid
                                     )
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
