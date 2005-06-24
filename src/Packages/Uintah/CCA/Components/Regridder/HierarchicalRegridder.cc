#include <Packages/Uintah/CCA/Components/Regridder/HierarchicalRegridder.h>
#include <Packages/Uintah/CCA/Components/Regridder/PerPatchVars.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Parallel/BufferInfo.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Mutex.h>

#include <mpi.h>

using namespace Uintah;
using namespace SCIRun;

extern DebugStream rdbg;
extern DebugStream dilate_dbg;
extern Mutex MPITypeLock;

HierarchicalRegridder::HierarchicalRegridder(const ProcessorGroup* pg) : RegridderCommon(pg)
{
  rdbg << "HierarchicalRegridder::HierarchicalRegridder() BGN" << endl;

  d_dilatedCellsCreationLabel  = VarLabel::create("DilatedCellsCreation",
                             CCVariable<int>::getTypeDescription());
  d_dilatedCellsDeletionLabel = VarLabel::create("DilatedCellsDeletion",
                             CCVariable<int>::getTypeDescription());
  d_activePatchesLabel = VarLabel::create("activePatches",
                             PerPatch<SubPatchFlag>::getTypeDescription());

  rdbg << "HierarchicalRegridder::HierarchicalRegridder() END" << endl;
}

HierarchicalRegridder::~HierarchicalRegridder()
{
  rdbg << "HierarchicalRegridder::~HierarchicalRegridder() BGN" << endl;
  VarLabel::destroy(d_dilatedCellsCreationLabel);
  VarLabel::destroy(d_dilatedCellsDeletionLabel);
  VarLabel::destroy(d_activePatchesLabel);

  rdbg << "HierarchicalRegridder::~HierarchicalRegridder() END" << endl;
}

Grid* HierarchicalRegridder::regrid(Grid* oldGrid, SchedulerP& scheduler, const ProblemSpecP& ups)
{
  rdbg << "HierarchicalRegridder::regrid() BGN" << endl;

  if (d_maxLevels <= 1)
    return oldGrid;

  ProblemSpecP grid_ps = ups->findBlock("Grid");
  if (!grid_ps) {
    throw InternalError("HierarchicalRegridder::regrid() Grid section of UPS file not found!");
  }

  // this is for dividing the entire regridding problem into patchwise domains
  DataWarehouse* parent_dw = scheduler->getLastDW();
  DataWarehouse::ScrubMode ParentNewDW_scrubmode =
                           parent_dw->setScrubbing(DataWarehouse::ScrubNone);


  SchedulerP tempsched = scheduler->createSubScheduler();

  // it's normally unconventional to pass the new_dw in in the old_dw's spot,
  // but we don't even use the old_dw and on the first timestep it could be null 
  // and not initialize the parent dws.
  tempsched->initialize(3, 1, parent_dw, parent_dw);

  tempsched->clearMappings();
  tempsched->mapDataWarehouse(Task::ParentOldDW, 0);
  tempsched->mapDataWarehouse(Task::ParentNewDW, 1);
  tempsched->mapDataWarehouse(Task::OldDW, 2);
  tempsched->mapDataWarehouse(Task::NewDW, 3);

  // make sure we have data in our subolddw
  tempsched->advanceDataWarehouse(oldGrid);
  tempsched->advanceDataWarehouse(oldGrid);

  tempsched->get_dw(2)->setScrubbing(DataWarehouse::ScrubNone);
  tempsched->get_dw(3)->setScrubbing(DataWarehouse::ScrubNone);
  

  int ngc;
  for ( int levelIndex = 0; levelIndex < oldGrid->numLevels() && levelIndex < d_maxLevels-1; levelIndex++ ) {
  // copy refine flags to the "old dw" so mpi copying will work correctly
    const PatchSubset* psub = scheduler->getLoadBalancer()->createPerProcessorPatchSet(oldGrid->getLevel(levelIndex))->getSubset(d_myworld->myrank());
    MaterialSubset* msub = scinew MaterialSubset;
    msub->add(0);
    DataWarehouse* old_dw = tempsched->get_dw(2);
    old_dw->transferFrom(parent_dw, d_sharedState->get_refineFlag_label(), psub, msub);
    delete msub;
                       
                       
    // dilate flagged cells on this level
    Task* dilate_task = scinew Task("RegridderCommon::Dilate2 Creation",
                                 dynamic_cast<RegridderCommon*>(this),
                                 &RegridderCommon::Dilate2, 
                                 DILATE_CREATION, old_dw);
    ngc = Max(d_cellCreationDilation.x(), d_cellCreationDilation.y());
    ngc = Max(ngc, d_cellCreationDilation.z());
    
    dilate_task->requires(Task::OldDW, d_sharedState->get_refineFlag_label(), d_sharedState->refineFlagMaterials(),
                          Ghost::AroundCells, ngc);
    dilate_task->computes(d_dilatedCellsCreationLabel);
    tempsched->addTask(dilate_task, oldGrid->getLevel(levelIndex)->eachPatch(), d_sharedState->allMaterials());
#if 0
    if (d_cellCreationDilation != d_cellDeletionDilation) {
      // dilate flagged cells (for deletion) on this level)
      Task* dilate_delete_task = scinew Task("RegridderCommon::Dilate2 Deletion",
                                          dynamic_cast<RegridderCommon*>(this),
                                          &RegridderCommon::Dilate2,
                                          DILATE_DELETION, old_dw);

      ngc = Max(d_cellDeletionDilation.x(), d_cellDeletionDilation.y());
      ngc = Max(ngc, d_cellDeletionDilation.z());

      dilate_delete_task->requires(Task::OldDW, d_sharedState->get_refineFlag_label(), 
                                   d_sharedState->refineFlagMaterials(), Ghost::AroundCells, ngc);
      dilate_delete_task->computes(d_dilatedCellsDeletionLabel);
      tempsched->addTask(dilate_delete_task, oldGrid->getLevel(levelIndex)->eachPatch(), 
                         d_sharedState->allMaterials());
    }
#endif
    // mark subpatches on this level (subpatches represent where patches on the next
    // level will be created).
    Task* mark_task = scinew Task("HierarchicalRegridder::MarkPatches2",
                               this, &HierarchicalRegridder::MarkPatches2);
    mark_task->requires(Task::NewDW, d_dilatedCellsCreationLabel, Ghost::None);
#if 0
    if (d_cellCreationDilation != d_cellDeletionDilation)
      mark_task->requires(Task::NewDW, d_dilatedCellsDeletionLabel, Ghost::None);
#endif

    mark_task->computes(d_activePatchesLabel, d_sharedState->refineFlagMaterials());
    tempsched->addTask(mark_task, oldGrid->getLevel(levelIndex)->eachPatch(), d_sharedState->allMaterials());
  }
  
  tempsched->compile();
  tempsched->execute();
  parent_dw->setScrubbing(ParentNewDW_scrubmode);
  GatherSubPatches(oldGrid, tempsched);
  return CreateGrid2(oldGrid, ups);  
}


IntVector HierarchicalRegridder::StartCellToLattice ( IntVector startCell, int levelIdx )
{
  return startCell / d_patchSize[levelIdx];
}

void HierarchicalRegridder::MarkPatches2(const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* ,
                                         DataWarehouse*,
                                         DataWarehouse* new_dw)
{
  rdbg << "MarkPatches2 BGN\n";
  const Level* oldLevel = getLevel(patches);
  int levelIdx = oldLevel->getIndex();
  IntVector subPatchSize = d_patchSize[levelIdx+1]/d_cellRefinementRatio[levelIdx];
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    IntVector startCell = patch->getInteriorCellLowIndex();
    IntVector endCell = patch->getInteriorCellHighIndex();
    IntVector latticeIdx = StartCellToLattice( startCell, levelIdx );
    IntVector realPatchSize = endCell - startCell; // + IntVector(1,1,1); // RNJ this is incorrect maybe?
    IntVector numSubPatches = realPatchSize / subPatchSize;

    IntVector startidx = latticeIdx * d_latticeRefinementRatio[levelIdx];

    SubPatchFlag* subpatches = scinew SubPatchFlag(startidx, startidx+numSubPatches);
    PerPatch<SubPatchFlagP> activePatches(subpatches);
    
    constCCVariable<int> dilatedCellsCreated;
    // FIX Deletion - CCVariable<int>* dilatedCellsDeleted;
    new_dw->put(activePatches, d_activePatchesLabel, 0, patch);
    new_dw->get(dilatedCellsCreated, d_dilatedCellsCreationLabel, 0, patch, Ghost::None, 0);
    
    if (d_cellCreationDilation != d_cellDeletionDilation) {
      //FIX Deletion
      //constCCVariable<int> dcd;
      //new_dw->get(dcd, d_dilatedCellsDeletionLabel, 0, patch, Ghost::None, 0);
      //dilatedCellsDeleted = dynamic_cast<CCVariable<int>*>(const_cast<CCVariableBase*>(dcd.clone()));
    }
    else {
      // Fix Deletion - dilatedCellsDeleted = dilatedCellsCreated;
    }

    for (CellIterator iter(IntVector(0,0,0), numSubPatches); !iter.done(); iter++) {
      IntVector idx(*iter);
      IntVector startCellSubPatch = startCell + idx * subPatchSize;
      IntVector endCellSubPatch = startCell + (idx + IntVector(1,1,1)) * subPatchSize - IntVector(1,1,1);
      IntVector latticeStartIdx = startidx + idx;
      //IntVector latticeEndIdx = latticeStartIdx + IntVector(1,1,1);
      
//       rdbg << "MarkPatches() startCell         = " << startCell         << endl;
//       rdbg << "MarkPatches() endCell           = " << endCell           << endl;
//       rdbg << "MarkPatches() latticeIdx        = " << latticeIdx        << endl;
//       rdbg << "MarkPatches() realPatchSize     = " << realPatchSize     << endl;
//       rdbg << "MarkPatches() numSubPatches     = " << numSubPatches     << endl;
//       rdbg << "MarkPatches() currentIdx        = " << idx               << endl;
//       rdbg << "MarkPatches() startCellSubPatch = " << startCellSubPatch << endl;
//       rdbg << "MarkPatches() endCellSubPatch   = " << endCellSubPatch   << endl;

      if (flaggedCellsExist(dilatedCellsCreated, startCellSubPatch, endCellSubPatch)) {
        rdbg << "Marking Active [ " << levelIdx+1 << " ]: " << latticeStartIdx << endl;
        subpatches->set(latticeStartIdx);
//       } else if (!flaggedCellsExist(*dilatedCellsDeleted, startCellSubPatch, endCellSubPatch)) {
//         // Do we need to check for flagged cells in the children?
//         IntVector childLatticeStartIdx = latticeStartIdx;
//         IntVector childLatticeEndIdx = latticeEndIdx;
//         for (int childLevelIdx = levelIdx+1; childLevelIdx < oldLevel->getGrid()->numLevels(); childLevelIdx++) {
//           for (CellIterator inner_iter(childLatticeStartIdx, childLatticeEndIdx); !inner_iter.done(); inner_iter++) {
//             IntVector inner_idx(*inner_iter);
//             rdbg << "Deleting Active [ " << childLevelIdx << " ]: " << inner_idx << endl;
//             subpatches->clear(inner_idx);
//           }
//           childLatticeStartIdx = childLatticeStartIdx * d_latticeRefinementRatio[childLevelIdx];
//           childLatticeEndIdx = childLatticeEndIdx * d_latticeRefinementRatio[childLevelIdx];
//         }
      } else {
        //        rdbg << "Not Marking or deleting [ " << levelIdx+1 << " ]: " << latticeStartIdx << endl;
      }
    }
  }
  rdbg << "MarkPatches2 END\n";
}

void HierarchicalRegridder::GatherSubPatches(const GridP& oldGrid, SchedulerP& sched)
{
  rdbg << "GatherSubPatches BGN\n";

  LoadBalancer* lb = sched->getLoadBalancer();

  // numLevels -1 because that is the index of the finest level, and
  // maxLevels -2 because we don't care about the subpatches on the 
  //   finest level
  int toplevel = Min(oldGrid->numLevels()-1, d_maxLevels-2);

  // make sure we get the right DW
  DataWarehouse* dw = sched->getLastDW();

  d_patches.clear();
  d_patches.resize(toplevel+2);

  for (int i = toplevel; i >= 0; i--) {
    rdbg << "  gathering on level TOP " << toplevel << " Level " << i << endl;
    const Level* l = oldGrid->getLevel(i).get_rep();
    
    // place to end up with all subpatches
    vector<SubPatchFlagP> allSubpatches(l->numPatches());
    if (d_myworld->size() > 1) {
      // get ordered list of patch-proc assignment
      vector<int> procAssignment(l->numPatches());
      for (Level::const_patchIterator iter = l->patchesBegin(); iter != l->patchesEnd(); iter++) {
        procAssignment[(*iter)->getLevelIndex()] = lb->getPatchwiseProcessorAssignment(*iter);
      }
      // gives us number of things on each proc
      vector<int> sorted_processorAssignment = procAssignment;
      sort(sorted_processorAssignment.begin(), sorted_processorAssignment.end());
      
      vector<int> displs;
      vector<int> recvcounts(d_myworld->size(),0); // init the counts to 0
      
      int numSubpatches = d_patchNum[i+1].x() * d_patchNum[i+1].y() * d_patchNum[i+1].z();

      // num subpatches per patch
      int nsppp = d_latticeRefinementRatio[i].x() * d_latticeRefinementRatio[i].y() * d_latticeRefinementRatio[i].z();

      int offsetProc = 0;
      for (int p = 0; p < (int)sorted_processorAssignment.size(); p++) {
        // set the offsets for the MPI_Gatherv
        while ( offsetProc <= sorted_processorAssignment[p]) {
          // the while part is in case there are fewer procs than patches, and we 
          // have the proc assignment being something like '0 2'
          displs.push_back(p*nsppp);
          offsetProc++;
        }
        recvcounts[sorted_processorAssignment[p]] += nsppp;
      }
    
      // create the buffers to send/recv the data
      int* recvbuf = scinew int[numSubpatches];
      int* sendbuf = scinew int[recvcounts[d_myworld->myrank()]];

      int sendbufindex = 0;

      // use this to sort the allSubPatchBuffer in order of processor.  The displs
      // array keeps track of the index of the final array that holds the first
      // item of each processor.  So use this to put allSubPatchBuffer in the right
      // order, by referencing it, and then the next item for that processor will
      // go in that value + nsppp.
      vector<int> subpatchSorter = displs;
      
      // for this mpi communication, we're going to Gather all subpatch information to allprocessors.
      // We're going to get what data we have from the DW and put its pointer data in sendbuf.  While
      // we're doing that, we need to prepare the receiving buffer to receive these, so we create a subPatchFlag
      // for each patch we're going to receive from, and then put it in the right order, as the 
      // MPI_Allgatherv puts them in order of processor.
      
      for (Level::const_patchIterator citer = l->patchesBegin();
           citer != l->patchesEnd(); citer++) {
        Patch *patch = *citer;

        // the subpatchSorter's index is in terms of the numbers of the subpatches
        int index = subpatchSorter[lb->getPatchwiseProcessorAssignment(patch)];

        subpatchSorter[lb->getPatchwiseProcessorAssignment(patch)]+=nsppp;
        IntVector patchIndex = patch->getInteriorCellLowIndex()/d_patchSize[i];

        // create the recv buffers to 
        SubPatchFlagP spf1 = scinew SubPatchFlag;
        spf1->initialize(d_latticeRefinementRatio[i]*patchIndex,
                         d_latticeRefinementRatio[i]*(patchIndex+IntVector(1,1,1)), &recvbuf[index]);

        allSubpatches[index/nsppp] = spf1;

        if (procAssignment[patch->getLevelIndex()] != d_myworld->myrank())
          continue;
        
        // get the variable and prepare to send it: put its base pointer in the send buffer
        PerPatch<SubPatchFlagP> spf2;
        dw->get(spf2, d_activePatchesLabel, 0, patch);
        for (int idx = 0; idx < nsppp; idx++) {
          sendbuf[idx + sendbufindex] = spf2.get()->subpatches_[idx];
        }
        sendbufindex += nsppp;
      }

      MPI_Allgatherv(sendbuf, sendbufindex, MPI_INT, 
                     recvbuf, &recvcounts[0], &displs[0], MPI_INT, d_myworld->getComm());

    }
    else {
      // for the single-proc case, just put the subpatches in the same container
      for (Level::const_patchIterator citer = l->patchesBegin();
           citer != l->patchesEnd(); citer++) {
        Patch *patch = *citer;
        PerPatch<SubPatchFlagP> spf;
        dw->get(spf, d_activePatchesLabel, 0, patch);
        allSubpatches[patch->getLevelIndex()] = spf.get();
      }
      rdbg << "   Got data\n";

    }
    // loop over each patch's subpatches (these will be the patches on level+1)
    IntVector periodic = oldGrid->getLevel(0)->getPeriodicBoundaries();
    for (unsigned j = 0; j < allSubpatches.size(); j++) {
      rdbg << "   Doing subpatches index " << j << endl;
      for (CellIterator iter(allSubpatches[j].get_rep()->low_, allSubpatches[j].get_rep()->high_); 
           !iter.done(); iter++) {
        IntVector idx(*iter);
        
        // if that subpatch is active...
        if ((*allSubpatches[j].get_rep())[idx]) {
          // add this subpatch to become the next level's patch
          rdbg << "Adding normal subpatch " << idx << " to level " << i+1 << endl;
          d_patches[i+1].insert(idx);
          
          if (i > 0) { // don't dilate onto level 0
            IntVector range = Ceil(d_minBoundaryCells.asVector()/d_patchSize[i].asVector());
            for (CellIterator inner(IntVector(-1,-1,-1)*range, range+IntVector(1,1,1)); !inner.done(); inner++) {
              // "dilate" each subpatch, adding it to the patches on the coarser level
              IntVector dilate_idx = (idx + *inner) / d_latticeRefinementRatio[i];
              // we need to wrap around for periodic boundaries
              if (((dilate_idx.x() < 0 || dilate_idx.x() >= d_patchNum[i].x()) && !periodic.x()) ||
                  ((dilate_idx.y() < 0 || dilate_idx.y() >= d_patchNum[i].y()) && !periodic.y()) ||
                  ((dilate_idx.z() < 0 || dilate_idx.z() >= d_patchNum[i].z()) && !periodic.z()))
                continue;
                
              // if it was periodic, get it in the proper range
              for (int d = 0; d < 3; d++) {
                while (dilate_idx[d] < 0) dilate_idx[d] += d_patchNum[i][d];
                while (dilate_idx[d] >= d_patchNum[i][d]) dilate_idx[d] -= d_patchNum[i][d];
              }

              rdbg << "  Adding dilated subpatch " << dilate_idx << endl;
              d_patches[i].insert(dilate_idx);
            }
          }
        } // end if allSubpatches[j][idx]
        else {
          //           rdbg << " NOT adding subpatch " << idx << " to level " << i+1 << endl;
        }
      } // end for celliterator
    } // end for unsigned j
  } // end for i = toplevel

  // put level 0's patches into the set so we can just iterate through the whole set later
  for (CellIterator iter(IntVector(0,0,0), d_patchNum[0]); !iter.done(); iter++) {
    IntVector idx(*iter);
    rdbg << "Adding patch " << idx << " to level 0" << endl;
    d_patches[0].insert(idx);
  }
  rdbg << "GatherSubPatches END\n";
}

Grid* HierarchicalRegridder::CreateGrid2(Grid* oldGrid, const ProblemSpecP& ups) 
{
  rdbg << "CreateGrid2 BGN\n";

  Grid* newGrid = scinew Grid();
  ProblemSpecP grid_ps = ups->findBlock("Grid");

  for (int levelIdx=0; levelIdx < (int)d_patches.size(); levelIdx++) {
    if (d_patches[levelIdx].size() == 0)
      break;

    Point anchor;
    Vector spacing;
    IntVector extraCells;

    if (levelIdx < oldGrid->numLevels()) {
      anchor = oldGrid->getLevel(levelIdx)->getAnchor();
      spacing = oldGrid->getLevel(levelIdx)->dCell();
      extraCells = oldGrid->getLevel(levelIdx)->getExtraCells();
    } else {
      anchor = newGrid->getLevel(levelIdx-1)->getAnchor();
      spacing = newGrid->getLevel(levelIdx-1)->dCell() / d_cellRefinementRatio[levelIdx-1];
      extraCells = newGrid->getLevel(levelIdx-1)->getExtraCells();
    }

    LevelP newLevel = newGrid->addLevel(anchor, spacing);
    newLevel->setTimeRefinementRatio(oldGrid->getLevel(0)->timeRefinementRatio());
    newLevel->setExtraCells(extraCells);

    rdbg << "HierarchicalRegridder::regrid(): Setting extra cells to be: " << extraCells << endl;

    for (subpatchset::iterator iter = d_patches[levelIdx].begin(); iter != d_patches[levelIdx].end(); iter++) {
      IntVector idx(*iter);
      rdbg << "   Creating patch "<< *iter << endl;
      IntVector startCell       = idx * d_patchSize[levelIdx];
      IntVector endCell         = (idx + IntVector(1,1,1)) * d_patchSize[levelIdx] - IntVector(1,1,1);
      IntVector inStartCell(startCell);
      IntVector inEndCell(endCell);

      // do extra cells - add if there is no neighboring patch
      IntVector low(0,0,0), high(0,0,0);
      if (d_patches[levelIdx].find(idx-IntVector(1,0,0)) == d_patches[levelIdx].end())
        low[0] = extraCells.x();
      if (d_patches[levelIdx].find(idx-IntVector(0,1,0)) == d_patches[levelIdx].end())
        low[1] = extraCells.y();
      if (d_patches[levelIdx].find(idx-IntVector(0,0,1)) == d_patches[levelIdx].end())
        low[2] = extraCells.z();

      if (d_patches[levelIdx].find(idx+IntVector(1,0,0)) == d_patches[levelIdx].end())
        high[0] = extraCells.x();
      if (d_patches[levelIdx].find(idx+IntVector(0,1,0)) == d_patches[levelIdx].end())
        high[1] = extraCells.y();
      if (d_patches[levelIdx].find(idx+IntVector(0,0,1)) == d_patches[levelIdx].end())
        high[2] = extraCells.z();

      if (idx.x() == d_patchNum[levelIdx](0)-1) endCell(0) = d_cellNum[levelIdx](0)-1;
      if (idx.y() == d_patchNum[levelIdx](1)-1) endCell(1) = d_cellNum[levelIdx](1)-1;
      if (idx.z() == d_patchNum[levelIdx](2)-1) endCell(2) = d_cellNum[levelIdx](2)-1;

      rdbg << "     Adding extra cells: " << low << ", " << high << endl;

      startCell -= low;
      endCell += high;

      newLevel->addPatch(startCell, endCell + IntVector(1,1,1), inStartCell, inEndCell + IntVector(1,1,1));
      //newPatch->setLayoutHint(oldPatch->layouthint);
    }
    IntVector periodic;
    if(levelIdx == 0){
      periodic = oldGrid->getLevel(0)->getPeriodicBoundaries();
    } else {
      periodic = newGrid->getLevel(levelIdx-1)->getPeriodicBoundaries();
    }
    newLevel->finalizeLevel(periodic.x(), periodic.y(), periodic.z());
    newLevel->assignBCS(grid_ps);
  }

  d_newGrid = true;
  d_lastRegridTimestep = d_sharedState->getCurrentTopLevelTimeStep();

  rdbg << "HierarchicalRegridder::regrid() END" << endl;

  rdbg << "CreateGrid2 END\n";
  if (*newGrid == *oldGrid) {
    d_newGrid = false;
    delete newGrid;
    return oldGrid;
  }

  newGrid->performConsistencyCheck();
  return newGrid;
}
