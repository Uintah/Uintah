#include <Packages/Uintah/CCA/Components/Regridder/HierarchicalRegridder.h>
#include <Packages/Uintah/CCA/Components/Regridder/PerPatchVars.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
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

#define BRYAN

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
  rdbg << "HierarchicalRegridder::~HierarchicalRegridder() END" << endl;
}

Grid* HierarchicalRegridder::regrid(Grid* oldGrid, SchedulerP& scheduler, const ProblemSpecP& ups)
{
  rdbg << "HierarchicalRegridder::regrid() BGN" << endl;

  ProblemSpecP grid_ps = ups->findBlock("Grid");
  if (!grid_ps) {
    throw InternalError("HierarchicalRegridder::regrid() Grid section of UPS file not found!");
  }

#ifdef BRYAN
  // this is for dividing the entire regridding problem into patchwise domains
  DataWarehouse* new_dw = scheduler->getLastDW();
  DataWarehouse::ScrubMode ParentNewDW_scrubmode =
                           new_dw->setScrubbing(DataWarehouse::ScrubNone);


  SchedulerP tempsched = scheduler->createSubScheduler();

  // it's normally unconventional to pass the new_dw in in the old_dw's spot,
  // but we don't even use the old_dw and on the first timestep it could be null 
  // and not initialize the parent dws.
  tempsched->initialize(3, 1, new_dw, new_dw);

  tempsched->clearMappings();
  tempsched->mapDataWarehouse(Task::ParentOldDW, 0);
  tempsched->mapDataWarehouse(Task::ParentNewDW, 1);
  tempsched->mapDataWarehouse(Task::OldDW, 2);
  tempsched->mapDataWarehouse(Task::NewDW, 3);

  // make sure we have data in our subolddw
  //tempsched->advanceDataWarehouse(oldGrid);
  tempsched->advanceDataWarehouse(oldGrid);
  
  tempsched->get_dw(3)->setScrubbing(DataWarehouse::ScrubNone);

  for (int i = 0; i < 4; i++)
    rdbg << " DW " << i << ": " << tempsched->get_dw(i) << endl;


  int ngc;
  for ( int levelIndex = 0; levelIndex < oldGrid->numLevels() && levelIndex < d_maxLevels-1; levelIndex++ ) {
    //Task* dummy_task = new Task("DUMMY", this, &HierarchicalRegridder::dummyTask);
    //dummy_task->computes(d_sharedState->get_refineFlag_label());
    //tempsched->addTask(dummy_task, oldGrid->getLevel(levelIndex)->eachPatch(), d_sharedState->allMaterials());

    // dilate flagged cells on this level
    Task* dilate_task = new Task("RegridderCommon::Dilate2 Creation",
                                 dynamic_cast<RegridderCommon*>(this),
                                 &RegridderCommon::Dilate2, 
                                 DILATE_CREATION, new_dw);
    ngc = Max(d_cellCreationDilation.x(), d_cellCreationDilation.y());
    ngc = Max(ngc, d_cellCreationDilation.z());
    
    dilate_task->requires(Task::ParentNewDW, d_sharedState->get_refineFlag_label(), Ghost::AroundCells, ngc);
    dilate_task->computes(d_dilatedCellsCreationLabel);
    tempsched->addTask(dilate_task, oldGrid->getLevel(levelIndex)->eachPatch(), d_sharedState->allMaterials());
    if (d_cellCreationDilation != d_cellDeletionDilation) {
      // dilate flagged cells (for deletion) on this level)
      Task* dilate_delete_task = new Task("RegridderCommon::Dilate2 Deletion",
                                          dynamic_cast<RegridderCommon*>(this),
                                          &RegridderCommon::Dilate2,
                                          DILATE_DELETION, new_dw);

      ngc = Max(d_cellDeletionDilation.x(), d_cellDeletionDilation.y());
      ngc = Max(ngc, d_cellDeletionDilation.z());

      dilate_delete_task->requires(Task::ParentNewDW, d_sharedState->get_refineFlag_label(), Ghost::AroundCells, ngc);
      dilate_delete_task->computes(d_dilatedCellsDeletionLabel);
      tempsched->addTask(dilate_delete_task, oldGrid->getLevel(levelIndex)->eachPatch(), d_sharedState->allMaterials());
    }
    // mark subpatches on this level (subpatches represent where patches on the next
    // level will be created).
    Task* mark_task = new Task("HierarchicalRegridder::MarkPatches2",
                               this, &HierarchicalRegridder::MarkPatches2);
    mark_task->requires(Task::NewDW, d_dilatedCellsCreationLabel, Ghost::None);
    if (d_cellCreationDilation != d_cellDeletionDilation)
      mark_task->requires(Task::NewDW, d_dilatedCellsDeletionLabel, Ghost::None);
    
    mark_task->computes(d_activePatchesLabel);
    tempsched->addTask(mark_task, oldGrid->getLevel(levelIndex)->eachPatch(), d_sharedState->allMaterials());
  }
  
  tempsched->compile();
  tempsched->execute();
  new_dw->setScrubbing(ParentNewDW_scrubmode);
  GatherSubPatches(oldGrid, tempsched);
  return CreateGrid2(oldGrid, ups);
#else

  int levelIdx;

  rdbg << "HierarchicalRegridder::regrid() BGN" << endl;

  //  DataWarehouse* dw = scheduler->get_dw(0);
  DataWarehouse* dw = scheduler->getLastDW();

  d_flaggedCells.resize(d_maxLevels);
  d_dilatedCellsCreated.resize(d_maxLevels);
  d_dilatedCellsDeleted.resize(d_maxLevels);
  d_numCreated.resize(d_maxLevels);
  d_numDeleted.resize(d_maxLevels);

  for (levelIdx = 0; levelIdx < oldGrid->numLevels() && levelIdx < d_maxLevels-1; levelIdx++) {
    rdbg << "HierarchicalRegridder::regrid() Level = " << levelIdx << endl;

    d_numCreated[levelIdx] = 0;
    d_numDeleted[levelIdx] = 0;

    GetFlaggedCells( oldGrid, levelIdx, dw );
    Dilate( *(d_flaggedCells[levelIdx]), *(d_dilatedCellsCreated[levelIdx]), d_creationFilter, d_cellCreationDilation );
    Dilate( *(d_flaggedCells[levelIdx]), *(d_dilatedCellsDeleted[levelIdx]), d_deletionFilter, d_cellDeletionDilation );
    MarkPatches( oldGrid, levelIdx  );
    ExtendPatches( oldGrid, levelIdx );
  }
  return CreateGrid(oldGrid, ups);

#endif
  
}

void HierarchicalRegridder::MarkPatches( const GridP& oldGrid, int levelIdx  )
{
  rdbg << "HierarchicalRegridder::MarkPatches() BGN" << endl;
  rdbg << "HierarchicalRegridder::MarkPatches() Level" << levelIdx << endl;

  LevelP level = oldGrid->getLevel(levelIdx);

  IntVector subPatchSize = d_patchSize[levelIdx+1]/d_cellRefinementRatio[levelIdx];

  for (Level::patchIterator patchIter = level->patchesBegin(); patchIter != level->patchesEnd(); patchIter++) {
    const Patch* patch = *patchIter;
    IntVector startCell = patch->getCellLowIndex();
    IntVector endCell = patch->getCellHighIndex();
    IntVector latticeIdx = StartCellToLattice( startCell, levelIdx );
    IntVector realPatchSize = endCell - startCell; // + IntVector(1,1,1); // RNJ this is incorrect maybe?
    IntVector realSubPatchNum = realPatchSize / subPatchSize;

    for (CellIterator iter(IntVector(0,0,0), realSubPatchNum); !iter.done(); iter++) {
      IntVector idx(*iter);
      IntVector startCellSubPatch = startCell + idx * subPatchSize;
      IntVector endCellSubPatch = startCell + (idx + IntVector(1,1,1)) * subPatchSize - IntVector(1,1,1);
      IntVector latticeStartIdx = latticeIdx * d_latticeRefinementRatio[levelIdx] + idx;
      IntVector latticeEndIdx = latticeStartIdx;
      
      rdbg << "MarkPatches() startCell         = " << startCell         << endl;
      rdbg << "MarkPatches() endCell           = " << endCell           << endl;
      rdbg << "MarkPatches() latticeIdx        = " << latticeIdx        << endl;
      rdbg << "MarkPatches() realPatchSize     = " << realPatchSize     << endl;
      rdbg << "MarkPatches() realSubPatchNum   = " << realSubPatchNum   << endl;
      rdbg << "MarkPatches() currentIdx        = " << idx               << endl;
      rdbg << "MarkPatches() startCellSubPatch = " << startCellSubPatch << endl;
      rdbg << "MarkPatches() endCellSubPatch   = " << endCellSubPatch   << endl;

      if (flaggedCellsExist(*d_dilatedCellsCreated[levelIdx], startCellSubPatch, endCellSubPatch)) {
        rdbg << "Marking Active [ " << levelIdx+1 << " ]: " << latticeStartIdx << endl;
        (*d_patchActive[levelIdx+1])[latticeStartIdx] = 1;
      } else if (!flaggedCellsExist(*d_dilatedCellsDeleted[levelIdx], startCellSubPatch, endCellSubPatch)) {
        // Do we need to check for flagged cells in the children?
        IntVector childLatticeStartIdx = latticeStartIdx;
        IntVector childLatticeEndIdx = latticeEndIdx;
        for (int childLevelIdx = levelIdx+1; childLevelIdx < oldGrid->numLevels(); childLevelIdx++) {
          for (CellIterator inner_iter(childLatticeStartIdx, childLatticeEndIdx+IntVector(1,1,1)); !inner_iter.done(); inner_iter++) {
            IntVector inner_idx(*inner_iter);
            rdbg << "Deleting Active [ " << childLevelIdx << " ]: " << inner_idx << endl;
            (*d_patchActive[childLevelIdx])[inner_idx] = 0;
          }
          childLatticeStartIdx = childLatticeStartIdx * d_latticeRefinementRatio[childLevelIdx];
          childLatticeEndIdx = childLatticeEndIdx * d_latticeRefinementRatio[childLevelIdx];
        }
      } else {
        rdbg << "Not Marking or deleting [ " << levelIdx+1 << " ]: " << latticeStartIdx << endl;
      }
    }
  }

  rdbg << "HierarchicalRegridder::MarkPatches() END" << endl;
}

void HierarchicalRegridder::ExtendPatches( const GridP& oldGrid, int levelIdx  )
  /*--------------------------------------------------------------------------
    Based on the patches marked at the current-finest level, extend the patches
    of all coarser levels, so that we have at least d_minBoundaryCells coarse
    cells beyond the boundaries of a fine patch, before yet-coarer-level patches
    exist.
    16-SEP-2004 (Oren)     Modified the entire routine to overcome a conceptual
                           bug: we can't dilate cell-wise at parentLevelIdx,
                           because this dilation is meaningful exactly for the
			   cells that are outside the existing cells of
			   parentLevel. So instead, we work with the entire
			   image (map) of the lattices at childLevelIdx and
			   parentLevelIdx.
    --------------------------------------------------------------------------*/
{
  rdbg << "HierarchicalRegridder::ExtendPatches() BGN" << endl;

  for (int childLevelIdx = levelIdx+1; childLevelIdx>0; childLevelIdx--) {                    // Main level loop from finest (parent=levelIdx) up to the coarsest (parent=0)
    /*
      Extend (childLevelIdx - 1)-level patches based on childLevelIdx-patches
    */
    rdbg << "Extend Patches Level: " << childLevelIdx << endl;
    int parentLevelIdx = childLevelIdx - 1;
    IntVector currentLatticeRefinementRatio = d_latticeRefinementRatio[parentLevelIdx];       // Number of normal child patches in a parent patch

    /*
      Dilate active patches on child level, with the filter size determined by the number of
      parent level cells required and the cell size of the sub-region of a child patch at the
      parent level.
    */

    // Compute a filter size that will put child patches on regions of the PARENT level as distant
    // as d_minBoundaryCells from the current active child patches
    IntVector extendFilterSize = (d_minBoundaryCells - IntVector(1,1,1)) / currentLatticeRefinementRatio + IntVector(1,1,1);
    if (d_minBoundaryCells.x() == 0) extendFilterSize(0) = 0;
    if (d_minBoundaryCells.y() == 0) extendFilterSize(1) = 0;
    if (d_minBoundaryCells.z() == 0) extendFilterSize(2) = 0;
    rdbg << "HierarchicalRegridder::ExtendPatches() extendFilterSize = " << extendFilterSize << endl;

    // Dilate patchActive at child into dilatedChildPatchActive
    rdbg << "DILATING LATTICE MAP ON LEVEL " << childLevelIdx << endl;
    CCVariable<int> dilatedChildPatchActive;
    dilatedChildPatchActive.rewindow(IntVector(0,0,0), d_patchNum[childLevelIdx]);
    Dilate(*d_patchActive[childLevelIdx], dilatedChildPatchActive, d_patchFilter, extendFilterSize);
    
    /*
      Add to patches patchActive at parent covering all patches marked in dilatedChildPatchActive
    */
    rdbg << "COMPUTING LATTICE MAP ON LEVEL " << parentLevelIdx << endl;
    rdbg << "d_patchNum[parentLevelIdx] = " << d_patchNum[parentLevelIdx] << endl;
    for (CellIterator iter(IntVector(0,0,0), d_patchNum[parentLevelIdx]); !iter.done(); iter++) {
      IntVector parentLatticeIdx(*iter);
      rdbg << "HierarchicalRegridder::ExtendPatches() parentLatticeIdx      = " << parentLatticeIdx   << endl;
      if (!(*d_patchActive[parentLevelIdx])[parentLatticeIdx]) {
	for (CellIterator subIter(IntVector(0,0,0), currentLatticeRefinementRatio); !subIter.done(); subIter++) {
	  IntVector subIdx(*subIter);
	  IntVector childLatticeIdx = parentLatticeIdx * currentLatticeRefinementRatio + subIdx;
	  rdbg << "HierarchicalRegridder::ExtendPatches()     childLatticeIdx   = " << childLatticeIdx    << endl;
	  if (dilatedChildPatchActive[childLatticeIdx]) {
	      rdbg << "Marking Parent Patch to be ACTIVE." << endl;
	      (*d_patchActive[parentLevelIdx])[parentLatticeIdx] = 1;
	      break;
	  }
	} // end for subIter
      } else {
	cout << "ALREADY ACTIVE." << endl;
      } // end if (*d_patchActive[parentLevelIdx])[parentLatticeIdx])
      rdbg << endl;
    } // end for iter
  } // end for childLevelIdx

  rdbg << "HierarchicalRegridder::ExtendPatches() END" << endl;
}

IntVector HierarchicalRegridder::StartCellToLattice ( IntVector startCell, int levelIdx )
{
  return startCell / d_patchSize[levelIdx];
}

void HierarchicalRegridder::MarkPatches2(const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* ,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  rdbg << "MarkPatches2 BGN\n";
  const Level* oldLevel = getLevel(patches);
  int levelIdx = oldLevel->getIndex();
  IntVector subPatchSize = d_patchSize[levelIdx+1]/d_cellRefinementRatio[levelIdx];
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    IntVector startCell = patch->getCellLowIndex();
    IntVector endCell = patch->getCellHighIndex();
    IntVector latticeIdx = StartCellToLattice( startCell, levelIdx );
    IntVector realPatchSize = endCell - startCell; // + IntVector(1,1,1); // RNJ this is incorrect maybe?
    IntVector numSubPatches = realPatchSize / subPatchSize;

    IntVector startidx = latticeIdx * d_latticeRefinementRatio[levelIdx];

    SubPatchFlag* subpatches = new SubPatchFlag(startidx, startidx+numSubPatches);
    PerPatch<SubPatchFlagP> activePatches(subpatches);
    
    // use pointers here to avoid a lot of const nonsense with the original version of Regridder
    constCCVariable<int> dcc;
    CCVariable<int>* dilatedCellsCreated;
    CCVariable<int>* dilatedCellsDeleted;
    new_dw->put(activePatches, d_activePatchesLabel, 0, patch);
    new_dw->get(dcc, d_dilatedCellsCreationLabel, 0, patch, Ghost::None, 0);
    dilatedCellsCreated = dynamic_cast<CCVariable<int>*>(const_cast<CCVariableBase*>(dcc.clone()));
    
    if (d_cellCreationDilation != d_cellDeletionDilation) {
      constCCVariable<int> dcd;
      new_dw->get(dcd, d_dilatedCellsDeletionLabel, 0, patch, Ghost::None, 0);
      dilatedCellsDeleted = dynamic_cast<CCVariable<int>*>(const_cast<CCVariableBase*>(dcd.clone()));
    }
    else
      dilatedCellsDeleted = dilatedCellsCreated;


    for (CellIterator iter(IntVector(0,0,0), numSubPatches); !iter.done(); iter++) {
      IntVector idx(*iter);
      IntVector startCellSubPatch = startCell + idx * subPatchSize;
      IntVector endCellSubPatch = startCell + (idx + IntVector(1,1,1)) * subPatchSize - IntVector(1,1,1);
      IntVector latticeStartIdx = startidx + idx;
      IntVector latticeEndIdx = latticeStartIdx + IntVector(1,1,1);
      
      rdbg << "MarkPatches() startCell         = " << startCell         << endl;
      rdbg << "MarkPatches() endCell           = " << endCell           << endl;
      rdbg << "MarkPatches() latticeIdx        = " << latticeIdx        << endl;
      rdbg << "MarkPatches() realPatchSize     = " << realPatchSize     << endl;
      rdbg << "MarkPatches() numSubPatches     = " << numSubPatches     << endl;
      rdbg << "MarkPatches() currentIdx        = " << idx               << endl;
      rdbg << "MarkPatches() startCellSubPatch = " << startCellSubPatch << endl;
      rdbg << "MarkPatches() endCellSubPatch   = " << endCellSubPatch   << endl;

      if (flaggedCellsExist(*dilatedCellsCreated, startCellSubPatch, endCellSubPatch)) {
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
        rdbg << "Not Marking or deleting [ " << levelIdx+1 << " ]: " << latticeStartIdx << endl;
      }
    }
  }
  rdbg << "MarkPatches2 END\n";
}

Grid* HierarchicalRegridder::CreateGrid(Grid* oldGrid, const ProblemSpecP& ups) 
{
  Grid* newGrid = scinew Grid();
  ProblemSpecP grid_ps = ups->findBlock("Grid");

  for (int levelIdx=0; levelIdx<d_maxLevels; levelIdx++) {

    bool thisLevelExists = false;

    for (CellIterator iter(IntVector(0,0,0), d_patchNum[levelIdx]); !iter.done(); iter++) {
      if ((*d_patchActive[levelIdx])[*iter]) {
        thisLevelExists = true;
        break;
      }
    }

    if (!thisLevelExists) break;
    
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

    // in the level code, the refinement ratio is the relation
    // between it and the coarser level.
    newLevel->setRefinementRatio(d_cellRefinementRatio[levelIdx]);
    newLevel->setExtraCells(extraCells);

    rdbg << "HierarchicalRegridder::regrid(): Setting extra cells to be: " << extraCells << endl;

    for (CellIterator iter(IntVector(0,0,0), d_patchNum[levelIdx]); !iter.done(); iter++) {
      IntVector idx(*iter);
      if ((*d_patchActive[levelIdx])[idx]) {
        IntVector startCell       = idx * d_patchSize[levelIdx];
        IntVector endCell         = (idx + IntVector(1,1,1)) * d_patchSize[levelIdx] - IntVector(1,1,1);
        if (idx.x() == d_patchNum[levelIdx](0)-1) endCell(0) = d_cellNum[levelIdx](0)-1;
        if (idx.y() == d_patchNum[levelIdx](1)-1) endCell(1) = d_cellNum[levelIdx](1)-1;
        if (idx.z() == d_patchNum[levelIdx](2)-1) endCell(2) = d_cellNum[levelIdx](2)-1;
        // endCell = Min (endCell, d_cellNum[levelIdx]);
        // ignores extra cells, boundary conditions.
        /*Patch* newPatch =*/ newLevel->addPatch(startCell-extraCells, endCell + IntVector(1,1,1) + extraCells , startCell, endCell + IntVector(1,1,1));
        //newPatch->setLayoutHint(oldPatch->layouthint);
      }
    }
    if((levelIdx < oldGrid->numLevels()) && oldGrid->getLevel(levelIdx)->getPeriodicBoundaries() != IntVector(0,0,0)) {
      newLevel->finalizeLevel(oldGrid->getLevel(levelIdx)->getPeriodicBoundaries().x() != 0,
			      oldGrid->getLevel(levelIdx)->getPeriodicBoundaries().y() != 0,
			      oldGrid->getLevel(levelIdx)->getPeriodicBoundaries().z() != 0);
    }
    else {
      newLevel->finalizeLevel();
    }
    newLevel->assignBCS(grid_ps);
  }

  d_newGrid = true;
  d_lastRegridTimestep = d_sharedState->getCurrentTopLevelTimeStep();

  rdbg << "HierarchicalRegridder::regrid() END" << endl;

  if (*newGrid == *oldGrid) {
    d_newGrid = false;
    delete newGrid;
    return oldGrid;
  }

  return newGrid;
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
    rdbg << "  gathering on level " << toplevel << i << endl;
    const Level* l = oldGrid->getLevel(i).get_rep();
    // gather sub patches from all procs
    
    // place to end up with all subpatches
    vector<SubPatchFlagP> allSubpatches(l->numPatches());
    if (d_myworld->size() > 1) {
      // get ordered list of patch-proc assignment
      vector<int> procAssignment(l->numPatches());
      for (Level::const_patchIterator iter = l->patchesBegin(); iter != l->patchesEnd(); iter++) {
        procAssignment[(*iter)->getLevelIndex()] = lb->getPatchwiseProcessorAssignment(*iter);
      }
      vector<int> sorted_processorAssignment = procAssignment;
      sort(sorted_processorAssignment.begin(), sorted_processorAssignment.end());
      
      vector<int> displs;
      vector<int> recvcounts(d_myworld->myrank(),0); // init the counts to 0
      
      int offsetProc = 0;
      for (i = 0; i < (int)procAssignment.size(); i++) {
        // set the offsets for the MPI_Gatherv
        if ( offsetProc == sorted_processorAssignment[i]) {
          displs.push_back(i);
          offsetProc++;
        }
        recvcounts[sorted_processorAssignment[i]]++;
      }
      
      // for this mpi communication, we're going to Gather all subpatch information to allprocessors.
      // We're going to get what data we have from the DW and put its pointer data in subPatchBuffer.  While
      // we're doing that, we need to prepare the receiving buffer to receive these, so we create a subPatchFlag
      // for each patch we're going to receive from, and then put it in the right order, as the 
      // MPI_Allgatherv puts them in order of processor.
      
      vector<void*> subPatchBuffer;
      vector<void*> allSubpatchBuffer(l->numPatches());
      
      // use this to sort the allSubPatchBuffer in order of processor.  The distls
      // array keeps track of the index of the final array that holds the first
      // item of each processor.  So use this to put allSubPatchBuffer in the right
      // order, by referencing it, and then the next item for that processor will
      // go in that value +1.
      vector<int> subpatchSorter = displs;
      
      
      // get ready to gather the subpatches
      for (Level::const_patchIterator citer = l->patchesBegin();
           citer != l->patchesEnd(); citer++) {
        Patch *patch = *citer;
        int index = subpatchSorter[lb->getPatchwiseProcessorAssignment(patch)]++;
        IntVector patchIndex = patch->getCellLowIndex()/d_patchSize[i];
        // prepare the receiving buffer (well, the buffer's buffer) to be the correct size
        SubPatchFlagP spf1 = new SubPatchFlag(d_latticeRefinementRatio[i]*patchIndex,
                                              d_latticeRefinementRatio[i]*(patchIndex+IntVector(1,1,1)));
        
        allSubpatches[index] = spf1;
        allSubpatchBuffer[index] = spf1.get_rep()->subpatches.getBasePointer();
        
        if (procAssignment[patch->getLevelIndex()] != d_myworld->myrank())
          continue;
        
        // get the variable and prepare to send it: put its base pointer in the send buffer
        PerPatch<SubPatchFlagP> spf2;
        dw->get(spf2, d_activePatchesLabel, 0, patch);
        subPatchBuffer.push_back(spf2.get().get_rep()->subpatches.getBasePointer());
      }
      
      //construct a mpi datatype for the subpatches, using a dummy SubPatchFlag
      // in case this proc doesn't have one on this level
      SubPatchFlagP dummy;
      dummy = new SubPatchFlag(IntVector(0,0,0), d_latticeRefinementRatio[i]);
      CCVariable<int> *ccDummy = &dummy.get_rep()->subpatches;
      MPI_Datatype subpatchtype;
      MPI_Datatype basetype=MPI_INT;
      IntVector low, high, s, strides, dataLow;
      ccDummy->getSizes(low, high, dataLow, s, strides);
      
      //IntVector off = low - dataLow;
      IntVector d = ccDummy->getHighIndex() - ccDummy->getLowIndex();
      MPI_Datatype type1d;
      MPITypeLock.lock();
      MPI_Type_hvector(d.x(), 1, strides.x(), basetype, &type1d);
      MPI_Datatype type2d;
      MPI_Type_hvector(d.y(), 1, strides.y(), type1d, &type2d);
      MPI_Type_free(&type1d);
      MPI_Type_hvector(d.z(), 1, strides.z(), type2d, &subpatchtype);
      MPI_Type_free(&type2d);
      MPI_Type_commit(&subpatchtype);
      MPITypeLock.unlock();  
      
      MPI_Allgatherv(&subPatchBuffer[0],subPatchBuffer.size(),subpatchtype,
                     &allSubpatchBuffer[0], &recvcounts[0], &displs[0], subpatchtype,
                     d_myworld->getComm());
      
      MPITypeLock.lock();
      MPI_Type_free(&subpatchtype);
      MPITypeLock.unlock();
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
    for (unsigned j = 0; j < allSubpatches.size(); j++) {
      rdbg << "   Doing subpatches index " << j << endl;
      for (CellIterator iter(allSubpatches[j].get_rep()->getLowIndex(), allSubpatches[j].get_rep()->getHighIndex()); 
           !iter.done(); iter++) {
        IntVector idx(*iter);
        // if that subpatch is active...
        if ((*allSubpatches[j].get_rep())[idx]) {
          // add this subpatch to become the next level's patch
          rdbg << "Adding normal subpatch " << idx << " to level " << i+1 << endl;
          d_patches[i+1].insert(idx);
          
          if (i > 0) { // don't dilate onto level 0
            IntVector range = Ceil(d_minBoundaryCells.asVector()/d_patchSize[i].asVector());
            for (CellIterator inner(IntVector(-1,-1,-1)*range, range); !inner.done(); inner++) {
              // "dilate" each subpatch, adding it to the patches on the coarser level
              IntVector idx = (idx + *inner) / d_latticeRefinementRatio[i];
              if ((idx.x() < 0 || idx.x() > d_patchNum[i].x()) ||
                  (idx.y() < 0 || idx.y() > d_patchNum[i].y()) ||
                  (idx.z() < 0 || idx.z() > d_patchNum[i].z()))
                break;
              d_patches[i].insert(idx);
            }
          }
        } // end if allSubpatches[j][idx]
        else
          rdbg << " NOT adding subpatch " << idx << " to level " << i+1 << endl;
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

    // in the level code, the refinement ratio is the relation
    // between it and the coarser level.
    newLevel->setRefinementRatio(d_cellRefinementRatio[levelIdx]);
    newLevel->setExtraCells(extraCells);

    rdbg << "HierarchicalRegridder::regrid(): Setting extra cells to be: " << extraCells << endl;

    for (subpatchset::iterator iter = d_patches[levelIdx].begin(); iter != d_patches[levelIdx].end(); iter++) {
      IntVector idx(*iter);
      rdbg << "   Creating patch "<< *iter << endl;
      IntVector startCell       = idx * d_patchSize[levelIdx];
      IntVector endCell         = (idx + IntVector(1,1,1)) * d_patchSize[levelIdx] - IntVector(1,1,1);
      if (idx.x() == d_patchNum[levelIdx](0)-1) endCell(0) = d_cellNum[levelIdx](0)-1;
      if (idx.y() == d_patchNum[levelIdx](1)-1) endCell(1) = d_cellNum[levelIdx](1)-1;
      if (idx.z() == d_patchNum[levelIdx](2)-1) endCell(2) = d_cellNum[levelIdx](2)-1;
      // endCell = Min (endCell, d_cellNum[levelIdx]);
      // ignores extra cells, boundary conditions.
      /*Patch* newPatch =*/ newLevel->addPatch(startCell-extraCells, endCell + IntVector(1,1,1) + extraCells , startCell, endCell + IntVector(1,1,1));
      //newPatch->setLayoutHint(oldPatch->layouthint);
    }
    if((levelIdx < oldGrid->numLevels()) && oldGrid->getLevel(levelIdx)->getPeriodicBoundaries() != IntVector(0,0,0)) {
      newLevel->finalizeLevel(oldGrid->getLevel(levelIdx)->getPeriodicBoundaries().x() != 0,
			      oldGrid->getLevel(levelIdx)->getPeriodicBoundaries().y() != 0,
			      oldGrid->getLevel(levelIdx)->getPeriodicBoundaries().z() != 0);
    }
    else {
      newLevel->finalizeLevel();
    }
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

  return newGrid;
}

