#include <Packages/Uintah/CCA/Components/Regridder/HierarchicalRegridder.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
using namespace SCIRun;

extern DebugStream rdbg;
extern DebugStream dilate_dbg;

HierarchicalRegridder::HierarchicalRegridder(const ProcessorGroup* pg) : RegridderCommon(pg)
{
  rdbg << "HierarchicalRegridder::HierarchicalRegridder() BGN" << endl;

  patchCells = VarLabel::create("PatchCells",
                             CCVariable<int>::getTypeDescription());
  dilatedCellsCreation  = VarLabel::create("DilatedCellsCreation",
                             CCVariable<int>::getTypeDescription());
  dilatedCellsDeletion = VarLabel::create("DilatedCellsDeletion",
                             CCVariable<int>::getTypeDescription());
  dilatedCellsPatch = VarLabel::create("DilatedCellsPatch",
                             CCVariable<int>::getTypeDescription());
  activePatches = VarLabel::create("activePatches",
                             CCVariable<int>::getTypeDescription());

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

  int levelIdx;

#ifdef BRYAN
  // this is for dividing the entire regridding problem into patchwise domains
  DataWarehouse* old_dw = scheduler->get_dw(0);
  DataWarehouse* new_dw = scheduler->getLastDW();
  SchedulerP tempsched = scheduler->createSubScheduler();
  tempsched->initialize(3, 1, old_dw, new_dw);
  tempsched->clearMappings();
  tempsched->mapDataWarehouse(Task::ParentOldDW, 0);
  tempsched->mapDataWarehouse(Task::ParentNewDW, 1);
  tempsched->mapDataWarehouse(Task::OldDW, 2);
  tempsched->mapDataWarehouse(Task::NewDW, 3);
  
  scheduler->initialize();

  int ngc;
  for ( int levelIndex = 0; levelIndex < oldGrid->numLevels(); levelIndex++ ) {
    //Task* dummy_task = new Task("DUMMY", this, &HierarchicalRegridder::dummyTask);
    //dummy_task->computes(d_sharedState->get_refineFlag_label());
    //scheduler->addTask(dummy_task, oldGrid->getLevel(levelIndex)->eachPatch(), d_sharedState->allMaterials());

    Task* dilate_task = new Task("RegridderCommon::Dilate2 Creation",
                                 dynamic_cast<RegridderCommon*>(this),
                                 &RegridderCommon::Dilate2, DILATE_CREATION, d_filterType,
                                 d_cellCreationDilation);
    ngc = Max(d_cellCreationDilation.x(), d_cellCreationDilation.y());
    ngc = Max(ngc, d_cellCreationDilation.z());
    
    dilate_task->requires(Task::ParentNewDW, d_sharedState->get_refineFlag_label(), Ghost::AroundCells, ngc);
    dilate_task->computes(dilatedCellsCreation);
    scheduler->addTask(dilate_task, oldGrid->getLevel(levelIndex)->eachPatch(), d_sharedState->allMaterials());
    if (d_cellCreationDilation != d_cellDeletionDilation) {
      // change somehow for deletion instead of creation
      Task* dilate_delete_task = new Task("RegridderCommon::Dilate2 Deletion",
                                          dynamic_cast<RegridderCommon*>(this),
                                          &RegridderCommon::Dilate2, DILATE_DELETION, d_filterType,
                                          d_cellDeletionDilation);
      ngc = Max(d_cellDeletionDilation.x(), d_cellDeletionDilation.y());
      ngc = Max(ngc, d_cellDeletionDilation.z());

      dilate_delete_task->requires(Task::ParentNewDW, d_sharedState->get_refineFlag_label(), Ghost::AroundCells, ngc);
      dilate_delete_task->computes(dilatedCellsDeletion);
      scheduler->addTask(dilate_delete_task, oldGrid->getLevel(levelIndex)->eachPatch(), d_sharedState->allMaterials());
    }
    Task* mark_task = new Task("HierarchicalRegridder::MarkPatches2",
                               this, &HierarchicalRegridder::MarkPatches2);
    mark_task->requires(Task::NewDW, dilatedCellsCreation, Ghost::None);
    if (d_cellCreationDilation != d_cellDeletionDilation)
      mark_task->requires(Task::NewDW, dilatedCellsDeletion, Ghost::None);
    
    mark_task->computes(activePatches);
    mark_task->computes(patchCells);
    scheduler->addTask(mark_task, oldGrid->getLevel(levelIndex)->eachPatch(), d_sharedState->allMaterials());


    
    if (levelIndex > 0 && levelIndex < d_maxLevels - 1) {
      Task* dilate_patch_task = new Task("RegridderCommon::Dilate2 Patch",
                                         dynamic_cast<RegridderCommon*>(this),
                                         &RegridderCommon::Dilate2, DILATE_PATCH, FILTER_BOX,
                                         d_minBoundaryCells);
      ngc = Max(d_minBoundaryCells.x(), d_minBoundaryCells.y());
      ngc = Max(ngc, d_minBoundaryCells.z());

      dilate_patch_task->requires(Task::NewDW, patchCells, Ghost::AroundCells, ngc);
      dilate_patch_task->computes(dilatedCellsPatch);
      scheduler->addTask(dilate_patch_task, oldGrid->getLevel(levelIndex)->eachPatch(), d_sharedState->allMaterials());
      
      Task* extend_task = new Task("HierarchicalRegridder::ExtendPatches2",
                                 this, &HierarchicalRegridder::ExtendPatches2);
      extend_task->requires(Task::NewDW, dilatedCellsPatch, Ghost::None);
      extend_task->requires(Task::NewDW, activePatches, 0, Task::FineLevel,
                            0, Task::NormalDomain, Ghost::None, 0);
      extend_task->modifies(activePatches);
      scheduler->addTask(extend_task, oldGrid->getLevel(levelIndex)->eachPatch(), d_sharedState->allMaterials());
    }
  }
  
  scheduler->compile();
  scheduler->execute();

  // now we need to gather the activePatches
#else

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
    Dilate( *(d_flaggedCells[levelIdx]), *(d_dilatedCellsCreated[levelIdx]), d_filterType, d_cellCreationDilation );
    Dilate( *(d_flaggedCells[levelIdx]), *(d_dilatedCellsDeleted[levelIdx]), d_filterType, d_cellDeletionDilation );
    MarkPatches( oldGrid, levelIdx  );
    ExtendPatches( oldGrid, levelIdx );
  }

#endif

  Grid* newGrid = scinew Grid();

  for (levelIdx=0; levelIdx<d_maxLevels; levelIdx++) {

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
{
  rdbg << "HierarchicalRegridder::ExtendPatches() BGN" << endl;

  for (int childLevelIdx = levelIdx+1; childLevelIdx>0; childLevelIdx--) {
    rdbg << "Extend Patches Level: " << childLevelIdx << endl;

    int parentLevelIdx = childLevelIdx - 1;
    IntVector currentLatticeRefinementRatio = d_latticeRefinementRatio[parentLevelIdx];
    IntVector currentPatchSize = d_patchSize[parentLevelIdx];
    CCVariable<int> patchCells;
    patchCells.rewindow(d_flaggedCells[parentLevelIdx]->getLowIndex(), d_flaggedCells[parentLevelIdx]->getHighIndex());
    patchCells.initialize(0);
    CCVariable<int> dilatedPatchCells;
    dilatedPatchCells.rewindow(d_flaggedCells[parentLevelIdx]->getLowIndex(), d_flaggedCells[parentLevelIdx]->getHighIndex());
    if ( rdbg.active() ) {
      
      rdbg << endl << "  ACTIVE PATCHES  " << endl;
      rdbg << d_patchNum[childLevelIdx] << endl;
      for (CellIterator iter(IntVector(0,0,0), d_patchNum[childLevelIdx]); !iter.done(); iter++) {
        IntVector idx(*iter);
        if ((*d_patchActive[childLevelIdx])[idx]) {
          rdbg << idx << " ACTIVE " << endl;
        }
      }
    }

    // Loop over child level patches and fill the corresponding parent level patches with PatchCell flags, then dilate
    rdbg << "Size of total lattice on level " << childLevelIdx << " = " << d_patchNum[childLevelIdx] << endl;
    for (CellIterator iter(IntVector(0,0,0), d_patchNum[childLevelIdx]); !iter.done(); iter++) {
      IntVector idx(*iter);
      if ((*d_patchActive[childLevelIdx])[idx]) { // only if Fine patch exists
        rdbg << "Marking child patch at: [ " << childLevelIdx << " ]: " << idx << endl;
	IntVector parentIdx       = idx / d_latticeRefinementRatio[parentLevelIdx];
        IntVector startCell       = idx * d_patchSize[childLevelIdx];
        IntVector endCell         = (idx + IntVector(1,1,1)) * d_patchSize[childLevelIdx] - IntVector(1,1,1);
        IntVector parentStartCell = startCell / d_cellRefinementRatio[parentLevelIdx];
        IntVector parentEndCell   = endCell / d_cellRefinementRatio[parentLevelIdx];

	rdbg << "HierarchicalRegridder::ExtendPatches() idx             = " << idx << endl;
	rdbg << "HierarchicalRegridder::ExtendPatches() parentIdx       = " << parentIdx << endl;
	rdbg << "HierarchicalRegridder::ExtendPatches() d_patchSize[ch] = " << d_patchSize[childLevelIdx] << endl;
	rdbg << "HierarchicalRegridder::ExtendPatches() d_cellRefine[p] = " << d_cellRefinementRatio[parentLevelIdx] << endl;
	rdbg << "HierarchicalRegridder::ExtendPatches() startCell       = " << startCell       << endl;
	rdbg << "HierarchicalRegridder::ExtendPatches() endtCell        = " << endCell         << endl;
	rdbg << "HierarchicalRegridder::ExtendPatches() parentStartCell = " << parentStartCell << endl;
	rdbg << "HierarchicalRegridder::ExtendPatches() parentEndCell   = " << parentEndCell   << endl;

        if (parentIdx.x() == d_patchNum[parentLevelIdx](0)-1) parentEndCell(0) = d_cellNum[parentLevelIdx](0)-1;
        if (parentIdx.y() == d_patchNum[parentLevelIdx](1)-1) parentEndCell(1) = d_cellNum[parentLevelIdx](1)-1;
        if (parentIdx.z() == d_patchNum[parentLevelIdx](2)-1) parentEndCell(2) = d_cellNum[parentLevelIdx](2)-1;

	rdbg << "HierarchicalRegridder::ExtendPatches() new_parentStartCell = " << parentStartCell << endl;
	rdbg << "HierarchicalRegridder::ExtendPatches() new_parentEndCell   = " << parentEndCell   << endl;

        // parentEndCell             = Min(parentEndCell, d_cellNum[parentLevelIdx]);

	rdbg << "HierarchicalRegridder::ExtendPatches() Before Cell Iterator" << endl;
        for (CellIterator parent_iter(parentStartCell, parentEndCell+IntVector(1,1,1)); !parent_iter.done(); parent_iter++) {
	  rdbg << "Marking patchCells " << *parent_iter << " to 1" << endl;
          patchCells[*parent_iter] = 1;
        }
	rdbg << "HierarchicalRegridder::ExtendPatches() After Cell Iterator" << endl;
        rdbg << "Done Marking child patch at: " << idx << endl;
      }
      else
        rdbg << "NOT Marking child patch at: [ " << childLevelIdx << " ]: " << idx << endl;
    }
    rdbg << "DONE MARKING PATCHES!\n";
    Dilate(patchCells, dilatedPatchCells, FILTER_BOX, d_minBoundaryCells);
    
    // Loop over parent level patches and mark them as active if their contain dilatedPatchCells

    for (CellIterator iter(d_flaggedCells[parentLevelIdx]->getLowIndex()/d_patchSize[parentLevelIdx],
                           d_flaggedCells[parentLevelIdx]->getHighIndex()/d_patchSize[parentLevelIdx]); 
                           !iter.done(); iter++) {
      IntVector idx(*iter);
      IntVector startCell       = idx * d_patchSize[parentLevelIdx];
      IntVector endCell         = (idx + IntVector(1,1,1)) * d_patchSize[parentLevelIdx] - IntVector(1,1,1);

      if (idx.x() == d_patchNum[parentLevelIdx](0)-1) endCell(0) = d_cellNum[parentLevelIdx](0)-1;
      if (idx.y() == d_patchNum[parentLevelIdx](1)-1) endCell(1) = d_cellNum[parentLevelIdx](1)-1;
      if (idx.z() == d_patchNum[parentLevelIdx](2)-1) endCell(2) = d_cellNum[parentLevelIdx](2)-1;
      // endCell                   = Min(endCell, d_cellNum[parentLevelIdx]);
      
      IntVector latticeIdx      = StartCellToLattice( startCell, parentLevelIdx );
      if (flaggedCellsExist(dilatedPatchCells, startCell, endCell)) {
        (*d_patchActive[parentLevelIdx])[latticeIdx] = 1;
      }
    }
  }

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
  rdbg << "MP2\n";
}
void HierarchicalRegridder::ExtendPatches2(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset* ,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  rdbg << "EP2\n";

}
