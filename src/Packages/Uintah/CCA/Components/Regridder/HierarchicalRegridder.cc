#include <Packages/Uintah/CCA/Components/Regridder/HierarchicalRegridder.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
using namespace SCIRun;

extern DebugStream rdbg;


HierarchicalRegridder::HierarchicalRegridder(const ProcessorGroup* pg) : RegridderCommon(pg)
{

}

HierarchicalRegridder::~HierarchicalRegridder()
{

}

Grid* HierarchicalRegridder::regrid(Grid* oldGrid, SchedulerP scheduler, const ProblemSpecP& ups)
{
  ProblemSpecP grid_ps = ups->findBlock("Grid");
  if (!grid_ps) {
    throw InternalError("HierarchicalRegridder::regrid() Grid section of UPS file not found!");
  }

  /*
  scheduler->initialize();

  for ( int levelIndex = 0; levelIndex < oldGrid->numLevels(); levelIndex++ ) {
    Task* task = new Task("RegridderCommon::Dilate2",
			  dynamic_cast<RegridderCommon*>(this),
			  &RegridderCommon::Dilate2);
    scheduler->addTask(task, oldGrid->getLevel(levelIndex)->eachPatch(), sharedState->allMaterials());
  }
	
  scheduler->compile();
  scheduler->execute();
  */

  rdbg << "HierarchicalRegridder::regrid() BGN" << endl;

  //  DataWarehouse* dw = scheduler->get_dw(0);
  DataWarehouse* dw = scheduler->getLastDW();

  d_flaggedCells.resize(d_maxLevels);
  d_dilatedCellsCreated.resize(d_maxLevels);
  d_dilatedCellsDeleted.resize(d_maxLevels);
  d_numCreated.resize(d_maxLevels);
  d_numDeleted.resize(d_maxLevels);

  int levelIdx;

  rdbg << "HierarchicalRegridder::regrid() HERE 1" << endl;

  for (levelIdx = 0; levelIdx < oldGrid->numLevels() && levelIdx < d_maxLevels-1; levelIdx++) {
    d_numCreated[levelIdx] = 0;
    d_numDeleted[levelIdx] = 0;

    GetFlaggedCells( oldGrid, levelIdx, dw );
    Dilate( *(d_flaggedCells[levelIdx]), *(d_dilatedCellsCreated[levelIdx]), d_filterType, d_cellCreationDilation );
    Dilate( *(d_flaggedCells[levelIdx]), *(d_dilatedCellsDeleted[levelIdx]), d_filterType, d_cellDeletionDilation );
    MarkPatches( oldGrid, levelIdx  );
    ExtendPatches( oldGrid, levelIdx );
  }

  rdbg << "HierarchicalRegridder::regrid() HERE 2" << endl;

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
        /*Patch* newPatch =*/ newLevel->addPatch(startCell, endCell + IntVector(1,1,1) , startCell, endCell + IntVector(1,1,1));
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

  return newGrid;
}

void HierarchicalRegridder::MarkPatches( const GridP& oldGrid, int levelIdx  )
{
  rdbg << "HierarchicalRegridder::MarkPatches() BGN" << endl;

  LevelP level = oldGrid->getLevel(levelIdx);

  IntVector subPatchSize = d_patchSize[levelIdx+1]/d_cellRefinementRatio[levelIdx];

  for (Level::patchIterator patchIter = level->patchesBegin(); patchIter != level->patchesEnd(); patchIter++) {
    const Patch* patch = *patchIter;
    IntVector startCell = patch->getCellLowIndex();
    IntVector endCell = patch->getCellHighIndex();
    IntVector latticeIdx = StartCellToLattice( startCell, levelIdx );
    IntVector realPatchSize = endCell - startCell + IntVector(1,1,1);
    IntVector realSubPatchNum = realPatchSize / subPatchSize;
    
    for (CellIterator iter(IntVector(0,0,0), realSubPatchNum); !iter.done(); iter++) {
      IntVector idx(*iter);
      IntVector startCellSubPatch = startCell + idx * subPatchSize;
      IntVector endCellSubPatch = startCell + (idx + IntVector(1,1,1)) * subPatchSize - IntVector(1,1,1);
      IntVector latticeStartIdx = latticeIdx * d_latticeRefinementRatio[levelIdx] + idx;
      IntVector latticeEndIdx = latticeStartIdx + IntVector(1,1,1);
      
      if (flaggedCellsExist(*d_dilatedCellsCreated[levelIdx], startCellSubPatch, endCellSubPatch)) {
        rdbg << "Marking Active [ " << levelIdx+1 << " ]: " << latticeStartIdx << endl;
        (*d_patchActive[levelIdx+1])[latticeStartIdx] = 1;
      }
      if (!flaggedCellsExist(*d_dilatedCellsDeleted[levelIdx], startCellSubPatch, endCellSubPatch)) {
        // Do we need to check for flagged cells in the children?
        IntVector childLatticeStartIdx = latticeStartIdx;
        IntVector childLatticeEndIdx = latticeEndIdx;
        for (int childLevelIdx = levelIdx+1; childLevelIdx < oldGrid->numLevels(); childLevelIdx++) {
          for (CellIterator inner_iter(childLatticeStartIdx, childLatticeEndIdx); !iter.done(); iter++) {
            IntVector inner_idx(*inner_iter);
            rdbg << "Deleting Active [ " << childLevelIdx << " ]: " << inner_idx << endl;
            (*d_patchActive[childLevelIdx])[inner_idx] = 0;
          }
          childLatticeStartIdx = childLatticeStartIdx * d_latticeRefinementRatio[childLevelIdx];
          childLatticeEndIdx = childLatticeEndIdx * d_latticeRefinementRatio[childLevelIdx];
        }
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

    // Loop over child level patches and fill the corresponding parent level patches with PatchCell flags, then dilate
    for (CellIterator iter(IntVector(0,0,0), d_patchNum[childLevelIdx]); !iter.done(); iter++) {
      IntVector idx(*iter);
      if (!(*d_patchActive[childLevelIdx])[idx]) { // Fine patch does not exist, do nothing
        continue;
      }
      rdbg << "Marking child patch at: " << idx << endl;
      IntVector startCell       = idx * d_patchSize[childLevelIdx];
      IntVector endCell         = (idx + IntVector(1,1,1)) * d_patchSize[childLevelIdx] - IntVector(1,1,1);

      IntVector parentStartCell = startCell / d_cellRefinementRatio[parentLevelIdx];
      IntVector parentEndCell   = endCell / d_cellRefinementRatio[parentLevelIdx];
      if (idx.x() == d_patchNum[parentLevelIdx](0)-1) parentEndCell(0) = d_cellNum[parentLevelIdx](0)-1;
      if (idx.y() == d_patchNum[parentLevelIdx](1)-1) parentEndCell(1) = d_cellNum[parentLevelIdx](1)-1;
      if (idx.z() == d_patchNum[parentLevelIdx](2)-1) parentEndCell(2) = d_cellNum[parentLevelIdx](2)-1;
      // parentEndCell             = Min(parentEndCell, d_cellNum[parentLevelIdx]);
      rdbg << " SC: " << startCell << " EC: " << endCell << " PSC: " << parentStartCell << " pec " << parentEndCell << endl;
      for (CellIterator parent_iter(parentStartCell, parentEndCell); !iter.done(); iter++) {
        IntVector parent_idx(*parent_iter);
        patchCells[parent_idx] = 1;
      }
      rdbg << "Done Marking child patch at: " << idx << endl;
    }
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
