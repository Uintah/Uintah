#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Regridder/RegridderCommon.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>

#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Components/Regridder/PerPatchVars.h>
#include <Core/Util/DebugStream.h>
#include <iostream>
#include <sstream>

using namespace std;
using namespace Uintah;

DebugStream rdbg("Regridder", false);
DebugStream dilate_dbg("Regridder_dilate", false);


RegridderCommon::RegridderCommon(const ProcessorGroup* pg) : Regridder(), UintahParallelComponent(pg)
{
  d_filterType = FILTER_STAR;
  d_lastRegridTimestep = 0;
}

RegridderCommon::~RegridderCommon()
{
  for (int k = 0; k < d_maxLevels; k++) {
    delete d_patchActive[k];
    delete d_patchCreated[k];
    delete d_patchDeleted[k];
  }  

  d_patchActive.clear();
  d_patchCreated.clear();
  d_patchDeleted.clear();
}

bool RegridderCommon::needRecompile(double time, double delt, const GridP& grid)
{
  bool retval = d_newGrid;
  d_newGrid = false;
  return retval;
}


bool RegridderCommon::needsToReGrid()
{
  // TODO - do some logic with last timestep and dilation layers
  return d_isAdaptive;
}

bool RegridderCommon::flaggedCellsOnFinestLevel(const GridP& grid, SchedulerP& sched)
{
  const Level* level = grid->getLevel(grid->numLevels()-1).get_rep();
  DataWarehouse* newDW = sched->getLastDW();

  // mpi version
  if (d_myworld->size() > 1) {
    int thisproc = false;
    int allprocs;
    for (Level::const_patchIterator iter=level->patchesBegin(); iter != level->patchesEnd(); iter++) {
      // here we assume that the per-patch has been set
      PerPatch<PatchFlagP> flaggedPatchCells;
      newDW->get(flaggedPatchCells, d_sharedState->get_refinePatchFlag_label(), 0, *iter);
      if (flaggedPatchCells.get().get_rep()->flag) {
        thisproc = true;
        break;
      }
    }
    MPI_Allreduce(&thisproc, &allprocs, 1, MPI_INT, MPI_MAX, d_myworld->getComm());
    return allprocs;
  }
  else { 
    
    for (Level::const_patchIterator iter=level->patchesBegin(); iter != level->patchesEnd(); iter++) {
      // here we assume that the per-patch has been set
      PerPatch<PatchFlagP> flaggedPatchCells;
      newDW->get(flaggedPatchCells, d_sharedState->get_refinePatchFlag_label(), 0, *iter);
      cout << "  finest level, patch " << (*iter)->getID() << flaggedPatchCells.get() << endl;
      if (flaggedPatchCells.get().get_rep()->flag)
        return true;
    }
    cout << " NO FLAGGED PATCHES!\n";
    return false;
  }
  
}

void RegridderCommon::problemSetup(const ProblemSpecP& params, 
                                   const GridP& oldGrid,
				   const SimulationStateP& state)

{
  d_sharedState = state;

  ProblemSpecP regrid_spec = params->findBlock("Regridder");
  if (!regrid_spec) {
    d_isAdaptive = false;
    if (d_myworld->myrank() == 0) {
      cout << "No Regridder section specified.  Using static Grid.\n";
    }
    return;
  }

  d_isAdaptive = true;

  // get max num levels
  regrid_spec->require("max_levels", d_maxLevels);

  // get cell refinement ratio
  // get simple ratio - allow user to just say '2'
  int simpleRatio = -1;
  int size;
  regrid_spec->get("simple_refinement_ratio", simpleRatio);
  regrid_spec->get("cell_refinement_ratio", d_cellRefinementRatio);
  size = (int) d_cellRefinementRatio.size();

  if (simpleRatio != -1 && size > 0)
    throw ProblemSetupException("Cannot specify both simple_refinement_ratio"
                                " and cell_refinement_ratio\n");
  if (simpleRatio == -1 && size == 0)
    throw ProblemSetupException("Must specify either simple_refinement_ratio"
                                " or cell_refinement_ratio\n");

  // as it is not required to have cellRefinementRatio specified to all levels,
  // expand it to all levels here for convenience in looking up later.
  if (simpleRatio != -1) {
    d_cellRefinementRatio.push_back(IntVector(simpleRatio, simpleRatio, simpleRatio));
    size = 1;
  }

  IntVector lastRatio = d_cellRefinementRatio[size - 1];
  if (size < d_maxLevels) {
    d_cellRefinementRatio.resize(d_maxLevels);
    for (int i = size; i < d_maxLevels; i++)
      d_cellRefinementRatio[i] = lastRatio;
  }
  
  // Check for existence TODO

  // get lattice refinement ratio, expand it to max levels
  regrid_spec->require("lattice_refinement_ratio", d_latticeRefinementRatio);
  size = (int) d_latticeRefinementRatio.size();
  lastRatio = d_latticeRefinementRatio[size - 1];
  if (size < d_maxLevels) {
    d_latticeRefinementRatio.resize(d_maxLevels);
    for (int i = size; i < d_maxLevels; i++)
      d_latticeRefinementRatio[i] = lastRatio;
  }

  // get other init parameters
  d_cellCreationDilation = IntVector(1,1,1);
  d_cellDeletionDilation = IntVector(1,1,1);
  d_minBoundaryCells = IntVector(1,1,1);

  regrid_spec->get("cell_creation_dilation", d_cellCreationDilation);
  regrid_spec->get("cell_deletion_dilation", d_cellDeletionDilation);
  regrid_spec->get("min_boundary_cells", d_minBoundaryCells);

  const LevelP level0 = oldGrid->getLevel(0);
  d_cellNum.resize(d_maxLevels);
  d_patchNum.resize(d_maxLevels);
  d_patchSize.resize(d_maxLevels);
  d_patchActive.resize(d_maxLevels);
  d_patchCreated.resize(d_maxLevels);
  d_patchDeleted.resize(d_maxLevels);
  
  // get level0 resolution
  IntVector low, high;
  level0->findCellIndexRange(low, high);
  d_cellNum[0] = high-low - level0->getExtraCells()*IntVector(2,2,2);
  const Patch* patch = level0->selectPatchForCellIndex(IntVector(0,0,0));
  d_patchSize[0] = patch->getInteriorCellHighIndex() - patch->getInteriorCellLowIndex();
  d_patchNum[0] = calculateNumberOfPatches(d_cellNum[0], d_patchSize[0]);
  d_patchActive[0] = new CCVariable<int>;
  d_patchCreated[0] = new CCVariable<int>;
  d_patchDeleted[0] = new CCVariable<int>;
  d_patchActive[0]->rewindow(IntVector(0,0,0), d_patchNum[0]);
  d_patchCreated[0]->rewindow(IntVector(0,0,0), d_patchNum[0]);
  d_patchDeleted[0]->rewindow(IntVector(0,0,0), d_patchNum[0]);
  d_patchActive[0]->initialize(1);
  d_patchCreated[0]->initialize(0);
  d_patchDeleted[0]->initialize(0);
  if ( Mod( d_patchSize[0], d_latticeRefinementRatio[0] ) != IntVector(0,0,0) ) {
    throw InternalError("Regridder: patch size is not divisible by lattice ratio on level 0!");
  }

  for (int k = 1; k < d_maxLevels; k++) {
    d_cellNum[k] = d_cellNum[k-1] * d_cellRefinementRatio[k-1];
    d_patchSize[k] = d_patchSize[k-1] * d_cellRefinementRatio[k-1] /
      d_latticeRefinementRatio[k-1];
    d_patchNum[k] = calculateNumberOfPatches(d_cellNum[k], d_patchSize[k]);
    d_patchActive[k] = new CCVariable<int>;
    d_patchCreated[k] = new CCVariable<int>;
    d_patchDeleted[k] = new CCVariable<int>;
    d_patchActive[k]->rewindow(IntVector(0,0,0), d_patchNum[k]);
    d_patchCreated[k]->rewindow(IntVector(0,0,0), d_patchNum[k]);
    d_patchDeleted[k]->rewindow(IntVector(0,0,0), d_patchNum[k]);
    d_patchActive[k]->initialize(0);
    d_patchCreated[k]->initialize(0);
    d_patchDeleted[k]->initialize(0);
    if (k < (d_maxLevels-1)) {
      if ( Mod( d_patchSize[k], d_latticeRefinementRatio[k] ) != IntVector(0,0,0) ) {
	ostringstream msg;
	msg << "Regridder: patch size is not divisible by lattice ratio on level " << k; 
	throw InternalError(msg.str());
      }
    }
  }
}

bool RegridderCommon::flaggedCellsExist(CCVariable<int>& flaggedCells, IntVector low, IntVector high)
{
  if (low > high) {
    throw InternalError("Regridder has given flagCellsExist incorrect parameters!");
  }

  for(CellIterator iter(low, high); !iter.done(); iter++) {
    IntVector idx(*iter);
    if (flaggedCells[idx]) {
      return true;
    }
  }

  return false;
}

IntVector RegridderCommon::calculateNumberOfPatches(IntVector& cellNum, IntVector& patchSize)
{
  rdbg << "RegridderCommon::calculateNumberOfPatches() BGN" << endl;

  IntVector patchNum = Ceil(Vector(cellNum.x(), cellNum.y(), cellNum.z()) / patchSize);
  IntVector remainder = Mod(cellNum, patchSize);
  IntVector small = And(Less(remainder, patchSize / IntVector(2,2,2)), Greater(remainder,IntVector(0,0,0)));
  
  for ( int i = 0; i < 3; i++ ) {
    if (small[i])
      patchNum[small[i]]--;
  }

  rdbg << "RegridderCommon::calculateNumberOfPatches() END" << endl;
  return patchNum;
}


IntVector RegridderCommon::Less    (const IntVector& a, const IntVector& b)
{
  return IntVector(a.x() < b.x(), a.y() < b.y(), a.z() < b.z());
}

IntVector RegridderCommon::Greater (const IntVector& a, const IntVector& b)
{
  return IntVector(a.x() > b.x(), a.y() > b.y(), a.z() > b.z());
}

IntVector RegridderCommon::And     (const IntVector& a, const IntVector& b)
{
  return IntVector(a.x() & b.x(), a.y() & b.y(), a.z() & b.z());
}

IntVector RegridderCommon::Mod     (const IntVector& a, const IntVector& b)
{
  return IntVector(a.x() % b.x(), a.y() % b.y(), a.z() % b.z());
}

IntVector RegridderCommon::Ceil    (const Vector& a)
{
  return IntVector(static_cast<int>(ceil(a.x())), static_cast<int>(ceil(a.y())), static_cast<int>(ceil(a.z())));
}

void RegridderCommon::GetFlaggedCells ( const GridP& oldGrid, int levelIdx, DataWarehouse* dw )
{
  rdbg << "RegridderCommon::GetFlaggedCells() BGN" << endl;

  // This needs to be fixed for Parallel cases.

  LevelP level = oldGrid->getLevel(levelIdx);

  IntVector minIdx = (*(level->patchesBegin()))->getCellLowIndex();
  IntVector maxIdx = (*(level->patchesBegin()))->getCellHighIndex();

  // This could be a problem because of extra cells.
  
  for ( Level::patchIterator patchIter = level->patchesBegin(); patchIter != level->patchesEnd(); patchIter++ ) {
    const Patch* patch = *patchIter;
    minIdx = Min( minIdx, patch->getCellLowIndex() );
    maxIdx = Max( maxIdx, patch->getCellHighIndex() );
  }

  d_flaggedCells[levelIdx] = new CCVariable<int>;
  d_dilatedCellsCreated[levelIdx] = new CCVariable<int>;
  d_dilatedCellsDeleted[levelIdx] = new CCVariable<int>;
  
  d_flaggedCells[levelIdx]->rewindow( minIdx, maxIdx );
  d_dilatedCellsCreated[levelIdx]->rewindow( minIdx, maxIdx );
  d_dilatedCellsDeleted[levelIdx]->rewindow( minIdx, maxIdx );

  d_flaggedCells[levelIdx]->initialize(0);
  d_dilatedCellsCreated[levelIdx]->initialize(0);
  d_dilatedCellsDeleted[levelIdx]->initialize(0);

  // This is only a first step, getting the dilation cells in serial.
  // This is a HUGE memory waste.

  for ( Level::patchIterator patchIter = level->patchesBegin(); patchIter != level->patchesEnd(); patchIter++ ) {
    const Patch* patch = *patchIter;
    IntVector l(patch->getCellLowIndex());
    IntVector h(patch->getCellHighIndex());

    constCCVariable<int> refineFlag;

    dw->get(refineFlag, d_sharedState->get_refineFlag_label(), 0, patch, Ghost::None, 0);

    for(CellIterator iter(l, h); !iter.done(); iter++){
      IntVector idx(*iter);
      if (refineFlag[idx])
	(*d_flaggedCells[levelIdx])[idx] = true;
    }
  }

  rdbg << "RegridderCommon::GetFlaggedCells() END" << endl;
}

void RegridderCommon::Dilate( CCVariable<int>& flaggedCells, CCVariable<int>& dilatedFlaggedCells, int filterType, IntVector depth )
{
  rdbg << "RegridderCommon::Dilate() BGN" << endl;

  if ((depth.x() < 0) || (depth.y() < 0) || (depth.z() < 0))  throw InternalError("Regridder given a bad dilation depth!");

  // we'll find a better way to do this
  CCVariable<int> filter;
  filter.rewindow(IntVector(0,0,0), IntVector(2,2,2)*depth+IntVector(2,2,2));
  filter.initialize(0);

  dilate_dbg << "Size of Filter = " << filter.size() << endl;

  switch (filterType) {

    case FILTER_STAR: {
      for (int x = 0; x < 2*depth.x()+1; x++) {
	for (int y = 0; y < 2*depth.y()+1; y++) {
	  for (int z = 0; z < 2*depth.z()+1; z++) {
	    if ((fabs(static_cast<float>(x - depth.x()))/(depth.x()+1e-16) +
		 fabs(static_cast<float>(y - depth.y()))/(depth.y()+1e-16) +
		 fabs(static_cast<float>(z - depth.z()))/(depth.z()+1e-16)) <= 1.0) {
	      filter[IntVector(x,y,z)] = 1;
	    }
	  }
	}
      }
      break;
    }

    case FILTER_BOX: {
      filter.initialize(1);
      break;
    }

    default: throw InternalError("Regridder given a bad dilation filter type!");
  }

  dilate_dbg << "----------------------------------------------------------------" << endl;
  dilate_dbg << "FILTER" << endl;

  for (int z = 0; z < 2*depth.z()+1; z++) {
    for (int y = 0; y < 2*depth.y()+1; y++) {
      for (int x = 0; x < 2*depth.x()+1; x++) {
	dilate_dbg << filter[IntVector(x,y,z)] << " ";
      }
      dilate_dbg << endl;
    }
    dilate_dbg << endl;
  }

  IntVector arraySize = flaggedCells.size();
  IntVector dilatedArraySize = dilatedFlaggedCells.size();

  if (arraySize != dilatedArraySize) {
    throw InternalError("Original flagged cell array and dilated flagged cell array do not possess the same size!");
  }

  IntVector flagLow = flaggedCells.getLowIndex();
  IntVector flagHigh = flaggedCells.getHighIndex();

  IntVector low, high;

  for(CellIterator iter(flagLow, flagHigh); !iter.done(); iter++) {
    IntVector idx(*iter);

    low = Max( flagLow, idx - depth );
    high = Min( flagHigh - IntVector(1,1,1), idx + depth );
    int temp = 0;
    for(CellIterator local_iter(low, high); !local_iter.done(); local_iter++) {
      IntVector local_idx(*local_iter);
      temp += flaggedCells[local_idx]*filter[local_idx-idx+depth];
    }
    dilatedFlaggedCells[idx] = static_cast<int>(temp > 0);
  }

  if (dilate_dbg.active()) {
    dilate_dbg << "----------------------------------------------------------------" << endl;
    dilate_dbg << "FLAGGED CELLS" << endl;
    
    for (int z = 0; z < arraySize.z(); z++) {
      for (int y = 0; y < arraySize.y(); y++) {
        for (int x = 0; x < arraySize.x(); x++) {
          dilate_dbg << flaggedCells[IntVector(x,y,z)] << " ";
        }
        dilate_dbg << endl;
      }
      dilate_dbg << endl;
    }
    
    dilate_dbg << "----------------------------------------------------------------" << endl;
    dilate_dbg << "DILATED FLAGGED CELLS" << endl;

    for (int z = 0; z < arraySize.z(); z++) {
      for (int y = 0; y < arraySize.y(); y++) {
        for (int x = 0; x < arraySize.x(); x++) {
          dilate_dbg << dilatedFlaggedCells[IntVector(x,y,z)] << " ";
        }
        dilate_dbg << endl;
      }
      dilate_dbg << endl;
    }
  }

  rdbg << "RegridderCommon::Dilate() END" << endl;
}

void RegridderCommon::Dilate2(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset* ,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw, DilationType dilate_which, FilterType filter_type,
                              IntVector depth)
{
  cout << "D2\n";
}
