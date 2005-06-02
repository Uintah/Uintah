#include <Packages/Uintah/CCA/Components/Regridder/RegridderCommon.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
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
  rdbg << "RegridderCommon::RegridderCommon() BGN" << endl;
  d_filterType = FILTER_STAR;
  d_lastRegridTimestep = 0;
  rdbg << "RegridderCommon::RegridderCommon() END" << endl;
}

RegridderCommon::~RegridderCommon()
{
  rdbg << "RegridderCommon::~RegridderCommon() BGN" << endl;
  for (int k = 0; k < d_maxLevels; k++) {
    delete d_patchActive[k];
    delete d_patchCreated[k];
    delete d_patchDeleted[k];
  }  

  d_patchActive.clear();
  d_patchCreated.clear();
  d_patchDeleted.clear();
  rdbg << "RegridderCommon::~RegridderCommon() END" << endl;
}

bool
RegridderCommon::needRecompile(double /*time*/, double /*delt*/, const GridP& /*grid*/)
{
  rdbg << "RegridderCommon::needRecompile() BGN" << endl;
  bool retval = d_newGrid;
  d_newGrid = false;
  rdbg << "RegridderCommon::needRecompile( " << retval << " ) END" << endl;
  return retval;
}


bool RegridderCommon::needsToReGrid()
{
  rdbg << "RegridderCommon::needsToReGrid() BGN" << endl;

  bool retval = false;
  if (!d_isAdaptive) {
    retval = false;
  } else if (d_lastRegridTimestep + d_maxTimestepsBetweenRegrids 
      <= d_sharedState->getCurrentTopLevelTimeStep()) {
    d_lastRegridTimestep = d_sharedState->getCurrentTopLevelTimeStep();
    retval = true;
  }
  rdbg << "RegridderCommon::needsToReGrid( " << retval << " ) END" << endl;
  return retval;
}

bool RegridderCommon::flaggedCellsOnFinestLevel(const GridP& grid, SchedulerP& sched)
{
  rdbg << "RegridderCommon::flaggedCellsOnFinestLevel() BGN" << endl;
  const Level* level = grid->getLevel(grid->numLevels()-1).get_rep();
  DataWarehouse* newDW = sched->getLastDW();

  // mpi version
  if (d_myworld->size() > 1) {
    int thisproc = false;
    int allprocs;
    for (Level::const_patchIterator iter=level->patchesBegin(); iter != level->patchesEnd(); iter++) {
      // here we assume that the per-patch has been set
      PerPatch<PatchFlagP> flaggedPatchCells;
      if (sched->getLoadBalancer()->getPatchwiseProcessorAssignment(*iter) == d_myworld->myrank()) {
        newDW->get(flaggedPatchCells, d_sharedState->get_refinePatchFlag_label(), 0, *iter);
        if (flaggedPatchCells.get().get_rep()->flag) {
          thisproc = true;
          break;
        }
      }
    }
    MPI_Allreduce(&thisproc, &allprocs, 1, MPI_INT, MPI_MAX, d_myworld->getComm());
    rdbg << "RegridderCommon::flaggedCellsOnFinestLevel() END" << endl;
    return allprocs;
  }
  else { 
    
    for (Level::const_patchIterator iter=level->patchesBegin(); iter != level->patchesEnd(); iter++) {
      // here we assume that the per-patch has been set
      PerPatch<PatchFlagP> flaggedPatchCells;
      newDW->get(flaggedPatchCells, d_sharedState->get_refinePatchFlag_label(), 0, *iter);
      rdbg << "  finest level, patch " << (*iter)->getID() << flaggedPatchCells.get() << endl;
      if (flaggedPatchCells.get().get_rep()->flag) {
	rdbg << "RegridderCommon::flaggedCellsOnFinestLevel( true ) END" << endl;
        return true;
      }
    }
    rdbg << " NO FLAGGED PATCHES!\n";
    rdbg << "RegridderCommon::flaggedCellsOnFinestLevel( false ) END" << endl;
    return false;
  }
  
}

void RegridderCommon::problemSetup(const ProblemSpecP& params, 
                                   const GridP& oldGrid,
				   const SimulationStateP& state)

{
  rdbg << "RegridderCommon::problemSetup() BGN" << endl;
  d_sharedState = state;

  ProblemSpecP regrid_spec = params->findBlock("Regridder");
  if (!regrid_spec) {
    d_isAdaptive = false;
    if (d_myworld->myrank() == 0) {
      rdbg << "No Regridder section specified.  Using static Grid.\n";
    }
    rdbg << "RegridderCommon::problemSetup() END" << endl;
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
  d_maxTimestepsBetweenRegrids = 1;

  regrid_spec->get("cell_creation_dilation", d_cellCreationDilation);
  regrid_spec->get("cell_deletion_dilation", d_cellDeletionDilation);
  regrid_spec->get("min_boundary_cells", d_minBoundaryCells);
  regrid_spec->get("max_timestep_interval", d_maxTimestepsBetweenRegrids);

  if (d_maxTimestepsBetweenRegrids > d_cellCreationDilation.x()+1 || 
      d_maxTimestepsBetweenRegrids > d_cellCreationDilation.y()+1 || 
      d_maxTimestepsBetweenRegrids > d_cellCreationDilation.z()+1) {
    throw ProblemSetupException("max_timestep_interval can be at most 1 greater than any component of \ncell_creation_dilation");
  }
    


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
    ostringstream msg;
    msg << "Problem Setup: Regridder: you've specified a patch size that is not divisible by the lattice ratio on level 0 \n"
        << " patch size " <<  d_patchSize[0] << " lattice refinement ratio " << d_latticeRefinementRatio[0] << endl;
    throw ProblemSetupException(msg.str());
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
	msg << "Regridder: patch size is not divisible by  the lattice ratio on level " << k;
       msg << "Problem Setup: Regridder: you've specified a patch size that is not divisible by lattice ratio on level" << k
           << "\npatch size " <<  d_patchSize[k] << " lattice refinement ratio " << d_latticeRefinementRatio[k] << endl; 
	throw ProblemSetupException(msg.str());
      }
    }
  }

  // set up filters
  dilate_dbg << "Initializing cell creation filter\n";
  initFilter(d_creationFilter, d_filterType, d_cellCreationDilation);
  dilate_dbg << "Initializing cell deletion filter\n";
  initFilter(d_deletionFilter, d_filterType, d_cellDeletionDilation);
  dilate_dbg << "Initializing patch extension filter\n";
  initFilter(d_patchFilter, FILTER_BOX, d_minBoundaryCells);
  rdbg << "RegridderCommon::problemSetup() END" << endl;
}

bool RegridderCommon::flaggedCellsExist(constCCVariable<int>& flaggedCells, IntVector low, IntVector high)
{
  //  rdbg << "RegridderCommon::flaggedCellsExist() BGN" << endl;

  if (low > high) {
    throw InternalError("Regridder has given flagCellsExist incorrect parameters!");
  }
  IntVector newHigh( high + IntVector( 1, 1, 1 ) );
  for ( CellIterator iter( low, newHigh ); !iter.done(); iter++ ) {
    IntVector idx( *iter );
    if (flaggedCells[idx]) {
      //      rdbg << "RegridderCommon::flaggedCellsExist( true ) END" << endl;
      return true;
    }
  }

  //  rdbg << "RegridderCommon::flaggedCellsExist( false ) END" << endl;
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

void RegridderCommon::initFilter(CCVariable<int>& filter, FilterType ft, IntVector& depth)
{
  if ((depth.x() < 0) || (depth.y() < 0) || (depth.z() < 0))  throw InternalError("Regridder given a bad dilation depth!");

  filter.rewindow(IntVector(0,0,0), IntVector(2,2,2)*depth+IntVector(2,2,2));
  filter.initialize(0);

  dilate_dbg << "Size of Filter = " << filter.size() << endl;

  switch (ft) {

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

  if (dilate_dbg.active()) {
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
  }
}


void RegridderCommon::Dilate( CCVariable<int>& flaggedCells, CCVariable<int>& dilatedFlaggedCells, CCVariable<int>& filter, IntVector depth )
{
  rdbg << "RegridderCommon::Dilate() BGN" << endl;

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
    for(CellIterator local_iter(low, high + IntVector(1,1,1)); !local_iter.done(); local_iter++) {
      IntVector local_idx(*local_iter);
      temp += flaggedCells[local_idx]*filter[local_idx-idx+depth];
    }
    dilatedFlaggedCells[idx] = static_cast<int>(temp > 0);
  }

  if (dilate_dbg.active()) {
    dilate_dbg << "----------------------------------------------------------------" << endl;
    dilate_dbg << "FLAGGED CELLS" << endl;

    IntVector low  = flaggedCells.getLowIndex();
    IntVector high = flaggedCells.getHighIndex();
    
    for (int z = high.z()-1; z >= low.z(); z--) {
      for (int y = high.y()-1; y >= low.y(); y--) {
        for (int x = low.x(); x < high.x(); x++) {
          dilate_dbg << flaggedCells[IntVector(x,y,z)] << " ";
        }
        dilate_dbg << endl;
      }
      dilate_dbg << endl;
    }
    
    dilate_dbg << "----------------------------------------------------------------" << endl;
    dilate_dbg << "DILATED FLAGGED CELLS" << endl;

    for (int z = high.z()-1; z >= low.z(); z--) {
      for (int y = high.y()-1; y >= low.y(); y--) {
        for (int x = low.x(); x < high.x(); x++) {
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
			      DataWarehouse* /*old_dw*/,
			      DataWarehouse* new_dw, DilationType type, DataWarehouse* get_dw)
{
  rdbg << "RegridderCommon::Dilate2() BGN" << endl;

  // change values based on which dilation it is
  const VarLabel* to_get = d_sharedState->get_refineFlag_label();
  const VarLabel* to_put;
  CCVariable<int>* filter;
  IntVector depth;

  switch (type) {
  case DILATE_CREATION:
    to_put = d_dilatedCellsCreationLabel;
    filter = &d_creationFilter;
    depth = d_cellCreationDilation;
    break;
  case DILATE_DELETION:
    to_put = d_dilatedCellsDeletionLabel;
    filter = &d_deletionFilter;
    depth = d_cellDeletionDilation;
    break;
  default:
    throw InternalError("Dilate not implemented for this Dilation Type");
  }

  int ngc;
  ngc = Max(depth.x(), depth.y());
  ngc = Max(ngc, depth.z());


  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    constCCVariable<int> flaggedCells;
    CCVariable<int> dilatedFlaggedCells;

    get_dw->get(flaggedCells, to_get, 0, patch, Ghost::AroundCells, ngc);
    new_dw->allocateAndPut(dilatedFlaggedCells, to_put, 0, patch);

    IntVector flagLow = patch->getLowIndex();
    IntVector flagHigh = patch->getHighIndex();

    IntVector low, high;
    for(CellIterator iter(flagLow, flagHigh); !iter.done(); iter++) {
      IntVector idx(*iter);

      low = Max(idx - depth, flaggedCells.getLowIndex());
      high = Min(idx + depth, flaggedCells.getHighIndex() - IntVector(1,1,1));
      int temp = 0;
      for(CellIterator local_iter(low, high + IntVector(1,1,1)); !local_iter.done(); local_iter++) {
        IntVector local_idx(*local_iter);
        temp += flaggedCells[local_idx]*(*filter)[local_idx-idx+depth];
      }
      dilatedFlaggedCells[idx] = static_cast<int>(temp > 0);
      //    dilate_dbg << idx << " = " << static_cast<int>(temp > 0) << endl;
    }

  rdbg << "G\n";
    if (dilate_dbg.active()) {
      dilate_dbg << "----------------------------------------------------------------" << endl;
      dilate_dbg << "FLAGGED CELLS" << endl;

      IntVector low  = flaggedCells.getLowIndex();
      IntVector high = flaggedCells.getHighIndex();
      
      for (int z = low.z(); z < high.z(); z++) {
        for (int y = low.y(); y < high.y(); y++) {
          for (int x = low.x(); x < high.x(); x++) {
            dilate_dbg << flaggedCells[IntVector(x,y,z)] << " ";
          }
          dilate_dbg << endl;
        }
        dilate_dbg << endl;
      }
      
      dilate_dbg << "----------------------------------------------------------------" << endl;
      dilate_dbg << "DILATED FLAGGED CELLS" << endl;

      for (int z = low.z(); z < high.z(); z++) {
        for (int y = low.y(); y < high.y(); y++) {
          for (int x = low.x(); x < high.x(); x++) {
            dilate_dbg << dilatedFlaggedCells[IntVector(x,y,z)] << " ";
          }
          dilate_dbg << endl;
        }
        dilate_dbg << endl;
      }
    }
  }
  rdbg << "RegridderCommon::Dilate2() END" << endl;
}

void RegridderCommon::scheduleInitializeErrorEstimate(SchedulerP& sched, const LevelP& level)
{
  Task* task = scinew Task("initializeErrorEstimate", this,
                           &RegridderCommon::initializeErrorEstimate);
 
  task->computes(d_sharedState->get_refineFlag_label(), d_sharedState->refineFlagMaterials());
  task->computes(d_sharedState->get_oldRefineFlag_label(), d_sharedState->refineFlagMaterials());
  task->computes(d_sharedState->get_refinePatchFlag_label(), d_sharedState->refineFlagMaterials());
  sched->addTask(task, level->eachPatch(), d_sharedState->allMaterials());
  
}

void RegridderCommon::initializeErrorEstimate(const ProcessorGroup*,
                                              const PatchSubset* patches,
                                              const MaterialSubset* /*matls*/,
                                              DataWarehouse*,
                                              DataWarehouse* new_dw)
{
  // only make one refineFlag per patch.  Do not loop over matls!
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    CCVariable<int> refineFlag;
    new_dw->allocateAndPut(refineFlag, d_sharedState->get_refineFlag_label(),
                           0, patch);
    refineFlag.initialize(0);
    CCVariable<int> oldRefineFlag;
    new_dw->allocateAndPut(oldRefineFlag, d_sharedState->get_oldRefineFlag_label(),
                           0, patch);
    oldRefineFlag.initialize(0);
    PerPatch<PatchFlagP> refinePatchFlag(new PatchFlag);
    new_dw->put(refinePatchFlag, d_sharedState->get_refinePatchFlag_label(),
                0, patch);
  }
}
