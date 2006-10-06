#include <Packages/Uintah/CCA/Components/Regridder/RegridderCommon.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Components/Regridder/PerPatchVars.h>
#include <Core/Util/DebugStream.h>
#include <iostream>
#include <sstream>
#include <deque>

using namespace std;
using namespace Uintah;

DebugStream rdbg("Regridder", false);
DebugStream dilate_dbg("Regridder_dilate", false);


RegridderCommon::RegridderCommon(const ProcessorGroup* pg) : Regridder(), UintahParallelComponent(pg)
{
  rdbg << "RegridderCommon::RegridderCommon() BGN" << endl;
  d_filterType = FILTER_STAR;
  d_lastRegridTimestep = 0;
  d_dilatedCellsCreationLabel  = VarLabel::create("DilatedCellsCreation",
                             CCVariable<int>::getTypeDescription());
#if 0
  d_dilatedCellsCreationOldLabel  = VarLabel::create("DilatedCellsCreationOld",
                             CCVariable<int>::getTypeDescription());
#endif
  d_dilatedCellsDeletionLabel = VarLabel::create("DilatedCellsDeletion",
                             CCVariable<int>::getTypeDescription());
  rdbg << "RegridderCommon::RegridderCommon() END" << endl;
}

RegridderCommon::~RegridderCommon()
{
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
  } else if ( d_sharedState->getCurrentTopLevelTimeStep() % d_maxTimestepsBetweenRegrids == 0) {
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

  ProblemSpecP amr_spec = params->findBlock("AMR");
  ProblemSpecP regrid_spec = amr_spec->findBlock("Regridder");
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
                                " and cell_refinement_ratio\n", __FILE__, __LINE__);
  if (simpleRatio == -1 && size == 0)
    throw ProblemSetupException("Must specify either simple_refinement_ratio"
                                " or cell_refinement_ratio\n", __FILE__, __LINE__);

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
  
  // get other init parameters
  d_cellNum.resize(d_maxLevels);

  IntVector low, high;
  oldGrid->getLevel(0)->findCellIndexRange(low, high);
  d_cellNum[0] = high-low - oldGrid->getLevel(0)->getExtraCells()*IntVector(2,2,2);
  for (int k = 1; k < d_maxLevels; k++) {
    d_cellNum[k] = d_cellNum[k-1] * d_cellRefinementRatio[k-1];
  }
  

  d_cellCreationDilation = IntVector(1,1,1);
  d_cellDeletionDilation = IntVector(1,1,1);
  d_minBoundaryCells = IntVector(1,1,1);
  d_maxTimestepsBetweenRegrids = 1;

  regrid_spec->get("cell_creation_dilation", d_cellCreationDilation);
  regrid_spec->get("cell_deletion_dilation", d_cellDeletionDilation);
  regrid_spec->get("min_boundary_cells", d_minBoundaryCells);
  regrid_spec->get("max_timestep_interval", d_maxTimestepsBetweenRegrids);

  // set up filters
  dilate_dbg << "Initializing cell creation filter\n";
  initFilter(d_creationFilter, d_filterType, d_cellCreationDilation);
  dilate_dbg << "Initializing cell deletion filter\n";
  initFilter(d_deletionFilter, d_filterType, d_cellDeletionDilation);
  dilate_dbg << "Initializing patch extension filter\n";
  initFilter(d_patchFilter, FILTER_BOX, d_minBoundaryCells);

  Scheduler* sched = dynamic_cast<Scheduler*>(getPort("scheduler"));
  // we need these so they don't get scrubbed
  sched->overrideVariableBehavior("DilatedCellsCreation", true, false, false);


  rdbg << "RegridderCommon::problemSetup() END" << endl;
}

//_________________________________________________________________
void RegridderCommon::problemSetup_BulletProofing(const int k){

  if(k == 0){  
    for(int dir = 0; dir <3; dir++){
      if (d_cellNum[k][dir] > 1 ) {  // ignore portions of this check for 1D and 2D problems
        if (d_maxTimestepsBetweenRegrids > (d_cellCreationDilation[dir] + 1)) {
          throw ProblemSetupException("Problem Setup: Regridder: max_timestep_interval can be at most 1 greater than any component of \ncell_creation_dilation", __FILE__, __LINE__);
        }
      }
    }
  }

  // For 2D problems the cell Creation/dilation & minBoundaryCells must be 0 in that plane
  for(int dir = 0; dir <3; dir++){
    if(d_cellNum[k][dir] == 1 && 
    (d_cellCreationDilation[dir] != 0 || d_minBoundaryCells[dir] != 0 || d_minBoundaryCells[dir] != 0 )){
    ostringstream msg;
    msg << "Problem Setup: Regridder: The problem you're running is 2D. \n"
        << " You must specifify cell_creation_dilation, cell_deletion_dilation & min_boundary_cells = 0 in that direction \n"
        << "Grid Size " << d_cellNum[k] 
        << " cell_creation_dilation " << d_cellCreationDilation
        << " cell_deletion_dilation " << d_cellDeletionDilation
        << " min_boundary_cells " << d_minBoundaryCells << endl;
    throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
    
    }
  }
}
//______________________________________________________________________
bool RegridderCommon::flaggedCellsExist(constCCVariable<int>& flaggedCells, IntVector low, IntVector high)
{
  //  rdbg << "RegridderCommon::flaggedCellsExist() BGN " << low << " " << high << endl;

  if (high < low) {
    throw InternalError("Regridder has given flagCellsExist incorrect parameters!", __FILE__, __LINE__);
  }
  IntVector newHigh( high + IntVector( 1, 1, 1 ) );
  for ( CellIterator iter( low, newHigh ); !iter.done(); iter++ ) {
    IntVector idx( *iter );
    if (flaggedCells[idx]) {
      //    rdbg << "RegridderCommon::flaggedCellsExist( true ) END " << idx << endl;
      return true;
    }
  }

  //  rdbg << "RegridderCommon::flaggedCellsExist( false ) END" << endl;
  return false;
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
  if ((depth.x() < 0) || (depth.y() < 0) || (depth.z() < 0))  throw InternalError("Regridder given a bad dilation depth!", __FILE__, __LINE__);

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

    default: throw InternalError("Regridder given a bad dilation filter type!", __FILE__, __LINE__);
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

void RegridderCommon::scheduleDilation(SchedulerP& sched, const LevelP& level)
{

  if (level->getIndex() >= d_maxLevels)
    return;

  // dilate flagged cells on this level
  Task* dilate_task = scinew Task("RegridderCommon::Dilate Creation", this,
				  &RegridderCommon::Dilate, DILATE_CREATION);

  int ngc = Max(d_cellCreationDilation.x(), d_cellCreationDilation.y());
  ngc = Max(ngc, d_cellCreationDilation.z());
  
  dilate_task->requires(Task::NewDW, d_sharedState->get_refineFlag_label(), d_sharedState->refineFlagMaterials(),
			Ghost::AroundCells, ngc);

  // we need this task on the init task, but will get bad if you require from old on the init task :)
#if 0
  if (sched->get_dw(0) != 0)
    dilate_task->requires(Task::OldDW, d_dilatedCellsCreationLabel, Ghost::None, 0);
  dilate_task->computes(d_dilatedCellsCreationOldLabel);
#endif
  dilate_task->computes(d_dilatedCellsCreationLabel);
  sched->addTask(dilate_task, level->eachPatch(), d_sharedState->allMaterials());
#if 0
  if (d_cellCreationDilation != d_cellDeletionDilation) {
    // dilate flagged cells (for deletion) on this level)
    Task* dilate_delete_task = scinew Task("RegridderCommon::Dilate Deletion",
					   dynamic_cast<RegridderCommon*>(this),
					   &RegridderCommon::Dilate,
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
  
}

void RegridderCommon::Dilate(const ProcessorGroup*,
			     const PatchSubset* patches,
			     const MaterialSubset* ,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw, DilationType type)
{
  rdbg << "RegridderCommon::Dilate() BGN" << endl;

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
    throw InternalError("Dilate not implemented for this Dilation Type", __FILE__, __LINE__);
  }

  int ngc;
  ngc = Max(depth.x(), depth.y());
  ngc = Max(ngc, depth.z());


  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    constCCVariable<int> flaggedCells;
    CCVariable<int> dilatedFlaggedCells;

#if 0    
    CCVariable<int> dilatedFlaggedOldCells;

    if (old_dw && old_dw->exists(d_dilatedCellsCreationLabel, 0, patch)) {
      constCCVariable<int> oldDilated;
      old_dw->get(oldDilated, d_dilatedCellsCreationLabel, 0, patch, Ghost::None, 0);
      dilatedFlaggedOldCells.copyPointer(oldDilated.castOffConst());
      new_dw->put(dilatedFlaggedOldCells, d_dilatedCellsCreationOldLabel, 0, patch);
    }
    else {
      new_dw->allocateAndPut(dilatedFlaggedOldCells, d_dilatedCellsCreationOldLabel, 0, patch);
      dilatedFlaggedOldCells.initialize(0);
    }
#endif
    new_dw->get(flaggedCells, to_get, 0, patch, Ghost::AroundCells, ngc);
    new_dw->allocateAndPut(dilatedFlaggedCells, to_put, 0, patch);

    IntVector flagLow = patch->getLowIndex();
    IntVector flagHigh = patch->getHighIndex();

    if (patch->getLevel()->getIndex() > 0 && ((d_filterType == FILTER_STAR && ngc > 2) || ngc > 1)) {
      // if we go diagonally along a patch corner where there is no patch, we will need to initialize those cells
      // (see OnDemandDataWarehouse comment about dead cells)
      deque<Box> b1, b2, difference;
      IntVector low = flaggedCells.getLowIndex(), high = flaggedCells.getHighIndex();
      b1.push_back(Box(Point(low(0), low(1), low(2)), 
                       Point(high(0), high(1), high(2))));
      Level::selectType n;
      patch->getLevel()->selectPatches(low, high, n);
      for (int i = 0; i < n.size(); i++) {
        const Patch* p = n[i];
        IntVector low = p->getLowIndex(Patch::CellBased, IntVector(0,0,0));
        IntVector high = p->getHighIndex(Patch::CellBased, IntVector(0,0,0));
        b2.push_back(Box(Point(low(0), low(1), low(2)), Point(high(0), high(1), high(2))));
      }
      difference = Box::difference(b1, b2);
      for (unsigned i = 0; i < difference.size(); i++) {
        Box b = difference[i];
        IntVector low((int)b.lower()(0), (int)b.lower()(1), (int)b.lower()(2));
        IntVector high((int)b.upper()(0), (int)b.upper()(1), (int)b.upper()(2));
        for (CellIterator iter(low, high); !iter.done(); iter++)
          flaggedCells.castOffConst()[*iter] = 0;
      }
    }

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
    if (dilate_dbg.active() && patch->getLevel()->getIndex() == 1) {
      IntVector low  = patch->getInteriorCellLowIndex();
      IntVector high = patch->getInteriorCellHighIndex();
      
      dilate_dbg << "----------------------------------------------------------------" << endl;
      dilate_dbg << "FLAGGED CELLS " << low << " " << high << endl;

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
      dilate_dbg << "DILATED FLAGGED CELLS " << low << " " << high << endl;

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
  rdbg << "RegridderCommon::Dilate() END" << endl;
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
