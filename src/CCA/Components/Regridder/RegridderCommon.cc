/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

//-- Uintah component includes --//
#include <CCA/Components/Regridder/RegridderCommon.h>
#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Scheduler.h>

//-- Uintah framework includes --//
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/PerPatchVars.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DebugStream.h>

//-- system includes --//
#include <iostream>
#include <sstream>
#include <vector>

using namespace Uintah;

static DebugStream rdbg(       "Regridder",        "Regridder", "Regridder debug stream",        false );
static DebugStream dilate_dbg( "Regridder_dilate", "Regridder", "Regridder dilate debug stream", false );
static DebugStream rreason(    "RegridReason",     "Regridder", "Regridder reason debug stream", false );

DebugStream regrider_dbg("GridDBG", "Regridder", "Regridder debug stream", false);

//______________________________________________________________________
//
RegridderCommon::RegridderCommon(const ProcessorGroup* pg)
  : UintahParallelComponent(pg),
    Regridder()
{
  rdbg << "RegridderCommon::RegridderCommon() BGN" << std::endl;
  d_filterType = FILTER_BOX;
  d_lastRegridTimestep = 0;
  d_dilationTimestep = 3;
  d_newGrid = true;
  d_regridOnce = false;
  d_forceRegridding = false;

  d_dilatedCellsStabilityLabel = VarLabel::create("DilatedCellsStability", CCVariable<int>::getTypeDescription());
  d_dilatedCellsRegridLabel = VarLabel::create("DilatedCellsRegrid", CCVariable<int>::getTypeDescription());
  d_dilatedCellsDeletionLabel = VarLabel::create("DilatedCellsDeletion", CCVariable<int>::getTypeDescription());

  m_refineFlagLabel =
    VarLabel::create("refineFlag",      CCVariable<int>::getTypeDescription());
  m_oldRefineFlagLabel =
    VarLabel::create("oldRefineFlag",   CCVariable<int>::getTypeDescription());
  m_refinePatchFlagLabel =
    VarLabel::create("refinePatchFlag", PerPatch<int>::getTypeDescription());

  rdbg << "RegridderCommon::RegridderCommon() END" << std::endl;

  //refine matl subset, only done on matl 0 (matl independent)
  refine_flag_matls = scinew MaterialSubset();
  refine_flag_matls->addReference();
  refine_flag_matls->add(0);
}

//______________________________________________________________________
//
RegridderCommon::~RegridderCommon()
{
  VarLabel::destroy(d_dilatedCellsStabilityLabel);
  VarLabel::destroy(d_dilatedCellsRegridLabel);
  VarLabel::destroy(d_dilatedCellsDeletionLabel);

  VarLabel::destroy(m_refineFlagLabel);
  VarLabel::destroy(m_oldRefineFlagLabel);
  VarLabel::destroy(m_refinePatchFlagLabel);
  
  //delete all filters that were created
  for (std::map<IntVector, CCVariable<int>*>::iterator filter = filters.begin(); filter != filters.end(); filter++) {
    delete (*filter).second;
  }
  filters.clear();

  if(refine_flag_matls && refine_flag_matls->removeReference()){
    delete refine_flag_matls;
  }
}

//______________________________________________________________________
//
void RegridderCommon::getComponents()
{
  m_scheduler = dynamic_cast< Scheduler * >( getPort( "scheduler" ) );

  if( !m_scheduler ) {
    throw InternalError("dynamic_cast of 'm_scheduler' failed!", __FILE__, __LINE__);
  }

  m_loadBalancer = dynamic_cast<LoadBalancer*>( getPort("load balancer") );

  if( !m_loadBalancer ) {
    throw InternalError("dynamic_cast of 'm_loadBalancer' failed!", __FILE__, __LINE__);
  }

  m_application = dynamic_cast<ApplicationInterface*>( getPort("application") );

  if( !m_application ) {
    throw InternalError("dynamic_cast of 'm_application' failed!", __FILE__, __LINE__);
  }
}

//______________________________________________________________________
//
void RegridderCommon::releaseComponents()
{
  releasePort( "scheduler" );
  releasePort( "load balancer" );
  releasePort( "application" );

  m_scheduler    = nullptr;
  m_loadBalancer = nullptr;
  m_application  = nullptr;

  d_materialManager = nullptr;
}

//______________________________________________________________________//
const MaterialSubset* RegridderCommon::refineFlagMaterials() const
{
  ASSERT(refine_flag_matls != 0);
  return refine_flag_matls;
}

//______________________________________________________________________
//
bool
RegridderCommon::needRecompile( const GridP& /*grid*/ )
{
  rdbg << "RegridderCommon::needRecompile() BGN" << std::endl;
  bool retval = d_newGrid;

  if (d_dynamicDilation) {
    if (m_application->getTimeStep() - d_dilationTimestep > 5)  //make sure a semi-decent sample has been taken
    {
      //compute the average overhead

      int numDims = m_loadBalancer->getNumDims();
      int *activeDims = m_loadBalancer->getActiveDims();      
      IntVector newDilation(0, 0, 0);
      
      //if above overhead threshold
      
      if (d_overheadAverage > d_amrOverheadHigh) {
        //increase dilation
        for (int d = 0; d < numDims; d++) {
          int dim = activeDims[d];
          //do not exceed maximum dilation
          if (d_cellRegridDilation[dim] + d_cellStabilityDilation[dim] < d_maxDilation[dim]) {
            newDilation[dim] = d_cellRegridDilation[dim] + 1;
          }
          else {
            newDilation[dim] = d_cellRegridDilation[dim];
          }
        }

        if (newDilation != d_cellRegridDilation) {
          proc0cout << "Increasing Regrid Dilation to:" << newDilation << std::endl;
          d_cellRegridDilation = newDilation;
          //reset the dilation overhead
          d_dilationTimestep = m_application->getTimeStep();
          retval = true;
        }
      }
      //if below overhead threshold
      else if (d_overheadAverage < d_amrOverheadLow) {
        //decrease dilation
        for (int d = 0; d < numDims; d++) {
          int dim = activeDims[d];
          //do not lower dilation to be less than 0
          if (d_cellRegridDilation[dim] > 0) {
            newDilation[dim] = d_cellRegridDilation[dim] - 1;
          }
          else {
            newDilation[dim] = d_cellRegridDilation[dim];
          }
        }
        if (newDilation != d_cellRegridDilation) {
          proc0cout << "Decreasing Regrid Dilation to:" << newDilation << std::endl;
          d_cellRegridDilation = newDilation;
          //reset the dilation overhead
          d_dilationTimestep = m_application->getTimeStep();
          retval = true;
        }
      }
    }
  }
  d_newGrid = false;
  rdbg << "RegridderCommon::needRecompile( " << retval << " ) END" << std::endl;
  return retval;
}

//______________________________________________________________________
//
bool
RegridderCommon::needsToReGrid(const GridP &oldGrid)
{
  rdbg << "RegridderCommon::needsToReGrid() BGN" << std::endl;

  int timeStepsSinceRegrid = m_application->getTimeStep() - d_lastRegridTimestep;

  int retval = false;

  if( d_forceRegridding )
  {
    retval = true;
  }
  else if (!d_isAdaptive || timeStepsSinceRegrid < d_minTimestepsBetweenRegrids) {
    retval = false;
    if (d_myworld->myRank() == 0)
      rreason << "Not regridding because timesteps since regrid is less than min timesteps between regrid\n";

  }
  else if (timeStepsSinceRegrid > d_maxTimestepsBetweenRegrids) {
    retval = true;
    if (d_myworld->myRank() == 0)
      rreason << "Regridding because timesteps since regrid is more than max timesteps between regrid\n";
  }
  else  //check if flags are contained within the finer levels patches
  {
    int result = false;
    DataWarehouse *dw = m_scheduler->getLastDW();
    //for each level finest to coarsest
    for (int l = oldGrid->numLevels() - 1; l >= 0; l--) {
      //if on finest level skip
      if (l == d_maxLevels - 1)
        continue;

      const LevelP coarse_level = oldGrid->getLevel(l);
      LevelP fine_level;

      //get fine level if it exists
      if (l < oldGrid->numLevels() - 1)
        fine_level = oldGrid->getLevel(l + 1);

      //get coarse level patches
      const PatchSubset *patches = m_loadBalancer->getPerProcessorPatchSet(coarse_level)->getSubset(d_myworld->myRank());

      //for each patch
      for (int p = 0; p < patches->size(); p++) {
        const Patch *patch = patches->get(p);
        std::vector < Region > difference;
        patch->getFinestRegionsOnPatch(difference);

        //get flags for coarse patch
        constCCVariable<int> flags;
        dw->get(flags, d_dilatedCellsStabilityLabel, 0, patch, Ghost::None, 0);

        //search non-overlapping
        for (std::vector<Region>::iterator region = difference.begin(); region != difference.end(); region++) {
          for (CellIterator ci(region->getLow(), region->getHigh()); !ci.done(); ci++) {
            if (flags[*ci]) {
              result = true;
              goto GATHER;
            }
          }
        }
      }
    }
    GATHER:
    //Only reduce if we are running in parallel
    if (d_myworld->nRanks() > 1) {
      Uintah::MPI::Allreduce(&result, &retval, 1, MPI_INT, MPI_LOR, d_myworld->getComm());
    }
    else {
      retval = result;
    }
    if (d_myworld->myRank() == 0 && retval) {
      rreason << "Regridding needed because refinement flag was found\n";
    }
  }

  if (retval == true) {
    d_lastRegridTimestep = m_application->getTimeStep();
  }

  rdbg << "RegridderCommon::needsToReGrid( " << retval << " ) END" << std::endl;

  return retval;
}

//______________________________________________________________________
//
bool
RegridderCommon::flaggedCellsOnFinestLevel(const GridP& grid)
{
  rdbg << "RegridderCommon::flaggedCellsOnFinestLevel() BGN" << std::endl;
  const Level* level = grid->getLevel(grid->numLevels() - 1).get_rep();
  DataWarehouse* newDW = m_scheduler->getLastDW();

  // mpi version
  if (d_myworld->nRanks() > 1) {
    int thisproc = false;
    int allprocs;
    for (Level::const_patch_iterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
      // here we assume that the per-patch has been set
      PerPatch<PatchFlagP> flaggedPatchCells;
      if (m_loadBalancer->getPatchwiseProcessorAssignment(*iter) == d_myworld->myRank()) {
        newDW->get(flaggedPatchCells, m_refinePatchFlagLabel, 0, *iter);
        if (flaggedPatchCells.get().get_rep()->flag) {
          thisproc = true;
          break;
        }
      }
    }
    Uintah::MPI::Allreduce(&thisproc, &allprocs, 1, MPI_INT, MPI_MAX, d_myworld->getComm());
    rdbg << "RegridderCommon::flaggedCellsOnFinestLevel() END" << std::endl;
    return allprocs;
  }
  else {
    for (Level::const_patch_iterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
      // here we assume that the per-patch has been set
      PerPatch<PatchFlagP> flaggedPatchCells;
      newDW->get(flaggedPatchCells, m_refinePatchFlagLabel, 0, *iter);
      rdbg << "  finest level, patch " << (*iter)->getID() << flaggedPatchCells.get() << std::endl;

      if (flaggedPatchCells.get().get_rep()->flag) {
        rdbg << "RegridderCommon::flaggedCellsOnFinestLevel( true ) END" << std::endl;
        return true;
      }
    }
    rdbg << " NO FLAGGED PATCHES!\n";
    rdbg << "RegridderCommon::flaggedCellsOnFinestLevel( false ) END" << std::endl;

    return false;
  }
}

//______________________________________________________________________
//
void
RegridderCommon::switchInitialize(const ProblemSpecP& params)
{
  // on a switch, for right now, only change adaptivity.  Maybe
  //   later we can change criteria, but min-patch-size and cell-refinement
  //   need to stay the same
  ProblemSpecP amr_spec = params->findBlock("AMR");
  ProblemSpecP regrid_spec = amr_spec->findBlock("Regridder");

  if (regrid_spec) {
    // only changes if "adaptive" is there
    bool adaptive = d_isAdaptive;
    regrid_spec->get("adaptive", adaptive);
    if (d_myworld->myRank() == 0 && d_isAdaptive != adaptive) {
      if (adaptive) {
        std::cout << "Enabling Regridder\n";
      }
      else {
        std::cout << "Disabling Regridder\n";
      }
    }
    d_isAdaptive = adaptive;
  }
}

//______________________________________________________________________
//
void
RegridderCommon::problemSetup(const ProblemSpecP& params, const GridP& oldGrid, const MaterialManagerP& materialManager)
{
  rdbg << "RegridderCommon::problemSetup() BGN" << std::endl;

  d_materialManager = materialManager;

  grid_ps_ = params->findBlock("Grid");

  ProblemSpecP    amr_spec = params->findBlock("AMR");
  ProblemSpecP regrid_spec = amr_spec->findBlock("Regridder");

  d_isAdaptive = true;  // use if "adaptive" not there
  regrid_spec->get("adaptive", d_isAdaptive);

  if (d_myworld->myRank() == 0 && !d_isAdaptive) {
    std::cout << "Regridder inactive.  Using static Grid.\n";
  }

  // get max num levels
  regrid_spec->require("max_levels", d_maxLevels);

  // get cell refinement ratio
  // get simple ratio - allow user to just say '2'
  int simpleRatio = -1;
  int size;
  regrid_spec->get("simple_refinement_ratio", simpleRatio);
  regrid_spec->get("cell_refinement_ratio", d_cellRefinementRatio);
  size = (int)d_cellRefinementRatio.size();

  if (simpleRatio != -1 && size > 0) {
    throw ProblemSetupException("Cannot specify both simple_refinement_ratio"
                                " and cell_refinement_ratio\n",
                                __FILE__, __LINE__);
  }

  if (simpleRatio == -1 && size == 0) {
    throw ProblemSetupException("Must specify either simple_refinement_ratio"
                                " or cell_refinement_ratio\n",
                                __FILE__, __LINE__);
  }
  // as it is not required to have cellRefinementRatio specified to all levels,
  // expand it to all levels here for convenience in looking up later.
  if (simpleRatio != -1) {
    d_cellRefinementRatio.push_back(IntVector(simpleRatio, simpleRatio, simpleRatio));
    size = 1;
  }

  IntVector lastRatio = d_cellRefinementRatio[size - 1];
  if (size < d_maxLevels) {
    d_cellRefinementRatio.resize(d_maxLevels);
    for (int i = size; i < d_maxLevels; i++) {
      d_cellRefinementRatio[i] = lastRatio;
    }
  }

  // get other init parameters
  d_cellNum.resize(d_maxLevels);

  IntVector low, high;
  oldGrid->getLevel(0)->findInteriorCellIndexRange(low, high);
  d_cellNum[0] = high - low;
  for (int k = 1; k < d_maxLevels; k++) {
    d_cellNum[k] = d_cellNum[k - 1] * d_cellRefinementRatio[k - 1];
  }

  d_maxDilation = IntVector(4, 4, 4);
  d_cellStabilityDilation = IntVector(1, 1, 1);
  d_cellRegridDilation = IntVector(0, 0, 0);
  d_cellDeletionDilation = IntVector(1, 1, 1);
  d_minBoundaryCells = IntVector(1, 1, 1);
  d_maxTimestepsBetweenRegrids = 50;
  d_minTimestepsBetweenRegrids = 1;
  d_overheadAverage = .0;
  d_amrOverheadLow  = .05;
  d_amrOverheadHigh = .15;

  d_dynamicDilation = false;
  regrid_spec->get("regrid_once", d_regridOnce);
  regrid_spec->get("cell_stability_dilation", d_cellStabilityDilation);
  regrid_spec->get("cell_regrid_dilation", d_cellRegridDilation);
  regrid_spec->get("cell_deletion_dilation", d_cellDeletionDilation);
  regrid_spec->get("min_boundary_cells", d_minBoundaryCells);
  regrid_spec->get("max_timestep_interval", d_maxTimestepsBetweenRegrids);
  regrid_spec->get("min_timestep_interval", d_minTimestepsBetweenRegrids);
  regrid_spec->get("dynamic_dilation", d_dynamicDilation);
  regrid_spec->get("amr_overhead_low", d_amrOverheadLow);
  regrid_spec->get("amr_overhead_high", d_amrOverheadHigh);
  regrid_spec->get("max_dilation", d_maxDilation);
  ASSERT(d_amrOverheadLow <= d_amrOverheadHigh);

  // set up filters
  dilate_dbg << "Initializing patch extension filter\n";
  initFilter(d_patchFilter, FILTER_BOX, d_minBoundaryCells);

  // we need these so they don't get scrubbed
  m_scheduler->overrideVariableBehavior("DilatedCellsStability", true, false, false, false, false);
  m_scheduler->overrideVariableBehavior("DilatedCellsRegrid", true, false, false, false, false);

  rdbg << "RegridderCommon::problemSetup() END" << std::endl;
}

//_________________________________________________________________
void
RegridderCommon::problemSetup_BulletProofing(const int k)
{

  if (k == 0) {
    for (int dir = 0; dir < 3; dir++) {
      if (d_cellNum[k][dir] > 1) {  // ignore portions of this check for 1D and 2D problems
        if (d_minTimestepsBetweenRegrids > (d_cellStabilityDilation[dir] + 1)) {
          throw ProblemSetupException(
              "Problem Setup: Regridder: min_timestep_interval can be at most 1 greater than any component of \ncell_stablity_dilation",
              __FILE__, __LINE__);
        }
      }
    }
  }

  // For 2D problems the cell Stability/dilation & minBoundaryCells must be 0 in that plane
  for (int dir = 0; dir < 3; dir++) {
    if (d_cellNum[k][dir] == 1
        && (d_cellStabilityDilation[dir] != 0 || d_cellRegridDilation[dir] != 0 || d_minBoundaryCells[dir] != 0
            || d_minBoundaryCells[dir] != 0)) {
      std::ostringstream msg;
      msg << "Problem Setup: Regridder: The problem you're running is 2D. \n"
          << " You must specifify cell_stablity_dilation & min_boundary_cells = 0 in that direction \n" << "Grid Size "
          << d_cellNum[k] << " cell_stablity_dilation " << d_cellStabilityDilation << " cell_regrid_dilation "
          << d_cellRegridDilation << " min_boundary_cells " << d_minBoundaryCells << std::endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);

    }
  }
}

//______________________________________________________________________
bool
RegridderCommon::flaggedCellsExist(constCCVariable<int>& flaggedCells, IntVector low, IntVector high)
{
  //  rdbg << "RegridderCommon::flaggedCellsExist() BGN " << low << " " << high << std::endl;

  if (high < low) {
    throw InternalError("Regridder has given flagCellsExist incorrect parameters!", __FILE__, __LINE__);
  }
  IntVector newHigh(high + IntVector(1, 1, 1));
  for (CellIterator iter(low, newHigh); !iter.done(); iter++) {
    IntVector idx(*iter);
    if (flaggedCells[idx]) {
      return true;
    }
  }
  return false;
}
//______________________________________________________________________
//  Helpers
IntVector
RegridderCommon::Less(const IntVector& a, const IntVector& b)
{
  return IntVector(a.x() < b.x(), a.y() < b.y(), a.z() < b.z());
}

IntVector
RegridderCommon::Greater(const IntVector& a, const IntVector& b)
{
  return IntVector(a.x() > b.x(), a.y() > b.y(), a.z() > b.z());
}

IntVector
RegridderCommon::And(const IntVector& a, const IntVector& b)
{
  return IntVector(a.x() & b.x(), a.y() & b.y(), a.z() & b.z());
}

IntVector
RegridderCommon::Mod(const IntVector& a, const IntVector& b)
{
  return IntVector(a.x() % b.x(), a.y() % b.y(), a.z() % b.z());
}

IntVector
RegridderCommon::Ceil(const Vector& a)
{
  return IntVector(static_cast<int>(ceil(a.x())), static_cast<int>(ceil(a.y())), static_cast<int>(ceil(a.z())));
}

//______________________________________________________________________
// d
void
RegridderCommon::GetFlaggedCells(const GridP& oldGrid, int levelIdx, DataWarehouse* dw)
{
  rdbg << "RegridderCommon::GetFlaggedCells() BGN" << std::endl;

  // This needs to be fixed for Parallel cases.

  LevelP level = oldGrid->getLevel(levelIdx);

  IntVector minIdx = (*(level->patchesBegin()))->getExtraCellLowIndex();
  IntVector maxIdx = (*(level->patchesBegin()))->getExtraCellHighIndex();

  // This could be a problem because of extra cells.

  for (Level::patch_iterator patchIter = level->patchesBegin(); patchIter != level->patchesEnd(); patchIter++) {
    const Patch* patch = *patchIter;
    minIdx = Min(minIdx, patch->getExtraCellLowIndex());
    maxIdx = Max(maxIdx, patch->getExtraCellHighIndex());
  }

  d_flaggedCells[levelIdx] = scinew CCVariable<int>;
  d_dilatedCellsStability[levelIdx] = scinew CCVariable<int>;
  d_dilatedCellsRegrid[levelIdx] = scinew CCVariable<int>;
  d_dilatedCellsDeleted[levelIdx] = scinew CCVariable<int>;

  d_flaggedCells[levelIdx]->rewindow(minIdx, maxIdx);
  d_dilatedCellsStability[levelIdx]->rewindow(minIdx, maxIdx);
  d_dilatedCellsRegrid[levelIdx]->rewindow(minIdx, maxIdx);
  d_dilatedCellsDeleted[levelIdx]->rewindow(minIdx, maxIdx);

  d_flaggedCells[levelIdx]->initialize(0);
  d_dilatedCellsStability[levelIdx]->initialize(0);
  d_dilatedCellsRegrid[levelIdx]->initialize(0);
  d_dilatedCellsDeleted[levelIdx]->initialize(0);

  // This is only a first step, getting the dilation cells in serial.
  // This is a HUGE memory waste.

  for (Level::patch_iterator patchIter = level->patchesBegin(); patchIter != level->patchesEnd(); patchIter++) {
    const Patch* patch = *patchIter;
    IntVector l(patch->getExtraCellLowIndex());
    IntVector h(patch->getExtraCellHighIndex());

    constCCVariable<int> refineFlag;

    dw->get(refineFlag, m_refineFlagLabel, 0, patch, Ghost::None, 0);

    for (CellIterator iter(l, h); !iter.done(); iter++) {
      IntVector idx(*iter);
      if (refineFlag[idx]) {
        (*d_flaggedCells[levelIdx])[idx] = true;
      }
    }
  }

  rdbg << "RegridderCommon::GetFlaggedCells() END" << std::endl;
}

//______________________________________________________________________
//
void
RegridderCommon::initFilter(CCVariable<int>& filter, FilterType ft, IntVector& depth)
{
  if ((depth.x() < 0) || (depth.y() < 0) || (depth.z() < 0)) {
    throw InternalError("Regridder given a bad dilation depth!", __FILE__, __LINE__);
  }

  filter.rewindow(IntVector(0, 0, 0), IntVector(2, 2, 2) * depth + IntVector(2, 2, 2));
  filter.initialize(0);

  dilate_dbg << "Size of Filter = " << filter.size() << std::endl;

  switch ( ft ) {

    case FILTER_STAR : {
      for (int x = 0; x < 2 * depth.x() + 1; x++) {
        for (int y = 0; y < 2 * depth.y() + 1; y++) {
          for (int z = 0; z < 2 * depth.z() + 1; z++) {
            if ((fabs(static_cast<float>(x - depth.x())) / (depth.x() + 1e-16)
                + fabs(static_cast<float>(y - depth.y())) / (depth.y() + 1e-16)
                + fabs(static_cast<float>(z - depth.z())) / (depth.z() + 1e-16)) <= 1.0) {
              filter[IntVector(x, y, z)] = 1;
            }
          }
        }
      }
      break;
    }

    case FILTER_BOX : {
      filter.initialize(1);
      break;
    }

    default :
      throw InternalError("Regridder given a bad dilation filter type!", __FILE__, __LINE__);
  }

  if (dilate_dbg.active()) {
    dilate_dbg << "----------------------------------------------------------------" << std::endl;
    dilate_dbg << "FILTER" << std::endl;

    for (int z = 0; z < 2 * depth.z() + 1; z++) {
      for (int y = 0; y < 2 * depth.y() + 1; y++) {
        for (int x = 0; x < 2 * depth.x() + 1; x++) {
          dilate_dbg << filter[IntVector(x, y, z)] << " ";
        }
        dilate_dbg << std::endl;
      }
      dilate_dbg << std::endl;
    }
  }
}

//______________________________________________________________________
//
void
RegridderCommon::scheduleDilation(const LevelP& level, const bool isLockstepAMR)
{

  GridP grid = level->getGrid();

  if (level->getIndex() >= d_maxLevels)
    return;

  IntVector stability_depth = d_cellStabilityDilation;
  IntVector regrid_depth = d_cellRegridDilation;
  //IntVector delete_depth=d_cellDeletionDilation;

  if (isLockstepAMR) {
    //scale regrid dilation according to level
    int max_level = std::min(grid->numLevels() - 1, d_maxLevels - 2);   //finest level that is dilated
    int my_level = level->getIndex();

    Vector div(1, 1, 1);
    //calculate divisor
    for (int i = max_level; i > my_level; i--) {
      div = div * d_cellRefinementRatio[i - 1].asVector();
    }
    regrid_depth = Ceil(regrid_depth.asVector() / div);
  }

  regrid_depth = regrid_depth + stability_depth;

  //create filters if needed
  if (filters.find(stability_depth) == filters.end()) {
    filters[stability_depth] = scinew CCVariable<int>;
    initFilter(*filters[stability_depth], d_filterType, stability_depth);
  }

  if (filters.find(regrid_depth) == filters.end()) {
    filters[regrid_depth] = scinew CCVariable<int>;
    initFilter(*filters[regrid_depth], d_filterType, regrid_depth);
  }
  /*
   if(filters.find(delete_depth)==filters.end())
   {
   filters[delete_depth]=scinew CCVariable<int>;
   initFilter(*filters[delete_depth], d_filterType, delete_depth);
   }
   */

  // dilate flagged cells on this level
  Task* dilate_stability_task = scinew Task("RegridderCommon::Dilate Stability", this, &RegridderCommon::Dilate,
                                            d_dilatedCellsStabilityLabel, filters[stability_depth], stability_depth);

  Task* dilate_regrid_task = scinew Task("RegridderCommon::Dilate Regrid", this, &RegridderCommon::Dilate,
                                         d_dilatedCellsRegridLabel, filters[regrid_depth], regrid_depth);

  int ngc_stability = Max(stability_depth.x(), stability_depth.y());
  ngc_stability = Max(ngc_stability, stability_depth.z());

  int ngc_regrid = Max(regrid_depth.x(), regrid_depth.y());
  ngc_regrid = Max(ngc_regrid, regrid_depth.z());

  dilate_stability_task->requires(Task::NewDW, m_refineFlagLabel, refine_flag_matls, Ghost::AroundCells, ngc_stability);
  dilate_regrid_task->requires(Task::NewDW, m_refineFlagLabel, refine_flag_matls, Ghost::AroundCells, ngc_regrid);

  const MaterialSet* all_matls = d_materialManager->allMaterials();

  dilate_stability_task->computes(d_dilatedCellsStabilityLabel, refine_flag_matls);
  m_scheduler->addTask(dilate_stability_task, level->eachPatch(), all_matls);

  dilate_regrid_task->computes(d_dilatedCellsRegridLabel, refine_flag_matls);
  m_scheduler->addTask(dilate_regrid_task, level->eachPatch(), all_matls);
}

//______________________________________________________________________
//
void
RegridderCommon::Dilate(const ProcessorGroup*,
                        const PatchSubset* patches,
                        const MaterialSubset*,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw,
                        const VarLabel* to_put,
                        CCVariable<int>* filter,
                        IntVector depth)

{
  rdbg << "RegridderCommon::Dilate() BGN" << std::endl;

  // change values based on which dilation it is
  int ngc;
  ngc = Max(depth.x(), depth.y());
  ngc = Max(ngc, depth.z());

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    constCCVariable<int> flaggedCells;
    CCVariable<int> dilatedFlaggedCells;

    new_dw->get(flaggedCells, m_refineFlagLabel, 0, patch, Ghost::AroundCells, ngc);
    new_dw->allocateAndPut(dilatedFlaggedCells, to_put, 0, patch);

    IntVector flagLow = patch->getExtraCellLowIndex();
    IntVector flagHigh = patch->getExtraCellHighIndex();

    if (patch->getLevel()->getIndex() > 0 && ((d_filterType == FILTER_STAR && ngc > 2) || ngc > 1)) {
      // if we go diagonally along a patch corner where there is no patch, we will need to initialize those cells
      // (see OnDemandDataWarehouse comment about dead cells)
      std::vector<Region> b1, b2, difference;
      IntVector low = flaggedCells.getLowIndex();
      IntVector high = flaggedCells.getHighIndex();

      b1.push_back(Region(low, high));
      Level::selectType n;
      patch->getLevel()->selectPatches(low, high, n);

      for (unsigned int i = 0; i < n.size(); i++) {
        const Patch* p = n[i];
        IntVector low = p->getExtraLowIndex(Patch::CellBased, IntVector(0, 0, 0));
        IntVector high = p->getExtraHighIndex(Patch::CellBased, IntVector(0, 0, 0));
        b2.push_back(Region(low, high));
      }

      difference = Region::difference(b1, b2);
      for (unsigned i = 0; i < difference.size(); i++) {
        Region b = difference[i];
        IntVector low = b.getLow();
        IntVector high = b.getHigh();

        for (CellIterator iter(low, high); !iter.done(); iter++) {
          flaggedCells.castOffConst()[*iter] = 0;
        }
      }
    }

    //__________________________________
    //
    IntVector low, high;
    for (CellIterator iter(flagLow, flagHigh); !iter.done(); iter++) {
      IntVector idx(*iter);

      low = Max(idx - depth, flaggedCells.getLowIndex());
      high = Min(idx + depth, flaggedCells.getHighIndex() - IntVector(1, 1, 1));
      int flag = 0;

      for (CellIterator local_iter(low, high + IntVector(1, 1, 1)); !local_iter.done(); local_iter++) {
        IntVector local_idx(*local_iter);

        if (flaggedCells[local_idx] * (*filter)[local_idx - idx + depth]) {
          flag = 1;
          break;
        }
      }
      dilatedFlaggedCells[idx] = static_cast<int>(flag);
      //    dilate_dbg << idx << " = " << static_cast<int>(temp > 0) << std::endl;
    }

    rdbg << "G\n";
    //__________________________________
    //  DEBUGGING CODE
    if (dilate_dbg.active() && patch->getLevel()->getIndex() == 1) {
      IntVector low = patch->getCellLowIndex();
      IntVector high = patch->getCellHighIndex();

      dilate_dbg << "----------------------------------------------------------------" << std::endl;
      dilate_dbg << "FLAGGED CELLS " << low << " " << high << std::endl;

      for (int z = low.z(); z < high.z(); z++) {
        for (int y = low.y(); y < high.y(); y++) {
          for (int x = low.x(); x < high.x(); x++) {
            dilate_dbg << flaggedCells[IntVector(x, y, z)] << " ";
          }
          dilate_dbg << std::endl;
        }
        dilate_dbg << std::endl;
      }

      dilate_dbg << "----------------------------------------------------------------" << std::endl;
      dilate_dbg << "DILATED FLAGGED CELLS " << low << " " << high << std::endl;

      for (int z = low.z(); z < high.z(); z++) {
        for (int y = low.y(); y < high.y(); y++) {
          for (int x = low.x(); x < high.x(); x++) {
            dilate_dbg << dilatedFlaggedCells[IntVector(x, y, z)] << " ";
          }
          dilate_dbg << std::endl;
        }
        dilate_dbg << std::endl;
      }
    }
  }
  rdbg << "RegridderCommon::Dilate() END" << std::endl;
}

//______________________________________________________________________
//
void
RegridderCommon::scheduleInitializeErrorEstimate(const LevelP& level)
{
  Task* task = scinew Task("RegridderCommon::initializeErrorEstimate", this, &RegridderCommon::initializeErrorEstimate);

  task->computes(     m_refineFlagLabel, refine_flag_matls);
  task->computes(  m_oldRefineFlagLabel, refine_flag_matls);
  task->computes(m_refinePatchFlagLabel, refine_flag_matls);

  m_scheduler->addTask(task, level->eachPatch(), d_materialManager->allMaterials());
}

//______________________________________________________________________
//
void
RegridderCommon::initializeErrorEstimate(const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* /*matls*/,
                                         DataWarehouse*,
                                         DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    CCVariable<int> refineFlag;
    new_dw->allocateAndPut(refineFlag, m_refineFlagLabel, 0, patch);
    refineFlag.initialize(0);

    CCVariable<int> oldRefineFlag;
    new_dw->allocateAndPut(oldRefineFlag, m_oldRefineFlagLabel, 0, patch);
    oldRefineFlag.initialize(0);

    PerPatch<PatchFlagP> refinePatchFlag(new PatchFlag);
    new_dw->put(refinePatchFlag, m_refinePatchFlagLabel, 0, patch);
  }
}
