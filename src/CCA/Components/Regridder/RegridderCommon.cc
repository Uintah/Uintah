/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <TauProfilerForSCIRun.h>
#include <CCA/Components/Regridder/RegridderCommon.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Components/Regridder/PerPatchVars.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Time.h> 
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;
using namespace Uintah;


DebugStream rdbg("Regridder", false);
DebugStream dilate_dbg("Regridder_dilate", false);
DebugStream rreason("RegridReason",false);

RegridderCommon::RegridderCommon(const ProcessorGroup* pg) : Regridder(), UintahParallelComponent(pg)
{
  rdbg << "RegridderCommon::RegridderCommon() BGN" << endl;
  d_filterType = FILTER_BOX;
  d_lastRegridTimestep = 0;
  d_dilationTimestep = 3;
  d_newGrid = true;
  d_dilatedCellsStabilityLabel  = VarLabel::create("DilatedCellsStability",
                             CCVariable<int>::getTypeDescription());
  d_dilatedCellsRegridLabel  = VarLabel::create("DilatedCellsRegrid",
                             CCVariable<int>::getTypeDescription());
#if 0
  d_dilatedCellsStablityOldLabel  = VarLabel::create("DilatedCellsStablityOld",
                             CCVariable<int>::getTypeDescription());
#endif
  d_dilatedCellsDeletionLabel = VarLabel::create("DilatedCellsDeletion",
                             CCVariable<int>::getTypeDescription());

  rdbg << "RegridderCommon::RegridderCommon() END" << endl;
}

RegridderCommon::~RegridderCommon()
{
  VarLabel::destroy(d_dilatedCellsStabilityLabel);
  VarLabel::destroy(d_dilatedCellsRegridLabel);
#if 0
  VarLabel::destroy(d_dilatedCellsStabilityOldLabel);
#endif
  VarLabel::destroy(d_dilatedCellsDeletionLabel);

  //delete all filters that were created
  for(map<IntVector,CCVariable<int>*>::iterator filter=filters.begin();filter!=filters.end();filter++)
  {
    delete (*filter).second;
  }
  filters.clear();
}

bool
RegridderCommon::needRecompile(double /*time*/, double /*delt*/, const GridP& /*grid*/)
{
  rdbg << "RegridderCommon::needRecompile() BGN" << endl;
  bool retval = d_newGrid;
 
  if(d_dynamicDilation)
  {
    if(d_sharedState->getCurrentTopLevelTimeStep()-d_dilationTimestep>5) //make sure a semi-decent sample has been taken
    {
      //compute the average overhead

      //if above overhead threshold
      if(d_sharedState->overheadAvg>d_amrOverheadHigh)
      {
        //increase dilation
        int numDims=d_sharedState->getNumDims();
        int *activeDims=d_sharedState->getActiveDims();
        IntVector newDilation;
        for(int d=0;d<numDims;d++)
        {
          int dim=activeDims[d];
          //do not exceed maximum dilation
          if(d_cellRegridDilation[dim]+d_cellStabilityDilation[dim]<d_maxDilation[dim])
            newDilation[dim]=d_cellRegridDilation[dim]+1;
          else
            newDilation[dim]=d_cellRegridDilation[dim];
        }
        if(newDilation!=d_cellRegridDilation)
        {
          if(d_myworld->myrank()==0)
            cout << "Increasing Regrid Dilation to:" << newDilation << endl;
          d_cellRegridDilation=newDilation;
           //reset the dilation overhead
           d_dilationTimestep=d_sharedState->getCurrentTopLevelTimeStep();
           retval=true;
         }
      }
      //if below overhead threshold
      else if(d_sharedState->overheadAvg<d_amrOverheadLow)
      {          
        //decrease dilation
        int numDims=d_sharedState->getNumDims();
        int *activeDims=d_sharedState->getActiveDims();
        IntVector newDilation(0,0,0);
        for(int d=0;d<numDims;d++)
        {
          int dim=activeDims[d];
          //do not lower dilation to be less than 0
          if(d_cellRegridDilation[dim]>0)
            newDilation[dim]=d_cellRegridDilation[dim]-1;
          else
            newDilation[dim]=d_cellRegridDilation[dim];
        }
        if(newDilation!=d_cellRegridDilation)
        {
          if(d_myworld->myrank()==0)
            cout << "Decreasing Regrid Dilation to:" << newDilation << endl;
          d_cellRegridDilation=newDilation;
          //reset the dilation overhead
          d_dilationTimestep=d_sharedState->getCurrentTopLevelTimeStep();
          retval=true;
        }
      }
    }
  }
  d_newGrid = false;
  rdbg << "RegridderCommon::needRecompile( " << retval << " ) END" << endl;
  return retval;
}


bool RegridderCommon::needsToReGrid(const GridP &oldGrid)
{
  TAU_PROFILE("RegridderCommon::needsToReGrid()", " ", TAU_USER);
  rdbg << "RegridderCommon::needsToReGrid() BGN" << endl;
  int timeStepsSinceRegrid=d_sharedState->getCurrentTopLevelTimeStep() - d_lastRegridTimestep;
  int retval = false;
  
  if (!d_isAdaptive || timeStepsSinceRegrid < d_minTimestepsBetweenRegrids) {
    if(d_myworld->myrank()==0)
      rreason << "Not regridding because timesteps since regrid is less than min timesteps between regrid\n";
    retval = false;
  } else if ( timeStepsSinceRegrid  > d_maxTimestepsBetweenRegrids ) {
    if(d_myworld->myrank()==0)
      rreason << "Regridding because timesteps since regrid is more than max timesteps between regrid\n";
    retval = true;
  }
  else //check if flags are contained within the finer levels patches
  {
    int result=false;
    DataWarehouse *dw=sched_->getLastDW();
    //for each level finest to coarsest
    for(int l=oldGrid->numLevels()-1; l>= 0; l--)
    {
      //if on finest level skip
      if(l==d_maxLevels-1)
        continue;

      const LevelP coarse_level=oldGrid->getLevel(l);
      LevelP fine_level;
      
      //get fine level if it exists
      if(l<oldGrid->numLevels()-1)
        fine_level=oldGrid->getLevel(l+1);
     
      //get coarse level patches
      const PatchSubset *patches=lb_->getPerProcessorPatchSet(coarse_level)->getSubset(d_myworld->myrank());
      
      //for each patch
      for(int p=0;p<patches->size();p++)
      {
        const Patch *patch=patches->get(p);
        vector<Region> difference;
        patch->getFinestRegionsOnPatch(difference);

        //get flags for coarse patch
        constCCVariable<int> flags;
        dw->get(flags, d_dilatedCellsStabilityLabel, 0, patch, Ghost::None, 0);

        //search non-overlapping
        for(vector<Region>::iterator region=difference.begin();region!=difference.end();region++)
        {
          for (CellIterator ci(region->getLow(), region->getHigh()); !ci.done(); ci++)
          {
            if (flags[*ci])
            {
              //rreason << d_myworld->myrank() << " refinement flag found on level:" << l << " at cell:" << *ci << endl;
              result=true;
              goto GATHER;
            }
          }
        }
      }
    }
    GATHER:
    //Only reduce if we are running in parallel
    if(d_myworld->size()>1)
    {
      MPI_Allreduce(&result,&retval,1,MPI_INT,MPI_LOR,d_myworld->getComm());
    }
    else
    {
      retval=result;
    }
    if(d_myworld->myrank()==0 && retval)
        rreason << "Regridding needed because refinement flag was found\n";
  }
  
  if(retval==true)
    d_lastRegridTimestep = d_sharedState->getCurrentTopLevelTimeStep();

  rdbg << "RegridderCommon::needsToReGrid( " << retval << " ) END" << endl;
  return retval;
}

bool RegridderCommon::flaggedCellsOnFinestLevel(const GridP& grid)
{
  rdbg << "RegridderCommon::flaggedCellsOnFinestLevel() BGN" << endl;
  const Level* level = grid->getLevel(grid->numLevels()-1).get_rep();
  DataWarehouse* newDW = sched_->getLastDW();

  // mpi version
  if (d_myworld->size() > 1) {
    int thisproc = false;
    int allprocs;
    for (Level::const_patchIterator iter=level->patchesBegin(); iter != level->patchesEnd(); iter++) {
      // here we assume that the per-patch has been set
      PerPatch<PatchFlagP> flaggedPatchCells;
      if (lb_->getPatchwiseProcessorAssignment(*iter) == d_myworld->myrank()) {
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

void RegridderCommon::switchInitialize(const ProblemSpecP& params)
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
    if (d_myworld->myrank() == 0 && d_isAdaptive != adaptive) {
      if (adaptive)
        cout << "Enabling Regridder\n";
      else
        cout << "Disabling Regridder\n";
    }
    d_isAdaptive = adaptive;
  }
}


void RegridderCommon::problemSetup(const ProblemSpecP& params, 
                                   const GridP& oldGrid,
				   const SimulationStateP& state)

{
  rdbg << "RegridderCommon::problemSetup() BGN" << endl;
  d_sharedState = state;

  grid_ps_ = params->findBlock("Grid");

  sched_=dynamic_cast<Scheduler*>(getPort("scheduler"));
  lb_=dynamic_cast<LoadBalancer*>(getPort("load balancer"));

  ProblemSpecP amr_spec = params->findBlock("AMR");
  ProblemSpecP regrid_spec = amr_spec->findBlock("Regridder");
  d_isAdaptive = true;  // use if "adaptive" not there
  regrid_spec->get("adaptive", d_isAdaptive);
  
  if (d_myworld->myrank() == 0 &&!d_isAdaptive) {
    cout << "Regridder inactive.  Using static Grid.\n";
  }

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
  oldGrid->getLevel(0)->findInteriorCellIndexRange(low, high);
  d_cellNum[0] = high-low;
  for (int k = 1; k < d_maxLevels; k++) {
    d_cellNum[k] = d_cellNum[k-1] * d_cellRefinementRatio[k-1];
  }
  

  d_maxDilation = IntVector(4,4,4);
  d_cellStabilityDilation = IntVector(1,1,1);
  d_cellRegridDilation = IntVector(0,0,0);
  d_cellDeletionDilation = IntVector(1,1,1);
  d_minBoundaryCells = IntVector(1,1,1);
  d_maxTimestepsBetweenRegrids = 50;
  d_minTimestepsBetweenRegrids = 1;
  d_amrOverheadLow=.05;
  d_amrOverheadHigh=.15;

  d_dynamicDilation=false;
  regrid_spec->get("cell_stability_dilation", d_cellStabilityDilation);
  regrid_spec->get("cell_regrid_dilation", d_cellRegridDilation);
  regrid_spec->get("cell_deletion_dilation", d_cellDeletionDilation);
  regrid_spec->get("min_boundary_cells", d_minBoundaryCells);
  regrid_spec->get("max_timestep_interval", d_maxTimestepsBetweenRegrids);
  regrid_spec->get("min_timestep_interval", d_minTimestepsBetweenRegrids);
  regrid_spec->get("dynamic_dilation",d_dynamicDilation);
  regrid_spec->get("amr_overhead_low",d_amrOverheadLow);
  regrid_spec->get("amr_overhead_high",d_amrOverheadHigh);
  regrid_spec->get("max_dilation",d_maxDilation);
  ASSERT(d_amrOverheadLow<=d_amrOverheadHigh);

  // set up filters
  /* 
  dilate_dbg << "Initializing cell deletion filter\n";
  initFilter(d_deletionFilter, d_filterType, d_cellDeletionDilation);
  */
  dilate_dbg << "Initializing patch extension filter\n";
  initFilter(d_patchFilter, FILTER_BOX, d_minBoundaryCells);

  // we need these so they don't get scrubbed
  sched_->overrideVariableBehavior("DilatedCellsStability", true, false, false, false, false);
  sched_->overrideVariableBehavior("DilatedCellsRegrid",    true, false, false, false, false);

  //d_lastRegridTimestep=d_sharedState->getCurrentTopLevelTimeStep();

  rdbg << "RegridderCommon::problemSetup() END" << endl;
}

//_________________________________________________________________
void RegridderCommon::problemSetup_BulletProofing(const int k){
  
  if(d_maxLevels < 2){
    ostringstream msg;
    msg << "\nProblem Setup: Regridder: <max_levels> must be > 1 to use a regridder with adaptive mesh refinement \n";
    throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
  }
  
  if(k == 0){  
    for(int dir = 0; dir <3; dir++){
      if (d_cellNum[k][dir] > 1 ) {  // ignore portions of this check for 1D and 2D problems
        if (d_minTimestepsBetweenRegrids > (d_cellStabilityDilation[dir] + 1)) {
          throw ProblemSetupException("Problem Setup: Regridder: min_timestep_interval can be at most 1 greater than any component of \ncell_stablity_dilation", __FILE__, __LINE__);
        }
      }
    }
  }

  // For 2D problems the cell Stability/dilation & minBoundaryCells must be 0 in that plane
  for(int dir = 0; dir <3; dir++){
    if(d_cellNum[k][dir] == 1 && 
    (d_cellStabilityDilation[dir] != 0 || d_cellRegridDilation[dir] != 0 || d_minBoundaryCells[dir] != 0 || d_minBoundaryCells[dir] != 0 )){
    ostringstream msg;
    msg << "Problem Setup: Regridder: The problem you're running is 2D. \n"
        << " You must specifify cell_stablity_dilation & min_boundary_cells = 0 in that direction \n"
        << "Grid Size " << d_cellNum[k] 
        << " cell_stablity_dilation " << d_cellStabilityDilation
        << " cell_regrid_dilation " << d_cellRegridDilation
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

  IntVector minIdx = (*(level->patchesBegin()))->getExtraCellLowIndex();
  IntVector maxIdx = (*(level->patchesBegin()))->getExtraCellHighIndex();

  // This could be a problem because of extra cells.
  
  for ( Level::patchIterator patchIter = level->patchesBegin(); patchIter != level->patchesEnd(); patchIter++ ) {
    const Patch* patch = *patchIter;
    minIdx = Min( minIdx, patch->getExtraCellLowIndex() );
    maxIdx = Max( maxIdx, patch->getExtraCellHighIndex() );
  }

  d_flaggedCells[levelIdx] = scinew CCVariable<int>;
  d_dilatedCellsStability[levelIdx] = scinew CCVariable<int>;
  d_dilatedCellsRegrid[levelIdx] = scinew CCVariable<int>;
  d_dilatedCellsDeleted[levelIdx] = scinew CCVariable<int>;
  
  d_flaggedCells[levelIdx]->rewindow( minIdx, maxIdx );
  d_dilatedCellsStability[levelIdx]->rewindow( minIdx, maxIdx );
  d_dilatedCellsRegrid[levelIdx]->rewindow( minIdx, maxIdx );
  d_dilatedCellsDeleted[levelIdx]->rewindow( minIdx, maxIdx );

  d_flaggedCells[levelIdx]->initialize(0);
  d_dilatedCellsStability[levelIdx]->initialize(0);
  d_dilatedCellsRegrid[levelIdx]->initialize(0);
  d_dilatedCellsDeleted[levelIdx]->initialize(0);

  // This is only a first step, getting the dilation cells in serial.
  // This is a HUGE memory waste.

  for ( Level::patchIterator patchIter = level->patchesBegin(); patchIter != level->patchesEnd(); patchIter++ ) {
    const Patch* patch = *patchIter;
    IntVector l(patch->getExtraCellLowIndex());
    IntVector h(patch->getExtraCellHighIndex());

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


void RegridderCommon::scheduleDilation(const LevelP& level)
{

  GridP grid=level->getGrid(); 
    
  if (level->getIndex() >= d_maxLevels)
    return;

  IntVector stability_depth=d_cellStabilityDilation;
  IntVector regrid_depth=d_cellRegridDilation;
  //IntVector delete_depth=d_cellDeletionDilation;

#if 1
  if(d_sharedState->d_lockstepAMR)
  {
    //scale regrid dilation according to level
    int max_level=min(grid->numLevels()-1,d_maxLevels-2);   //finest level that is dilated
    int my_level=level->getIndex();
    
    Vector div(1,1,1);
    //calculate divisor
    for(int i=max_level;i>my_level;i--)
    {
      div=div*d_cellRefinementRatio[i-1].asVector();
    }
    regrid_depth=Ceil(regrid_depth.asVector()/div);
  }
#endif  
  regrid_depth=regrid_depth+stability_depth;
  //create filters if needed
  if(filters.find(stability_depth)==filters.end())
  {
    filters[stability_depth]=scinew CCVariable<int>;
    initFilter(*filters[stability_depth], d_filterType, stability_depth);
  }
  if(filters.find(regrid_depth)==filters.end())
  {
    filters[regrid_depth]=scinew CCVariable<int>;
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
  Task* dilate_stability_task = scinew Task("RegridderCommon::Dilate Stability", this,
				  &RegridderCommon::Dilate, d_dilatedCellsStabilityLabel, filters[stability_depth], stability_depth);
  
  Task* dilate_regrid_task = scinew Task("RegridderCommon::Dilate Regrid", this,
				  &RegridderCommon::Dilate, d_dilatedCellsRegridLabel, filters[regrid_depth], regrid_depth);

  int ngc_stability = Max(stability_depth.x(), stability_depth.y());
  ngc_stability = Max(ngc_stability, stability_depth.z());
  
  int ngc_regrid = Max(regrid_depth.x(), regrid_depth.y());
  ngc_regrid = Max(ngc_regrid, regrid_depth.z());
  
  dilate_stability_task->requires(Task::NewDW, d_sharedState->get_refineFlag_label(), d_sharedState->refineFlagMaterials(),
			Ghost::AroundCells, ngc_stability);
  dilate_regrid_task->requires(Task::NewDW, d_sharedState->get_refineFlag_label(), d_sharedState->refineFlagMaterials(),
			Ghost::AroundCells, ngc_regrid);

  // we need this task on the init task, but will get bad if you require from old on the init task :)
#if 0
  if (sched_->get_dw(0) != 0)
    dilate_task->requires(Task::OldDW, d_dilatedCellsStabilityLabel, Ghost::None, 0);
  dilate_task->computes(d_dilatedCellsStabilityOldLabel);
#endif
  dilate_stability_task->computes(d_dilatedCellsStabilityLabel, d_sharedState->refineFlagMaterials());
  sched_->addTask(dilate_stability_task, level->eachPatch(), d_sharedState->allMaterials());
  
  dilate_regrid_task->computes(d_dilatedCellsRegridLabel, d_sharedState->refineFlagMaterials());
  sched_->addTask(dilate_regrid_task, level->eachPatch(), d_sharedState->allMaterials());
#if 0
  if (stability_depth != delete_depth) {
    // dilate flagged cells (for deletion) on this level)
    Task* dilate_delete_task = scinew Task("RegridderCommon::Dilate Deletion",
	  		   dynamic_cast<RegridderCommon*>(this),
	  		   &RegridderCommon::Dilate, d_dilatedCellsDeletionLabel, filters[delete_depth],
           delete_depth);
    
    ngc = Max(delete_depth.x(), delete_depth.y());
    ngc = Max(ngc, delete_depth.z());
   
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
			     DataWarehouse* new_dw, 
           const VarLabel* to_put,
           CCVariable<int>* filter,
           IntVector depth)
 
{
  rdbg << "RegridderCommon::Dilate() BGN" << endl;

  // change values based on which dilation it is
  const VarLabel* to_get = d_sharedState->get_refineFlag_label();

  int ngc;
  ngc = Max(depth.x(), depth.y());
  ngc = Max(ngc, depth.z());


  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    constCCVariable<int> flaggedCells;
    CCVariable<int> dilatedFlaggedCells;

#if 0    
    CCVariable<int> dilatedFlaggedOldCells;

    if (old_dw && old_dw->exists(d_dilatedCellsStabilityLabel, 0, patch)) {
      constCCVariable<int> oldDilated;
      old_dw->get(oldDilated, d_dilatedCellsStabilityLabel, 0, patch, Ghost::None, 0);
      dilatedFlaggedOldCells.copyPointer(oldDilated.castOffConst());
      new_dw->put(dilatedFlaggedOldCells, d_dilatedCellsStablityOldLabel, 0, patch);
    }
    else {
      new_dw->allocateAndPut(dilatedFlaggedOldCells, d_dilatedCellsStablityOldLabel, 0, patch);
      dilatedFlaggedOldCells.initialize(0);
    }
#endif
    new_dw->get(flaggedCells, to_get, 0, patch, Ghost::AroundCells, ngc);
    new_dw->allocateAndPut(dilatedFlaggedCells, to_put, 0, patch);

    IntVector flagLow = patch->getExtraCellLowIndex();
    IntVector flagHigh = patch->getExtraCellHighIndex();

    if (patch->getLevel()->getIndex() > 0 && ((d_filterType == FILTER_STAR && ngc > 2) || ngc > 1)) {
      // if we go diagonally along a patch corner where there is no patch, we will need to initialize those cells
      // (see OnDemandDataWarehouse comment about dead cells)
      vector<Region> b1, b2, difference;
      IntVector low = flaggedCells.getLowIndex(), high = flaggedCells.getHighIndex();
      b1.push_back(Region(low,high));
      Level::selectType n;
      patch->getLevel()->selectPatches(low, high, n);
      for (int i = 0; i < n.size(); i++) {
        const Patch* p = n[i];
        IntVector low = p->getExtraLowIndex(Patch::CellBased, IntVector(0,0,0));
        IntVector high = p->getExtraHighIndex(Patch::CellBased, IntVector(0,0,0));
        b2.push_back(Region(low,high));
      }
      difference = Region::difference(b1, b2);
      for (unsigned i = 0; i < difference.size(); i++) {
        Region b = difference[i];
        IntVector low=b.getLow();
        IntVector high=b.getHigh();
        for (CellIterator iter(low, high); !iter.done(); iter++)
          flaggedCells.castOffConst()[*iter] = 0;
      }
    }

    IntVector low, high;
    for(CellIterator iter(flagLow, flagHigh); !iter.done(); iter++) {
      IntVector idx(*iter);

      low = Max(idx - depth, flaggedCells.getLowIndex());
      high = Min(idx + depth, flaggedCells.getHighIndex() - IntVector(1,1,1));
      int flag=0;
      for(CellIterator local_iter(low, high + IntVector(1,1,1)); !local_iter.done(); local_iter++) {
        IntVector local_idx(*local_iter);
        if(flaggedCells[local_idx]*(*filter)[local_idx-idx+depth])
        {
          flag=1;
          break;
        }
      }
      dilatedFlaggedCells[idx] = static_cast<int>(flag);
      //    dilate_dbg << idx << " = " << static_cast<int>(temp > 0) << endl;
    }

  rdbg << "G\n";
    if (dilate_dbg.active() && patch->getLevel()->getIndex() == 1) {
      IntVector low  = patch->getCellLowIndex();
      IntVector high = patch->getCellHighIndex();
      
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

void RegridderCommon::scheduleInitializeErrorEstimate(const LevelP& level)
{
  Task* task = scinew Task("initializeErrorEstimate", this,
                           &RegridderCommon::initializeErrorEstimate);
 
  task->computes(d_sharedState->get_refineFlag_label(), d_sharedState->refineFlagMaterials());
  task->computes(d_sharedState->get_oldRefineFlag_label(), d_sharedState->refineFlagMaterials());
  task->computes(d_sharedState->get_refinePatchFlag_label(), d_sharedState->refineFlagMaterials());
  sched_->addTask(task, level->eachPatch(), d_sharedState->allMaterials());
  
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
