/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef UINTAH_HOMEBREW_REGRIDDERCOMMON_H
#define UINTAH_HOMEBREW_REGRIDDERCOMMON_H

#include <CCA/Ports/Regridder.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Geometry/IntVector.h>
#include <vector>

using std::vector;

namespace Uintah {

class DataWarehouse;
class Patch;
class VarLabel;

typedef vector<SCIRun::IntVector> SizeList;

/**************************************

CLASS
   RegridderCommon
   
   Short description...

GENERAL INFORMATION

   RegridderCommon.h

   Bryan Worthen
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   RegridderCommon

DESCRIPTION
   Long description...
  
WARNING

****************************************/

  class ProcessorGroup;
  class LoadBalancer;
  class Scheduler;

  //! Takes care of AMR Regridding.  Parent class which takes care
  //! of common regridding functionality.
  class RegridderCommon : public Regridder, public UintahParallelComponent {
  public:
    RegridderCommon(const ProcessorGroup* pg);
    virtual ~RegridderCommon();

    //! Initialize with regridding parameters from ups file
    virtual void problemSetup(const ProblemSpecP& params,
			      const GridP& grid,
			      const SimulationStateP& state);

    //! On a Switch, basically asks whether to turn off/on the Regridding
    virtual void switchInitialize(const ProblemSpecP& params);

    //! Asks if we need to recompile the task graph.
    //! Will return true if we did a regrid
    virtual bool needRecompile(double time, double delt,
			       const GridP& grid);

    //! Do we need to regrid this timestep?
    virtual bool needsToReGrid(const GridP& grid);

    //! Asks if we are going to do regridding
    virtual bool isAdaptive() { return d_isAdaptive; }

    //! Schedules task to initialize the error flags to 0
    virtual void scheduleInitializeErrorEstimate(const LevelP& level);

    //! Schedules task to dilate existing error flags
    virtual void scheduleDilation(const LevelP& level);

    //! Asks if we are going to do regridding
    virtual bool flaggedCellsOnFinestLevel(const GridP& grid);

    //! Returns the max number of levels this regridder will store
    virtual int maxLevels() { return d_maxLevels; }

    virtual bool useDynamicDilation() { return d_dynamicDilation; }

    enum FilterType {
      FILTER_STAR,
      FILTER_BOX
    };

    enum DilationType {
      DILATE_STABILITY,
      DILATE_REGRID,
      DILATE_DELETION,
      DILATE_PATCH
    };
  
    //! initialize the refineFlag variable for this domain (a task callback)
    void initializeErrorEstimate(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse*, DataWarehouse* new_dw);

    void Dilate(const ProcessorGroup*,
                const PatchSubset* patches,
                const MaterialSubset* ,
                DataWarehouse* old_dw,
                DataWarehouse* new_dw,
                const VarLabel* to_put,
                CCVariable<int>* filter,
                IntVector depth);



  protected:

    ProblemSpecP grid_ps_;
    LoadBalancer *lb_;
    Scheduler *sched_;

    SimulationStateP d_sharedState; ///< to keep track of timesteps
    bool d_isAdaptive; //!< if false, do not regrid (stick with what you got)

    // input parameters from ups file
    bool  d_dynamicDilation;
    IntVector d_maxDilation;
    SizeList  d_cellNum; 
    SizeList  d_cellRefinementRatio;
    IntVector d_cellStabilityDilation;
    IntVector d_cellRegridDilation;
    IntVector d_cellDeletionDilation;
    IntVector d_minBoundaryCells; //! min # of cells to be between levels' boundaries
    FilterType d_filterType;

    vector< CCVariable<int>* > d_flaggedCells;
    vector< CCVariable<int>* > d_dilatedCellsStability;
    vector< CCVariable<int>* > d_dilatedCellsRegrid;
    vector< CCVariable<int>* > d_dilatedCellsDeleted;

    map<IntVector,CCVariable<int>* > filters;
    CCVariable<int> d_patchFilter;

    int d_maxLevels;

    // var labels for interior task graph
    const VarLabel* d_dilatedCellsStabilityLabel;
    const VarLabel* d_dilatedCellsStabilityOldLabel;
    const VarLabel* d_dilatedCellsRegridLabel;
    const VarLabel* d_dilatedCellsDeletionLabel;

    vector<int> d_numStability;
    vector<int> d_numRegrid;
    vector<int> d_numDeleted;

    bool d_newGrid;
    int d_lastRegridTimestep;         //The last time the full regridder was called (grid may not change)
    int d_dilationTimestep;           //The last timestep that the dilation was changed
    int d_maxTimestepsBetweenRegrids;
    int d_minTimestepsBetweenRegrids;
    double d_amrOverheadLow;          //Percentage low target for AMR overhead
    double d_amrOverheadHigh;         //Percentage high target for AMR overhead

    bool flaggedCellsExist(constCCVariable<int>& flaggedCells, IntVector low, IntVector high);

    IntVector Less    (const IntVector& a, const IntVector& b);
    IntVector Greater (const IntVector& a, const IntVector& b);
    IntVector And     (const IntVector& a, const IntVector& b);
    IntVector Mod     (const IntVector& a, const IntVector& b);
    IntVector Ceil    (const Vector& a);

    void problemSetup_BulletProofing(const int k);
    void GetFlaggedCells ( const GridP& origGrid, int levelIdx, DataWarehouse* dw );
    void initFilter(CCVariable<int>& filter, FilterType ft, IntVector& depth);
  };

} // End namespace Uintah

#endif
