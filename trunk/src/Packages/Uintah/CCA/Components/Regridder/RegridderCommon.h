#ifndef UINTAH_HOMEBREW_REGRIDDERCOMMON_H
#define UINTAH_HOMEBREW_REGRIDDERCOMMON_H

#include <Packages/Uintah/CCA/Ports/Regridder.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
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
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   RegridderCommon

DESCRIPTION
   Long description...
  
WARNING

****************************************/

  class ProcessorGroup;

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

    //! Asks if we need to recompile the task graph.
    //! Will return true if we did a regrid
    virtual bool needRecompile(double time, double delt,
			       const GridP& grid);

    //! Do we need to regrid this timestep?
    virtual bool needsToReGrid();

    //! Asks if we are going to do regridding
    virtual bool isAdaptive() { return d_isAdaptive; }

    //! Schedules task to initialize the error flags to 0
    virtual void scheduleInitializeErrorEstimate(SchedulerP& sched, const LevelP& level);

    //! Asks if we are going to do regridding
    virtual bool flaggedCellsOnFinestLevel(const GridP& grid, SchedulerP& sched);

    //! Returns the max number of levels this regridder will store
    virtual int maxLevels() { return d_maxLevels; }

    enum FilterType {
      FILTER_STAR,
      FILTER_BOX
    };

    enum DilationType {
      DILATE_CREATION,
      DILATE_DELETION,
      DILATE_PATCH
    };
  
    //! initialize the refineFlag variable for this domain (a task callback)
    void initializeErrorEstimate(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse*, DataWarehouse* new_dw);

    void Dilate2(const ProcessorGroup*,
                 const PatchSubset* patches,
                 const MaterialSubset* ,
                 DataWarehouse* old_dw,
                 DataWarehouse* new_dw, DilationType type, DataWarehouse* get_dw);

  protected:
     SimulationStateP d_sharedState; ///< to keep track of timesteps
     bool d_isAdaptive; //!< if false, do not regrid (stick with what you got)

    // input parameters from ups file
    SizeList  d_cellRefinementRatio;
    IntVector d_cellCreationDilation;
    IntVector d_cellDeletionDilation;
    vector<int> d_timeRefinementRatio;
    IntVector d_minBoundaryCells; //! min # of cells to be between levels' boundaries
    FilterType d_filterType;

    vector< CCVariable<int>* > d_flaggedCells;
    vector< CCVariable<int>* > d_dilatedCellsCreated;
    vector< CCVariable<int>* > d_dilatedCellsDeleted;

    CCVariable<int> d_creationFilter;
    CCVariable<int> d_deletionFilter;
    CCVariable<int> d_patchFilter;

    //! ratio to divide each patch (inner vector is for x,y,z ratio, 
    //! outer vector is a subsequent value per level)
    vector<SCIRun::IntVector> d_latticeRefinementRatio;
    int d_maxLevels;

    // these are structures derived from the code
    SizeList d_cellNum;
    SizeList d_patchNum;
    SizeList d_patchSize;

    vector< CCVariable<int>* > d_patchActive;
    vector< CCVariable<int>* > d_patchCreated;
    vector< CCVariable<int>* > d_patchDeleted;

    // var labels for interior task graph
    const VarLabel* d_dilatedCellsCreationLabel;
    const VarLabel* d_dilatedCellsDeletionLabel;

    vector<int> d_numCreated;
    vector<int> d_numDeleted;

    bool d_newGrid;
    int d_lastRegridTimestep;
    int d_maxTimestepsBetweenRegrids;

    bool flaggedCellsExist(constCCVariable<int>& flaggedCells, IntVector low, IntVector high);
    SCIRun::IntVector calculateNumberOfPatches(SCIRun::IntVector& cell_num, SCIRun::IntVector& patch_size);

    IntVector Less    (const IntVector& a, const IntVector& b);
    IntVector Greater (const IntVector& a, const IntVector& b);
    IntVector And     (const IntVector& a, const IntVector& b);
    IntVector Mod     (const IntVector& a, const IntVector& b);
    IntVector Ceil    (const Vector& a);

    void Dilate( CCVariable<int>& flaggedCells, CCVariable<int>& dilatedFlaggedCells, CCVariable<int>& filter, IntVector depth );
    void GetFlaggedCells ( const GridP& origGrid, int levelIdx, DataWarehouse* dw );
    void initFilter(CCVariable<int>& filter, FilterType ft, IntVector& depth);
  };

} // End namespace Uintah

#endif
