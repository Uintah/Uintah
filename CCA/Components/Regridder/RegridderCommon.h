#ifndef UINTAH_HOMEBREW_REGRIDDERCOMMON_H
#define UINTAH_HOMEBREW_REGRIDDERCOMMON_H

#include <Packages/Uintah/CCA/Ports/Regridder.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Core/Geometry/IntVector.h>
#include <vector>

using std::vector;

namespace Uintah {

class DataWarehouse;
class Patch;

typedef vector<SCIRun::IntVector> IndexList;
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

    enum {
      FILTER_STAR,
      FILTER_BOX
    };

  private:
     SimulationStateP d_sharedState; ///< to keep track of timesteps

    // input parameters from ups file
    SizeList d_cellRefinementRatio;
    int d_cellCreationDilation;
    int d_cellDeletionDilation;
    int d_minBoundaryCells; //! min # of cells to be between levels' boundaries

    //! ratio to divide each patch (inner vector is for x,y,z ratio, 
    //! outer vector is a subsequent value per level)
    vector<SCIRun::IntVector> d_latticeRefinementRatio;
    int d_maxLevels;

    // these are structures derived from the code
    SizeList cell_num;
    SizeList patch_num;
    SizeList patch_size;

    vector<IndexList> cell_error;
    vector<IndexList> cell_error_create;
    vector<IndexList> cell_error_delete;
    vector<IndexList> cell_error_boundary;

    vector<IndexList> patch_active;
    vector<IndexList> patch_created;
    vector<IndexList> patch_deleted;

    vector<int> num_created;
    vector<int> num_deleted;

    bool newGrid;


    bool flagCellsExist(DataWarehouse* dw, Patch* patch);
    SCIRun::IntVector calculateNumberOfPatches(SCIRun::IntVector& cell_num, SCIRun::IntVector& patch_size);

    IntVector Less    (const IntVector& a, const IntVector& b);
    IntVector Greater (const IntVector& a, const IntVector& b);
    IntVector And     (const IntVector& a, const IntVector& b);
    IntVector Mod     (const IntVector& a, const IntVector& b);
    IntVector Ceil    (const Vector& a);

    void Dilate( IndexList& flaggedCells, IndexList& dilatedflaggedCells, int filterType );
  };

} // End namespace Uintah

#endif
