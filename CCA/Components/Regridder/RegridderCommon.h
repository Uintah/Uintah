#ifndef UINTAH_HOMEBREW_REGRIDDERCOMMON_H
#define UINTAH_HOMEBREW_REGRIDDERCOMMON_H

#include <Packages/Uintah/CCA/Ports/Regridder.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Core/Geometry/IntVector.h>
#include <vector>

using std::vector;
using namespace SCIRun;

namespace Uintah {

class DataWarehouse;
class Patch;

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
    RegridderCommon(ProcessorGroup* pg);
    virtual ~RegridderCommon();

    //! Initialize with regridding parameters from ups file
    virtual void problemSetup(const ProblemSpecP& params, const GridP& grid);

    //! Asks if we need to recompile the task graph.
    //! Will return true if we did a regrid
    virtual bool needRecompile(double time, double delt,
			       const GridP& grid);

  private:
    // input parameters from ups file
    vector<IntVector> d_cellRefinementRatio;
    int d_cellCreationDilation;
    int d_cellDeletionDilation;
    int d_minBoundaryCells; //! min # of cells to be between levels' boundaries

    //! ratio to divide each patch (inner vector is for x,y,z ratio, 
    //! outer vector is a subsequent value per level)
    vector<IntVector> d_latticeRefinementRatio;
    int d_maxLevels;

    // these are structures derived from the code
    vector<IntVector> cell_num;
    vector<IntVector> patch_num;
    vector<IntVector> patch_size;
    

    bool newGrid;

    bool flagCellsExist(DataWarehouse* dw, Patch* patch);
    int  calculateNumberOfPatches();
  };

} // End namespace Uintah

#endif
