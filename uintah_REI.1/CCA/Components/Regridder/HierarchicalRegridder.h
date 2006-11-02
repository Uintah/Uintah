#ifndef UINTAH_HOMEBREW_HIERARCHICALREGRIDDER_H
#define UINTAH_HOMEBREW_HIERARCHICALREGRIDDER_H

#include <Packages/Uintah/CCA/Components/Regridder/RegridderCommon.h>

#include <set>
using std::set;

namespace Uintah {

/**************************************

CLASS
   HierarchicalRegridder
   
   Short description...

GENERAL INFORMATION

   HierarchicalRegridder.h

   Bryan Worthen
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   HierarchicalRegridder

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class VarLabel;

  //! Takes care of AMR Regridding, using a hierarchical algorithm
  class HierarchicalRegridder : public RegridderCommon {
  public:
    HierarchicalRegridder(const ProcessorGroup* pg);
    virtual ~HierarchicalRegridder();

    void problemSetup(const ProblemSpecP& params, const GridP& oldGrid,
                      const SimulationStateP& state);

      
    //! Create a new Grid
    virtual Grid* regrid(Grid* oldGrid, SchedulerP& sched, const ProblemSpecP& ups);

  private:
    Grid* CreateGrid2(Grid* oldGrid, const ProblemSpecP& ups);
    void GatherSubPatches(const GridP& origGrid, SchedulerP& sched);
    void MarkPatches2(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* ,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw);

    inline IntVector StartCellToLattice ( SCIRun::IntVector startCell, int levelIdx );
    SCIRun::IntVector calculateNumberOfPatches(SCIRun::IntVector& cell_num, SCIRun::IntVector& patch_size);
    void problemSetup_BulletProofing(const int k);
    
    //! ratio to divide each patch (inner vector is for x,y,z ratio, 
    //! outer vector is a subsequent value per level)
    vector<SCIRun::IntVector> d_latticeRefinementRatio;

    // these are structures derived from the code
    SizeList d_patchNum;
    SizeList d_patchSize;
    SizeList d_maxPatchSize;
    SizeList d_patchesToCombine;

    vector< CCVariable<int>* > d_patchActive;
    vector< CCVariable<int>* > d_patchCreated;
    vector< CCVariable<int>* > d_patchDeleted;

    // activePatches will not act as a normal variable.  It will only be as large as the number
    // patches a level can be divided into.
    const VarLabel* d_activePatchesLabel;

    typedef set<IntVector> subpatchset;
    vector<subpatchset> d_patches;    
  };

} // End namespace Uintah

#endif
