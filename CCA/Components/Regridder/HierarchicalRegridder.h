/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_HOMEBREW_HIERARCHICALREGRIDDER_H
#define UINTAH_HOMEBREW_HIERARCHICALREGRIDDER_H

#include <CCA/Components/Regridder/RegridderCommon.h>

#include <set>

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
    virtual Grid* regrid(Grid* oldGrid);

    virtual std::vector<IntVector> getMinPatchSize() {return d_patchSize;};

  private:
    Grid* CreateGrid2(Grid* oldGrid);
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
    std::vector<SCIRun::IntVector> d_latticeRefinementRatio;

    // these are structures derived from the code
    SizeList d_patchNum;
    SizeList d_patchSize;
    SizeList d_maxPatchSize;
    SizeList d_patchesToCombine;

    std::vector< CCVariable<int>* > d_patchActive;
    std::vector< CCVariable<int>* > d_patchCreated;
    std::vector< CCVariable<int>* > d_patchDeleted;

    // activePatches will not act as a normal variable.  It will only be as large as the number
    // patches a level can be divided into.
    const VarLabel* d_activePatchesLabel;

    typedef std::set<IntVector> subpatchset;
    std::vector<subpatchset> d_patches;    
  };

} // End namespace Uintah

#endif
