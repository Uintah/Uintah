/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef UINTAH_HOMEBREW_REGRIDDER_H
#define UINTAH_HOMEBREW_REGRIDDER_H

#include <Core/Parallel/UintahParallelPort.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <vector>

namespace Uintah {

  class UintahParallelComponent;
  class VarLabel;
  
/**************************************

CLASS
   Regridder
   
   Short description...

GENERAL INFORMATION

   Regridder.h

   Bryan Worthen
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   Regridder

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  //! Takes care of AMR Regridding.
  class Regridder : public UintahParallelPort {
  public:
    Regridder();
    virtual ~Regridder();

    virtual std::string getName() = 0;

    // Methods for managing the components attached via the ports.
    virtual void setComponents( UintahParallelComponent *comp ) = 0;
    virtual void getComponents() = 0;
    virtual void releaseComponents() = 0;

    //! Initialize with regridding parameters from ups file
    virtual void problemSetup(const ProblemSpecP& params, const GridP&,
                              const MaterialManagerP& materialManager) = 0;

    virtual void switchInitialize(const ProblemSpecP& params) = 0;

    //! Asks if we need to recompile the task graph.
    virtual bool needRecompile(const GridP& grid) = 0;

    //! Do we need to regrid this timestep?
    virtual bool needsToReGrid(const GridP&) = 0;

    //! Asks if we are going to do regridding
    virtual bool isAdaptive() = 0;
    
    //! switch for setting adaptivity
    virtual void setAdaptivity(const bool ans) = 0;
    
    //! Ask if regridding only once.
    virtual bool doRegridOnce() = 0;

    //! Asks if we are going to do regridding
    virtual bool forceRegridding() = 0;
    
    //! force regridding to happen.
    virtual void setForceRegridding(const bool val) = 0;
    
    //! Schedules task to initialize the error flags to 0
    virtual void scheduleInitializeErrorEstimate(const LevelP& level) = 0;

    //! Schedules task to dilate existing error flags
    virtual void scheduleDilation(const LevelP& level,
				  const bool isLockstepAMR) = 0;

    //! Asks if we are going to do regridding
    virtual bool flaggedCellsOnFinestLevel(const GridP& grid) = 0;

    //! Returns the max number of levels this regridder will store
    virtual int maxLevels() = 0;

    //! Create a new Grid
    virtual Grid* regrid(Grid* oldGrid, const int timeStep) = 0;

    //! If the Regridder set up the load balance in the process of Regridding
    virtual bool isLoadBalanced() { return false; }

    virtual bool useDynamicDilation() = 0;

    virtual std::vector<Uintah::IntVector> getMinPatchSize() = 0;

    virtual void setOverheadAverage(double val) = 0;
    
    virtual const MaterialSubset* refineFlagMaterials() const = 0;
    
    virtual const VarLabel* getRefineFlagLabel() const = 0;
    virtual const VarLabel* getOldRefineFlagLabel() const = 0;
    virtual const VarLabel* getRefinePatchFlagLabel() const = 0;
    
  private:
    Regridder(const Regridder&);
    Regridder& operator=(const Regridder&);
  };

} // End namespace Uintah

#endif
