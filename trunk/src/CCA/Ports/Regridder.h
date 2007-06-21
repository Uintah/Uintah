#ifndef UINTAH_HOMEBREW_REGRIDDER_H
#define UINTAH_HOMEBREW_REGRIDDER_H

#include <Core/Parallel/UintahParallelPort.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/SchedulerP.h>

#include <CCA/Ports/uintahshare.h>

namespace Uintah {

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
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Regridder

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  //! Takes care of AMR Regridding.
  class UINTAHSHARE Regridder : public UintahParallelPort {
  public:
    Regridder();
    virtual ~Regridder();

    //! Initialize with regridding parameters from ups file
    virtual void problemSetup(const ProblemSpecP& params, const GridP&,
			      const SimulationStateP& state) = 0;

    virtual void switchInitialize(const ProblemSpecP& params) = 0;

    //! Asks if we need to recompile the task graph.
    virtual bool needRecompile(double time, double delt,
			       const GridP& grid) = 0;

    //! Do we need to regrid this timestep?
    virtual bool needsToReGrid(const GridP&) = 0;

    //! Asks if we are going to do regridding
    virtual bool isAdaptive() = 0;

    //! Schedules task to initialize the error flags to 0
    virtual void scheduleInitializeErrorEstimate(const LevelP& level) = 0;

    //! Schedules task to dilate existing error flags
    virtual void scheduleDilation(const LevelP& level) = 0;

    //! Asks if we are going to do regridding
    virtual bool flaggedCellsOnFinestLevel(const GridP& grid) = 0;

    //! Returns the max number of levels this regridder will store
    virtual int maxLevels() = 0;

    //! Create a new Grid
    virtual Grid* regrid(Grid* oldGrid) = 0;

    //! If the Regridder set up the load balance in the process of Regridding
    virtual bool isLoadBalanced() { return false; }
  private:
    Regridder(const Regridder&);
    Regridder& operator=(const Regridder&);
  };

} // End namespace Uintah

#endif
