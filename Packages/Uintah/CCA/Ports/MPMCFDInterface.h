#ifndef UINTAH_HOMEBREW_MPMCFDInterface_H
#define UINTAH_HOMEBREW_MPMCFDInterface_H

#include <Packages/Uintah/Core/Parallel/UintahParallelPort.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/ProblemSpec/Handle.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>

namespace Uintah {
/**************************************

CLASS
   MPMCFDInterface
   
   Short description...

GENERAL INFORMATION

   MPMCFDInterface.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   MPM_CFD_Interface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class MPMCFDInterface : public UintahParallelPort {
   public:
      MPMCFDInterface();
      virtual ~MPMCFDInterface();
      
      //////////
      // Insert Documentation Here:
      virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
				SimulationStateP& state) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleInitialize(const LevelP& level,
				      SchedulerP&) = 0;

      //////////
      // Insert Documentation Here:
      virtual void scheduleComputeStableTimestep(const LevelP& level,
						 SchedulerP&) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleTimeAdvance(double t, double dt,

				       const LevelP& level, SchedulerP&) = 0;

   private:
      MPMCFDInterface(const MPMCFDInterface&);
      MPMCFDInterface& operator=(const MPMCFDInterface&);
   };
} // End namespace Uintah
   
#endif
