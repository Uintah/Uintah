#ifndef UINTAH_HOMEBREW_MPMInterface_H
#define UINTAH_HOMEBREW_MPMInterface_H

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
   MPMInterface
   
   Short description...

GENERAL INFORMATION

   MPMInterface.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   MPM_Interface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class MPMInterface : public UintahParallelPort {
   public:
      MPMInterface();
      virtual ~MPMInterface();
      
      //////////
      // Insert Documentation Here:
      virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
				SimulationStateP& state) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleInitialize(const LevelP& level,
				      SchedulerP&,
				      DataWarehouseP&) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleComputeStableTimestep(const LevelP& level,
						 SchedulerP&,
						 DataWarehouseP&) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleTimeAdvance(double t, double dt,
				       const LevelP& level, SchedulerP&,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw) = 0;
   private:
      MPMInterface(const MPMInterface&);
      MPMInterface& operator=(const MPMInterface&);
   };
} // End namespace Uintah
   


#endif
