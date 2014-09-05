#ifndef UINTAH_HOMEBREW_SimulationInterface_H
#define UINTAH_HOMEBREW_SimulationInterface_H

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
   SimulationInterface
   
   Short description...

GENERAL INFORMATION

   SimulationInterface.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Simulation_Interface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class SimulationInterface : public UintahParallelPort {
   public:
      SimulationInterface();
      virtual ~SimulationInterface();
      
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
      virtual void restartInitialize() {}
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleComputeStableTimestep(const LevelP& level,
						 SchedulerP&) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleTimeAdvance(const LevelP& level, SchedulerP&) = 0;


   private:
      SimulationInterface(const SimulationInterface&);
      SimulationInterface& operator=(const SimulationInterface&);
   };
} // End namespace Uintah
   


#endif
