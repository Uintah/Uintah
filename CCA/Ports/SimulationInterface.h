#ifndef UINTAH_HOMEBREW_SimulationInterface_H
#define UINTAH_HOMEBREW_SimulationInterface_H

#include <Packages/Uintah/Core/Parallel/UintahParallelPort.h>
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

  class DataWarehouse;
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
     virtual void scheduleTimeAdvance(const LevelP& level, SchedulerP&,
				      int step, int nsteps);
     virtual void scheduleRefine(const LevelP& fineLevel, 
				 SchedulerP& scheduler);
     virtual void scheduleRefineInterface(const LevelP& fineLevel, 
				          SchedulerP& scheduler,
					  int step, int nsteps);
     virtual void scheduleCoarsen(const LevelP& coarseLevel, 
				  SchedulerP& scheduler);
     virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
					SchedulerP& sched);

     //////////
     // ask the component if it needs to be recompiled
     virtual bool needRecompile(double /*time*/, double /*dt*/,
				   const GridP& /*grid*/) {return false;}
   private:
     SimulationInterface(const SimulationInterface&);
     SimulationInterface& operator=(const SimulationInterface&);
   };
} // End namespace Uintah
   


#endif
