#ifndef UINTAH_HOMEBREW_SimulationInterface_H
#define UINTAH_HOMEBREW_SimulationInterface_H

#include <Packages/Uintah/Core/Parallel/UintahParallelPort.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Util/Handle.h>
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
     virtual void scheduleInitializeAddedMaterial(const LevelP& level,
                                                  SchedulerP&);
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
     virtual void scheduleRefine(const PatchSet* patches, 
				 SchedulerP& scheduler);
     virtual void scheduleRefineInterface(const LevelP& fineLevel, 
				          SchedulerP& scheduler,
					  int step, int nsteps);
     virtual void scheduleCoarsen(const LevelP& coarseLevel, 
				  SchedulerP& scheduler);

     /// Schedule to mark flags for AMR regridding
     virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
					SchedulerP& sched);

     /// Schedule to mark initial flags for AMR regridding
     virtual void scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                               SchedulerP& sched);

     // Redo a timestep if current time advance is not converging.
     // Returned time is the new dt to use.
     virtual double recomputeTimestep(double delt);
     virtual bool restartableTimesteps();

     // if we have a "special" lockstep component, have it take care of its own
     // task graph scheduling rather than have the SimController do it
     virtual bool useLockstepTimeAdvance() { return false;}
     virtual void scheduleLockstepTimeAdvance(const GridP& grid, SchedulerP& sched);

     //////////
     // ask the component if it needs to be recompiled
     virtual bool needRecompile(double /*time*/, double /*dt*/,
				const GridP& /*grid*/) {return false;}

     // direct component to add a new material
     virtual void addMaterial(const ProblemSpecP& params, GridP& grid,
                              SimulationStateP& state);

     virtual void scheduleSwitchTest(const LevelP& /*level*/, SchedulerP& /*sched*/)
       {};

     virtual void addToTimestepXML(ProblemSpecP&) {};
     virtual void readFromTimestepXML(const ProblemSpecP&) {};
 
   private:
     SimulationInterface(const SimulationInterface&);
     SimulationInterface& operator=(const SimulationInterface&);
   };
} // End namespace Uintah
   


#endif
