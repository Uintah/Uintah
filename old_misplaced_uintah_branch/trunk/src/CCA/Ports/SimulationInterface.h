#ifndef UINTAH_HOMEBREW_SimulationInterface_H
#define UINTAH_HOMEBREW_SimulationInterface_H

#include <Core/Parallel/UintahParallelPort.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Util/Handle.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/SchedulerP.h>
#include <SCIRun/Core/OS/Dir.h>
#include <CCA/Ports/uintahshare.h>

namespace Uintah {

  using SCIRun::Dir;
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
   class UINTAHSHARE SimulationInterface : public UintahParallelPort {
   public:
     SimulationInterface();
     virtual ~SimulationInterface();
      
     //////////
     // Insert Documentation Here:
     virtual void problemSetup(const ProblemSpecP& params, 
                               const ProblemSpecP& restart_prob_spec,
                               GridP& grid, SimulationStateP& state) = 0;

     virtual void outputProblemSpec(ProblemSpecP& ps) {}
     virtual void outputPS(Dir& dir) {}
      
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

     virtual void switchInitialize(const LevelP& level,SchedulerP&) {}
      
     //////////
     // Insert Documentation Here:
     virtual void scheduleComputeStableTimestep(const LevelP& level,
						SchedulerP&) = 0;
      
     //////////
     // Insert Documentation Here:
     virtual void scheduleTimeAdvance(const LevelP& level, SchedulerP&);

     // this is for wrapping up a timestep when it can't be done in scheduleTimeAdvance.
     virtual void scheduleFinalizeTimestep(const LevelP& level, SchedulerP&) {}
     virtual void scheduleRefine(const PatchSet* patches, 
				 SchedulerP& scheduler);
     virtual void scheduleRefineInterface(const LevelP& fineLevel, 
				          SchedulerP& scheduler,
					  bool needCoarseOld, bool needCoarseNew);
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

     // use this to get the progress ratio of an AMR subcycle
     double getSubCycleProgress(DataWarehouse* fineNewDW);


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
     virtual void readFromTimestepXML(const ProblemSpecP&,SimulationStateP& state) {};
 
   private:
     SimulationInterface(const SimulationInterface&);
     SimulationInterface& operator=(const SimulationInterface&);
   };
} // End namespace Uintah
   


#endif
