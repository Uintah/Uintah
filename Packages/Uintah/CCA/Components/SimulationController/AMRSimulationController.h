#ifndef UINTAH_HOMEBREW_AMRSIMULATIONCONTROLLER_H
#define UINTAH_HOMEBREW_AMRSIMULATIONCONTROLLER_H

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Components/SimulationController/SimulationController.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>

namespace Uintah {

class SimulationInterface;
class Output;
class LoadBalancer;  
/**************************************
      
  CLASS
       AMRSimulationController
      
       Short description...
      
  GENERAL INFORMATION
      
       AMRSimulationController.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
       
       Copyright (C) 2000 SCI Group
      
  KEYWORDS
       Simulation_Controller
      
  DESCRIPTION
       Long description...
     
  WARNING
      
****************************************/
   class AMRSimulationController : public SimulationController {
   public:
      AMRSimulationController(const ProcessorGroup* myworld);
      virtual ~AMRSimulationController();

      virtual void doRestart(std::string restartFromDir, int timestep,
		     bool fromScratch, bool removeOldDir);
      virtual void run();

   private:
      bool needRecompile(double t, double delt, const GridP& level,
			 SimulationInterface* cfd, Output* output,
			 LoadBalancer* lb, std::vector<int>& levelids);
      AMRSimulationController(const AMRSimulationController&);
      AMRSimulationController& operator=(const AMRSimulationController&);

      void subCycle(GridP& grid, SchedulerP& scheduler,
		    SimulationStateP& sharedState,
		    int startDW, int dwStride, int numLevel,
		    SimulationInterface* sim);

      void initializeErrorEstimate(const ProcessorGroup*,
				   const PatchSubset* patches,
				   const MaterialSubset* matls,
				   DataWarehouse*, DataWarehouse* new_dw,
				   SimulationStateP sharedState);

      /* for restarting */
      bool           d_restarting;
      std::string d_restartFromDir;
      int d_restartTimestep;

      // If d_restartFromScratch is true then don't copy or move any of
      // the old timesteps or dat files from the old directory.  Run as
      // as if it were running from scratch but with initial conditions
      // given by the restart checkpoint.
      bool d_restartFromScratch;

      // If !d_restartFromScratch, then this indicates whether to move
      // or copy the old timesteps.
      bool d_restartRemoveOldDir;
   };

} // End namespace Uintah

#endif
