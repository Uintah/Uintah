#ifndef UINTAH_HOMEBREW_SIMPLESIMULATIONCONTROLLER_H
#define UINTAH_HOMEBREW_SIMPLESIMULATIONCONTROLLER_H

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Components/SimulationController/SimulationController.h>

namespace Uintah {

class SimulationInterface;
class Output;
class LoadBalancer;

/**************************************
      
  CLASS
       SimulationController
      
       Short description...
      
  GENERAL INFORMATION
      
       SimulationController.h
      
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
    
   class SimpleSimulationController : public SimulationController {
   public:
      SimpleSimulationController(const ProcessorGroup* myworld);
      virtual ~SimpleSimulationController();

      virtual void doRestart(std::string restartFromDir, int timestep,
		     bool fromScratch, bool removeOldDir);
      virtual void doCombinePatches(std::string fromDir);
      virtual void run();

   private:
      bool needRecompile(double t, double delt, const GridP& grid,
			 SimulationInterface* cfd, Output* output,
			 LoadBalancer* lb);
      SimpleSimulationController(const SimpleSimulationController&);
      SimpleSimulationController& operator=(const SimpleSimulationController&);

      /* for restarting */
      bool           d_restarting;
      std::string d_fromDir;
      int d_restartTimestep;

      /* for patch combining mode */
      bool d_combinePatches;
      // also use d_fromDir
        
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
