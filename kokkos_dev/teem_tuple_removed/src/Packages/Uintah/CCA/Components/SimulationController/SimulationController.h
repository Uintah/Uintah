#ifndef UINTAH_HOMEBREW_SIMULATIONCONTROLLER_H
#define UINTAH_HOMEBREW_SIMULATIONCONTROLLER_H

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

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
       Abstract baseclass for the SimulationControllers.
       Introduced to make the "old" SimulationController
       and the new AMRSimulationController interchangeable.
     
  WARNING
      
****************************************/
    
   class SimulationController : public UintahParallelComponent {
   public:
      SimulationController(const ProcessorGroup* myworld);
      virtual ~SimulationController();

      virtual void doRestart(std::string restartFromDir, int timestep,
		     bool fromScratch, bool removeOldDir) = 0;
      virtual void run() = 0;

      // default just gives an error
      virtual void doCombinePatches(std::string fromDir);
     
      // for calculating memory usage when sci-malloc is disabled.
      static char* start_addr;
   private:
   /*
      void problemSetup(const ProblemSpecP&, GridP&) = 0;
      bool needRecompile(double t, double delt, const LevelP& level,
			  SimulationInterface* cfd, Output* output,
			  LoadBalancer* lb) = 0;
      SimulationController(const SimulationController&) = 0;
      SimulationController& operator=(const SimulationController&) = 0;
      */
   };

} // End namespace Uintah

#endif
