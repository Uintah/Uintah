#ifndef UINTAH_HOMEBREW_SIMULATIONCONTROLLER_H
#define UINTAH_HOMEBREW_SIMULATIONCONTROLLER_H

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

class CFDInterface;
class MPMInterface;
class MPMCFDInterface;
  class Output;

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
    
   class SimulationController : public UintahParallelComponent {
   public:
      SimulationController(const ProcessorGroup* myworld);
      virtual ~SimulationController();

      void doRestart(std::string restartFromDir, int timestep,
		     bool fromScratch, bool removeOldDir);
      void run();

      // for calculating memory usage when sci-malloc is disabled.
      static char* start_addr;
   private:
      void problemSetup(const ProblemSpecP&, GridP&);
      void scheduleInitialize(LevelP&, SchedulerP&,
			      CFDInterface*,
			      MPMInterface*,
			      MPMCFDInterface*);
      void scheduleComputeStableTimestep(LevelP&, SchedulerP&,
					 CFDInterface*,
					 MPMInterface*,
					 MPMCFDInterface*);
      void scheduleTimeAdvance(LevelP&, SchedulerP&,
			       CFDInterface*, MPMInterface*,
			       MPMCFDInterface*);

     bool need_recompile(double t, double delt, const LevelP& level,
			 CFDInterface* cfd, MPMInterface* mpm,
			 MPMCFDInterface* mpmcfd,  Output* output);
      SimulationController(const SimulationController&);
      SimulationController& operator=(const SimulationController&);

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
