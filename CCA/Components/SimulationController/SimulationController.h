#ifndef UINTAH_HOMEBREW_SIMULATIONCONTROLLER_H
#define UINTAH_HOMEBREW_SIMULATIONCONTROLLER_H

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
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

   //! The main component that controls the execution of the 
   //! entire simulation.
   class SimulationController : public UintahParallelComponent {
   public:
      SimulationController(const ProcessorGroup* myworld);
      virtual ~SimulationController();

      //! Notifies (before calling run) the SimulationController
      //! that this is simulation is a restart.
      virtual void doRestart(std::string restartFromDir, int timestep,
		     bool fromScratch, bool removeOldDir) = 0;

      //! Execute the simulation
      virtual void run() = 0;

      // notifies (before calling run) the simulationController
      //! that this run is a combinePatches run.
      virtual void doCombinePatches(std::string fromDir);
     
      // for calculating memory usage when sci-malloc is disabled.
      static char* start_addr;

   protected:

      double getWallTime     ( void );
      void   calcWallTime    ( void );

      double getStartTime    ( void );
      void   calcStartTime   ( void );
      void   setStartSimTime ( double t );

      void initSimulationStatsVars ( void );
      void printSimulationStats    ( Uintah::SimulationStateP sharedState, double delt, double time );

   private:

      int    d_n;
      double d_wallTime;              // current wall time
      double d_startTime;             // starting wall time
      double d_startSimTime;          // starting sim time
      double d_prevWallTime;
      double d_sumOfWallTimes;
      double d_sumOfWallTimeSquares;
      
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
