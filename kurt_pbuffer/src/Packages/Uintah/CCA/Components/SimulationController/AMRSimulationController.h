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
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>

namespace Uintah {

class SimulationInterface;
class Output;
class LoadBalancer;  
class Regridder;
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

   //! Controls the execution of an AMR Simulation
   class AMRSimulationController : public SimulationController {
   public:
      AMRSimulationController(const ProcessorGroup* myworld, bool doAMR);
      virtual ~AMRSimulationController();

      virtual void run();

   private:
      //! Set up, compile, and execute initial timestep
      void doInitialTimestep(GridP& grid, double& t);

      //! Does regridding based on initialization timestep
      //! Return true if a new grid is created.
      bool doInitialTimestepRegridding(GridP& grid);

      void doRegridding(GridP& grid);

      void recompile(double t, double delt, GridP& currentGrid, int totalFine);

      void executeTimestep(double t, double& delt, GridP& currentGrid, int totalFine);

      //! Asks a variety of components if one of them needs the taskgraph
      //! to recompile.
      bool needRecompile(double t, double delt, const GridP& level);
      AMRSimulationController(const AMRSimulationController&);
      AMRSimulationController& operator=(const AMRSimulationController&);

      //! recursively schedule refinement, coarsening, and time advances for
      //! finer levels - compensating for time refinement.
      void subCycle(GridP& grid, int startDW, int dwStride, int numLevel);
   };

} // End namespace Uintah

#endif
