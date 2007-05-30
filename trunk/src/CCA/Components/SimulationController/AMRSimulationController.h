#ifndef UINTAH_HOMEBREW_AMRSIMULATIONCONTROLLER_H
#define UINTAH_HOMEBREW_AMRSIMULATIONCONTROLLER_H

#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Ports/SchedulerP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/SimulationController/SimulationController.h>
#include <Core/Grid/Variables/ComputeSet.h>

#include <CCA/Components/SimulationController/share.h>
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
   class SCISHARE AMRSimulationController : public SimulationController {
   public:
      AMRSimulationController(const ProcessorGroup* myworld, bool doAMR, ProblemSpecP pspec);
      virtual ~AMRSimulationController();

      virtual void run();

   private:
      //! Set up, compile, and execute initial timestep
      void doInitialTimestep(GridP& grid, double& t);

      bool doRegridding(GridP& grid, bool initialTimestep);

      void recompile(double t, double delt, GridP& currentGrid, int totalFine);

      void executeTimestep(double t, double& delt, GridP& currentGrid, int totalFine);

      //! Asks a variety of components if one of them needs the taskgraph
      //! to recompile.
      bool needRecompile(double t, double delt, const GridP& level);
      AMRSimulationController(const AMRSimulationController&);
      AMRSimulationController& operator=(const AMRSimulationController&);

      //! recursively schedule refinement, coarsening, and time advances for
      //! finer levels - compensating for time refinement.  Builds one taskgraph
      void subCycleCompile(GridP& grid, int startDW, int dwStride, int step, int numLevel);

      //! recursively executes taskgraphs, as several were executed.  Similar to subCycleCompile,
      //! except that this executes the recursive taskgraphs, and compile builds one taskgraph
      //! (to exsecute once) recursively.
      void subCycleExecute(GridP& grid, int startDW, int dwStride, int numLevel, bool rootCycle);

      void scheduleComputeStableTimestep(const GridP& grid,
                                         SchedulerP&);
      void coarsenDelt(const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset* /*matls*/,
                       DataWarehouse* /*old_dw*/,
                       DataWarehouse* new_dw);

   };

} // End namespace Uintah

#endif
