#ifndef UINTAH_HOMEBREW_SIMULATIONCONTROLLER_H
#define UINTAH_HOMEBREW_SIMULATIONCONTROLLER_H

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Interface/ProblemSpecP.h>

namespace Uintah {

namespace Grid {
  class VarLabel;
}

namespace Interface {
  class CFDInterface;
  class MPMInterface;
}

namespace Components {
      
using Uintah::Parallel::UintahParallelComponent;
using Uintah::Interface::ProblemSpecP;
using Uintah::Grid::LevelP;
using Uintah::Grid::GridP;
using Uintah::Grid::VarLabel;
using Uintah::Interface::SchedulerP;
using Uintah::Interface::DataWarehouseP;
using Uintah::Interface::MPMInterface;
using Uintah::Interface::CFDInterface;
      
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
  SimulationController( int MpiRank, int MpiProcesses);
  virtual ~SimulationController();
      
  void run();

private:
  void problemSetup(const ProblemSpecP&, GridP&);
  void scheduleInitialize(LevelP&, SchedulerP&,
			  DataWarehouseP&,
			  CFDInterface*, MPMInterface*);
  void scheduleComputeStableTimestep(LevelP&, SchedulerP&,
				     DataWarehouseP&, const VarLabel*,
				     CFDInterface*, MPMInterface*);
  void scheduleTimeAdvance(double t, double delt, LevelP&, SchedulerP&,
			   const DataWarehouseP&, DataWarehouseP&,
			   CFDInterface*, MPMInterface*);
	 
  SimulationController(const SimulationController&);
  SimulationController& operator=(const SimulationController&);

  bool restarting;
  const VarLabel* delt_label;
};
      
} // end namespace Components
} // end namespace Uintah

//
// $Log$
// Revision 1.7  2000/04/19 20:59:25  dav
// adding MPI support
//
// Revision 1.6  2000/04/19 05:26:13  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.5  2000/04/13 06:50:59  sparker
// More implementation to get this to work
//
// Revision 1.4  2000/03/23 20:42:17  sparker
// Added copy ctor to exception classes (for Linux/g++)
// Helped clean up move of ProblemSpec from Interface to Grid
//
// Revision 1.3  2000/03/17 20:58:31  dav
// namespace updates
//
//

#endif
