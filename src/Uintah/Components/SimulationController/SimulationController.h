#ifndef UINTAH_HOMEBREW_SIMULATIONCONTROLLER_H
#define UINTAH_HOMEBREW_SIMULATIONCONTROLLER_H

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Interface/ProblemSpecP.h>

namespace Uintah {
   class CFDInterface;
   class MPMInterface;
   class MDInterface;

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
      
      void run();
      
   private:
      void problemSetup(const ProblemSpecP&, GridP&);
      void scheduleInitialize(LevelP&, SchedulerP&,
			      DataWarehouseP&,
			      CFDInterface*, MPMInterface*, MDInterface*);
      void scheduleComputeStableTimestep(LevelP&, SchedulerP&,
					 DataWarehouseP&, CFDInterface*,
					 MPMInterface*, MDInterface*);
      void scheduleTimeAdvance(double t, double delt, LevelP&, SchedulerP&,
			       DataWarehouseP& old_ds,
			       DataWarehouseP& new_ds,
			       CFDInterface*, MPMInterface*, MDInterface*);
      
      SimulationController(const SimulationController&);
      SimulationController& operator=(const SimulationController&);
      
      bool           d_restarting;
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.14  2000/08/24 20:51:47  dav
// Removed DWMpiHandler.
//
// Revision 1.13  2000/07/28 07:37:50  bbanerje
// Rajesh must have missed these .. adding the changed version of
// createDataWarehouse calls
//
// Revision 1.12  2000/06/17 07:06:43  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.11  2000/06/09 17:06:17  tan
// Added MD(molecular dynamics) module to SimulationController.
//
// Revision 1.10  2000/05/11 20:10:20  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.9  2000/04/26 06:48:36  sparker
// Streamlined namespaces
//
// Revision 1.8  2000/04/20 18:56:28  sparker
// Updates to MPM
//
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
