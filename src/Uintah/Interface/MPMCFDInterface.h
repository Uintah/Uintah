#ifndef UINTAH_HOMEBREW_MPMCFDInterface_H
#define UINTAH_HOMEBREW_MPMCFDInterface_H

#include <Uintah/Parallel/UintahParallelPort.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/Handle.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/SchedulerP.h>

namespace Uintah {

/**************************************

CLASS
   MPMCFDInterface
   
   Short description...

GENERAL INFORMATION

   MPMCFDInterface.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   MPM_CFD_Interface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class MPMCFDInterface : public UintahParallelPort {
   public:
      MPMCFDInterface();
      virtual ~MPMCFDInterface();
      
      //////////
      // Insert Documentation Here:
      virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
				SimulationStateP& state) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleInitialize(const LevelP& level,
				      SchedulerP&,
				      DataWarehouseP&) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleComputeStableTimestep(const LevelP& level,
						 SchedulerP&,
						 DataWarehouseP&) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleTimeAdvance(double t, double dt,
				       const LevelP& level, SchedulerP&,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw) = 0;
   private:
      MPMCFDInterface(const MPMCFDInterface&);
      MPMCFDInterface& operator=(const MPMCFDInterface&);
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/12/01 23:02:37  guilkey
// Adding stuff for coupled MPM and CFD.
//

#endif
