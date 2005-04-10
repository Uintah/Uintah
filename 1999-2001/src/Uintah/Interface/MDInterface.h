#ifndef UINTAH_HOMEBREW_MDInterface_H
#define UINTAH_HOMEBREW_MDInterface_H

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
   MDInterface
   
   Short description...

GENERAL INFORMATION

   MDInterface.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   MD_Interface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class MDInterface : public UintahParallelPort {
   public:
      MDInterface();
      virtual ~MDInterface();
      
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
      MDInterface(const MDInterface&);
      MDInterface& operator=(const MDInterface&);
   };
   
} // end namespace Uintah

#endif

//
// $Log$
// Revision 1.1  2000/06/09 16:33:17  tan
// Created MDInterface for molecular dynamics simulation.
//
