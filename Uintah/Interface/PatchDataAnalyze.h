#ifndef Uintah_PatchDataAnalyze
#define Uintah_PatchDataAnalyze

#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Grid/GridP.h>

namespace Uintah {

/**************************************

CLASS
   PatchDataAnalyze
   
   Short description...

GENERAL INFORMATION

   PatchDataAnalyze.h

   Honglai Tan
   Department of Materials Science and Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   PatchDataAnalyze

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class Patch;

   class PatchDataAnalyze {
   public:      
      virtual void problemSetup(const ProblemSpecP& problemSpec,
                        GridP& grid,
			SimulationStateP& state) = 0;
      
      virtual void performAnalyze(
				  const Patch* patch,
				  SchedulerP&,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw) = 0;      
   private:
   };

} // end namespace Uintah

#endif

//
// $Log$
// Revision 1.1  2000/11/21 23:52:23  tan
// Implemented different models for fracture simulations.  SimpleFracture model
// is for the simulation where the resolution focus only on macroscopic major
// cracks. NormalFracture and ExplosionFracture models are more sophiscated
// and specific fracture models that are currently underconstruction.
//
