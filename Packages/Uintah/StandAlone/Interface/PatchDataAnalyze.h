#ifndef Packages/Uintah_PatchDataAnalyze
#define Packages/Uintah_PatchDataAnalyze

#include <Packages/Uintah/Interface/DataWarehouseP.h>
#include <Packages/Uintah/Interface/ProblemSpecP.h>
#include <Packages/Uintah/Grid/SimulationStateP.h>
#include <Packages/Uintah/Interface/SchedulerP.h>
#include <Packages/Uintah/Grid/GridP.h>

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
} // End namespace Uintah


#endif

