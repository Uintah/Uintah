#ifndef UINTAH_PATCHDATAANALYZE_H
#define UINTAH_PATCHDATAANALYZE_H

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>

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
      
      virtual void performAnalyze(SchedulerP& sched,
                         const PatchSet* patches,
			 const MaterialSet* matls) = 0;
   private:
   };
} // End namespace Uintah

#endif
