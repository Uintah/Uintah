#ifndef UINTAH_HOMEBREW_ANALYZE_H
#define UINTAH_HOMEBREW_ANALYZE_H

#include <Packages/Uintah/Core/Parallel/UintahParallelPort.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/ProblemSpec/Handle.h>

namespace Uintah {
/**************************************

CLASS
   Analyze
   
   Short description...

GENERAL INFORMATION

   Analyze.h

   Honglai Tan
   Department of Materials Science and Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Analyze

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class Grid;
   class SimulationState;

   class Analyze : public UintahParallelPort {
   public:
      Analyze();
      virtual ~Analyze();
      
      virtual void problemSetup(const ProblemSpecP& problemSpec,
                        GridP& grid,
			SimulationStateP& state) = 0;
      
      virtual void performAnalyze(double t, double dt,
				  const LevelP& level, SchedulerP&) = 0;

      virtual void showStepInformation() const = 0;
      
   private:
      Analyze(const Analyze&);
      Analyze& operator=(const Analyze&);
   };
} // End namespace Uintah

#endif
