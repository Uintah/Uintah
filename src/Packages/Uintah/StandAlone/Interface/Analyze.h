#ifndef UINTAH_HOMEBREW_ANALYZE_H
#define UINTAH_HOMEBREW_ANALYZE_H

#include <Uintah/Parallel/UintahParallelPort.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Handle.h>

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
				  const LevelP& level, SchedulerP&,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw) = 0;

      virtual void showStepInformation() const = 0;
      
   private:
      Analyze(const Analyze&);
      Analyze& operator=(const Analyze&);
   };

} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/09/04 23:20:35  tan
// Control the information showing at each step in SimulationController by
// Analyze module.
//
// Revision 1.2  2000/09/04 00:37:49  tan
// Modified Analyze interface for scientific debugging under both
// sigle processor and mpi environment.
//
// Revision 1.1  2000/07/17 23:37:26  tan
// Added Analyze interface that will be especially useful for debugging
// on scitific results.
//

#endif
