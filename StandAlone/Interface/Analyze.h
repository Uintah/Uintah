#ifndef UINTAH_HOMEBREW_ANALYZE_H
#define UINTAH_HOMEBREW_ANALYZE_H

#include <Uintah/Parallel/UintahParallelPort.h>
#include <Uintah/Interface/DataWarehouseP.h>
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
      
      void setup( const Grid& grid,
                  const SimulationState& sharedState,
                  DataWarehouseP dw);
      
      virtual void performAnalyze() = 0;
      
   private:
      Analyze(const Analyze&);
      Analyze& operator=(const Analyze&);
      
   protected:
      const Grid*             d_grid;
      const SimulationState*  d_sharedState;
      DataWarehouseP          d_dw;
   };

} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/07/17 23:37:26  tan
// Added Analyze interface that will be especially useful for debugging
// on scitific results.
//

#endif
