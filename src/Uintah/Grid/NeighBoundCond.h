#ifndef UINTAH_GRID_NeighBoundCond_H
#define UINTAH_GRID_NeighBoundCond_H

#include <Uintah/Grid/BoundCondBase.h>
#include <Uintah/Interface/ProblemSpecP.h>
using namespace Uintah;

namespace Uintah {

   
/**************************************

CLASS
   NeighBoundCond
   
   
GENERAL INFORMATION

   NeighBoundCond.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   NeighBoundCond

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class NeighBoundCond : public BoundCondBase {
   public:
      NeighBoundCond() : BoundCondBase("Neighbor") {}; 
      NeighBoundCond(ProblemSpecP& ps) {};
      virtual ~NeighBoundCond() {};

   private:
      NeighBoundCond(const NeighBoundCond&);
      NeighBoundCond& operator=(const NeighBoundCond&);
      
     
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/11/02 21:25:55  jas
// Rearranged the boundary conditions so there is consistency between ICE
// and MPM.  Added fillFaceFlux for the Neumann BC condition.  BCs are now
// declared differently in the *.ups file.
//
// Revision 1.1  2000/06/27 22:31:50  jas
// Grid boundary conditions that are stored at the patch level.
//

#endif




