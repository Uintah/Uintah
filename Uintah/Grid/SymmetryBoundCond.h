#ifndef UINTAH_GRID_SymmetryBoundCond_H
#define UINTAH_GRID_SymmetryBoundCond_H

#include <Uintah/Grid/BoundCondBase.h>
#include <Uintah/Interface/ProblemSpecP.h>
using namespace Uintah;

namespace Uintah {

   
/**************************************

CLASS
   SymmetryBoundCond
   
   
GENERAL INFORMATION

   SymmetryBoundCond.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SymmetryBoundCond

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class SymmetryBoundCond : public BoundCondBase  {
   public:
      SymmetryBoundCond():BoundCondBase("Symmetry") {};
      SymmetryBoundCond(ProblemSpecP&) {d_type = "Symmetric";};
      virtual ~SymmetryBoundCond() {};
         
   private:
      SymmetryBoundCond(const SymmetryBoundCond&);
      SymmetryBoundCond& operator=(const SymmetryBoundCond&);
      
     
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.4  2000/12/07 23:47:55  guilkey
// Inlined [] operator in Array3.  Fixed (I think) SymmetryBoundCond so
// it actually does something now.
//
// Revision 1.3  2000/11/02 21:25:55  jas
// Rearranged the boundary conditions so there is consistency between ICE
// and MPM.  Added fillFaceFlux for the Neumann BC condition.  BCs are now
// declared differently in the *.ups file.
//
// Revision 1.2  2000/10/14 17:09:59  sparker
// Added get() method to PerPatch
// Fixed warning in SymmetryBoundCond
//
// Revision 1.1  2000/06/27 22:31:50  jas
// Grid boundary conditions that are stored at the patch level.
//

#endif




