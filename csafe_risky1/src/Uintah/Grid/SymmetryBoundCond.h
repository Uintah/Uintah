#ifndef UINTAH_GRID_SymmetryBoundCond_H
#define UINTAH_GRID_SymmetryBoundCond_H

#include <Uintah/Grid/BoundCond.h>
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

   class SymmetryBoundCond : public BoundCond  {
   public:
      SymmetryBoundCond() {};
      SymmetryBoundCond(ProblemSpecP& ps) {};
      virtual ~SymmetryBoundCond() {};
      virtual std::string getType() const {
	return "Symmetry";
      };
         
   private:
      SymmetryBoundCond(const SymmetryBoundCond&);
      SymmetryBoundCond& operator=(const SymmetryBoundCond&);
      
     
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/06/27 22:31:50  jas
// Grid boundary conditions that are stored at the patch level.
//

#endif




