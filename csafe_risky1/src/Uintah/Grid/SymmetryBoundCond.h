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
      SymmetryBoundCond() {}
      SymmetryBoundCond(ProblemSpecP&) {}
      virtual ~SymmetryBoundCond() {}
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
// Revision 1.1.4.2  2000/10/19 05:18:04  sparker
// Merge changes from main branch into csafe_risky1
//
// Revision 1.1.4.1  2000/10/10 05:28:08  sparker
// Added support for NullScheduler (used for profiling taskgraph overhead)
//
// Revision 1.2  2000/10/14 17:09:59  sparker
// Added get() method to PerPatch
// Fixed warning in SymmetryBoundCond
//
// Revision 1.1  2000/06/27 22:31:50  jas
// Grid boundary conditions that are stored at the patch level.
//

#endif




