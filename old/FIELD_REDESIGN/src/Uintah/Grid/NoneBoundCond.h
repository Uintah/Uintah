#ifndef UINTAH_GRID_NoneBoundCond_H
#define UINTAH_GRID_NoneBoundCond_H

#include <Uintah/Grid/BoundCond.h>
#include <Uintah/Interface/ProblemSpecP.h>

using namespace Uintah;

namespace Uintah {

   
/**************************************

CLASS
   NoneBoundCond
   
   
GENERAL INFORMATION

   NoneBoundCond.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   NoneBoundCond

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class NoneBoundCond : public BoundCond  {
   public:
      NoneBoundCond() {};
      NoneBoundCond(ProblemSpecP& ps) {};
      virtual ~NoneBoundCond() {};
      virtual std::string getType() const {
	return "None";
      };
         
   private:
      NoneBoundCond(const NoneBoundCond&);
      NoneBoundCond& operator=(const NoneBoundCond&);
      
     
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/06/27 22:31:50  jas
// Grid boundary conditions that are stored at the patch level.
//

#endif




