#ifndef UINTAH_GRID_NoneBoundCond_H
#define UINTAH_GRID_NoneBoundCond_H

#include <Packages/Uintah/Core/Grid/BoundCondBase.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

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

   class NoneBoundCond : public BoundCondBase  {
   public:
      NoneBoundCond() : BoundCondBase("None") {};
      NoneBoundCond(ProblemSpecP& ps){};
      virtual ~NoneBoundCond() {};
               
   private:
      NoneBoundCond(const NoneBoundCond&);
      NoneBoundCond& operator=(const NoneBoundCond&);
      
     
   };
} // End namespace Uintah

#endif
