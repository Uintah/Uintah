#ifndef UINTAH_GRID_SpecificVolBoundCond_H
#define UINTAH_GRID_SpecificVolBoundCond_H

#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCond.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {
   
/**************************************

CLASS
   SpecificVolBoundCond
   
   
GENERAL INFORMATION

   SpecificVolBoundCond.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SpecificVolBoundCond

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class SpecificVolBoundCond : public BoundCond<double>  {
   public:
     SpecificVolBoundCond(ProblemSpecP& ps,std::string& kind);
     virtual ~SpecificVolBoundCond();
     virtual SpecificVolBoundCond* clone();
   };
} // End namespace Uintah

#endif
