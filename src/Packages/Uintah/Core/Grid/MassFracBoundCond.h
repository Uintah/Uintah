#ifndef UINTAH_GRID_massFracBoundCond_H
#define UINTAH_GRID_massFracBoundCond_H

#include <Packages/Uintah/Core/Grid/BoundCond.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {
   
/**************************************

CLASS
   massFracBoundCond
   
   
GENERAL INFORMATION

   massFractionBoundCond.h

   Todd Harman
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   massFractBoundCond

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class MassFractionBoundCond : public BoundCond<double>  {
   public:
      MassFractionBoundCond(ProblemSpecP& ps,std::string& kind,
                            std::string& variableName);
      virtual ~MassFractionBoundCond();
      virtual MassFractionBoundCond* clone();
      
   };
} // End namespace Uintah

#endif
