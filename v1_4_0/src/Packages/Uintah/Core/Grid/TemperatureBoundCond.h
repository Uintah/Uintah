#ifndef UINTAH_GRID_TemperatureBoundCond_H
#define UINTAH_GRID_TemperatureBoundCond_H

#include <Packages/Uintah/Core/Grid/BoundCond.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {
   
/**************************************

CLASS
   TemperatureBoundCond
   
   
GENERAL INFORMATION

   TemperatureBoundCond.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   TemperatureBoundCond

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class TemperatureBoundCond : public BoundCond<double>  {
   public:
      TemperatureBoundCond(ProblemSpecP& ps,std::string& kind);
      virtual ~TemperatureBoundCond();
      virtual double getValue() const;
      
   private:
#if 0
      TemperatureBoundCond(const TemperatureBoundCond&);
      TemperatureBoundCond& operator=(const TemperatureBoundCond&);
#endif

      double d_temp;
   };
} // End namespace Uintah

#endif
