#ifndef UINTAH_GRID_ThermalBoundCond_H
#define UINTAH_GRID_ThermalBoundCond_H

#include <Packages/Uintah/Core/Grid/BoundCond.h>

namespace Uintah {
   
/**************************************

CLASS
   ThermalBoundCond
   
   
GENERAL INFORMATION

   ThermalBoundCond.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   ThermalBoundCond

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class ThermalBoundCond : public BoundCond  {
   public:
      ThermalBoundCond() {};
      virtual ~ThermalBoundCond() {};
      virtual std::string getType() const = 0;
         
   private:
#if 0
      ThermalBoundCond(const ThermalBoundCond&);
      ThermalBoundCond& operator=(const ThermalBoundCond&);
#endif
      
   };
} // End namespace Uintah

#endif
