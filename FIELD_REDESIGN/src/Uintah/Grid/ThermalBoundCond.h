#ifndef UINTAH_GRID_ThermalBoundCond_H
#define UINTAH_GRID_ThermalBoundCond_H

#include <Uintah/Grid/BoundCond.h>

using namespace Uintah;

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
      ThermalBoundCond(const ThermalBoundCond&);
      ThermalBoundCond& operator=(const ThermalBoundCond&);
      
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/06/27 22:31:50  jas
// Grid boundary conditions that are stored at the patch level.
//

#endif




