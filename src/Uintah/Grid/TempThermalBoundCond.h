#ifndef UINTAH_GRID_TempThermalBoundCond_H
#define UINTAH_GRID_TempThermalBoundCond_H

#include <Uintah/Grid/ThermalBoundCond.h>
#include <Uintah/Interface/ProblemSpecP.h>

using namespace Uintah;

namespace Uintah {

   
/**************************************

CLASS
   TempThermalBoundCond
   
   
GENERAL INFORMATION

   TempThermalBoundCond.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   TempThermalBoundCond

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class TempThermalBoundCond : public ThermalBoundCond  {
   public:
      TempThermalBoundCond(double& t);
      TempThermalBoundCond(ProblemSpecP& ps);
      virtual ~TempThermalBoundCond();
      virtual std::string getType() const;
      
      double getTemp() const;
      
   private:
      TempThermalBoundCond(const TempThermalBoundCond&);
      TempThermalBoundCond& operator=(const TempThermalBoundCond&);

      double d_temp;
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/06/27 22:31:50  jas
// Grid boundary conditions that are stored at the patch level.
//

#endif




