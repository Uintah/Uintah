#ifndef UINTAH_GRID_TemperatureBoundCond_H
#define UINTAH_GRID_TemperatureBoundCond_H

#include <Uintah/Grid/BoundCond.h>
#include <Uintah/Interface/ProblemSpecP.h>

using namespace Uintah;

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
      TemperatureBoundCond(const TemperatureBoundCond&);
      TemperatureBoundCond& operator=(const TemperatureBoundCond&);

      double d_temp;
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/11/02 21:25:55  jas
// Rearranged the boundary conditions so there is consistency between ICE
// and MPM.  Added fillFaceFlux for the Neumann BC condition.  BCs are now
// declared differently in the *.ups file.
//


#endif




