#ifndef UINTAH_GRID_FluxThermalBoundCond_H
#define UINTAH_GRID_FluxThermalBoundCond_H

#include <Uintah/Grid/ThermalBoundCond.h>
#include <Uintah/Interface/ProblemSpecP.h>

using namespace Uintah;

namespace Uintah {

   
/**************************************

CLASS
   FluxThermalBoundCond
   
   
GENERAL INFORMATION

   FluxThermalBoundCond.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   FluxThermalBoundCond

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class FluxThermalBoundCond : public ThermalBoundCond  {
   public:
      FluxThermalBoundCond(double& t);
      FluxThermalBoundCond(ProblemSpecP &ps);
      virtual ~FluxThermalBoundCond();
      virtual std::string getType() const;
      
      double getFlux() const;
      
   private:
      FluxThermalBoundCond(const FluxThermalBoundCond&);
      FluxThermalBoundCond& operator=(const FluxThermalBoundCond&);

      double d_flux;
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/06/27 22:31:49  jas
// Grid boundary conditions that are stored at the patch level.
//

#endif




