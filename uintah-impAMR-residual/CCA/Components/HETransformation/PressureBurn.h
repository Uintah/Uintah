#ifndef __PRESSURE_BURN_H__
#define __PRESSURE_BURN_H__

#include <Packages/Uintah/CCA/Components/HETransformation/Burn.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <math.h>

namespace Uintah {
/**************************************

CLASS
   PressureBurn
   
   This is a minor derivative of the SimpleBurn class in which the user
   specifies both the pressure exponent and burn coefficient.

GENERAL INFORMATION

   PressureBurn.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Burn_Model_Pressure

DESCRIPTION
  This burn component operates by computing the steady combustion rate
  (mass transfer of propellant to gas) by an empirical pressure equation
  in which the surface regression rate r is proportional to P^n:
      r = A*P^n
  The pressure passed to this burn model is the actual pressure (e.g., 
  in Pascal) divided by the reference pressure defined by the ICE component.
  
WARNING
  
****************************************/

    class PressureBurn : public Burn { 
    private:
      double thresholdTemp, thresholdPressure, Enthalpy, BurnCoeff, pressureExponent, refPressure;
      
      // Prevent copying of this class
      // copy constructor

      PressureBurn(const PressureBurn &burn);
      PressureBurn & operator=(const PressureBurn &burn);
      
    public:
      // Constructor
      PressureBurn(ProblemSpecP& ps);
      
      // Destructor
      ~PressureBurn();

      void computeBurn(double gasTemperature,
		       double gasPressure,
		       double materialMass,
		       double materialPressure,
		       double &burnedMass,
		       double &releasedHeat,
		       double &delT,
		       double &surfaceArea);

       double getThresholdTemperature();

    };
} // End namespace Uintah
    


#endif /* __PRESSURE_BURN_H__*/

