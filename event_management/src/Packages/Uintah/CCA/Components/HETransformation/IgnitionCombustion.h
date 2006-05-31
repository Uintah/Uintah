#ifndef __IGNITION_COMBUSTION_H__
#define __IGNITION_COMBUSTION_H__

#include <Packages/Uintah/CCA/Components/HETransformation/Burn.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <math.h>

namespace Uintah {
/**************************************

CLASS
   SimpleBurn
   
   Short description...

GENERAL INFORMATION

   SimpleBurn.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Burn_Model_Simple

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

    class IgnitionCombustion : public Burn { 
    private:
      double activationEnergy, preExponent;
      double thresholdTemp, thresholdPressure, BurnCoeff;
      double Enthalpy;
      
      // Prevent copying of this class
      // copy constructor

      IgnitionCombustion(const IgnitionCombustion &burn);
      IgnitionCombustion & operator=(const IgnitionCombustion &burn);
      
    public:
      // Constructor
      IgnitionCombustion(ProblemSpecP& ps);
      
      // Destructor
      ~IgnitionCombustion();

      void computeBurn(double gasTemperature,
		       double gasPressure,
		       double materialMass,
		       double materialTemp,  // this doesn't make any sence
		       double &burnedMass,
		       double &releasedHeat,
		       double &delT,
		       double &surfaceArea);

       double getThresholdTemperature();

    };
} // End namespace Uintah
    


#endif /* __SIMPLE_BURN_H__*/

